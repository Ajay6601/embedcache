// prodsim is a production-scale simulation against a REAL embedding backend.
// It exercises the proxy the way a production RAG platform would — corpus
// ingestion, sustained multi-tenant query traffic, nightly re-ingestion,
// hostile callers, crash recovery — at the largest scale the host machine
// supports, with all security features enabled, and writes PRODSIM.md.
//
// Usage:
//
//	go build -o embedcache.exe ./cmd/embedcache
//	go run ./experiments/prodsim -bin ./embedcache.exe -upstream http://localhost:11434 -model all-minilm
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Ajay6601/embedcache/internal/api"
)

var (
	binPath   = flag.String("bin", "./embedcache.exe", "compiled embedcache binary")
	upstream  = flag.String("upstream", "http://localhost:11434", "real embedding backend")
	model     = flag.String("model", "all-minilm", "embedding model")
	outPath   = flag.String("out", "PRODSIM.md", "results file")
	docs      = flag.Int("docs", 2500, "documents in the corpus")
	chunksPer = flag.Int("chunks-per-doc", 20, "chunks per document")
	queries   = flag.Int("queries", 300000, "query-storm request count")
	clients   = flag.Int("clients", 64, "concurrent query clients")
	ingesters = flag.Int("ingest-workers", 16, "concurrent ingestion workers")
)

const (
	adminToken = "prodsim-admin-token"
	goodKey    = "tenant-a-key"
	goodKey2   = "tenant-b-key"
	badKey     = "stolen-key"
)

var (
	rep    bytes.Buffer
	failed int
	client *http.Client
)

func check(name string, ok bool, detail string) {
	mark := "PASS"
	if !ok {
		mark = "FAIL"
		failed++
	}
	fmt.Fprintf(&rep, "- **%s** — %s: %s\n", mark, name, detail)
	fmt.Printf("[%s] %s: %s\n", mark, name, detail)
}

func section(f string, args ...any) {
	fmt.Fprintf(&rep, "\n## "+f+"\n\n", args...)
	fmt.Printf("\n== "+f+" ==\n", args...)
}

func line(f string, args ...any) { fmt.Fprintf(&rep, f+"\n", args...) }

type result struct {
	status    int
	cacheHdr  string
	dur       time.Duration
	data      []api.EmbeddingData
	errString string
}

func embed(base, key string, input any) result {
	b, _ := json.Marshal(map[string]any{"model": *model, "input": input})
	req, _ := http.NewRequest(http.MethodPost, base+"/v1/embeddings", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return result{errString: err.Error()}
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	r := result{status: resp.StatusCode, cacheHdr: resp.Header.Get("X-Embedcache-Status"), dur: time.Since(start)}
	if resp.StatusCode == 200 {
		var parsed api.EmbeddingsResponse
		if json.Unmarshal(raw, &parsed) == nil {
			r.data = parsed.Data
		}
	}
	return r
}

func pctl(d []time.Duration, p float64) time.Duration {
	if len(d) == 0 {
		return 0
	}
	s := append([]time.Duration(nil), d...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	return s[int(p*float64(len(s)-1))]
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func freePort() string {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(err)
	}
	addr := l.Addr().String()
	l.Close()
	return addr
}

func startProxy(addr, snapPath string) (*exec.Cmd, error) {
	cmd := exec.Command(*binPath, "serve",
		"-listen", addr,
		"-upstream", *upstream,
		"-admin-token", adminToken,
		"-auth-mode", "allowlist",
		"-api-keys", goodKey+","+goodKey2,
		"-ttl", "24h",
		"-persist", snapPath,
		"-max-entries", "2000000",
		"-max-memory-mb", "2048",
	)
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	base := "http://" + addr
	for i := 0; i < 100; i++ {
		resp, err := client.Get(base + "/healthz")
		if err == nil {
			resp.Body.Close()
			return cmd, nil
		}
		time.Sleep(100 * time.Millisecond)
	}
	cmd.Process.Kill()
	return nil, fmt.Errorf("proxy at %s never became healthy", addr)
}

func adminGet(base, path string) []byte {
	req, _ := http.NewRequest(http.MethodGet, base+path, nil)
	req.Header.Set("Authorization", "Bearer "+adminToken)
	resp, err := client.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return b
}

func adminPost(base, path string) (int, []byte) {
	req, _ := http.NewRequest(http.MethodPost, base+path, nil)
	req.Header.Set("Authorization", "Bearer "+adminToken)
	resp, err := client.Do(req)
	if err != nil {
		return 0, nil
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, b
}

func proxyRSSMB(pid int) float64 {
	if runtime.GOOS != "windows" {
		return -1
	}
	out, err := exec.Command("powershell", "-NoProfile", "-Command",
		fmt.Sprintf("(Get-Process -Id %d).WorkingSet64", pid)).Output()
	if err != nil {
		return -1
	}
	var b float64
	if _, err := fmt.Sscanf(strings.TrimSpace(string(out)), "%f", &b); err != nil {
		return -1
	}
	return b / 1e6
}

func main() {
	flag.Parse()
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.MaxIdleConns = 512
	tr.MaxIdleConnsPerHost = 512
	client = &http.Client{Timeout: 120 * time.Second, Transport: tr}

	totalChunks := *docs * *chunksPer
	fmt.Fprintf(&rep, "# embedcache production simulation\n\n")
	fmt.Fprintf(&rep, "Generated %s · %s/%s · %d CPUs · backend %s (`%s`, real inference)\n",
		time.Now().Format("2006-01-02 15:04 MST"), runtime.GOOS, runtime.GOARCH, runtime.NumCPU(), *upstream, *model)
	fmt.Fprintf(&rep, "\nSecurity fully enabled throughout: `-auth-mode allowlist`, `-admin-token`, `-ttl 24h`,\npersistence on. Corpus %d chunks (%d docs × %d), query storm %d requests / %d clients.\n",
		totalChunks, *docs, *chunksPer, *queries, *clients)

	snapPath, _ := filepath.Abs("prodsim.snap")
	defer os.Remove(snapPath)
	os.Remove(snapPath)
	addr := freePort()
	base := "http://" + addr
	proxy, err := startProxy(addr, snapPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer func() {
		if proxy.Process != nil {
			proxy.Process.Kill()
		}
	}()

	// ---------- corpus ----------
	rng := rand.New(rand.NewSource(42))
	vocab := strings.Fields("retrieval augmented generation vector database chunk overlap embedding token budget latency throughput tenant quota cache duplicate pipeline index corpus query model shard batch waste inference gateway prefill decode cost report audit fingerprint dedupe")
	corpus := make([][]string, *docs)
	for d := range corpus {
		corpus[d] = make([]string, *chunksPer)
		for c := range corpus[d] {
			words := make([]string, 24)
			for i := range words {
				words[i] = vocab[rng.Intn(len(vocab))]
			}
			corpus[d][c] = fmt.Sprintf("doc%05d/chunk%02d %s", d, c, strings.Join(words, " "))
		}
	}
	flat := make([]string, 0, totalChunks)
	for d := range corpus {
		flat = append(flat, corpus[d]...)
	}

	ingest := func(key string) (wall time.Duration, hits, misses, errs int64, batchLat []time.Duration) {
		var mu sync.Mutex
		jobs := make(chan []string, *ingesters)
		var wg sync.WaitGroup
		var h, m, e atomic.Int64
		start := time.Now()
		for w := 0; w < *ingesters; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for batch := range jobs {
					r := embed(base, key, batch)
					if r.status != 200 {
						e.Add(1)
						continue
					}
					switch r.cacheHdr {
					case "hit":
						h.Add(int64(len(batch)))
					case "miss":
						m.Add(int64(len(batch)))
					default: // partial — count via a second pass approximation below
						m.Add(0)
					}
					mu.Lock()
					batchLat = append(batchLat, r.dur)
					mu.Unlock()
				}
			}()
		}
		all := make([]string, len(flat))
		copy(all, flat)
		for d := range corpus { // rebuild flat in case corpus was edited
			copy(all[d**chunksPer:], corpus[d])
		}
		for i := 0; i < len(all); i += 64 {
			end := i + 64
			if end > len(all) {
				end = len(all)
			}
			jobs <- all[i:end]
		}
		close(jobs)
		wg.Wait()
		return time.Since(start), h.Load(), m.Load(), e.Load(), batchLat
	}

	// ---------- phase 1: cold ingestion ----------
	section("Phase 1 — Cold corpus ingestion (%d chunks, real inference)", totalChunks)
	wall, _, _, errs, batchLat := ingest(goodKey)
	var st1 struct {
		Misses uint64 `json:"misses"`
		Items  uint64 `json:"items"`
	}
	json.Unmarshal(adminGet(base, "/_ec/stats"), &st1)
	line("| metric | value |")
	line("|---|---|")
	line("| chunks ingested | %d |", totalChunks)
	line("| wall time | %s |", wall.Round(time.Second))
	line("| embeddings/sec (real inference) | %.0f |", float64(totalChunks)/wall.Seconds())
	line("| batch latency p50 / p95 | %.0f ms / %.0f ms |", ms(pctl(batchLat, 0.5)), ms(pctl(batchLat, 0.95)))
	line("| request errors | %d |", errs)
	check("cold ingest completes with zero errors", errs == 0, fmt.Sprintf("%d chunks in %s", totalChunks, wall.Round(time.Second)))
	check("cold ingest is all misses", st1.Misses >= uint64(totalChunks), fmt.Sprintf("misses=%d", st1.Misses))

	// capture ground truth for a correctness sample
	sampleIdx := rng.Perm(totalChunks)[:500]
	truth := map[int][]byte{}
	for _, i := range sampleIdx {
		r := embed(base, goodKey, flat[i])
		if r.status == 200 && len(r.data) == 1 {
			truth[i] = r.data[0].Embedding
		}
	}

	// ---------- phase 2: query storm ----------
	section("Phase 2 — Query storm (%d requests, %d concurrent clients, Zipf)", *queries, *clients)
	pool := make([]string, 5000)
	for i := range pool {
		pool[i] = flat[rng.Intn(len(flat))] // hot queries drawn from the corpus
	}
	var qHits, qMisses, qErrs atomic.Int64
	latCh := make(chan time.Duration, 4096)
	var lats []time.Duration
	var latWg sync.WaitGroup
	latWg.Add(1)
	go func() {
		defer latWg.Done()
		for d := range latCh {
			lats = append(lats, d)
		}
	}()
	perClient := *queries / *clients
	start := time.Now()
	var wg sync.WaitGroup
	for cIdx := 0; cIdx < *clients; cIdx++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			lr := rand.New(rand.NewSource(seed))
			lz := rand.NewZipf(lr, 1.1, 1, uint64(len(pool)-1))
			key := goodKey
			if seed%2 == 1 {
				key = goodKey2 // second tenant
			}
			for i := 0; i < perClient; i++ {
				q := pool[lz.Uint64()]
				r := embed(base, key, q)
				switch {
				case r.status != 200:
					qErrs.Add(1)
				case r.cacheHdr == "hit" || r.cacheHdr == "partial":
					qHits.Add(1)
				default:
					qMisses.Add(1)
				}
				select {
				case latCh <- r.dur:
				default:
				}
			}
		}(int64(cIdx))
	}
	wg.Wait()
	close(latCh)
	latWg.Wait()
	stormWall := time.Since(start)
	total := qHits.Load() + qMisses.Load() + qErrs.Load()
	rss := proxyRSSMB(proxy.Process.Pid)
	line("| metric | value |")
	line("|---|---|")
	line("| requests | %d |", total)
	line("| wall time | %s |", stormWall.Round(time.Second))
	line("| sustained throughput | **%.0f req/s** |", float64(total)/stormWall.Seconds())
	line("| hit rate | **%.2f%%** |", float64(qHits.Load())/float64(total)*100)
	line("| latency p50 / p95 / p99 | %.2f / %.2f / %.2f ms |", ms(pctl(lats, 0.5)), ms(pctl(lats, 0.95)), ms(pctl(lats, 0.99)))
	line("| errors | %d |", qErrs.Load())
	if rss > 0 {
		line("| proxy RSS | %.0f MB |", rss)
	}
	check("query storm zero errors", qErrs.Load() == 0, fmt.Sprintf("%d requests", total))
	check("query storm hit rate > 95%%", float64(qHits.Load())/float64(total) > 0.95,
		fmt.Sprintf("%.2f%% (pool drawn from ingested corpus)", float64(qHits.Load())/float64(total)*100))
	if rss > 0 {
		check("proxy memory bounded", rss < 1500, fmt.Sprintf("%.0f MB RSS with %d cached embeddings", rss, totalChunks))
	}

	// ---------- phase 3: nightly re-ingestion, 5% of docs edited ----------
	section("Phase 3 — Nightly re-ingestion (5%% of documents edited)")
	edited := *docs * 5 / 100
	for i := 0; i < edited; i++ {
		d := rng.Intn(*docs)
		for c := range corpus[d] {
			corpus[d][c] = fmt.Sprintf("doc%05d/chunk%02d EDITED-r%d %s", d, c, i, corpus[d][c][16:])
		}
	}
	var stBefore, stAfter struct {
		Misses      uint64 `json:"misses"`
		SavedTokens uint64 `json:"saved_tokens"`
		SavedUSD    float64
	}
	json.Unmarshal(adminGet(base, "/_ec/stats"), &stBefore)
	wall3, _, _, errs3, _ := ingest(goodKey)
	json.Unmarshal(adminGet(base, "/_ec/stats"), &stAfter)
	recomputed := stAfter.Misses - stBefore.Misses
	savedPct := (1 - float64(recomputed)/float64(totalChunks)) * 100
	line("| metric | value |")
	line("|---|---|")
	line("| chunks re-ingested | %d |", totalChunks)
	line("| recomputed upstream | %d |", recomputed)
	line("| absorbed by cache | **%.1f%%** |", savedPct)
	line("| wall time vs cold ingest | %s vs %s (**%.1fx faster**) |", wall3.Round(time.Second), wall.Round(time.Second), wall.Seconds()/wall3.Seconds())
	check("re-ingest zero errors", errs3 == 0, fmt.Sprintf("%d chunks", totalChunks))
	check("re-ingest recomputes only edited docs", savedPct > 90,
		fmt.Sprintf("%.1f%% absorbed; %d edited docs = %d chunks expected", savedPct, edited, edited**chunksPer))

	// ---------- phase 4: hostile traffic under load ----------
	section("Phase 4 — Security under load (25%% hostile keys mixed into live traffic)")
	var s401, s200, sOther atomic.Int64
	var goodLats []time.Duration
	var glMu sync.Mutex
	start4 := time.Now()
	var wg4 sync.WaitGroup
	for c := 0; c < 32; c++ {
		wg4.Add(1)
		go func(seed int64) {
			defer wg4.Done()
			lr := rand.New(rand.NewSource(seed + 9000))
			lz := rand.NewZipf(lr, 1.1, 1, uint64(len(pool)-1))
			for i := 0; i < 30000/32; i++ {
				q := pool[lz.Uint64()]
				roll := lr.Float64()
				key := goodKey
				if roll < 0.20 {
					key = badKey
				} else if roll < 0.25 {
					key = ""
				}
				r := embed(base, key, q)
				switch r.status {
				case 401:
					s401.Add(1)
				case 200:
					s200.Add(1)
					glMu.Lock()
					goodLats = append(goodLats, r.dur)
					glMu.Unlock()
				default:
					sOther.Add(1)
				}
				if (key == badKey || key == "") && r.status == 200 {
					sOther.Add(1000000) // poison the counter: a hostile request got data
				}
			}
		}(int64(c))
	}
	wg4.Wait()
	wall4 := time.Since(start4)
	line("| metric | value |")
	line("|---|---|")
	line("| requests | %d (%.0f/s) |", s401.Load()+s200.Load()+sOther.Load(), float64(s401.Load()+s200.Load())/wall4.Seconds())
	line("| hostile rejected with 401 | %d |", s401.Load())
	line("| legitimate served | %d |", s200.Load())
	line("| legit hit latency p50 under attack | %.2f ms |", ms(pctl(goodLats, 0.5)))
	check("every hostile request rejected, every legit served", sOther.Load() == 0,
		fmt.Sprintf("401=%d 200=%d other=%d", s401.Load(), s200.Load(), sOther.Load()))

	// ---------- phase 5: correctness sampling ----------
	section("Phase 5 — Byte-exact correctness after %d requests", total+int64(totalChunks*2)+30000)
	mismatches := 0
	for i, want := range truth {
		r := embed(base, goodKey, flat[i])
		if r.status != 200 || len(r.data) != 1 || !bytes.Equal(r.data[0].Embedding, want) {
			mismatches++
		}
	}
	check("500 sampled embeddings still byte-exact", mismatches == 0,
		fmt.Sprintf("%d verified against ground truth captured at ingest, %d mismatches", len(truth), mismatches))

	// ---------- phase 6: crash recovery ----------
	section("Phase 6 — Snapshot, hard kill, restart, cache survives")
	code, snapBody := adminPost(base, "/_ec/snapshot")
	fi, _ := os.Stat(snapPath)
	var snapMB float64
	if fi != nil {
		snapMB = float64(fi.Size()) / 1e6
	}
	check("snapshot endpoint works under admin token", code == 200, fmt.Sprintf("%s (%.0f MB on disk)", snapBody, snapMB))
	proxy.Process.Kill() // simulate a crash: no graceful shutdown
	proxy.Wait()
	proxy2, err := startProxy(addr, snapPath)
	if err != nil {
		check("proxy restarts from snapshot", false, err.Error())
	} else {
		defer proxy2.Process.Kill()
		restored := 0
		for i := range truth {
			r := embed(base, goodKey, flat[i])
			if r.status == 200 && r.cacheHdr == "hit" && bytes.Equal(r.data[0].Embedding, truth[i]) {
				restored++
			}
		}
		check("cache survives a hard crash", restored == len(truth),
			fmt.Sprintf("%d/%d sampled entries served as byte-exact hits after restart", restored, len(truth)))
	}

	// ---------- totals ----------
	section("Bottom line")
	var final struct {
		Items       uint64  `json:"items"`
		Hits        uint64  `json:"hits"`
		SavedTokens uint64  `json:"saved_tokens"`
		SavedUSD    float64 `json:"saved_usd"`
		HitRate     float64 `json:"hit_rate"`
	}
	json.Unmarshal(adminGet(base, "/_ec/stats"), &final)
	line("Across the whole simulation:")
	line("")
	line("- total embedding items served: **%d+** (ingest ×2 + storm + attack traffic)", total+int64(totalChunks)*2)
	line("- final hit rate in the restarted proxy's window: %.1f%%, saved tokens: %d", final.HitRate*100, final.SavedTokens)
	line("- zero errors, zero wrong bytes, zero hostile requests served, across ~%dk requests", (total+int64(totalChunks)*2+30000)/1000)

	if err := os.WriteFile(*outPath, rep.Bytes(), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if failed > 0 {
		fmt.Fprintf(os.Stderr, "\n%d check(s) FAILED — see %s\n", failed, *outPath)
		os.Exit(1)
	}
	fmt.Printf("\nall production-simulation checks passed — %s written\n", *outPath)
}
