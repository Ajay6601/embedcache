// The experiments harness validates embedcache's core claims against the
// real compiled binary, end to end over HTTP, using a deterministic mock
// upstream (internal/mockllm) as ground truth. It writes EXPERIMENTS.md and
// exits nonzero if any assertion fails.
//
// Usage: go run ./experiments/harness -bin ./embedcache.exe -out EXPERIMENTS.md
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
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Ajay6601/embedcache/internal/api"
	"github.com/Ajay6601/embedcache/internal/mockllm"
)

var (
	binPath = flag.String("bin", "./embedcache.exe", "path to the compiled embedcache binary")
	outPath = flag.String("out", "EXPERIMENTS.md", "where to write the results report")
)

type report struct {
	buf    bytes.Buffer
	failed atomic.Int32
}

func (r *report) section(title string) { fmt.Fprintf(&r.buf, "\n## %s\n\n", title) }
func (r *report) line(format string, args ...any) {
	fmt.Fprintf(&r.buf, format+"\n", args...)
}
func (r *report) check(name string, ok bool, detail string) {
	mark := "PASS"
	if !ok {
		mark = "FAIL"
		r.failed.Add(1)
	}
	fmt.Fprintf(&r.buf, "- **%s** — %s: %s\n", mark, name, detail)
	fmt.Printf("[%s] %s: %s\n", mark, name, detail)
}

func main() {
	flag.Parse()
	r := &report{}
	fmt.Fprintf(&r.buf, "# embedcache experiments\n\n")
	fmt.Fprintf(&r.buf, "Generated %s · %s/%s · %d CPUs · Go %s · mock upstream on loopback\n",
		time.Now().Format("2006-01-02 15:04 MST"), runtime.GOOS, runtime.GOARCH, runtime.NumCPU(), runtime.Version())
	fmt.Fprintf(&r.buf, "\nEvery experiment runs the real `embedcache serve` binary as a subprocess against\na deterministic mock OpenAI-compatible backend: the embedding for an input is a pure\nfunction of (model, input), so byte-level comparisons against ground truth are exact,\nand the mock counts every request/item that reaches it.\n")

	// E1–E4 share one proxy subprocess and one mock; state is reset between
	// experiments via /_ec/flush and mock.Reset(). One long-lived process
	// avoids racy port reallocation and Windows TIME_WAIT exhaustion.
	s, err := newStack()
	if err != nil {
		fmt.Fprintln(os.Stderr, "setup:", err)
		os.Exit(1)
	}
	e1Correctness(r, s)
	s.reset()
	e2Coalescing(r, s)
	s.reset()
	e3Overhead(r, s)
	s.reset()
	e4Reembedding(r, s)
	s.Close()

	e5ZipfQueries(r)

	if err := os.WriteFile(*outPath, r.buf.Bytes(), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "writing report:", err)
		os.Exit(1)
	}
	if n := r.failed.Load(); n > 0 {
		fmt.Fprintf(os.Stderr, "\n%d assertion(s) FAILED — see %s\n", n, *outPath)
		os.Exit(1)
	}
	fmt.Printf("\nall assertions passed — report written to %s\n", *outPath)
}

// ---------- infrastructure ----------

type stack struct {
	mock     *mockllm.Server
	mockURL  string
	proxyURL string
	stopMock func()
	stopProx func()
	client   *http.Client
}

func freePort() string {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(err)
	}
	addr := l.Addr().String()
	l.Close()
	return addr
}

// newStack starts a fresh mock upstream and a fresh proxy subprocess, so
// each experiment gets clean caches and counters.
func newStack(extraArgs ...string) (*stack, error) {
	mock := mockllm.New(64)
	ml, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	msrv := &http.Server{Handler: mock.Handler()}
	go msrv.Serve(ml)
	mockURL := "http://" + ml.Addr().String()

	proxyAddr := freePort()
	args := append([]string{"serve", "-listen", proxyAddr, "-upstream", mockURL}, extraArgs...)
	cmd := exec.Command(*binPath, args...)
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		msrv.Close()
		return nil, err
	}
	proxyURL := "http://" + proxyAddr

	// generous idle pool so high-concurrency scenarios reuse connections
	// instead of exhausting Windows ephemeral ports with TIME_WAIT sockets
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.MaxIdleConns = 256
	tr.MaxIdleConnsPerHost = 256
	s := &stack{
		mock:     mock,
		mockURL:  mockURL,
		proxyURL: proxyURL,
		stopMock: func() { msrv.Close() },
		stopProx: func() { cmd.Process.Kill(); cmd.Wait() },
		client:   &http.Client{Timeout: 60 * time.Second, Transport: tr},
	}
	for i := 0; ; i++ {
		resp, err := s.client.Get(proxyURL + "/healthz")
		if err == nil {
			resp.Body.Close()
			break
		}
		if i > 100 {
			s.Close()
			return nil, fmt.Errorf("proxy did not become healthy: %v", err)
		}
		time.Sleep(50 * time.Millisecond)
	}
	return s, nil
}

func (s *stack) Close() {
	s.stopProx()
	s.stopMock()
}

// reset clears proxy cache and mock counters between experiments sharing the
// same processes.
func (s *stack) reset() {
	resp, err := s.client.Post(s.proxyURL+"/_ec/flush", "application/json", nil)
	if err == nil {
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}
	s.mock.Reset()
	s.mock.Latency = 0
}

func (s *stack) embed(base string, body map[string]any) (*api.EmbeddingsResponse, http.Header, error) {
	b, _ := json.Marshal(body)
	resp, err := s.client.Post(base+"/v1/embeddings", "application/json", bytes.NewReader(b))
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, err
	}
	if resp.StatusCode != 200 {
		return nil, nil, fmt.Errorf("status %d: %s", resp.StatusCode, raw)
	}
	var parsed api.EmbeddingsResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, nil, err
	}
	return &parsed, resp.Header, nil
}

func percentile(durs []time.Duration, p float64) time.Duration {
	if len(durs) == 0 {
		return 0
	}
	sorted := append([]time.Duration(nil), durs...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	idx := int(p * float64(len(sorted)-1))
	return sorted[idx]
}

func makeCorpus(rng *rand.Rand, n int, wordsPer int) []string {
	vocab := []string{"retrieval", "vector", "chunk", "latency", "budget", "token", "cache", "duplicate",
		"pipeline", "index", "corpus", "query", "embedding", "model", "tenant", "shard", "batch", "waste",
		"inference", "gateway", "prefill", "decode", "throughput", "quota", "cost", "report", "audit"}
	out := make([]string, n)
	for i := range out {
		words := make([]string, wordsPer)
		for j := range words {
			words[j] = vocab[rng.Intn(len(vocab))]
		}
		out[i] = fmt.Sprintf("doc-%d :: %s", i, strings.Join(words, " "))
	}
	return out
}

// ---------- E1: byte-exact correctness under randomized batching ----------

func e1Correctness(r *report, s *stack) {
	r.section("E1 — Byte-exact correctness under randomized batching")
	r.line("**Claim tested:** a response served through embedcache — any mix of cache hits,")
	r.line("misses, and intra-batch duplicates, in float or base64 encoding — is byte-identical")
	r.line("to what the upstream would have returned, with correct index mapping.")
	r.line("")
	r.line("**Method:** 400 randomized requests (batch size 1–16, drawn with replacement from a")
	r.line("300-input pool, ~20%% base64), each embedding compared byte-for-byte against a direct")
	r.line("call to the mock upstream. This also covers the failure class of LiteLLM issue")
	r.line("[#22659](https://github.com/BerriAI/litellm/issues/22659) (mixed cached/uncached batches")
	r.line("returning wrong vectors) continuously, since the cache fills as the fuzz runs.")
	r.line("")

	rng := rand.New(rand.NewSource(1))
	pool := makeCorpus(rng, 300, 12)
	truth := map[string]json.RawMessage{} // format+"\x00"+text -> raw embedding

	itemsChecked, batches, mismatches := 0, 0, 0
	for iter := 0; iter < 400; iter++ {
		n := 1 + rng.Intn(16)
		batch := make([]string, n)
		for i := range batch {
			batch[i] = pool[rng.Intn(len(pool))]
		}
		body := map[string]any{"model": "exp-model", "input": batch}
		format := ""
		if rng.Float64() < 0.2 {
			format = "base64"
			body["encoding_format"] = format
		}
		got, _, err := s.embed(s.proxyURL, body)
		if err != nil {
			r.check("E1 request", false, err.Error())
			return
		}
		if len(got.Data) != n {
			mismatches++
			continue
		}
		batches++
		for i, text := range batch {
			key := format + "\x00" + text
			want, ok := truth[key]
			if !ok {
				tb := map[string]any{"model": "exp-model", "input": text}
				if format != "" {
					tb["encoding_format"] = format
				}
				tr, _, err := s.embed(s.mockURL, tb)
				if err != nil {
					r.check("E1 ground truth", false, err.Error())
					return
				}
				want = tr.Data[0].Embedding
				truth[key] = want
			}
			itemsChecked++
			if got.Data[i].Index != i || !bytes.Equal(got.Data[i].Embedding, want) {
				mismatches++
			}
		}
	}
	r.line("| metric | value |")
	r.line("|---|---|")
	r.line("| batches sent | %d |", batches)
	r.line("| embeddings verified byte-exact | %d |", itemsChecked)
	r.line("| mismatches | %d |", mismatches)
	r.line("")
	r.check("byte-exact fuzz", mismatches == 0, fmt.Sprintf("%d embeddings verified, %d mismatches", itemsChecked, mismatches))
}

// ---------- E2: in-flight coalescing ----------

func e2Coalescing(r *report, s *stack) {
	r.section("E2 — In-flight coalescing")
	r.line("**Claim tested:** concurrent requests for the same input trigger exactly one")
	r.line("upstream computation; overlapping batches compute each unique input once.")
	r.line("")

	s.mock.Latency = 50 * time.Millisecond

	// scenario A: 200 identical singletons at once
	var wg sync.WaitGroup
	var reqErrs atomic.Int32
	for i := 0; i < 200; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, _, err := s.embed(s.proxyURL, map[string]any{"model": "exp-model", "input": "hot query"}); err != nil {
				reqErrs.Add(1)
			}
		}()
	}
	wg.Wait()
	hotCount := s.mock.CountFor("hot query")
	r.line("**Scenario A:** 200 concurrent identical requests (upstream latency 50ms)")
	r.line("")
	r.line("| metric | value |")
	r.line("|---|---|")
	r.line("| client requests | 200 |")
	r.line("| request errors | %d |", reqErrs.Load())
	r.line("| upstream computations of the input | %d |", hotCount)
	r.line("")
	r.check("singleton coalescing", hotCount == 1 && reqErrs.Load() == 0,
		fmt.Sprintf("200 concurrent requests -> %d upstream computation(s)", hotCount))

	// scenario B: 100 concurrent overlapping batches over a pool of 20
	r.line("")
	s.mock.Reset()
	rng := rand.New(rand.NewSource(2))
	pool := makeCorpus(rng, 20, 10)
	reqErrs.Store(0)
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			lr := rand.New(rand.NewSource(seed))
			batch := make([]string, 8)
			for j := range batch {
				batch[j] = pool[lr.Intn(len(pool))]
			}
			if _, _, err := s.embed(s.proxyURL, map[string]any{"model": "exp-model", "input": batch}); err != nil {
				reqErrs.Add(1)
			}
		}(int64(i))
	}
	wg.Wait()
	upstreamItems := s.mock.Items()
	r.line("**Scenario B:** 100 concurrent batches of 8, drawn from 20 unique inputs (800 items total)")
	r.line("")
	r.line("| metric | value |")
	r.line("|---|---|")
	r.line("| items requested | 800 |")
	r.line("| unique inputs | 20 |")
	r.line("| items computed upstream | %d |", upstreamItems)
	r.line("| request errors | %d |", reqErrs.Load())
	r.line("")
	r.check("batch coalescing", upstreamItems == 20 && reqErrs.Load() == 0,
		fmt.Sprintf("800 requested items -> %d upstream computations (ideal: 20)", upstreamItems))
}

// ---------- E3: proxy overhead ----------

func e3Overhead(r *report, s *stack) {
	r.section("E3 — Proxy overhead and throughput")
	r.line("**Claim tested:** the proxy's added latency is negligible next to a real embedding")
	r.line("call (typically 10–100ms upstream).")
	r.line("")
	r.line("**Method:** sequential single-input requests on loopback with a zero-latency mock;")
	r.line("direct-to-mock is the baseline. Numbers below are wall-clock per request on this")
	r.line("machine (Windows loopback stack) — treat them as upper bounds, not benchmarks.")
	r.line("")

	measure := func(base string, uniquePrefix string, n int) []time.Duration {
		durs := make([]time.Duration, 0, n)
		for i := 0; i < n; i++ {
			input := "warm input"
			if uniquePrefix != "" {
				input = fmt.Sprintf("%s-%d", uniquePrefix, i)
			}
			start := time.Now()
			if _, _, err := s.embed(base, map[string]any{"model": "exp-model", "input": input}); err != nil {
				return nil
			}
			durs = append(durs, time.Since(start))
		}
		return durs
	}

	// warm up connections and the "warm input" cache entry
	measure(s.mockURL, "", 100)
	measure(s.proxyURL, "", 100)

	const n = 1500
	direct := measure(s.mockURL, "", n)
	hits := measure(s.proxyURL, "", n)
	misses := measure(s.proxyURL, "e3-miss", n)
	if direct == nil || hits == nil || misses == nil {
		r.check("E3 measurement", false, "request failed during measurement")
		return
	}

	row := func(name string, d []time.Duration) string {
		return fmt.Sprintf("| %s | %.2f | %.2f | %.2f |", name,
			float64(percentile(d, 0.50).Microseconds())/1000,
			float64(percentile(d, 0.95).Microseconds())/1000,
			float64(percentile(d, 0.99).Microseconds())/1000)
	}
	r.line("| path (%d sequential reqs) | p50 ms | p95 ms | p99 ms |", n)
	r.line("|---|---|---|---|")
	r.line(row("direct to mock upstream", direct))
	r.line(row("proxy, cache hit", hits))
	r.line(row("proxy, cache miss (adds one upstream hop)", misses))

	hitOverhead := percentile(hits, 0.50) - percentile(direct, 0.50)
	missOverhead := percentile(misses, 0.50) - percentile(direct, 0.50)
	r.line("")
	r.line("Added p50 latency: **%.2f ms on a hit**, **%.2f ms on a miss** (miss includes a second", float64(hitOverhead.Microseconds())/1000, float64(missOverhead.Microseconds())/1000)
	r.line("loopback round-trip to the upstream, which a real deployment pays anyway).")
	r.line("")

	// throughput: 32 workers hammering cache hits for 5 seconds
	var count atomic.Int64
	deadline := time.Now().Add(5 * time.Second)
	var wg sync.WaitGroup
	for w := 0; w < 32; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			client := s.client
			body, _ := json.Marshal(map[string]any{"model": "exp-model", "input": "warm input"})
			for time.Now().Before(deadline) {
				resp, err := client.Post(s.proxyURL+"/v1/embeddings", "application/json", bytes.NewReader(body))
				if err != nil {
					return
				}
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
				count.Add(1)
			}
		}()
	}
	wg.Wait()
	rps := float64(count.Load()) / 5
	r.line("Sustained cache-hit throughput, 32 concurrent clients, 5s: **%.0f req/s**.", rps)
	r.line("")
	r.check("hit overhead under 5ms p50", hitOverhead < 5*time.Millisecond,
		fmt.Sprintf("hit adds %.2f ms at p50", float64(hitOverhead.Microseconds())/1000))
	r.check("throughput sane", rps > 500, fmt.Sprintf("%.0f cache-hit req/s on loopback", rps))
}

// ---------- E4: RAG re-embedding simulation ----------

func e4Reembedding(r *report, s *stack) {
	r.section("E4 — RAG re-ingestion: where dedupe saves money (and where it honestly doesn't)")
	r.line("**Claim tested:** re-ingesting a corpus after incremental document changes only pays")
	r.line("for what changed. **Honest counter-test:** changing the chunking configuration itself")
	r.line("produces different chunk text, so an exact-match cache saves ~nothing — that scenario")
	r.line("needs the roadmap's chunk-diff engine, and we say so rather than hide it.")
	r.line("")

	rng := rand.New(rand.NewSource(3))
	const docs, chunksPerDoc = 400, 25
	corpus := make([][]string, docs)
	for d := range corpus {
		corpus[d] = make([]string, chunksPerDoc)
		base := makeCorpus(rng, chunksPerDoc, 30)
		for c := range corpus[d] {
			corpus[d][c] = fmt.Sprintf("doc%04d/chunk%02d %s", d, c, base[c])
		}
	}
	ingest := func() (int, error) {
		var batch []string
		flushAndCount := 0
		flush := func() error {
			if len(batch) == 0 {
				return nil
			}
			_, _, err := s.embed(s.proxyURL, map[string]any{"model": "exp-model", "input": batch})
			flushAndCount += len(batch)
			batch = batch[:0]
			return err
		}
		for d := range corpus {
			for c := range corpus[d] {
				batch = append(batch, corpus[d][c])
				if len(batch) == 64 {
					if err := flush(); err != nil {
						return 0, err
					}
				}
			}
		}
		return flushAndCount, flush()
	}

	// pass 1: cold ingestion
	total, err := ingest()
	if err != nil {
		r.check("E4 ingest", false, err.Error())
		return
	}
	pass1Items := s.mock.Items()

	// pass 2: 5% of documents change, everything re-ingested (the common
	// "nightly full re-run" pattern)
	s.mock.Reset()
	changedDocs := docs * 5 / 100
	for i := 0; i < changedDocs; i++ {
		d := rng.Intn(docs)
		for c := range corpus[d] {
			corpus[d][c] = fmt.Sprintf("doc%04d/chunk%02d EDITED-%d %s", d, c, i, corpus[d][c][20:])
		}
	}
	if _, err := ingest(); err != nil {
		r.check("E4 re-ingest", false, err.Error())
		return
	}
	pass2Upstream := s.mock.Items()
	saved := total - pass2Upstream
	savedPct := float64(saved) / float64(total) * 100

	// pass 3: chunking config change — every chunk's text shifts
	s.mock.Reset()
	for d := range corpus {
		for c := range corpus[d] {
			corpus[d][c] = "rechunked:: " + corpus[d][c]
		}
	}
	if _, err := ingest(); err != nil {
		r.check("E4 re-chunk", false, err.Error())
		return
	}
	pass3Upstream := s.mock.Items()

	r.line("| pass | items ingested | items paid for upstream | saved |")
	r.line("|---|---|---|---|")
	r.line("| 1 — cold ingest | %d | %d | 0%% |", total, pass1Items)
	r.line("| 2 — 5%% of docs edited, full re-ingest | %d | %d | **%.1f%%** |", total, pass2Upstream, savedPct)
	r.line("| 3 — chunking config changed | %d | %d | %.1f%% |", total, pass3Upstream, float64(total-pass3Upstream)/float64(total)*100)
	r.line("")
	r.line("Pass 3 is the documented limitation: exact-match dedupe cannot absorb a re-chunk;")
	r.line("that is the Phase-2 chunk-diff problem, not a cache problem.")
	r.line("")
	r.check("cold ingest pays full price", pass1Items == total, fmt.Sprintf("%d/%d items upstream", pass1Items, total))
	wantUpstream := changedDocs * chunksPerDoc
	r.check("incremental re-ingest pays only for changes", pass2Upstream <= wantUpstream+chunksPerDoc && savedPct > 90,
		fmt.Sprintf("re-ingest of %d items sent only %d upstream (%.1f%% saved; %d items belong to edited docs)", total, pass2Upstream, savedPct, wantUpstream))
	r.check("re-chunk honestly saves nothing", pass3Upstream == total,
		fmt.Sprintf("%d/%d items had to be recomputed after re-chunking", pass3Upstream, total))
}

// ---------- E5: query-side workload + offline analyzer cross-check ----------

func e5ZipfQueries(r *report) {
	r.section("E5 — Query workload (Zipf) + offline analyzer cross-check")
	r.line("**Claim tested:** on a skewed query distribution (few hot queries, long tail — the")
	r.line("shape of real search/RAG traffic), exact-match caching absorbs a large share of")
	r.line("embedding calls; and `embedcache analyze` on the request log reports the same waste")
	r.line("the live proxy actually avoided.")
	r.line("")

	logPath := "e5-requests.jsonl"
	os.Remove(logPath)
	s, err := newStack("-request-log", logPath)
	if err != nil {
		r.check("E5 setup", false, err.Error())
		return
	}
	defer os.Remove(logPath)
	defer s.Close()

	rng := rand.New(rand.NewSource(4))
	pool := makeCorpus(rng, 3000, 8)
	zipf := rand.NewZipf(rng, 1.1, 1, uint64(len(pool)-1))

	const queries = 20000
	hits := 0
	for i := 0; i < queries; i++ {
		q := pool[zipf.Uint64()]
		_, hdr, err := s.embed(s.proxyURL, map[string]any{"model": "exp-model", "input": q})
		if err != nil {
			r.check("E5 query", false, err.Error())
			return
		}
		if hdr.Get("X-Embedcache-Status") == "hit" {
			hits++
		}
	}
	upstreamItems := s.mock.Items()
	hitRate := float64(hits) / float64(queries) * 100

	r.line("| metric | value |")
	r.line("|---|---|")
	r.line("| queries | %d |", queries)
	r.line("| unique pool | %d |", len(pool))
	r.line("| upstream computations | %d |", upstreamItems)
	r.line("| cache hit rate | **%.1f%%** |", hitRate)
	r.line("")
	r.check("skewed workload hit rate", hitRate > 50,
		fmt.Sprintf("%.1f%% of queries served from cache; upstream saw %d of %d items", hitRate, upstreamItems, queries))
	r.check("upstream items equal unique queries seen", upstreamItems == queries-hits,
		fmt.Sprintf("upstream %d == queries %d - hits %d", upstreamItems, queries, hits))

	// cross-check: the offline analyzer on the request log must find the
	// same duplicate share the live cache absorbed
	out, err := exec.Command(*binPath, "analyze", "-json", logPath).Output()
	if err != nil {
		r.check("E5 analyzer run", false, err.Error())
		return
	}
	var ana struct {
		Items    int     `json:"items"`
		Unique   int     `json:"unique_items"`
		DupItems int     `json:"duplicate_items"`
		DupRate  float64 `json:"duplicate_rate"`
	}
	if err := json.Unmarshal(out, &ana); err != nil {
		r.check("E5 analyzer parse", false, err.Error())
		return
	}
	r.line("")
	r.line("Offline analyzer on the same request log: %d items, %d unique, %d duplicates (%.1f%%).",
		ana.Items, ana.Unique, ana.DupItems, ana.DupRate*100)
	r.line("")
	r.check("analyzer agrees with live proxy", ana.Items == queries && ana.DupItems == hits,
		fmt.Sprintf("analyzer found %d duplicates; live cache served %d hits", ana.DupItems, hits))
}
