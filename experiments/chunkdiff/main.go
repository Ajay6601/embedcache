// chunkdiff proves the chunk-diff engine's actual value end to end: fetch
// ONE real Wikipedia article, make ONE realistic single-sentence edit, and
// measure — through the real embedcache proxy in front of real Ollama
// inference — how much of the re-ingested corpus hits the cache under
// content-defined chunking versus naive fixed-size chunking.
//
// No synthetic or bulk-generated text anywhere: the corpus is one live
// article, the edit is a single realistic sentence a human editor would
// plausibly add, matching how REALTEST.md validated the base cache.
//
// Usage:
//
//	go build -o embedcache.exe ./cmd/embedcache
//	go run ./experiments/chunkdiff -bin ./embedcache.exe -upstream http://localhost:11434
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/Ajay6601/embedcache/internal/chunker"
)

var (
	binPath  = flag.String("bin", "./embedcache.exe", "compiled embedcache binary")
	upstream = flag.String("upstream", "http://localhost:11434", "real embedding backend")
	model    = flag.String("model", "all-minilm", "embedding model")
	article  = flag.String("article", "Retrieval-augmented generation", "a live Wikipedia article title")
	out      = flag.String("out", "CHUNKDIFF.md", "results file")
)

const pipelineKey = "pipeline-key"

func fetchArticle(title string) (string, error) {
	u := "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&redirects=1&format=json&titles=" + url.QueryEscape(title)
	req, _ := http.NewRequest(http.MethodGet, u, nil)
	req.Header.Set("User-Agent", "embedcache-chunkdiff/0.1 (https://github.com/Ajay6601/embedcache)")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	var parsed struct {
		Query struct {
			Pages map[string]struct {
				Extract string `json:"extract"`
			} `json:"pages"`
		} `json:"query"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return "", err
	}
	for _, p := range parsed.Query.Pages {
		if len(p.Extract) < 500 {
			return "", fmt.Errorf("article %q too short or missing", title)
		}
		return p.Extract, nil
	}
	return "", fmt.Errorf("no page for %q", title)
}

// fixedSizeSplit mirrors what a typical RAG pipeline does today: slice by
// byte count with no regard for content, so any insertion shifts every
// downstream boundary.
func fixedSizeSplit(text string, size int) []string {
	var out []string
	b := []byte(text)
	for len(b) > 0 {
		n := size
		if n > len(b) {
			n = len(b)
		}
		out = append(out, string(b[:n]))
		b = b[n:]
	}
	return out
}

func cdcSplit(text string) []string {
	chunks := chunker.Split([]byte(text), chunker.Options{})
	out := make([]string, len(chunks))
	for i, c := range chunks {
		out[i] = string(c.Data)
	}
	return out
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

type proxy struct {
	base   string
	cmd    *exec.Cmd
	client *http.Client
}

func startProxy() (*proxy, error) {
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.MaxIdleConns, tr.MaxIdleConnsPerHost = 128, 128
	client := &http.Client{Timeout: 120 * time.Second, Transport: tr}

	addr := freePort()
	cmd := exec.Command(*binPath, "serve", "-listen", addr, "-upstream", *upstream,
		"-auth-mode", "allowlist", "-api-keys", pipelineKey)
	cmd.Stdout, cmd.Stderr = io.Discard, io.Discard
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	base := "http://" + addr
	for i := 0; ; i++ {
		resp, err := client.Get(base + "/healthz")
		if err == nil {
			resp.Body.Close()
			break
		}
		if i > 100 {
			cmd.Process.Kill()
			return nil, fmt.Errorf("proxy never became healthy: %v", err)
		}
		time.Sleep(100 * time.Millisecond)
	}
	return &proxy{base: base, cmd: cmd, client: client}, nil
}

func (p *proxy) close() { p.cmd.Process.Kill(); p.cmd.Wait() }

func (p *proxy) flush() {
	req, _ := http.NewRequest(http.MethodPost, p.base+"/_ec/flush", nil)
	if resp, err := p.client.Do(req); err == nil {
		resp.Body.Close()
	}
}

type ingestResult struct {
	items, hits, misses int
}

func (p *proxy) ingest(chunks []string) (ingestResult, error) {
	var r ingestResult
	for i := 0; i < len(chunks); i += 16 {
		end := i + 16
		if end > len(chunks) {
			end = len(chunks)
		}
		body, _ := json.Marshal(map[string]any{"model": *model, "input": chunks[i:end]})
		req, _ := http.NewRequest(http.MethodPost, p.base+"/v1/embeddings", strings.NewReader(string(body)))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+pipelineKey)
		resp, err := p.client.Do(req)
		if err != nil {
			return r, err
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 {
			return r, fmt.Errorf("status %d", resp.StatusCode)
		}
		var h, m int
		fmt.Sscanf(resp.Header.Get("X-Embedcache-Hits"), "%d", &h)
		fmt.Sscanf(resp.Header.Get("X-Embedcache-Misses"), "%d", &m)
		r.items += end - i
		r.hits += h
		r.misses += m
	}
	return r, nil
}

func main() {
	flag.Parse()
	var rep strings.Builder
	failed := 0
	check := func(name string, ok bool, detail string) {
		mark := "PASS"
		if !ok {
			mark = "FAIL"
			failed++
		}
		fmt.Fprintf(&rep, "- **%s** — %s: %s\n", mark, name, detail)
		fmt.Printf("[%s] %s: %s\n", mark, name, detail)
	}

	fmt.Fprintf(&rep, "# chunk-diff engine: real-data validation\n\n")
	fmt.Fprintf(&rep, "Generated %s. Corpus: one live Wikipedia article (%q). Edit: one realistic\n", time.Now().Format("2006-01-02 15:04 MST"), *article)
	fmt.Fprintf(&rep, "single-sentence insertion — no synthetic or bulk-generated text. All embeddings\n")
	fmt.Fprintf(&rep, "computed by real Ollama (`%s`) through the real embedcache proxy.\n", *model)

	v1, err := fetchArticle(*article)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	// one realistic edit: insert a single on-topic sentence at a paragraph
	// break partway through the real article
	idx := strings.Index(v1[len(v1)/3:], "\n\n")
	if idx < 0 {
		fmt.Fprintln(os.Stderr, "no paragraph break found to edit at")
		os.Exit(1)
	}
	idx += len(v1) / 3
	edit := "\n\nThis remains an active area of practical application as of 2026.\n\n"
	v2 := v1[:idx] + edit + v1[idx:]

	fmt.Fprintf(&rep, "\nArticle length: %d bytes. Edit: +%d bytes at one point.\n", len(v1), len(edit))

	px, err := startProxy()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer px.close()

	var fixedHitRate, cdcHitRate float64
	px.flush()
	{
		v1c := fixedSizeSplit(v1, 800)
		cold, err := px.ingest(v1c)
		if err != nil {
			check("fixed-size: cold ingest", false, err.Error())
		} else {
			v2c := fixedSizeSplit(v2, 800)
			warm, err := px.ingest(v2c)
			if err != nil {
				check("fixed-size: re-ingest", false, err.Error())
			} else {
				fixedHitRate = float64(warm.hits) / float64(warm.items) * 100
				fmt.Fprintf(&rep, "\n## Fixed-size chunking (naive baseline, 800-byte blocks)\n\n")
				fmt.Fprintf(&rep, "| metric | value |\n|---|---|\n")
				fmt.Fprintf(&rep, "| v1 chunks (cold ingest) | %d |\n", cold.items)
				fmt.Fprintf(&rep, "| v2 chunks (after the edit) | %d |\n", warm.items)
				fmt.Fprintf(&rep, "| v2 cache hits | %d |\n", warm.hits)
				fmt.Fprintf(&rep, "| **hit rate on re-ingest after one edit** | **%.1f%%** |\n", fixedHitRate)
			}
		}
	}

	px.flush()
	{
		v1c := cdcSplit(v1)
		cold, err := px.ingest(v1c)
		if err != nil {
			check("content-defined: cold ingest", false, err.Error())
		} else {
			v2c := cdcSplit(v2)
			warm, err := px.ingest(v2c)
			if err != nil {
				check("content-defined: re-ingest", false, err.Error())
			} else {
				cdcHitRate = float64(warm.hits) / float64(warm.items) * 100
				fmt.Fprintf(&rep, "\n## Content-defined chunking (embedcache's chunker)\n\n")
				fmt.Fprintf(&rep, "| metric | value |\n|---|---|\n")
				fmt.Fprintf(&rep, "| v1 chunks (cold ingest) | %d |\n", cold.items)
				fmt.Fprintf(&rep, "| v2 chunks (after the edit) | %d |\n", warm.items)
				fmt.Fprintf(&rep, "| v2 cache hits | %d |\n", warm.hits)
				fmt.Fprintf(&rep, "| **hit rate on re-ingest after one edit** | **%.1f%%** |\n", cdcHitRate)
			}
		}
	}

	fmt.Fprintf(&rep, "\n## Bottom line\n\n")
	fmt.Fprintf(&rep, "Same real article, same single realistic edit: content-defined chunking absorbed\n")
	fmt.Fprintf(&rep, "**%.1f%%** of the re-ingest from cache; fixed-size chunking absorbed only **%.1f%%**.\n", cdcHitRate, fixedHitRate)
	check("content-defined chunking beats fixed-size on a real edit", cdcHitRate > fixedHitRate,
		fmt.Sprintf("cdc=%.1f%% fixed=%.1f%% on real article %q", cdcHitRate, fixedHitRate, *article))
	check("content-defined chunking absorbs most of the re-ingest", cdcHitRate > 60,
		fmt.Sprintf("%.1f%% hit rate after a single-sentence edit", cdcHitRate))

	if err := os.WriteFile(*out, []byte(rep.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if failed > 0 {
		fmt.Fprintf(os.Stderr, "\n%d check(s) FAILED — see %s\n", failed, *out)
		os.Exit(1)
	}
	fmt.Printf("\nall checks passed — %s written\n", *out)
}
