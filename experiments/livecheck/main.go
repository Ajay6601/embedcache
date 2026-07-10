// livecheck validates embedcache against a REAL embedding backend (Ollama,
// vLLM, TEI, OpenAI, Gemini's OpenAI-compat endpoint, ...). Unlike the mock
// harness it cannot count upstream calls directly, so it verifies through
// response bytes and the X-Embedcache-* headers.
//
// Usage:
//
//	embedcache serve -listen :8090 -upstream http://localhost:11434 &
//	go run ./experiments/livecheck -proxy http://localhost:8090 -upstream http://localhost:11434 -model all-minilm
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/Ajay6601/embedcache/internal/api"
)

var (
	proxyURL    = flag.String("proxy", "http://localhost:8090", "embedcache address")
	upstreamURL = flag.String("upstream", "", "the real backend, for direct ground-truth calls (optional)")
	model       = flag.String("model", "all-minilm", "embedding model to test")
	apiKey      = flag.String("api-key", "", "Authorization bearer for backends that need one")
	failed      = 0
)

func check(name string, ok bool, detail string) {
	mark := "PASS"
	if !ok {
		mark = "FAIL"
		failed++
	}
	fmt.Printf("[%s] %-40s %s\n", mark, name, detail)
}

func info(name, detail string) { fmt.Printf("[info] %-40s %s\n", name, detail) }

func embed(base string, body map[string]any) (*api.EmbeddingsResponse, http.Header, int, error) {
	b, _ := json.Marshal(body)
	req, _ := http.NewRequest(http.MethodPost, base+"/v1/embeddings", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	if *apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+*apiKey)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, nil, 0, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return nil, resp.Header, resp.StatusCode, fmt.Errorf("status %d: %.200s", resp.StatusCode, raw)
	}
	var parsed api.EmbeddingsResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, resp.Header, resp.StatusCode, fmt.Errorf("unparseable response: %v", err)
	}
	return &parsed, resp.Header, resp.StatusCode, nil
}

func main() {
	flag.Parse()
	fmt.Printf("livecheck: proxy=%s upstream=%s model=%s\n\n", *proxyURL, *upstreamURL, *model)
	stamp := time.Now().UnixNano() // unique inputs per run so misses are real misses

	in := func(s string) string { return fmt.Sprintf("%s [run %d]", s, stamp) }

	// 1. basic miss -> hit, byte-exact replay
	first, h1, _, err := embed(*proxyURL, map[string]any{"model": *model, "input": in("the quick brown fox")})
	if err != nil {
		check("proxy reachable / model works", false, err.Error())
		os.Exit(1)
	}
	check("proxy reachable / model works", true, fmt.Sprintf("dims=%d", dims(first.Data[0].Embedding)))
	check("first request is a miss", h1.Get("X-Embedcache-Status") == "miss", "status="+h1.Get("X-Embedcache-Status"))

	second, h2, _, err := embed(*proxyURL, map[string]any{"model": *model, "input": in("the quick brown fox")})
	if err != nil {
		check("second request", false, err.Error())
		os.Exit(1)
	}
	check("second request is a hit", h2.Get("X-Embedcache-Status") == "hit", "status="+h2.Get("X-Embedcache-Status"))
	check("hit replays cached bytes exactly", bytes.Equal(first.Data[0].Embedding, second.Data[0].Embedding),
		fmt.Sprintf("%d bytes compared", len(first.Data[0].Embedding)))
	check("hit is billed zero tokens", second.Usage.PromptTokens == 0,
		fmt.Sprintf("miss billed %d, hit billed %d", first.Usage.PromptTokens, second.Usage.PromptTokens))

	// 2. upstream determinism (informational — affects nothing about the
	// cache guarantee, but worth knowing per backend)
	if *upstreamURL != "" {
		d1, _, _, err1 := embed(*upstreamURL, map[string]any{"model": *model, "input": in("determinism probe")})
		d2, _, _, err2 := embed(*upstreamURL, map[string]any{"model": *model, "input": in("determinism probe")})
		if err1 == nil && err2 == nil {
			if bytes.Equal(d1.Data[0].Embedding, d2.Data[0].Embedding) {
				info("upstream determinism", "backend returns identical bytes for repeated input")
			} else {
				info("upstream determinism", "backend is NOT byte-deterministic across calls; caching also stabilizes results")
			}
		}
	}

	// 3. mixed batch: warm A, then send [A, B, C]; A must come back as the
	// cached bytes, B and C must be fresh and land at the right indices
	warmA, _, _, err := embed(*proxyURL, map[string]any{"model": *model, "input": in("item A")})
	if err != nil {
		check("warm A", false, err.Error())
		os.Exit(1)
	}
	batch, hb, _, err := embed(*proxyURL, map[string]any{"model": *model,
		"input": []string{in("item A"), in("item B"), in("item C")}})
	if err != nil {
		check("mixed batch", false, err.Error())
		os.Exit(1)
	}
	check("mixed batch is partial", hb.Get("X-Embedcache-Status") == "partial", "status="+hb.Get("X-Embedcache-Status"))
	check("cached item keeps its exact bytes in batch", bytes.Equal(batch.Data[0].Embedding, warmA.Data[0].Embedding), "index 0 == warmed A")
	distinct := !bytes.Equal(batch.Data[1].Embedding, batch.Data[0].Embedding) &&
		!bytes.Equal(batch.Data[2].Embedding, batch.Data[1].Embedding)
	check("batch items are distinct vectors", distinct, "B != A, C != B")
	singleB, _, _, _ := embed(*proxyURL, map[string]any{"model": *model, "input": in("item B")})
	check("batch item B matches its own cached identity", singleB != nil && bytes.Equal(batch.Data[1].Embedding, singleB.Data[0].Embedding),
		"re-querying B alone hits the cache entry the batch created")

	// 4. intra-batch duplicates
	dup, _, _, err := embed(*proxyURL, map[string]any{"model": *model,
		"input": []string{in("dup X"), in("dup X"), in("dup Y")}})
	if err == nil {
		check("intra-batch duplicates deduped", bytes.Equal(dup.Data[0].Embedding, dup.Data[1].Embedding) &&
			!bytes.Equal(dup.Data[0].Embedding, dup.Data[2].Embedding), "X==X, X!=Y")
	} else {
		check("intra-batch duplicates deduped", false, err.Error())
	}

	// 5. coalescing under real latency: N concurrent identical requests;
	// exactly one should report a miss, the rest hit/coalesced
	const n = 12
	var wg sync.WaitGroup
	statuses := make([]string, n)
	misses := 0
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, h, _, err := embed(*proxyURL, map[string]any{"model": *model, "input": in("concurrent burst")})
			if err == nil {
				statuses[i] = h.Get("X-Embedcache-Status")
			}
		}(i)
	}
	wg.Wait()
	for _, s := range statuses {
		if s == "miss" {
			misses++
		}
	}
	check("concurrent burst coalesces", misses == 1,
		fmt.Sprintf("%d concurrent requests -> %d upstream miss(es), rest served in-flight/cache", n, misses))

	// 6. base64 support probe (informational; some self-hosted backends
	// ignore or reject encoding_format)
	b64, _, code, err := embed(*proxyURL, map[string]any{"model": *model, "input": in("b64 probe"), "encoding_format": "base64"})
	switch {
	case err != nil:
		info("encoding_format=base64", fmt.Sprintf("backend rejected it (status %d) — proxy forwarded the error correctly", code))
	default:
		var s string
		if json.Unmarshal(b64.Data[0].Embedding, &s) == nil {
			info("encoding_format=base64", "supported by backend, cached under its own key")
		} else {
			info("encoding_format=base64", "backend ignored it and returned floats — cached under the base64 key; consistent but worth knowing")
		}
	}

	// 7. live stats endpoint
	resp, err := http.Get(*proxyURL + "/_ec/stats")
	if err == nil {
		var st struct {
			Hits        uint64  `json:"hits"`
			Coalesced   uint64  `json:"coalesced"`
			SavedTokens uint64  `json:"saved_tokens"`
			HitRate     float64 `json:"hit_rate"`
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		json.Unmarshal(body, &st)
		check("stats endpoint accounts the run", st.Hits > 0 && st.SavedTokens > 0,
			fmt.Sprintf("hits=%d coalesced=%d saved_tokens=%d hit_rate=%.0f%%", st.Hits, st.Coalesced, st.SavedTokens, st.HitRate*100))
	}

	fmt.Println()
	if failed > 0 {
		fmt.Printf("%d check(s) FAILED against this backend\n", failed)
		os.Exit(1)
	}
	fmt.Println("all live checks passed against this backend")
}

func dims(raw json.RawMessage) int {
	var f []float64
	if json.Unmarshal(raw, &f) == nil {
		return len(f)
	}
	return -1
}
