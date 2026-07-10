package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"embedcache/internal/api"
	"embedcache/internal/auth"
	"embedcache/internal/breaker"
	"embedcache/internal/cache"
	"embedcache/internal/fingerprint"
	"embedcache/internal/mockllm"
	"embedcache/internal/pricing"
	"embedcache/internal/stats"
	"embedcache/internal/upstream"
)

type stack struct {
	mock      *mockllm.Server
	mockSrv   *httptest.Server
	proxySrv  *httptest.Server
	collector *stats.Collector
}

func newStack(t *testing.T) *stack {
	t.Helper()
	mock := mockllm.New(32)
	mockSrv := httptest.NewServer(mock.Handler())
	t.Cleanup(mockSrv.Close)

	up, err := upstream.New(mockSrv.URL, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	st := stats.New()
	p := New(cache.New(0, 0), up, st, fingerprint.Normalizer{})
	proxySrv := httptest.NewServer(http.HandlerFunc(p.ServeEmbeddings))
	t.Cleanup(proxySrv.Close)
	return &stack{mock: mock, mockSrv: mockSrv, proxySrv: proxySrv, collector: st}
}

func embed(t *testing.T, baseURL string, body map[string]any) (*api.EmbeddingsResponse, *http.Response) {
	t.Helper()
	b, _ := json.Marshal(body)
	resp, err := http.Post(baseURL+"/v1/embeddings", "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %s", resp.StatusCode, raw)
	}
	var parsed api.EmbeddingsResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatalf("unmarshal: %v (%s)", err, raw)
	}
	return &parsed, resp
}

// direct asks the mock itself, bypassing the proxy, for ground truth.
func direct(t *testing.T, s *stack, body map[string]any) *api.EmbeddingsResponse {
	t.Helper()
	parsed, _ := embed(t, s.mockSrv.URL, body)
	return parsed
}

func TestByteExactHit(t *testing.T) {
	s := newStack(t)
	req := map[string]any{"model": "m1", "input": "the quick brown fox"}

	truth := direct(t, s, req)
	first, r1 := embed(t, s.proxySrv.URL, req)
	second, r2 := embed(t, s.proxySrv.URL, req)

	if !bytes.Equal(first.Data[0].Embedding, truth.Data[0].Embedding) {
		t.Fatal("miss response must be byte-exact vs upstream")
	}
	if !bytes.Equal(second.Data[0].Embedding, truth.Data[0].Embedding) {
		t.Fatal("hit response must be byte-exact vs upstream")
	}
	if got := r1.Header.Get("X-Embedcache-Status"); got != "miss" {
		t.Fatalf("first status = %q", got)
	}
	if got := r2.Header.Get("X-Embedcache-Status"); got != "hit" {
		t.Fatalf("second status = %q", got)
	}
	// direct(1) + proxy first(1); the hit must not call upstream
	if s.mock.CountFor("the quick brown fox") != 2 {
		t.Fatalf("upstream saw the input %d times", s.mock.CountFor("the quick brown fox"))
	}
}

// TestMixedBatchOrdering is the regression test for the failure class in
// LiteLLM issue #22659: when a batch mixes cached and uncached items, every
// item must still receive ITS OWN vector at its original index.
func TestMixedBatchOrdering(t *testing.T) {
	s := newStack(t)
	// warm item A alone
	embed(t, s.proxySrv.URL, map[string]any{"model": "m1", "input": "A"})

	// now request [A, B, C] — A is cached, B and C are not
	got, resp := embed(t, s.proxySrv.URL, map[string]any{"model": "m1", "input": []string{"A", "B", "C"}})
	truth := direct(t, s, map[string]any{"model": "m1", "input": []string{"A", "B", "C"}})

	for i := range truth.Data {
		if got.Data[i].Index != i {
			t.Fatalf("data[%d] has index %d", i, got.Data[i].Index)
		}
		if !bytes.Equal(got.Data[i].Embedding, truth.Data[i].Embedding) {
			t.Fatalf("item %d got the wrong vector in a mixed hit/miss batch", i)
		}
	}
	if resp.Header.Get("X-Embedcache-Status") != "partial" {
		t.Fatalf("status = %q", resp.Header.Get("X-Embedcache-Status"))
	}
	// A was warmed once by the proxy, once by direct(); the mixed batch
	// must NOT have re-sent it upstream
	if n := s.mock.CountFor("A"); n != 2 {
		t.Fatalf("cached item was re-sent upstream (saw A %d times)", n)
	}
}

func TestIntraBatchDuplicates(t *testing.T) {
	s := newStack(t)
	got, _ := embed(t, s.proxySrv.URL, map[string]any{"model": "m1", "input": []string{"X", "X", "Y", "X"}})
	if len(got.Data) != 4 {
		t.Fatalf("want 4 embeddings, got %d", len(got.Data))
	}
	if !bytes.Equal(got.Data[0].Embedding, got.Data[1].Embedding) || !bytes.Equal(got.Data[0].Embedding, got.Data[3].Embedding) {
		t.Fatal("duplicate inputs must yield identical embeddings")
	}
	if bytes.Equal(got.Data[0].Embedding, got.Data[2].Embedding) {
		t.Fatal("distinct inputs must yield distinct embeddings")
	}
	// X must reach upstream once, not three times
	if n := s.mock.CountFor("X"); n != 1 {
		t.Fatalf("upstream saw X %d times", n)
	}
	if s.mock.Items() != 2 {
		t.Fatalf("upstream items = %d, want 2 (X and Y)", s.mock.Items())
	}
}

func TestConcurrentIdenticalRequestsCoalesce(t *testing.T) {
	s := newStack(t)
	s.mock.Latency = 30 * time.Millisecond // so requests overlap

	const n = 50
	var wg sync.WaitGroup
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			b, _ := json.Marshal(map[string]any{"model": "m1", "input": "popular query"})
			resp, err := http.Post(s.proxySrv.URL+"/v1/embeddings", "application/json", bytes.NewReader(b))
			if err != nil {
				errs <- err
				return
			}
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if resp.StatusCode != 200 {
				errs <- fmt.Errorf("status %d", resp.StatusCode)
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	if got := s.mock.CountFor("popular query"); got != 1 {
		t.Fatalf("upstream computed the same input %d times; coalescing failed", got)
	}
}

func TestEncodingFormatsAreSeparateEntries(t *testing.T) {
	s := newStack(t)
	f, _ := embed(t, s.proxySrv.URL, map[string]any{"model": "m1", "input": "fmt-test"})
	b, _ := embed(t, s.proxySrv.URL, map[string]any{"model": "m1", "input": "fmt-test", "encoding_format": "base64"})
	if bytes.Equal(f.Data[0].Embedding, b.Data[0].Embedding) {
		t.Fatal("float and base64 responses must differ")
	}
	var s64 string
	if err := json.Unmarshal(b.Data[0].Embedding, &s64); err != nil {
		t.Fatalf("base64 response is not a JSON string: %v", err)
	}
	truth := direct(t, s, map[string]any{"model": "m1", "input": "fmt-test", "encoding_format": "base64"})
	if !bytes.Equal(b.Data[0].Embedding, truth.Data[0].Embedding) {
		t.Fatal("base64 response must be byte-exact vs upstream")
	}
}

func TestUsageReflectsOnlyBilledTokens(t *testing.T) {
	s := newStack(t)
	warm, _ := embed(t, s.proxySrv.URL, map[string]any{"model": "m1", "input": "already cached input"})
	if warm.Usage.PromptTokens == 0 {
		t.Fatal("miss must report billed tokens")
	}
	hit, _ := embed(t, s.proxySrv.URL, map[string]any{"model": "m1", "input": "already cached input"})
	if hit.Usage.PromptTokens != 0 {
		t.Fatalf("pure hit billed %d tokens; cache hits are free", hit.Usage.PromptTokens)
	}
}

func TestTokenArrayInputs(t *testing.T) {
	s := newStack(t)
	req := map[string]any{"model": "m1", "input": [][]int{{1, 2, 3}, {4, 5}}}
	got, _ := embed(t, s.proxySrv.URL, req)
	truth := direct(t, s, req)
	for i := range truth.Data {
		if !bytes.Equal(got.Data[i].Embedding, truth.Data[i].Embedding) {
			t.Fatalf("token input %d mismatched", i)
		}
	}
	again, r := embed(t, s.proxySrv.URL, req)
	if r.Header.Get("X-Embedcache-Status") != "hit" {
		t.Fatal("token inputs must be cacheable")
	}
	if !bytes.Equal(again.Data[0].Embedding, truth.Data[0].Embedding) {
		t.Fatal("cached token embedding mismatched")
	}
}

func TestUpstreamErrorPassthrough(t *testing.T) {
	failing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(429)
		w.Write([]byte(`{"error":{"message":"rate limited","type":"rate_limit"}}`))
	}))
	defer failing.Close()
	up, _ := upstream.New(failing.URL, "", 0)
	p := New(cache.New(0, 0), up, stats.New(), fingerprint.Normalizer{})
	srv := httptest.NewServer(http.HandlerFunc(p.ServeEmbeddings))
	defer srv.Close()

	b, _ := json.Marshal(map[string]any{"model": "m1", "input": "x"})
	resp, err := http.Post(srv.URL+"/v1/embeddings", "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 429 {
		t.Fatalf("status = %d, want upstream's 429", resp.StatusCode)
	}
	if !bytes.Contains(body, []byte("rate limited")) {
		t.Fatalf("upstream error body not forwarded: %s", body)
	}
}

func TestAllowlistGuardsHitsAndMisses(t *testing.T) {
	s := newStack(t)
	authorizer, err := auth.New(auth.ModeAllowlist, []string{"sk-good"}, "", 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	// rebuild the proxy handler with auth enabled
	up, _ := upstream.New(s.mockSrv.URL, "", 0)
	p := New(cache.New(0, 0), up, stats.New(), fingerprint.Normalizer{})
	p.Auth = authorizer
	srv := httptest.NewServer(http.HandlerFunc(p.ServeEmbeddings))
	defer srv.Close()

	post := func(key string) int {
		b, _ := json.Marshal(map[string]any{"model": "m1", "input": "guarded"})
		req, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/embeddings", bytes.NewReader(b))
		req.Header.Set("Content-Type", "application/json")
		if key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		return resp.StatusCode
	}

	if code := post(""); code != 401 {
		t.Fatalf("no key: %d, want 401", code)
	}
	if code := post("sk-evil"); code != 401 {
		t.Fatalf("wrong key: %d, want 401", code)
	}
	if code := post("sk-good"); code != 200 {
		t.Fatalf("good key miss: %d, want 200", code)
	}
	// the critical case: the entry is now CACHED; a bad key must still not
	// be able to read it
	if code := post("sk-evil"); code != 401 {
		t.Fatalf("wrong key on cache hit: %d, want 401", code)
	}
	if code := post("sk-good"); code != 200 {
		t.Fatalf("good key hit: %d, want 200", code)
	}
}

func TestBodySizeLimit(t *testing.T) {
	s := newStack(t)
	up, _ := upstream.New(s.mockSrv.URL, "", 0)
	p := New(cache.New(0, 0), up, stats.New(), fingerprint.Normalizer{})
	p.MaxBody = 256
	srv := httptest.NewServer(http.HandlerFunc(p.ServeEmbeddings))
	defer srv.Close()

	big := strings.Repeat("x", 500)
	b, _ := json.Marshal(map[string]any{"model": "m1", "input": big})
	resp, err := http.Post(srv.URL+"/v1/embeddings", "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversized body: %d, want 413", resp.StatusCode)
	}
}

func TestBatchItemLimit(t *testing.T) {
	s := newStack(t)
	up, _ := upstream.New(s.mockSrv.URL, "", 0)
	p := New(cache.New(0, 0), up, stats.New(), fingerprint.Normalizer{})
	p.MaxBatchItems = 4
	srv := httptest.NewServer(http.HandlerFunc(p.ServeEmbeddings))
	defer srv.Close()

	b, _ := json.Marshal(map[string]any{"model": "m1", "input": []string{"a", "b", "c", "d", "e"}})
	resp, err := http.Post(srv.URL+"/v1/embeddings", "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Fatalf("oversized batch: %d, want 400", resp.StatusCode)
	}
}

func TestCacheTTLEndToEnd(t *testing.T) {
	s := newStack(t)
	up, _ := upstream.New(s.mockSrv.URL, "", 0)
	p := New(cache.New(0, 0), up, stats.New(), fingerprint.Normalizer{})
	p.CacheTTL = 40 * time.Millisecond
	srv := httptest.NewServer(http.HandlerFunc(p.ServeEmbeddings))
	defer srv.Close()

	_, r1 := embed(t, srv.URL, map[string]any{"model": "m1", "input": "ttl probe"})
	if r1.Header.Get("X-Embedcache-Status") != "miss" {
		t.Fatal("first must miss")
	}
	_, r2 := embed(t, srv.URL, map[string]any{"model": "m1", "input": "ttl probe"})
	if r2.Header.Get("X-Embedcache-Status") != "hit" {
		t.Fatal("within TTL must hit")
	}
	time.Sleep(60 * time.Millisecond)
	_, r3 := embed(t, srv.URL, map[string]any{"model": "m1", "input": "ttl probe"})
	if r3.Header.Get("X-Embedcache-Status") != "miss" {
		t.Fatalf("after TTL must miss, got %q", r3.Header.Get("X-Embedcache-Status"))
	}
}

func TestCircuitOpenReturns503(t *testing.T) {
	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(503)
	}))
	defer dead.Close()
	up, _ := upstream.New(dead.URL, "", 0)
	up.Retries = 0
	up.Breaker = breaker.New(1, time.Hour)
	p := New(cache.New(0, 0), up, stats.New(), fingerprint.Normalizer{})
	srv := httptest.NewServer(http.HandlerFunc(p.ServeEmbeddings))
	defer srv.Close()

	post := func(input string) int {
		b, _ := json.Marshal(map[string]any{"model": "m1", "input": input})
		resp, err := http.Post(srv.URL+"/v1/embeddings", "application/json", bytes.NewReader(b))
		if err != nil {
			t.Fatal(err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		return resp.StatusCode
	}
	if code := post("first"); code != 503 {
		t.Fatalf("upstream 503 should pass through, got %d", code)
	}
	// breaker is now open: the next request must fail fast with 503 without
	// waiting on the dead upstream
	if code := post("second"); code != 503 {
		t.Fatalf("open circuit should 503, got %d", code)
	}
	if p.Stats.Snapshot(nil2(), 0, 0).FastFails != 1 {
		t.Fatal("fast fail not recorded")
	}
}

func nil2() *pricing.Table { return pricing.Default() }

func TestRequestLogRotation(t *testing.T) {
	s := newStack(t)
	up, _ := upstream.New(s.mockSrv.URL, "", 0)
	p := New(cache.New(0, 0), up, stats.New(), fingerprint.Normalizer{})
	dir := t.TempDir()
	logPath := filepath.Join(dir, "req.jsonl")
	if err := p.OpenRequestLog(logPath, 400); err != nil { // tiny budget
		t.Fatal(err)
	}
	defer p.CloseRequestLog()
	srv := httptest.NewServer(http.HandlerFunc(p.ServeEmbeddings))
	defer srv.Close()

	for i := 0; i < 20; i++ {
		embed(t, srv.URL, map[string]any{"model": "m1", "input": fmt.Sprintf("rotation filler line %02d", i)})
	}
	fi, err := os.Stat(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Size() > 400 {
		t.Fatalf("live log exceeded budget: %d bytes", fi.Size())
	}
	if _, err := os.Stat(logPath + ".1"); err != nil {
		t.Fatal("rotated backup missing")
	}
}

func TestBadRequests(t *testing.T) {
	s := newStack(t)
	for name, body := range map[string]string{
		"invalid json":  `{`,
		"missing model": `{"input":"x"}`,
		"missing input": `{"model":"m"}`,
		"empty array":   `{"model":"m","input":[]}`,
	} {
		resp, err := http.Post(s.proxySrv.URL+"/v1/embeddings", "application/json", bytes.NewReader([]byte(body)))
		if err != nil {
			t.Fatal(err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 400 {
			t.Errorf("%s: status = %d, want 400", name, resp.StatusCode)
		}
	}
}
