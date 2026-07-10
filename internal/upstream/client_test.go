package upstream

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Ajay6601/embedcache/internal/api"
	"github.com/Ajay6601/embedcache/internal/breaker"
)

func okBody(n int) []byte {
	data := make([]api.EmbeddingData, n)
	for i := range data {
		data[i] = api.EmbeddingData{Object: "embedding", Index: i, Embedding: json.RawMessage(`[0.1,0.2]`)}
	}
	b, _ := json.Marshal(api.EmbeddingsResponse{Object: "list", Data: data, Usage: api.Usage{PromptTokens: n}})
	return b
}

func call(c *Client) (*api.EmbeddingsResponse, error) {
	return c.Embeddings(context.Background(), "/v1/embeddings",
		api.EmbeddingsRequest{Model: "m"}, []api.InputItem{{Text: "x"}}, "")
}

func TestRetriesTransientFailures(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) <= 2 {
			w.WriteHeader(503)
			return
		}
		w.Write(okBody(1))
	}))
	defer srv.Close()
	c, _ := New(srv.URL, "", 0)
	c.Retries = 2
	var retried atomic.Int32
	c.OnRetry = func() { retried.Add(1) }

	resp, err := call(c)
	if err != nil {
		t.Fatalf("expected success after retries: %v", err)
	}
	if len(resp.Data) != 1 || calls.Load() != 3 || retried.Load() != 2 {
		t.Fatalf("calls=%d retries=%d", calls.Load(), retried.Load())
	}
}

func TestNoRetryOnClientError(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(400)
		w.Write([]byte(`{"error":{"message":"bad input"}}`))
	}))
	defer srv.Close()
	c, _ := New(srv.URL, "", 0)
	c.Retries = 3

	_, err := call(c)
	var he *HTTPError
	if !errors.As(err, &he) || he.Status != 400 {
		t.Fatalf("want HTTPError 400, got %v", err)
	}
	if calls.Load() != 1 {
		t.Fatalf("a 400 must not be retried; upstream saw %d calls", calls.Load())
	}
}

func TestRetryAfterHeaderHonored(t *testing.T) {
	var calls atomic.Int32
	var firstRetryAt time.Time
	start := time.Now()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(429)
			return
		}
		firstRetryAt = time.Now()
		w.Write(okBody(1))
	}))
	defer srv.Close()
	c, _ := New(srv.URL, "", 0)
	c.Retries = 1

	if _, err := call(c); err != nil {
		t.Fatal(err)
	}
	if wait := firstRetryAt.Sub(start); wait < 900*time.Millisecond {
		t.Fatalf("Retry-After: 1 not honored; retried after %v", wait)
	}
}

func TestBreakerOpensAndFailsFast(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(503)
	}))
	defer srv.Close()
	c, _ := New(srv.URL, "", 0)
	c.Retries = 0
	c.Breaker = breaker.New(3, time.Hour)

	for i := 0; i < 3; i++ {
		if _, err := call(c); err == nil {
			t.Fatal("expected failure")
		}
	}
	before := calls.Load()
	_, err := call(c)
	if !errors.Is(err, ErrCircuitOpen) {
		t.Fatalf("want ErrCircuitOpen, got %v", err)
	}
	if calls.Load() != before {
		t.Fatal("open circuit must not dial the upstream")
	}
}

func TestBreakerRecovers(t *testing.T) {
	var healthy atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if healthy.Load() {
			w.Write(okBody(1))
			return
		}
		w.WriteHeader(503)
	}))
	defer srv.Close()
	c, _ := New(srv.URL, "", 0)
	c.Retries = 0
	c.Breaker = breaker.New(1, 30*time.Millisecond)

	call(c) // opens
	if _, err := call(c); !errors.Is(err, ErrCircuitOpen) {
		t.Fatalf("want fast fail, got %v", err)
	}
	healthy.Store(true)
	time.Sleep(40 * time.Millisecond)
	if _, err := call(c); err != nil {
		t.Fatalf("probe should succeed and close the circuit: %v", err)
	}
	if _, err := call(c); err != nil {
		t.Fatalf("circuit should be closed: %v", err)
	}
}
