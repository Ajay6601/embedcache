// Package upstream calls the OpenAI-compatible embedding backend.
package upstream

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"time"

	"embedcache/internal/api"
	"embedcache/internal/breaker"
)

// HTTPError carries a non-2xx upstream response so the proxy can forward it
// to the client unchanged.
type HTTPError struct {
	Status      int
	ContentType string
	Body        []byte
	RetryAfter  time.Duration // parsed from the Retry-After header, 0 if absent
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("upstream returned %d: %s", e.Status, truncate(string(e.Body), 200))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// ErrCircuitOpen is returned without dialing when the breaker is open.
var ErrCircuitOpen = errors.New("upstream circuit is open; failing fast until the upstream recovers")

type Client struct {
	Base   *url.URL
	APIKey string // optional override; otherwise the client's Authorization header is forwarded
	HTTP   *http.Client

	// Retries is how many extra attempts transient failures get (network
	// errors, 5xx, 429). Embedding calls are idempotent, so retrying is safe.
	Retries int
	// Breaker fails calls fast while the upstream is down; nil disables.
	Breaker *breaker.Breaker
	// OnRetry, if set, is called once per retry attempt (metrics hook).
	OnRetry func()
}

func New(base string, apiKey string, timeout time.Duration) (*Client, error) {
	u, err := url.Parse(base)
	if err != nil {
		return nil, fmt.Errorf("invalid upstream URL %q: %w", base, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return nil, fmt.Errorf("upstream URL must be http(s), got %q", base)
	}
	// the default transport keeps only 2 idle conns per host, which churns
	// sockets into TIME_WAIT under concurrent misses; raise it since we talk
	// to exactly one host
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.MaxIdleConns = 256
	tr.MaxIdleConnsPerHost = 256
	return &Client{
		Base:   u,
		APIKey: apiKey,
		HTTP:   &http.Client{Timeout: timeout, Transport: tr},
	}, nil
}

// Embeddings sends one embeddings request for the given items and returns the
// parsed response with data sorted by index. path is the path the client hit
// (mirrored so vLLM/Ollama/TEI route variants all work). Transient failures
// are retried with exponential backoff; sustained failures open the breaker.
func (c *Client) Embeddings(ctx context.Context, path string, req api.EmbeddingsRequest, items []api.InputItem, clientAuth string) (*api.EmbeddingsResponse, error) {
	if !c.Breaker.Allow() {
		return nil, ErrCircuitOpen
	}
	var lastErr error
	for attempt := 0; attempt <= c.Retries; attempt++ {
		if attempt > 0 {
			if c.OnRetry != nil {
				c.OnRetry()
			}
			select {
			case <-time.After(backoffFor(attempt, lastErr)):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		resp, err := c.embedOnce(ctx, path, req, items, clientAuth)
		if err == nil {
			c.Breaker.Success()
			return resp, nil
		}
		if ctx.Err() != nil {
			return nil, err // caller went away; not the upstream's fault
		}
		if isUpstreamFault(err) {
			c.Breaker.Failure()
		} else {
			// a definitive 4xx means the upstream is alive and this request
			// is just wrong — no retry, no breaker count
			c.Breaker.Success()
			return nil, err
		}
		lastErr = err
	}
	return nil, lastErr
}

// isUpstreamFault reports whether the error indicates the upstream itself is
// unhealthy or overloaded (worth retrying and counting against the breaker).
func isUpstreamFault(err error) bool {
	var he *HTTPError
	if errors.As(err, &he) {
		return he.Status == http.StatusTooManyRequests || he.Status >= 500
	}
	return true // network error, unparseable response, item-count mismatch
}

// backoffFor returns the wait before the given retry attempt: exponential
// with jitter, honoring a Retry-After header when the upstream sent one.
func backoffFor(attempt int, lastErr error) time.Duration {
	var he *HTTPError
	if errors.As(lastErr, &he) && he.RetryAfter > 0 {
		if he.RetryAfter > 5*time.Second {
			return 5 * time.Second
		}
		return he.RetryAfter
	}
	base := 250 * time.Millisecond << (attempt - 1)
	if base > 4*time.Second {
		base = 4 * time.Second
	}
	return base + time.Duration(rand.Int63n(int64(base/4+1)))
}

func (c *Client) embedOnce(ctx context.Context, path string, req api.EmbeddingsRequest, items []api.InputItem, clientAuth string) (*api.EmbeddingsResponse, error) {
	input, err := api.MarshalInputs(items)
	if err != nil {
		return nil, err
	}
	req.Input = input
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	u := *c.Base
	u.Path = strings.TrimRight(u.Path, "/") + path
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, u.String(), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	switch {
	case c.APIKey != "":
		hreq.Header.Set("Authorization", "Bearer "+c.APIKey)
	case clientAuth != "":
		hreq.Header.Set("Authorization", clientAuth)
	}

	resp, err := c.HTTP.Do(hreq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 256<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		he := &HTTPError{Status: resp.StatusCode, ContentType: resp.Header.Get("Content-Type"), Body: respBody}
		if secs, err := strconv.Atoi(resp.Header.Get("Retry-After")); err == nil && secs > 0 {
			he.RetryAfter = time.Duration(secs) * time.Second
		}
		return nil, he
	}
	var parsed api.EmbeddingsResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, fmt.Errorf("upstream returned unparseable response: %w", err)
	}
	if len(parsed.Data) != len(items) {
		return nil, fmt.Errorf("upstream returned %d embeddings for %d inputs", len(parsed.Data), len(items))
	}
	sort.Slice(parsed.Data, func(i, j int) bool { return parsed.Data[i].Index < parsed.Data[j].Index })
	return &parsed, nil
}
