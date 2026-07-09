// Package upstream calls the OpenAI-compatible embedding backend.
package upstream

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"

	"embedcache/internal/api"
)

// HTTPError carries a non-2xx upstream response so the proxy can forward it
// to the client unchanged.
type HTTPError struct {
	Status      int
	ContentType string
	Body        []byte
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

type Client struct {
	Base   *url.URL
	APIKey string // optional override; otherwise the client's Authorization header is forwarded
	HTTP   *http.Client
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
// (mirrored so vLLM/Ollama/TEI route variants all work).
func (c *Client) Embeddings(ctx context.Context, path string, req api.EmbeddingsRequest, items []api.InputItem, clientAuth string) (*api.EmbeddingsResponse, error) {
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
		return nil, &HTTPError{Status: resp.StatusCode, ContentType: resp.Header.Get("Content-Type"), Body: respBody}
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
