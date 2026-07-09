package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"embedcache/internal/cache"
	"embedcache/internal/fingerprint"
	"embedcache/internal/pricing"
	"embedcache/internal/proxy"
	"embedcache/internal/stats"
	"embedcache/internal/upstream"
)

func newTestServer(t *testing.T, adminToken string) *httptest.Server {
	t.Helper()
	up, err := upstream.New("http://127.0.0.1:1", "", 0) // never dialed in these tests
	if err != nil {
		t.Fatal(err)
	}
	base, _ := url.Parse("http://127.0.0.1:1")
	s := New(proxy.New(cache.New(0, 0), up, stats.New(), fingerprint.Normalizer{}), stats.New(), pricing.Default(), base)
	s.AdminToken = adminToken
	srv := httptest.NewServer(s.Handler())
	t.Cleanup(srv.Close)
	return srv
}

func get(t *testing.T, url, token string) int {
	t.Helper()
	req, _ := http.NewRequest(http.MethodGet, url, nil)
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	return resp.StatusCode
}

func TestAdminEndpointsRequireToken(t *testing.T) {
	srv := newTestServer(t, "secret-token")
	for _, path := range []string{"/_ec/stats", "/stats", "/_ec/report", "/report", "/metrics"} {
		if code := get(t, srv.URL+path, ""); code != 401 {
			t.Errorf("%s without token: %d, want 401", path, code)
		}
		if code := get(t, srv.URL+path, "wrong"); code != 401 {
			t.Errorf("%s with wrong token: %d, want 401", path, code)
		}
		if code := get(t, srv.URL+path, "secret-token"); code != 200 {
			t.Errorf("%s with token: %d, want 200", path, code)
		}
	}
}

func TestFlushRequiresToken(t *testing.T) {
	srv := newTestServer(t, "secret-token")
	resp, err := http.Post(srv.URL+"/_ec/flush", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 401 {
		t.Fatalf("flush without token: %d, want 401", resp.StatusCode)
	}
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/_ec/flush", nil)
	req.Header.Set("Authorization", "Bearer secret-token")
	resp2, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp2.Body)
	resp2.Body.Close()
	if resp2.StatusCode != 200 || !strings.Contains(string(body), "flushed") {
		t.Fatalf("flush with token: %d %s", resp2.StatusCode, body)
	}
}

func TestSnapshotEndpoint(t *testing.T) {
	srv := newTestServer(t, "secret-token")
	// not configured -> 400 even with a valid token
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/_ec/snapshot", nil)
	req.Header.Set("Authorization", "Bearer secret-token")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Fatalf("snapshot without persistence: %d, want 400", resp.StatusCode)
	}
	// guarded like other admin routes
	resp2, err := http.Post(srv.URL+"/_ec/snapshot", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp2.Body.Close()
	if resp2.StatusCode != 401 {
		t.Fatalf("snapshot without token: %d, want 401", resp2.StatusCode)
	}
}

func TestHealthzAlwaysOpen(t *testing.T) {
	srv := newTestServer(t, "secret-token")
	if code := get(t, srv.URL+"/healthz", ""); code != 200 {
		t.Fatalf("healthz must stay open for probes, got %d", code)
	}
}

func TestNoTokenMeansOpenAdmin(t *testing.T) {
	srv := newTestServer(t, "")
	if code := get(t, srv.URL+"/stats", ""); code != 200 {
		t.Fatalf("empty AdminToken should keep current behavior, got %d", code)
	}
}
