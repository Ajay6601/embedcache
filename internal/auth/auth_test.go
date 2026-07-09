package auth

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestOffAllowsEverything(t *testing.T) {
	a, err := New(ModeOff, nil, "", 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := a.Allow(context.Background(), ""); !ok {
		t.Fatal("off mode must allow requests without keys")
	}
}

func TestAllowlist(t *testing.T) {
	a, err := New(ModeAllowlist, []string{"sk-good", " sk-other "}, "", 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		header string
		want   bool
	}{
		{"Bearer sk-good", true},
		{"sk-good", true}, // raw header without Bearer prefix
		{"Bearer sk-other", true},
		{"Bearer sk-evil", false},
		{"", false},
	}
	for _, c := range cases {
		if ok, _ := a.Allow(context.Background(), c.header); ok != c.want {
			t.Errorf("header %q: allowed=%v want %v", c.header, ok, c.want)
		}
	}
}

func TestAllowlistRequiresKeys(t *testing.T) {
	if _, err := New(ModeAllowlist, nil, "", 0, nil); err == nil {
		t.Fatal("allowlist without keys must be a config error")
	}
}

func TestVerifyCachesVerdicts(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.Header.Get("Authorization") == "Bearer sk-valid" {
			w.WriteHeader(200)
			return
		}
		w.WriteHeader(401)
	}))
	defer upstream.Close()

	a, err := New(ModeVerify, nil, upstream.URL, time.Minute, nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		if ok, _ := a.Allow(ctx, "Bearer sk-valid"); !ok {
			t.Fatal("valid key rejected")
		}
	}
	if calls.Load() != 1 {
		t.Fatalf("upstream verified %d times for 5 requests; verdict caching broken", calls.Load())
	}

	for i := 0; i < 3; i++ {
		if ok, _ := a.Allow(ctx, "Bearer sk-bad"); ok {
			t.Fatal("invalid key accepted")
		}
	}
	if calls.Load() != 2 {
		t.Fatalf("upstream called %d times; negative verdicts must also cache", calls.Load())
	}

	if ok, _ := a.Allow(ctx, ""); ok {
		t.Fatal("verify mode must reject missing keys without asking upstream")
	}
}

func TestVerifyFailsClosedOnUpstreamOutage(t *testing.T) {
	dead := httptest.NewServer(nil)
	dead.Close() // now unreachable
	a, err := New(ModeVerify, nil, dead.URL, time.Minute, &http.Client{Timeout: 200 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := a.Allow(context.Background(), "Bearer sk-any"); ok {
		t.Fatal("must fail closed when the upstream cannot be reached")
	}
}
