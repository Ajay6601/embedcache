package rediscache

import (
	"bytes"
	"os"
	"testing"
	"time"

	"github.com/Ajay6601/embedcache/internal/cache"
)

// redisAddr is where the tests look for a real Redis. They skip (not fail) when
// none is reachable, so CI without Redis stays green while a dev box with Redis
// running exercises the real protocol.
func testClient(t *testing.T) *Client {
	t.Helper()
	addr := os.Getenv("EMBEDCACHE_TEST_REDIS")
	if addr == "" {
		addr = "127.0.0.1:6379"
	}
	c, err := New(Options{Addr: addr, Prefix: "ectest:", TTL: time.Minute})
	if err != nil {
		t.Skipf("no Redis at %s (%v); skipping shared-cache tests", addr, err)
	}
	return c
}

func TestRoundTripByteExact(t *testing.T) {
	c := testClient(t)
	key := "roundtrip-" + time.Now().Format("150405.000000")
	want := cache.Entry{Raw: []byte(`[0.0125,-0.98,0.5]`), Tokens: 7, ExpiresAt: 0}
	c.Set(key, want)
	got, ok := c.Get(key)
	if !ok {
		t.Fatal("expected a shared hit after Set")
	}
	if !bytes.Equal(got.Raw, want.Raw) {
		t.Fatalf("raw bytes not preserved: got %q want %q", got.Raw, want.Raw)
	}
	if got.Tokens != want.Tokens {
		t.Fatalf("tokens not preserved: got %d want %d", got.Tokens, want.Tokens)
	}
}

func TestMissReturnsNoHit(t *testing.T) {
	c := testClient(t)
	if _, ok := c.Get("definitely-not-present-" + time.Now().Format("150405.000000000")); ok {
		t.Fatal("expected miss on an absent key")
	}
}

func TestExpiryDropsEntry(t *testing.T) {
	addr := os.Getenv("EMBEDCACHE_TEST_REDIS")
	if addr == "" {
		addr = "127.0.0.1:6379"
	}
	c, err := New(Options{Addr: addr, Prefix: "ectest:", TTL: 300 * time.Millisecond})
	if err != nil {
		t.Skipf("no Redis at %s (%v)", addr, err)
	}
	key := "expiry-" + time.Now().Format("150405.000000")
	c.Set(key, cache.Entry{Raw: []byte("x"), Tokens: 1})
	if _, ok := c.Get(key); !ok {
		t.Fatal("expected hit immediately after Set")
	}
	time.Sleep(500 * time.Millisecond)
	if _, ok := c.Get(key); ok {
		t.Fatal("expected the entry to expire via Redis TTL")
	}
}

func TestEntryEncodingIsReversible(t *testing.T) {
	// pure unit test of the wire encoding, no Redis needed
	cases := []cache.Entry{
		{Raw: []byte{}, Tokens: 0, ExpiresAt: 0},
		{Raw: []byte(`{"embedding":[1,2,3]}`), Tokens: 42, ExpiresAt: 1_700_000_000_000_000_000},
		{Raw: bytes.Repeat([]byte{0x00, 0xff, 0x7f}, 100), Tokens: -1, ExpiresAt: -5},
	}
	for _, want := range cases {
		got, ok := decodeEntry(encodeEntry(want))
		if !ok {
			t.Fatalf("decode failed for %+v", want)
		}
		if !bytes.Equal(got.Raw, want.Raw) || got.Tokens != want.Tokens || got.ExpiresAt != want.ExpiresAt {
			t.Fatalf("round trip mismatch: got %+v want %+v", got, want)
		}
	}
}
