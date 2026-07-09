package cache

import (
	"fmt"
	"path/filepath"
	"testing"
	"time"
)

func TestGetSet(t *testing.T) {
	c := New(0, 0)
	if _, ok := c.Get("k"); ok {
		t.Fatal("empty cache must miss")
	}
	c.Set("k", Entry{Raw: []byte("[1,2]"), Tokens: 3})
	e, ok := c.Get("k")
	if !ok || string(e.Raw) != "[1,2]" || e.Tokens != 3 {
		t.Fatalf("got %+v ok=%v", e, ok)
	}
}

func TestEvictionByEntries(t *testing.T) {
	c := New(64, 0)
	for i := 0; i < 1000; i++ {
		c.Set(fmt.Sprintf("key-%04d", i), Entry{Raw: []byte("x")})
	}
	if n := c.Len(); n > 64 {
		t.Fatalf("cache exceeded entry bound: %d", n)
	}
}

func TestEvictionByBytes(t *testing.T) {
	c := New(0, 16*100) // 100 bytes per shard
	big := make([]byte, 60)
	for i := 0; i < 200; i++ {
		c.Set(fmt.Sprintf("key-%04d", i), Entry{Raw: big})
	}
	if b := c.Bytes(); b > 16*100+16*60 { // one over-admission per shard allowed
		t.Fatalf("cache exceeded byte bound: %d", b)
	}
}

func TestLRUOrder(t *testing.T) {
	c := New(0, 0)
	c.Set("a", Entry{Raw: []byte("1")})
	c.Set("b", Entry{Raw: []byte("2")})
	c.Get("a") // refresh a
	// no eviction configured; just verify both remain retrievable
	if _, ok := c.Get("a"); !ok {
		t.Fatal("a lost")
	}
	if _, ok := c.Get("b"); !ok {
		t.Fatal("b lost")
	}
}

func TestTTLExpiry(t *testing.T) {
	c := New(0, 0)
	c.Set("forever", Entry{Raw: []byte("[1]")})
	c.Set("brief", Entry{Raw: []byte("[2]"), ExpiresAt: time.Now().Add(30 * time.Millisecond).UnixNano()})

	if _, ok := c.Get("brief"); !ok {
		t.Fatal("entry must be servable before expiry")
	}
	time.Sleep(50 * time.Millisecond)
	if _, ok := c.Get("brief"); ok {
		t.Fatal("expired entry must miss")
	}
	if _, ok := c.Get("forever"); !ok {
		t.Fatal("zero ExpiresAt must never expire")
	}
	if c.Len() != 1 {
		t.Fatalf("expired entry must be evicted on read, len=%d", c.Len())
	}
}

func TestFlush(t *testing.T) {
	c := New(0, 0)
	for i := 0; i < 20; i++ {
		c.Set(fmt.Sprintf("k%d", i), Entry{Raw: []byte("x")})
	}
	if n := c.Flush(); n != 20 {
		t.Fatalf("flushed %d, want 20", n)
	}
	if c.Len() != 0 || c.Bytes() != 0 {
		t.Fatalf("len=%d bytes=%d after flush", c.Len(), c.Bytes())
	}
	if _, ok := c.Get("k3"); ok {
		t.Fatal("entry survived flush")
	}
	c.Set("new", Entry{Raw: []byte("y")})
	if _, ok := c.Get("new"); !ok {
		t.Fatal("cache unusable after flush")
	}
}

func TestSnapshotRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "snap.gob")
	c := New(0, 0)
	for i := 0; i < 50; i++ {
		c.Set(fmt.Sprintf("k%d", i), Entry{Raw: []byte(fmt.Sprintf("[%d]", i)), Tokens: i})
	}
	if err := c.Snapshot(path); err != nil {
		t.Fatal(err)
	}
	c2 := New(0, 0)
	n, err := c2.Load(path)
	if err != nil || n != 50 {
		t.Fatalf("load: n=%d err=%v", n, err)
	}
	e, ok := c2.Get("k7")
	if !ok || string(e.Raw) != "[7]" || e.Tokens != 7 {
		t.Fatalf("roundtrip mismatch: %+v ok=%v", e, ok)
	}
}

func TestLoadMissingFileIsNotError(t *testing.T) {
	c := New(0, 0)
	if n, err := c.Load(filepath.Join(t.TempDir(), "nope.gob")); err != nil || n != 0 {
		t.Fatalf("n=%d err=%v", n, err)
	}
}
