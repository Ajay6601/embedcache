package coalesce

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestSingleOwnerAcrossConcurrentClaims(t *testing.T) {
	g := New[int]()
	var owners atomic.Int32
	var wg sync.WaitGroup
	results := make([]int, 100)
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			owned, waits := g.Claim([]string{"k"})
			if len(owned) == 1 {
				owners.Add(1)
				time.Sleep(5 * time.Millisecond) // let others pile up as waiters
				g.Fulfill("k", 42)
				results[i] = 42
				return
			}
			v, err := waits["k"].Wait(context.Background())
			if err != nil {
				t.Error(err)
			}
			results[i] = v
		}(i)
	}
	wg.Wait()
	if owners.Load() != 1 {
		t.Fatalf("expected exactly 1 owner, got %d", owners.Load())
	}
	for i, v := range results {
		if v != 42 {
			t.Fatalf("caller %d got %d", i, v)
		}
	}
}

func TestFailPropagates(t *testing.T) {
	g := New[int]()
	owned, _ := g.Claim([]string{"k"})
	if len(owned) != 1 {
		t.Fatal("expected ownership")
	}
	_, waits := g.Claim([]string{"k"})
	boom := errors.New("boom")
	go g.Fail("k", boom)
	if _, err := waits["k"].Wait(context.Background()); !errors.Is(err, boom) {
		t.Fatalf("want boom, got %v", err)
	}
}

func TestWaitRespectsContext(t *testing.T) {
	g := New[int]()
	g.Claim([]string{"k"}) // owned, never resolved
	_, waits := g.Claim([]string{"k"})
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := waits["k"].Wait(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("want deadline exceeded, got %v", err)
	}
}

func TestResolvedKeyCanBeClaimedAgain(t *testing.T) {
	g := New[int]()
	g.Claim([]string{"k"})
	g.Fulfill("k", 1)
	owned, _ := g.Claim([]string{"k"})
	if len(owned) != 1 {
		t.Fatal("a resolved key must be claimable again")
	}
}

func TestBatchClaimPartition(t *testing.T) {
	g := New[int]()
	ownedA, _ := g.Claim([]string{"a", "b"})
	if len(ownedA) != 2 {
		t.Fatalf("first claim should own both, got %v", ownedA)
	}
	ownedB, waits := g.Claim([]string{"b", "c"})
	if len(ownedB) != 1 || ownedB[0] != "c" {
		t.Fatalf("second claim should own only c, got %v", ownedB)
	}
	if _, ok := waits["b"]; !ok {
		t.Fatal("second claim should wait on b")
	}
}
