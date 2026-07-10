package breaker

import (
	"testing"
	"time"
)

func TestOpensAfterThreshold(t *testing.T) {
	b := New(3, time.Hour)
	for i := 0; i < 2; i++ {
		b.Failure()
		if !b.Allow() {
			t.Fatalf("must stay closed below threshold (failure %d)", i+1)
		}
	}
	b.Failure() // third
	if b.Allow() {
		t.Fatal("must open at threshold")
	}
	if b.State() != Open {
		t.Fatalf("state = %v", b.State())
	}
}

func TestSuccessResetsCounter(t *testing.T) {
	b := New(3, time.Hour)
	b.Failure()
	b.Failure()
	b.Success()
	b.Failure()
	b.Failure()
	if !b.Allow() {
		t.Fatal("success must reset the consecutive-failure count")
	}
}

func TestHalfOpenSingleProbe(t *testing.T) {
	b := New(1, 30*time.Millisecond)
	b.Failure()
	if b.Allow() {
		t.Fatal("open circuit must fail fast")
	}
	time.Sleep(40 * time.Millisecond)
	if !b.Allow() {
		t.Fatal("cooldown elapsed: one probe must be admitted")
	}
	if b.Allow() {
		t.Fatal("only ONE probe while half-open")
	}
	b.Success()
	if !b.Allow() {
		t.Fatal("successful probe must close the circuit")
	}
}

func TestFailedProbeReopens(t *testing.T) {
	b := New(1, 20*time.Millisecond)
	b.Failure()
	time.Sleep(30 * time.Millisecond)
	if !b.Allow() {
		t.Fatal("probe expected")
	}
	b.Failure()
	if b.Allow() {
		t.Fatal("failed probe must reopen immediately")
	}
	time.Sleep(30 * time.Millisecond)
	if !b.Allow() {
		t.Fatal("next probe after another cooldown")
	}
}

func TestDisabledAndNil(t *testing.T) {
	b := New(0, time.Second)
	for i := 0; i < 100; i++ {
		b.Failure()
	}
	if !b.Allow() {
		t.Fatal("threshold 0 disables the breaker")
	}
	var nb *Breaker
	if !nb.Allow() {
		t.Fatal("nil breaker must allow")
	}
	nb.Failure() // must not panic
	nb.Success()
}
