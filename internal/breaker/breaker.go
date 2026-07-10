// Package breaker is a minimal circuit breaker for the upstream path.
//
// After Threshold consecutive failures the circuit opens and calls fail fast
// instead of stacking timeouts on a dead backend. After Cooldown, exactly one
// probe call is admitted; its outcome closes the circuit or reopens it.
package breaker

import (
	"sync"
	"time"
)

type State int

const (
	Closed State = iota
	Open
	HalfOpen
)

func (s State) String() string {
	switch s {
	case Open:
		return "open"
	case HalfOpen:
		return "half-open"
	default:
		return "closed"
	}
}

type Breaker struct {
	threshold int
	cooldown  time.Duration

	mu       sync.Mutex
	state    State
	failures int
	openedAt time.Time
	probe    bool // a half-open probe is currently in flight
}

// New returns a breaker; threshold <= 0 disables it (always allows).
func New(threshold int, cooldown time.Duration) *Breaker {
	return &Breaker{threshold: threshold, cooldown: cooldown}
}

// Allow reports whether a call may proceed right now. While open it returns
// false until the cooldown elapses, then admits a single probe; further calls
// keep failing fast until that probe resolves.
func (b *Breaker) Allow() bool {
	if b == nil || b.threshold <= 0 {
		return true
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	switch b.state {
	case Closed:
		return true
	case Open:
		if time.Since(b.openedAt) >= b.cooldown {
			b.state = HalfOpen
			b.probe = true
			return true
		}
		return false
	default: // HalfOpen
		if !b.probe {
			b.probe = true
			return true
		}
		return false
	}
}

// Success reports a completed upstream call; it closes the circuit.
func (b *Breaker) Success() {
	if b == nil || b.threshold <= 0 {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.failures = 0
	b.state = Closed
	b.probe = false
}

// Failure reports an upstream-fault failure (network error or 5xx — a 4xx
// means the upstream is alive and must not be counted).
func (b *Breaker) Failure() {
	if b == nil || b.threshold <= 0 {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.state == HalfOpen {
		b.state = Open
		b.openedAt = time.Now()
		b.probe = false
		return
	}
	b.failures++
	if b.failures >= b.threshold {
		b.state = Open
		b.openedAt = time.Now()
	}
}

func (b *Breaker) State() State {
	if b == nil || b.threshold <= 0 {
		return Closed
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	// reflect cooldown expiry in reads too
	if b.state == Open && time.Since(b.openedAt) >= b.cooldown {
		return HalfOpen
	}
	return b.state
}
