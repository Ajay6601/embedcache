// Package budget enforces hard per-key token budgets on upstream spend.
//
// A budget bounds what a caller may SPEND, not what it may read: cache hits
// cost nothing upstream and are served even after a key's budget is
// exhausted. Only requests that would trigger new upstream computation are
// rejected. Counters live in memory and reset at fixed window boundaries
// (and on restart) — the same durability contract as any in-process rate
// limiter.
package budget

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"sync"
	"time"
)

type Enforcer struct {
	window       time.Duration
	defaultLimit int64
	limits       map[string]int64 // sha256(bare key) -> tokens per window

	mu          sync.Mutex
	spent       map[string]int64
	windowStart time.Time
}

// Status is one key's budget state, as exposed on /_ec/stats.
type Status struct {
	Limit          int64 `json:"limit"`
	Spent          int64 `json:"spent"`
	Remaining      int64 `json:"remaining"`
	ResetsInSecond int64 `json:"resets_in_seconds"`
}

// New creates an enforcer. defaultLimit applies to every key without an
// explicit limit; 0 means those keys are unlimited.
func New(defaultLimit int64, window time.Duration) *Enforcer {
	if window <= 0 {
		window = 24 * time.Hour
	}
	return &Enforcer{
		window:       window,
		defaultLimit: defaultLimit,
		limits:       map[string]int64{},
		spent:        map[string]int64{},
		windowStart:  time.Now(),
	}
}

// SetLimit assigns a per-window limit to one raw API key. 0 makes that key
// explicitly unlimited even when a default limit is set.
func (e *Enforcer) SetLimit(rawKey string, tokens int64) {
	e.limits[hashKey(rawKey)] = tokens
}

// keyFor normalizes a caller credential (Authorization header value or bare
// api-key) to the budget bucket it belongs to. All anonymous callers share
// one bucket.
func keyFor(credential string) string {
	if t, ok := strings.CutPrefix(credential, "Bearer "); ok {
		credential = t
	}
	return hashKey(credential)
}

func hashKey(k string) string {
	sum := sha256.Sum256([]byte(k))
	return hex.EncodeToString(sum[:])
}

func (e *Enforcer) limitFor(key string) int64 {
	if l, ok := e.limits[key]; ok {
		return l
	}
	return e.defaultLimit
}

// rollLocked advances the window if it has elapsed, clearing all counters.
// Boundaries stay fixed relative to the first window's start.
func (e *Enforcer) rollLocked(now time.Time) {
	elapsed := now.Sub(e.windowStart)
	if elapsed < e.window {
		return
	}
	steps := elapsed / e.window
	e.windowStart = e.windowStart.Add(steps * e.window)
	e.spent = map[string]int64{}
}

// Allow reports whether the caller may incur new upstream spend. When the
// budget is exhausted it returns how long until the window resets.
func (e *Enforcer) Allow(credential string) (ok bool, retryAfter time.Duration) {
	if e == nil {
		return true, 0
	}
	now := time.Now()
	key := keyFor(credential)
	e.mu.Lock()
	defer e.mu.Unlock()
	e.rollLocked(now)
	limit := e.limitFor(key)
	if limit <= 0 || e.spent[key] < limit {
		return true, 0
	}
	return false, e.window - now.Sub(e.windowStart)
}

// Record adds upstream-billed tokens to the caller's window. A request that
// starts under budget may finish over it; the overshoot counts and the next
// spend attempt is rejected.
func (e *Enforcer) Record(credential string, tokens int) {
	if e == nil || tokens <= 0 {
		return
	}
	now := time.Now()
	key := keyFor(credential)
	e.mu.Lock()
	defer e.mu.Unlock()
	e.rollLocked(now)
	e.spent[key] += int64(tokens)
}

// Remaining returns the caller's remaining budget this window, and whether a
// limit applies at all.
func (e *Enforcer) Remaining(credential string) (int64, bool) {
	if e == nil {
		return 0, false
	}
	now := time.Now()
	key := keyFor(credential)
	e.mu.Lock()
	defer e.mu.Unlock()
	e.rollLocked(now)
	limit := e.limitFor(key)
	if limit <= 0 {
		return 0, false
	}
	rem := limit - e.spent[key]
	if rem < 0 {
		rem = 0
	}
	return rem, true
}

// Report exposes the state of every key that has a limit or spend this
// window, keyed by a short display hash.
func (e *Enforcer) Report() map[string]Status {
	if e == nil {
		return nil
	}
	now := time.Now()
	e.mu.Lock()
	defer e.mu.Unlock()
	e.rollLocked(now)
	resets := int64((e.window - now.Sub(e.windowStart)).Seconds())
	out := map[string]Status{}
	add := func(key string) {
		disp := key[:8]
		if _, done := out[disp]; done {
			return
		}
		limit := e.limitFor(key)
		spent := e.spent[key]
		rem := limit - spent
		if limit <= 0 {
			rem = -1 // unlimited
		} else if rem < 0 {
			rem = 0
		}
		out[disp] = Status{Limit: limit, Spent: spent, Remaining: rem, ResetsInSecond: resets}
	}
	for k := range e.limits {
		add(k)
	}
	for k := range e.spent {
		add(k)
	}
	return out
}
