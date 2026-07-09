// Package coalesce deduplicates concurrent work by key, batch-aware.
//
// Unlike classic singleflight, callers claim many keys at once: the caller
// becomes owner of the keys nobody is computing and a waiter on the keys
// already in flight. Owners must resolve every owned key with Fulfill or
// Fail, or waiters would block until their context expires.
package coalesce

import (
	"context"
	"sync"
)

type Group[T any] struct {
	mu       sync.Mutex
	inflight map[string]*Call[T]
}

type Call[T any] struct {
	done chan struct{}
	val  T
	err  error
}

// Wait blocks until the owner resolves this call or ctx is done.
func (c *Call[T]) Wait(ctx context.Context) (T, error) {
	select {
	case <-c.done:
		return c.val, c.err
	case <-ctx.Done():
		var zero T
		return zero, ctx.Err()
	}
}

func New[T any]() *Group[T] {
	return &Group[T]{inflight: map[string]*Call[T]{}}
}

// Claim partitions keys into those this caller now owns and those already in
// flight elsewhere. Keys must be unique within one call.
func (g *Group[T]) Claim(keys []string) (owned []string, waits map[string]*Call[T]) {
	g.mu.Lock()
	defer g.mu.Unlock()
	waits = map[string]*Call[T]{}
	for _, k := range keys {
		if c, ok := g.inflight[k]; ok {
			waits[k] = c
			continue
		}
		g.inflight[k] = &Call[T]{done: make(chan struct{})}
		owned = append(owned, k)
	}
	return owned, waits
}

func (g *Group[T]) Fulfill(key string, val T) { g.resolve(key, val, nil) }

func (g *Group[T]) Fail(key string, err error) {
	var zero T
	g.resolve(key, zero, err)
}

func (g *Group[T]) resolve(key string, val T, err error) {
	g.mu.Lock()
	c, ok := g.inflight[key]
	delete(g.inflight, key)
	g.mu.Unlock()
	if !ok {
		return
	}
	c.val = val
	c.err = err
	close(c.done)
}
