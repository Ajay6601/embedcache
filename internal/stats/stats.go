// Package stats accumulates the numbers that make the waste report: what was
// asked for, what was served from cache, what was actually paid for upstream.
package stats

import (
	"fmt"
	"io"
	"sort"
	"sync"
	"time"

	"github.com/Ajay6601/embedcache/internal/budget"
	"github.com/Ajay6601/embedcache/internal/pricing"
)

type Bucket struct {
	Items       uint64 `json:"items"`
	Hits        uint64 `json:"hits"`
	Misses      uint64 `json:"misses"`
	SavedTokens uint64 `json:"saved_tokens"`
	SpentTokens uint64 `json:"spent_tokens"`
}

const maxCallers = 1000

type Collector struct {
	mu    sync.Mutex
	start time.Time

	Requests    uint64
	Passthrough uint64
	Errors      uint64

	Items     uint64
	Hits      uint64
	Misses    uint64
	Coalesced uint64 // items served by waiting on another request's upstream call

	UpstreamCalls uint64
	UpstreamItems uint64

	Retries       uint64 // upstream attempts beyond the first
	FastFails     uint64 // requests rejected because the circuit was open
	BudgetRejects uint64 // requests rejected because the key's budget was spent

	SavedTokens uint64
	SpentTokens uint64

	perModel  map[string]*Bucket
	perCaller map[string]*Bucket
}

func New() *Collector {
	return &Collector{
		start:     time.Now(),
		perModel:  map[string]*Bucket{},
		perCaller: map[string]*Bucket{},
	}
}

type RequestRecord struct {
	Model         string
	Caller        string // hashed identity of the API key, "" if none
	Hits          int
	Misses        int
	Coalesced     int
	SavedTokens   int
	SpentTokens   int
	UpstreamCalls int
	UpstreamItems int
}

func (c *Collector) Record(r RequestRecord) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.Requests++
	items := uint64(r.Hits + r.Misses + r.Coalesced)
	c.Items += items
	c.Hits += uint64(r.Hits)
	c.Misses += uint64(r.Misses)
	c.Coalesced += uint64(r.Coalesced)
	c.UpstreamCalls += uint64(r.UpstreamCalls)
	c.UpstreamItems += uint64(r.UpstreamItems)
	c.SavedTokens += uint64(r.SavedTokens)
	c.SpentTokens += uint64(r.SpentTokens)

	mb := c.bucket(c.perModel, r.Model)
	cb := c.bucket(c.perCaller, r.Caller)
	for _, b := range []*Bucket{mb, cb} {
		if b == nil {
			continue
		}
		b.Items += items
		b.Hits += uint64(r.Hits + r.Coalesced)
		b.Misses += uint64(r.Misses)
		b.SavedTokens += uint64(r.SavedTokens)
		b.SpentTokens += uint64(r.SpentTokens)
	}
}

func (c *Collector) bucket(m map[string]*Bucket, key string) *Bucket {
	if key == "" {
		return nil
	}
	b, ok := m[key]
	if !ok {
		if len(m) >= maxCallers {
			return nil
		}
		b = &Bucket{}
		m[key] = b
	}
	return b
}

func (c *Collector) RecordPassthrough() {
	c.mu.Lock()
	c.Passthrough++
	c.mu.Unlock()
}

func (c *Collector) RecordError() {
	c.mu.Lock()
	c.Errors++
	c.mu.Unlock()
}

func (c *Collector) RecordRetry() {
	c.mu.Lock()
	c.Retries++
	c.mu.Unlock()
}

func (c *Collector) RecordFastFail() {
	c.mu.Lock()
	c.FastFails++
	c.mu.Unlock()
}

func (c *Collector) RecordBudgetReject() {
	c.mu.Lock()
	c.BudgetRejects++
	c.mu.Unlock()
}

type Report struct {
	UptimeSeconds float64                  `json:"uptime_seconds"`
	Requests      uint64                   `json:"requests"`
	Passthrough   uint64                   `json:"passthrough_requests"`
	Errors        uint64                   `json:"errors"`
	Items         uint64                   `json:"items"`
	Hits          uint64                   `json:"hits"`
	Misses        uint64                   `json:"misses"`
	Coalesced     uint64                   `json:"coalesced"`
	HitRate       float64                  `json:"hit_rate"`
	UpstreamCalls uint64                   `json:"upstream_calls"`
	UpstreamItems uint64                   `json:"upstream_items"`
	Retries       uint64                   `json:"upstream_retries"`
	FastFails     uint64                   `json:"breaker_fast_fails"`
	BreakerOpen   bool                     `json:"breaker_open"`
	BudgetRejects uint64                   `json:"budget_rejects"`
	Budgets       map[string]budget.Status `json:"budgets,omitempty"`
	SavedTokens   uint64                   `json:"saved_tokens"`
	SpentTokens   uint64                   `json:"spent_tokens"`
	SavedUSD      float64                  `json:"saved_usd"`
	SpentUSD      float64                  `json:"spent_usd"`
	CacheEntries  int                      `json:"cache_entries"`
	CacheBytes    int64                    `json:"cache_bytes"`
	PerModel      map[string]*Bucket       `json:"per_model"`
	PerCaller     map[string]*Bucket       `json:"per_caller,omitempty"`
}

func (c *Collector) Snapshot(table *pricing.Table, cacheEntries int, cacheBytes int64) Report {
	c.mu.Lock()
	defer c.mu.Unlock()
	r := Report{
		UptimeSeconds: time.Since(c.start).Seconds(),
		Requests:      c.Requests,
		Passthrough:   c.Passthrough,
		Errors:        c.Errors,
		Items:         c.Items,
		Hits:          c.Hits,
		Misses:        c.Misses,
		Coalesced:     c.Coalesced,
		UpstreamCalls: c.UpstreamCalls,
		UpstreamItems: c.UpstreamItems,
		Retries:       c.Retries,
		FastFails:     c.FastFails,
		BudgetRejects: c.BudgetRejects,
		SavedTokens:   c.SavedTokens,
		SpentTokens:   c.SpentTokens,
		CacheEntries:  cacheEntries,
		CacheBytes:    cacheBytes,
		PerModel:      map[string]*Bucket{},
		PerCaller:     map[string]*Bucket{},
	}
	if r.Items > 0 {
		r.HitRate = float64(c.Hits+c.Coalesced) / float64(r.Items)
	}
	for k, b := range c.perModel {
		cp := *b
		r.PerModel[k] = &cp
		r.SavedUSD += table.Cost(k, int64(b.SavedTokens))
		r.SpentUSD += table.Cost(k, int64(b.SpentTokens))
	}
	for k, b := range c.perCaller {
		cp := *b
		r.PerCaller[k] = &cp
	}
	return r
}

// RenderText prints the human waste report.
func (r Report) RenderText(w io.Writer) {
	fmt.Fprintf(w, "embedcache waste report (uptime %s)\n", time.Duration(r.UptimeSeconds*float64(time.Second)).Round(time.Second))
	fmt.Fprintf(w, "================================================\n")
	fmt.Fprintf(w, "embedding requests     %d   (passthrough: %d, errors: %d)\n", r.Requests, r.Passthrough, r.Errors)
	fmt.Fprintf(w, "embedding items        %d\n", r.Items)
	fmt.Fprintf(w, "  served from cache    %d\n", r.Hits)
	fmt.Fprintf(w, "  coalesced in flight  %d\n", r.Coalesced)
	fmt.Fprintf(w, "  sent upstream        %d  (in %d calls, %d retries)\n", r.Misses, r.UpstreamCalls, r.Retries)
	if r.FastFails > 0 || r.BreakerOpen {
		state := "closed"
		if r.BreakerOpen {
			state = "OPEN"
		}
		fmt.Fprintf(w, "circuit breaker        %s   (%d requests failed fast)\n", state, r.FastFails)
	}
	if r.BudgetRejects > 0 || len(r.Budgets) > 0 {
		fmt.Fprintf(w, "budget rejections      %d\n", r.BudgetRejects)
		keys := make([]string, 0, len(r.Budgets))
		for k := range r.Budgets {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			b := r.Budgets[k]
			if b.Remaining < 0 {
				fmt.Fprintf(w, "  key %s          unlimited (spent %d this window)\n", k, b.Spent)
				continue
			}
			fmt.Fprintf(w, "  key %s          %d / %d tokens spent, resets in %s\n",
				k, b.Spent, b.Limit, time.Duration(b.ResetsInSecond)*time.Second)
		}
	}
	fmt.Fprintf(w, "hit rate               %.1f%%\n", r.HitRate*100)
	fmt.Fprintf(w, "tokens paid upstream   %d   ($%.4f)\n", r.SpentTokens, r.SpentUSD)
	fmt.Fprintf(w, "tokens saved           %d   ($%.4f)\n", r.SavedTokens, r.SavedUSD)
	if r.SpentTokens+r.SavedTokens > 0 {
		pct := float64(r.SavedTokens) / float64(r.SpentTokens+r.SavedTokens) * 100
		fmt.Fprintf(w, "duplicate share        %.1f%% of embedding spend was duplicate work\n", pct)
	}
	fmt.Fprintf(w, "cache                  %d entries, %.1f MB\n", r.CacheEntries, float64(r.CacheBytes)/1e6)
	if len(r.PerModel) > 0 {
		fmt.Fprintf(w, "\nper model:\n")
		keys := make([]string, 0, len(r.PerModel))
		for k := range r.PerModel {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			b := r.PerModel[k]
			fmt.Fprintf(w, "  %-30s items=%-8d hits=%-8d saved_tokens=%d\n", k, b.Items, b.Hits, b.SavedTokens)
		}
	}
}

// RenderPrometheus emits counters in Prometheus text exposition format.
func (r Report) RenderPrometheus(w io.Writer) {
	fmt.Fprintf(w, "# TYPE embedcache_requests_total counter\nembedcache_requests_total %d\n", r.Requests)
	fmt.Fprintf(w, "# TYPE embedcache_passthrough_requests_total counter\nembedcache_passthrough_requests_total %d\n", r.Passthrough)
	fmt.Fprintf(w, "# TYPE embedcache_errors_total counter\nembedcache_errors_total %d\n", r.Errors)
	fmt.Fprintf(w, "# TYPE embedcache_items_total counter\n")
	fmt.Fprintf(w, "embedcache_items_total{result=\"hit\"} %d\n", r.Hits)
	fmt.Fprintf(w, "embedcache_items_total{result=\"miss\"} %d\n", r.Misses)
	fmt.Fprintf(w, "embedcache_items_total{result=\"coalesced\"} %d\n", r.Coalesced)
	fmt.Fprintf(w, "# TYPE embedcache_upstream_calls_total counter\nembedcache_upstream_calls_total %d\n", r.UpstreamCalls)
	fmt.Fprintf(w, "# TYPE embedcache_upstream_retries_total counter\nembedcache_upstream_retries_total %d\n", r.Retries)
	fmt.Fprintf(w, "# TYPE embedcache_breaker_fast_fails_total counter\nembedcache_breaker_fast_fails_total %d\n", r.FastFails)
	fmt.Fprintf(w, "# TYPE embedcache_budget_rejects_total counter\nembedcache_budget_rejects_total %d\n", r.BudgetRejects)
	open := 0
	if r.BreakerOpen {
		open = 1
	}
	fmt.Fprintf(w, "# TYPE embedcache_breaker_open gauge\nembedcache_breaker_open %d\n", open)
	fmt.Fprintf(w, "# TYPE embedcache_upstream_items_total counter\nembedcache_upstream_items_total %d\n", r.UpstreamItems)
	fmt.Fprintf(w, "# TYPE embedcache_saved_tokens_total counter\nembedcache_saved_tokens_total %d\n", r.SavedTokens)
	fmt.Fprintf(w, "# TYPE embedcache_spent_tokens_total counter\nembedcache_spent_tokens_total %d\n", r.SpentTokens)
	fmt.Fprintf(w, "# TYPE embedcache_saved_usd gauge\nembedcache_saved_usd %f\n", r.SavedUSD)
	fmt.Fprintf(w, "# TYPE embedcache_cache_entries gauge\nembedcache_cache_entries %d\n", r.CacheEntries)
	fmt.Fprintf(w, "# TYPE embedcache_cache_bytes gauge\nembedcache_cache_bytes %d\n", r.CacheBytes)
}
