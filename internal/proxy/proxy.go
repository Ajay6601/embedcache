// Package proxy implements the /v1/embeddings data plane: fingerprint every
// input item, serve what's cached, coalesce what's in flight, and send only
// the remainder upstream — then reassemble the response in original order.
//
// Correctness invariants (see proxy_test.go and the experiments suite):
//   - cached embeddings are byte-exact replicas of the upstream JSON value
//   - batch responses preserve input order and index mapping under any mix
//     of hits, misses, and intra-batch duplicates
//   - a key is computed upstream at most once across concurrent requests
package proxy

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"time"

	"embedcache/internal/api"
	"embedcache/internal/auth"
	"embedcache/internal/cache"
	"embedcache/internal/coalesce"
	"embedcache/internal/fingerprint"
	"embedcache/internal/stats"
	"embedcache/internal/tokens"
	"embedcache/internal/upstream"
)

var errLeaderAborted = errors.New("the request computing this embedding was aborted")

type Proxy struct {
	Cache    *cache.Cache
	Group    *coalesce.Group[cache.Entry]
	Upstream *upstream.Client
	Stats    *stats.Collector
	Norm     fingerprint.Normalizer

	// Auth validates client keys before any cache read; nil means off.
	Auth *auth.Authorizer
	// CacheTTL bounds how long an entry may be served; 0 = forever.
	CacheTTL time.Duration
	// MaxBatchItems rejects oversized batches; 0 = unlimited.
	MaxBatchItems int
	// MaxBody rejects oversized request bodies.
	MaxBody int64

	logMu      sync.Mutex
	requestLog *os.File
}

func New(c *cache.Cache, up *upstream.Client, st *stats.Collector, norm fingerprint.Normalizer) *Proxy {
	return &Proxy{
		Cache:    c,
		Group:    coalesce.New[cache.Entry](),
		Upstream: up,
		Stats:    st,
		Norm:     norm,
		MaxBody:  64 << 20,
	}
}

// SetRequestLog enables JSONL request logging compatible with the offline
// analyzer (`embedcache analyze`).
func (p *Proxy) SetRequestLog(f *os.File) { p.requestLog = f }

func (p *Proxy) ServeEmbeddings(w http.ResponseWriter, r *http.Request) {
	if p.Auth != nil {
		if ok, reason := p.Auth.Allow(r.Context(), r.Header.Get("Authorization")); !ok {
			p.Stats.RecordError()
			writeError(w, http.StatusUnauthorized, reason)
			return
		}
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, p.MaxBody+1))
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body: "+err.Error())
		p.Stats.RecordError()
		return
	}
	if int64(len(body)) > p.MaxBody {
		writeError(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("request body exceeds %d bytes", p.MaxBody))
		p.Stats.RecordError()
		return
	}
	var req api.EmbeddingsRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		p.Stats.RecordError()
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "missing required field: model")
		p.Stats.RecordError()
		return
	}
	items, err := api.SplitInput(req.Input)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		p.Stats.RecordError()
		return
	}
	if p.MaxBatchItems > 0 && len(items) > p.MaxBatchItems {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("batch of %d items exceeds the limit of %d", len(items), p.MaxBatchItems))
		p.Stats.RecordError()
		return
	}
	p.logRequest(&req)

	keys := make([]string, len(items))
	for i := range items {
		keys[i] = fingerprint.Key(req.Model, req.Dimensions, req.EncodingFormat, items[i], p.Norm)
	}

	entries := make([]cache.Entry, len(items))
	var uniqueMissing []string
	missingIdx := map[string][]int{}
	hits, savedTokens := 0, 0
	for i, k := range keys {
		if e, ok := p.Cache.Get(k); ok {
			entries[i] = e
			hits++
			savedTokens += e.Tokens
			continue
		}
		if _, seen := missingIdx[k]; !seen {
			uniqueMissing = append(uniqueMissing, k)
		}
		missingIdx[k] = append(missingIdx[k], i)
	}

	var (
		misses, coalesced, spentTokens int
		upstreamCalls, upstreamItems   int
		respModel                      = req.Model
	)

	if len(uniqueMissing) > 0 {
		owned, waits := p.Group.Claim(uniqueMissing)

		// If this handler exits without resolving an owned key (upstream
		// error path, panic), fail it so waiters do not hang. Keys resolved
		// normally are removed from pending first, so this never touches a
		// call claimed later by someone else.
		pending := make(map[string]bool, len(owned))
		for _, k := range owned {
			pending[k] = true
		}
		defer func() {
			for k := range pending {
				p.Group.Fail(k, errLeaderAborted)
			}
		}()

		if len(owned) > 0 {
			ownedItems := make([]api.InputItem, len(owned))
			for j, k := range owned {
				ownedItems[j] = items[missingIdx[k][0]]
			}
			ureq := api.EmbeddingsRequest{
				Model:          req.Model,
				EncodingFormat: req.EncodingFormat,
				Dimensions:     req.Dimensions,
				User:           req.User,
			}
			uresp, err := p.Upstream.Embeddings(r.Context(), r.URL.Path, ureq, ownedItems, r.Header.Get("Authorization"))
			if err != nil {
				for _, k := range owned {
					p.Group.Fail(k, err)
					delete(pending, k)
				}
				p.Stats.RecordError()
				writeUpstreamError(w, err)
				return
			}
			upstreamCalls++
			upstreamItems += len(owned)
			if uresp.Model != "" {
				respModel = uresp.Model
			}
			perItem := tokens.Apportion(uresp.Usage.PromptTokens, ownedItems)
			var expires int64
			if p.CacheTTL > 0 {
				expires = time.Now().Add(p.CacheTTL).UnixNano()
			}
			for j, k := range owned {
				e := cache.Entry{Raw: uresp.Data[j].Embedding, Tokens: perItem[j], ExpiresAt: expires}
				p.Cache.Set(k, e)
				p.Group.Fulfill(k, e)
				delete(pending, k)
				for _, idx := range missingIdx[k] {
					entries[idx] = e
				}
				misses++
				spentTokens += e.Tokens
				// duplicate occurrences of the same input within this batch
				// were computed once and reused
				if extra := len(missingIdx[k]) - 1; extra > 0 {
					coalesced += extra
					savedTokens += extra * e.Tokens
				}
			}
		}

		for k, call := range waits {
			e, err := call.Wait(r.Context())
			if err != nil {
				p.Stats.RecordError()
				writeError(w, http.StatusBadGateway, "coalesced upstream call failed: "+err.Error())
				return
			}
			for _, idx := range missingIdx[k] {
				entries[idx] = e
			}
			n := len(missingIdx[k])
			coalesced += n
			savedTokens += n * e.Tokens
		}
	}

	p.Stats.Record(stats.RequestRecord{
		Model:         req.Model,
		Caller:        callerHash(r.Header.Get("Authorization")),
		Hits:          hits,
		Misses:        misses,
		Coalesced:     coalesced,
		SavedTokens:   savedTokens,
		SpentTokens:   spentTokens,
		UpstreamCalls: upstreamCalls,
		UpstreamItems: upstreamItems,
	})

	data := make([]api.EmbeddingData, len(items))
	for i := range items {
		data[i] = api.EmbeddingData{Object: "embedding", Index: i, Embedding: entries[i].Raw}
	}
	out := api.EmbeddingsResponse{
		Object: "list",
		Data:   data,
		Model:  respModel,
		// usage reflects what THIS request was billed upstream; cache and
		// coalesced items cost nothing
		Usage: api.Usage{PromptTokens: spentTokens, TotalTokens: spentTokens},
	}
	h := w.Header()
	h.Set("Content-Type", "application/json")
	h.Set("X-Embedcache-Hits", fmt.Sprint(hits))
	h.Set("X-Embedcache-Misses", fmt.Sprint(misses))
	h.Set("X-Embedcache-Coalesced", fmt.Sprint(coalesced))
	h.Set("X-Embedcache-Saved-Tokens", fmt.Sprint(savedTokens))
	h.Set("X-Embedcache-Status", cacheStatus(hits, misses, coalesced))
	json.NewEncoder(w).Encode(out)
}

func cacheStatus(hits, misses, coalesced int) string {
	switch {
	case misses == 0 && hits+coalesced > 0:
		return "hit"
	case hits == 0 && coalesced == 0:
		return "miss"
	default:
		return "partial"
	}
}

func callerHash(auth string) string {
	if auth == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(auth))
	return hex.EncodeToString(sum[:4])
}

func (p *Proxy) logRequest(req *api.EmbeddingsRequest) {
	if p.requestLog == nil {
		return
	}
	line, err := json.Marshal(req)
	if err != nil {
		return
	}
	p.logMu.Lock()
	p.requestLog.Write(append(line, '\n'))
	p.logMu.Unlock()
}

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(api.ErrorResponse{Error: api.ErrorDetail{Message: msg, Type: "embedcache_error"}})
}

func writeUpstreamError(w http.ResponseWriter, err error) {
	var he *upstream.HTTPError
	if errors.As(err, &he) {
		ct := he.ContentType
		if ct == "" {
			ct = "application/json"
		}
		w.Header().Set("Content-Type", ct)
		w.WriteHeader(he.Status)
		w.Write(he.Body)
		return
	}
	writeError(w, http.StatusBadGateway, "upstream request failed: "+err.Error())
}
