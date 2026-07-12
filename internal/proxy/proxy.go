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
	"log"
	"math"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/Ajay6601/embedcache/internal/api"
	"github.com/Ajay6601/embedcache/internal/auth"
	"github.com/Ajay6601/embedcache/internal/budget"
	"github.com/Ajay6601/embedcache/internal/cache"
	"github.com/Ajay6601/embedcache/internal/coalesce"
	"github.com/Ajay6601/embedcache/internal/fingerprint"
	"github.com/Ajay6601/embedcache/internal/semantic"
	"github.com/Ajay6601/embedcache/internal/stats"
	"github.com/Ajay6601/embedcache/internal/tokens"
	"github.com/Ajay6601/embedcache/internal/upstream"
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
	// Budget enforces per-key limits on upstream spend; nil means off.
	Budget *budget.Enforcer
	// CacheTTL bounds how long an entry may be served; 0 = forever.
	CacheTTL time.Duration
	// MaxBatchItems rejects oversized batches; 0 = unlimited.
	MaxBatchItems int
	// MaxBody rejects oversized request bodies.
	MaxBody int64

	// Semantic near-duplicate matching. Off unless SemanticMode is "shadow" or
	// "active" and Semantic is non-nil. In shadow mode nothing about serving
	// changes — the proxy only measures how far a near-neighbor's vector is from
	// the real one, so you can decide whether active mode is safe on your data.
	// In active mode a near-duplicate above SemanticThreshold is served the
	// neighbor's cached vector instead of being computed. Only text inputs
	// participate; token-array inputs never do.
	Semantic          *semantic.Index
	SemanticMode      string  // "off" | "shadow" | "active"
	SemanticThreshold float64 // Jaccard similarity in [0,1]

	logMu   sync.Mutex
	logFile *os.File
	logPath string
	logSize int64
	logMax  int64
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

// OpenRequestLog enables JSONL request logging compatible with the offline
// analyzer (`embedcache analyze`). When the log reaches maxBytes it is
// rotated to path+".1" (one generation kept), so it cannot fill the disk.
func (p *Proxy) OpenRequestLog(path string, maxBytes int64) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return err
	}
	p.logMu.Lock()
	defer p.logMu.Unlock()
	p.logFile, p.logPath, p.logSize, p.logMax = f, path, fi.Size(), maxBytes
	return nil
}

func (p *Proxy) CloseRequestLog() {
	p.logMu.Lock()
	defer p.logMu.Unlock()
	if p.logFile != nil {
		p.logFile.Close()
		p.logFile = nil
	}
}

// rotateLogLocked is called with logMu held when the size budget is spent.
func (p *Proxy) rotateLogLocked() {
	p.logFile.Close()
	old := p.logPath + ".1"
	os.Remove(old)
	if err := os.Rename(p.logPath, old); err != nil {
		log.Printf("request log rotation failed: %v", err)
	}
	f, err := os.OpenFile(p.logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		log.Printf("request log reopen failed, logging disabled: %v", err)
		p.logFile = nil
		return
	}
	p.logFile = f
	p.logSize = 0
}

func (p *Proxy) ServeEmbeddings(w http.ResponseWriter, r *http.Request) {
	// Azure-style clients authenticate with an api-key header instead of
	// Authorization; treat either as the caller credential
	credential := r.Header.Get("Authorization")
	if credential == "" {
		credential = r.Header.Get("api-key")
	}
	if p.Auth != nil {
		if ok, reason := p.Auth.Allow(r.Context(), credential); !ok {
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

	paramsDigest := fingerprint.ParamsDigest(req.Extra)
	keys := make([]string, len(items))
	for i := range items {
		keys[i] = fingerprint.Key(req.Model, req.Dimensions, req.EncodingFormat, paramsDigest, items[i], p.Norm)
	}

	entries := make([]cache.Entry, len(items))
	var uniqueMissing []string
	missingIdx := map[string][]int{}
	hits, savedTokens, semanticHits := 0, 0, 0
	// shadow-mode observations: for a missed item that had a near neighbor, the
	// item index, that neighbor's cached vector, and their text similarity — so
	// after the real vector is computed we can measure how wrong serving the
	// neighbor would have been, without actually serving it.
	type shadowCand struct {
		idx      int
		neighbor cache.Entry
		sim      float64
	}
	var shadow []shadowCand
	semOn := p.Semantic != nil && (p.SemanticMode == "shadow" || p.SemanticMode == "active")
	for i, k := range keys {
		if e, ok := p.Cache.Get(k); ok {
			entries[i] = e
			hits++
			savedTokens += e.Tokens
			continue
		}
		// exact-match miss. If semantic matching is on and this is a text input,
		// look for a near-duplicate already in the cache.
		if semOn && !items[i].IsTokens {
			if nk, sim, ok := p.Semantic.Nearest(items[i].Text); ok && sim >= p.SemanticThreshold {
				if ne, cached := p.Cache.Get(nk); cached {
					if p.SemanticMode == "active" {
						// serve the neighbor's vector instead of computing
						entries[i] = ne
						semanticHits++
						savedTokens += ne.Tokens
						continue
					}
					// shadow: remember it, but fall through and compute for real
					shadow = append(shadow, shadowCand{idx: i, neighbor: ne, sim: sim})
				}
			}
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

	// Budgets bound upstream SPEND, not reads: a fully-cached request is
	// served even when the caller's budget is exhausted. Only requests that
	// need new computation are rejected — the cache keeps a capped tenant's
	// existing workload alive while blocking new cost.
	if len(uniqueMissing) > 0 {
		if ok, retryAfter := p.Budget.Allow(credential); !ok {
			p.Stats.RecordBudgetReject()
			w.Header().Set("Retry-After", fmt.Sprint(int(retryAfter.Seconds())+1))
			writeError(w, http.StatusTooManyRequests,
				fmt.Sprintf("token budget exhausted for this key; window resets in %s", retryAfter.Round(time.Second)))
			return
		}
	}

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
				Extra:          req.Extra,
			}
			meta := upstream.RequestMeta{
				Path:          r.URL.Path,
				RawQuery:      r.URL.RawQuery,
				Authorization: r.Header.Get("Authorization"),
				APIKeyHeader:  r.Header.Get("api-key"), // Azure OpenAI style
			}
			uresp, err := p.Upstream.Embeddings(r.Context(), meta, ureq, ownedItems)
			if err != nil {
				for _, k := range owned {
					p.Group.Fail(k, err)
					delete(pending, k)
				}
				p.Stats.RecordError()
				if errors.Is(err, upstream.ErrCircuitOpen) {
					p.Stats.RecordFastFail()
				}
				writeUpstreamError(w, err)
				return
			}
			upstreamCalls++
			upstreamItems += len(owned)
			if uresp.Model != "" {
				respModel = uresp.Model
			}
			// some providers (Voyage) report only total_tokens
			billed := uresp.Usage.PromptTokens
			if billed == 0 {
				billed = uresp.Usage.TotalTokens
			}
			perItem := tokens.Apportion(billed, ownedItems)
			var expires int64
			if p.CacheTTL > 0 {
				expires = time.Now().Add(p.CacheTTL).UnixNano()
			}
			for j, k := range owned {
				e := cache.Entry{Raw: uresp.Data[j].Embedding, Tokens: perItem[j], ExpiresAt: expires}
				p.Cache.Set(k, e)
				p.Group.Fulfill(k, e)
				delete(pending, k)
				// index this input's text so future near-duplicates can find it
				if semOn {
					if it := items[missingIdx[k][0]]; !it.IsTokens {
						p.Semantic.Add(k, it.Text)
					}
				}
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
			p.Budget.Record(credential, spentTokens)
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

	// shadow mode: now that the real vectors exist, measure how close each near
	// neighbor's cached vector was to the truth. This is the number that tells
	// you whether active semantic caching would be safe on your data — recorded,
	// never served.
	for _, c := range shadow {
		real, ok1 := parseVector(entries[c.idx].Raw)
		cand, ok2 := parseVector(c.neighbor.Raw)
		if ok1 && ok2 {
			p.Stats.RecordSemanticShadow(c.sim, cosineSim(real, cand))
		}
	}

	p.Stats.Record(stats.RequestRecord{
		Model:         req.Model,
		Caller:        callerHash(credential),
		Hits:          hits,
		Misses:        misses,
		Coalesced:     coalesced,
		SavedTokens:   savedTokens,
		SpentTokens:   spentTokens,
		SemanticHits:  semanticHits,
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
	// a semantic (near-duplicate) hit is served from cache like an exact hit, so
	// it counts toward Hits and the overall status; the separate header lets a
	// caller see how many were approximate.
	h.Set("X-Embedcache-Hits", fmt.Sprint(hits+semanticHits))
	h.Set("X-Embedcache-Misses", fmt.Sprint(misses))
	h.Set("X-Embedcache-Coalesced", fmt.Sprint(coalesced))
	if semanticHits > 0 {
		h.Set("X-Embedcache-Semantic-Hits", fmt.Sprint(semanticHits))
	}
	h.Set("X-Embedcache-Saved-Tokens", fmt.Sprint(savedTokens))
	h.Set("X-Embedcache-Status", cacheStatus(hits+semanticHits, misses, coalesced))
	if rem, limited := p.Budget.Remaining(credential); limited {
		h.Set("X-Embedcache-Budget-Remaining", fmt.Sprint(rem))
	}
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

// parseVector decodes a float-array embedding from its raw JSON. Base64-encoded
// vectors return ok=false (shadow evaluation simply skips them).
func parseVector(raw []byte) ([]float64, bool) {
	var v []float64
	if json.Unmarshal(raw, &v) == nil && len(v) > 0 {
		return v, true
	}
	return nil, false
}

func cosineSim(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// callerHash normalizes a credential to a short display hash. The Bearer
// prefix is stripped so the same key hashes identically however it arrives,
// and so per-caller stats line up with the budget report's key hashes.
func callerHash(credential string) string {
	if credential == "" {
		return ""
	}
	if t, ok := strings.CutPrefix(credential, "Bearer "); ok {
		credential = t
	}
	sum := sha256.Sum256([]byte(credential))
	return hex.EncodeToString(sum[:4])
}

func (p *Proxy) logRequest(req *api.EmbeddingsRequest) {
	p.logMu.Lock()
	defer p.logMu.Unlock()
	if p.logFile == nil {
		return
	}
	line, err := json.Marshal(req)
	if err != nil {
		return
	}
	line = append(line, '\n')
	if p.logMax > 0 && p.logSize+int64(len(line)) > p.logMax {
		p.rotateLogLocked()
		if p.logFile == nil {
			return
		}
	}
	n, _ := p.logFile.Write(line)
	p.logSize += int64(n)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(api.ErrorResponse{Error: api.ErrorDetail{Message: msg, Type: "embedcache_error"}})
}

func writeUpstreamError(w http.ResponseWriter, err error) {
	if errors.Is(err, upstream.ErrCircuitOpen) {
		writeError(w, http.StatusServiceUnavailable, err.Error())
		return
	}
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
