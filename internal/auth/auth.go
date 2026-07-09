// Package auth validates client API keys on the data plane. Without it,
// cache hits are served to anyone who can reach the port — only misses get
// (implicitly) validated by the upstream.
//
// Three modes:
//
//	off       — current behavior, no validation (trusted network)
//	allowlist — the proxy holds the list of accepted client keys; right for
//	            self-hosted upstreams that have no auth of their own
//	verify    — the caller's key is checked against the upstream (GET
//	            /v1/models) and the verdict is cached; right for hosted
//	            providers where the upstream owns the key space
package auth

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

type Mode string

const (
	ModeOff       Mode = "off"
	ModeAllowlist Mode = "allowlist"
	ModeVerify    Mode = "verify"
)

const (
	negativeTTL = time.Minute // how long a rejected key stays rejected without re-checking
	maxVerdicts = 10000       // bound on the verdict cache
)

type verdict struct {
	ok    bool
	until time.Time
}

type Authorizer struct {
	mode      Mode
	allowed   map[string]bool // set of sha256(client key)
	verifyURL string
	http      *http.Client
	ttl       time.Duration

	mu       sync.Mutex
	verdicts map[string]verdict
}

// New builds an authorizer. keys are the raw client keys for allowlist mode;
// upstreamBase is the backend base URL for verify mode; ttl is how long a
// positive verify verdict is trusted.
func New(mode Mode, keys []string, upstreamBase string, ttl time.Duration, client *http.Client) (*Authorizer, error) {
	a := &Authorizer{
		mode:     mode,
		allowed:  map[string]bool{},
		ttl:      ttl,
		http:     client,
		verdicts: map[string]verdict{},
	}
	switch mode {
	case ModeOff:
	case ModeAllowlist:
		for _, k := range keys {
			k = strings.TrimSpace(k)
			if k != "" {
				a.allowed[hashKey(k)] = true
			}
		}
		if len(a.allowed) == 0 {
			return nil, fmt.Errorf("auth-mode allowlist requires at least one key (-api-keys or -api-keys-file)")
		}
	case ModeVerify:
		if upstreamBase == "" {
			return nil, fmt.Errorf("auth-mode verify requires an upstream")
		}
		a.verifyURL = strings.TrimRight(upstreamBase, "/") + "/v1/models"
		if a.http == nil {
			a.http = &http.Client{Timeout: 10 * time.Second}
		}
	default:
		return nil, fmt.Errorf("unknown auth mode %q (valid: off, allowlist, verify)", mode)
	}
	return a, nil
}

func hashKey(k string) string {
	sum := sha256.Sum256([]byte(k))
	return hex.EncodeToString(sum[:])
}

// bearerToken extracts the credential from an Authorization header value.
func bearerToken(header string) string {
	if t, ok := strings.CutPrefix(header, "Bearer "); ok {
		return t
	}
	return header
}

// Allow reports whether the request carrying this Authorization header may
// use the data plane. The reason is safe to return to the client.
func (a *Authorizer) Allow(ctx context.Context, authHeader string) (bool, string) {
	switch a.mode {
	case ModeOff:
		return true, ""
	case ModeAllowlist:
		tok := bearerToken(authHeader)
		if tok == "" {
			return false, "missing API key"
		}
		if !a.allowed[hashKey(tok)] {
			return false, "API key not in allowlist"
		}
		return true, ""
	case ModeVerify:
		if authHeader == "" {
			return false, "missing API key"
		}
		return a.verify(ctx, authHeader)
	}
	return false, "authorization misconfigured"
}

func (a *Authorizer) verify(ctx context.Context, authHeader string) (bool, string) {
	key := hashKey(authHeader)
	now := time.Now()

	a.mu.Lock()
	if v, ok := a.verdicts[key]; ok && now.Before(v.until) {
		a.mu.Unlock()
		if v.ok {
			return true, ""
		}
		return false, "API key rejected by upstream"
	}
	a.mu.Unlock()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, a.verifyURL, nil)
	if err != nil {
		return false, "authorization check failed"
	}
	req.Header.Set("Authorization", authHeader)
	resp, err := a.http.Do(req)
	if err != nil {
		// fail closed, but do not cache: a transient upstream problem should
		// not lock a valid key out for the negative TTL
		return false, "authorization check unavailable"
	}
	resp.Body.Close()

	switch {
	case resp.StatusCode >= 200 && resp.StatusCode < 300:
		a.store(key, verdict{ok: true, until: now.Add(a.ttl)})
		return true, ""
	case resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden:
		a.store(key, verdict{ok: false, until: now.Add(negativeTTL)})
		return false, "API key rejected by upstream"
	default:
		return false, "authorization check unavailable"
	}
}

func (a *Authorizer) store(key string, v verdict) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.verdicts) >= maxVerdicts {
		// drop expired entries; if nothing expired, drop everything rather
		// than grow without bound
		now := time.Now()
		for k, old := range a.verdicts {
			if now.After(old.until) {
				delete(a.verdicts, k)
			}
		}
		if len(a.verdicts) >= maxVerdicts {
			a.verdicts = map[string]verdict{}
		}
	}
	a.verdicts[key] = v
}
