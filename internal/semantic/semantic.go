// Package semantic finds near-duplicate inputs by cheap lexical similarity, so
// embedcache can (opt-in) treat "how do I reset my password" and "how to reset
// my password?" as the same request instead of paying to embed both.
//
// The similarity signal is deliberately NOT another embedding — computing one
// is exactly the cost we are trying to avoid. It is character-trigram Jaccard
// over the raw input text, retrieved through an inverted trigram index. That is
// cheap, needs no model, and catches the punctuation/whitespace/word-order
// near-duplicates that exact-match caching misses.
//
// Because returning a neighbor's vector for a not-actually-identical input can
// silently poison a vector store, this package only *finds* candidates and
// reports a similarity; whether to serve one (active) or merely measure how
// wrong it would have been (shadow) is the proxy's decision, off by default.
package semantic

import (
	"sort"
	"strings"
	"sync"
)

// Index is a concurrency-safe near-duplicate index over input texts.
type Index struct {
	mu       sync.RWMutex
	trigrams map[string]map[string]struct{} // trigram -> set of keys containing it
	shingles map[string]map[string]struct{} // key -> its trigram set
	maxKeys  int                            // 0 = unbounded; else stop indexing new keys past this
}

// New returns an empty index. maxKeys bounds memory (0 = unbounded).
func New(maxKeys int) *Index {
	return &Index{
		trigrams: map[string]map[string]struct{}{},
		shingles: map[string]map[string]struct{}{},
		maxKeys:  maxKeys,
	}
}

// shingle turns text into its set of character trigrams over a normalized form
// (lowercased, whitespace collapsed, space-padded so word edges form trigrams).
func shingle(text string) map[string]struct{} {
	norm := " " + strings.Join(strings.Fields(strings.ToLower(text)), " ") + " "
	rs := []rune(norm)
	set := map[string]struct{}{}
	if len(rs) < 3 {
		set[string(rs)] = struct{}{}
		return set
	}
	for i := 0; i+3 <= len(rs); i++ {
		set[string(rs[i:i+3])] = struct{}{}
	}
	return set
}

// Add indexes the text under key. Re-adding an existing key is a no-op.
func (ix *Index) Add(key, text string) {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	if _, exists := ix.shingles[key]; exists {
		return
	}
	if ix.maxKeys > 0 && len(ix.shingles) >= ix.maxKeys {
		return
	}
	set := shingle(text)
	ix.shingles[key] = set
	for tg := range set {
		keys := ix.trigrams[tg]
		if keys == nil {
			keys = map[string]struct{}{}
			ix.trigrams[tg] = keys
		}
		keys[key] = struct{}{}
	}
}

// Nearest returns the indexed key most similar to text and its Jaccard
// similarity in [0,1]. ok is false when the index is empty or nothing shares a
// trigram with text. An exact-key match is intentionally not special-cased:
// callers use this only after an exact-match cache miss.
func (ix *Index) Nearest(text string) (key string, similarity float64, ok bool) {
	q := shingle(text)
	ix.mu.RLock()
	defer ix.mu.RUnlock()

	// candidate keys: those sharing at least one trigram with the query
	overlap := map[string]int{}
	for tg := range q {
		for k := range ix.trigrams[tg] {
			overlap[k]++
		}
	}
	best, bestSim := "", 0.0
	for k, inter := range overlap {
		s := ix.shingles[k]
		union := len(q) + len(s) - inter
		if union == 0 {
			continue
		}
		sim := float64(inter) / float64(union)
		if sim > bestSim || (sim == bestSim && k < best) {
			best, bestSim = k, sim
		}
	}
	if best == "" {
		return "", 0, false
	}
	return best, bestSim, true
}

// Len reports how many keys are indexed.
func (ix *Index) Len() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return len(ix.shingles)
}

// jaccard is exported-for-test convenience: similarity of two raw texts.
func jaccard(a, b string) float64 {
	sa, sb := shingle(a), shingle(b)
	inter := 0
	for tg := range sa {
		if _, ok := sb[tg]; ok {
			inter++
		}
	}
	union := len(sa) + len(sb) - inter
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

// sortedKeys is a small helper kept for deterministic debugging output.
func sortedKeys(m map[string]struct{}) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
