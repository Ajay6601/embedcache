// Package chunker implements content-defined chunking (FastCDC) so editing
// one part of a document only shifts chunk boundaries near the edit.
//
// Fixed-size chunking re-slices the whole document from the edit point
// onward, so every downstream chunk becomes new text and misses the cache —
// this is the re-chunk gap documented in EXPERIMENTS.md (E4 pass 3, 0%
// absorbed). Content-defined chunking places boundaries based on a rolling
// hash of local bytes, so an insertion or deletion only perturbs the one or
// two chunks that actually changed; everything else re-chunks to
// byte-identical text and hits embedcache's existing exact-match cache.
package chunker

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
)

// gear is a table of 256 pseudo-random 64-bit values for the rolling hash.
// Derived deterministically so chunking behavior is fixed across processes.
var gear [256]uint64

func init() {
	for i := range gear {
		h := sha256.Sum256([]byte{byte(i)})
		gear[i] = binary.LittleEndian.Uint64(h[:8])
	}
}

const normalizationLevel = 2

// Options bounds chunk size in bytes. Zero values fall back to defaults
// sized for embedding-scale text (roughly paragraph size).
type Options struct {
	Min int
	Avg int
	Max int
}

const (
	DefaultMin = 200
	DefaultAvg = 800
	DefaultMax = 2000
)

func (o Options) withDefaults() Options {
	if o.Min <= 0 {
		o.Min = DefaultMin
	}
	if o.Avg <= 0 {
		o.Avg = DefaultAvg
	}
	if o.Max <= 0 {
		o.Max = DefaultMax
	}
	if o.Min > o.Max {
		o.Min = o.Max
	}
	if o.Avg < o.Min {
		o.Avg = o.Min
	}
	if o.Avg > o.Max {
		o.Avg = o.Max
	}
	return o
}

// Chunk is one content-defined chunk. Hash is the hex SHA-256 of Data — a
// position-independent identity used to detect unchanged chunks across edits.
type Chunk struct {
	Data []byte
	Hash string
}

func onesMask(n int) uint64 {
	if n <= 0 {
		return 0
	}
	if n >= 64 {
		return ^uint64(0)
	}
	return (uint64(1) << uint(n)) - 1
}

// maskBits returns the two FastCDC "normalized chunking" masks: maskS (more
// ones, harder to satisfy — used below the average-size point to discourage
// small chunks) and maskL (fewer ones, easier to satisfy — used above it to
// pull the boundary in before the max-size cutoff).
func maskBits(avg int) (maskS, maskL uint64) {
	bits := 0
	for (1 << uint(bits+1)) <= avg {
		bits++
	}
	return onesMask(bits + normalizationLevel), onesMask(bits - normalizationLevel)
}

// Split breaks data into content-defined chunks.
func Split(data []byte, opts Options) []Chunk {
	opts = opts.withDefaults()
	maskS, maskL := maskBits(opts.Avg)
	var out []Chunk
	for len(data) > 0 {
		n := cutPoint(data, opts, maskS, maskL)
		sum := sha256.Sum256(data[:n])
		out = append(out, Chunk{Data: data[:n], Hash: hex.EncodeToString(sum[:])})
		data = data[n:]
	}
	return out
}

func cutPoint(data []byte, opts Options, maskS, maskL uint64) int {
	n := len(data)
	if n <= opts.Min {
		return n
	}
	max := n
	if max > opts.Max {
		max = opts.Max
	}
	center := opts.Avg
	if center > max {
		center = max
	}

	var fp uint64
	i := opts.Min
	for ; i < center; i++ {
		fp = (fp << 1) + gear[data[i]]
		if fp&maskS == 0 {
			return i + 1
		}
	}
	for ; i < max; i++ {
		fp = (fp << 1) + gear[data[i]]
		if fp&maskL == 0 {
			return i + 1
		}
	}
	return max
}

// HashSet extracts the set of chunk hashes, for recording after ingestion or
// comparing against a prior version.
func HashSet(chunks []Chunk) map[string]bool {
	m := make(map[string]bool, len(chunks))
	for _, c := range chunks {
		m[c.Hash] = true
	}
	return m
}

// DiffResult summarizes how much of a re-chunked document is already known.
type DiffResult struct {
	Total     int
	Unchanged int // chunks whose hash existed in the prior set — skip re-embedding
	Changed   int // chunks whose content is new
}

// Diff compares a new chunk set against a previously recorded set of chunk
// hashes and reports how many are already known.
func Diff(before map[string]bool, after []Chunk) DiffResult {
	r := DiffResult{Total: len(after)}
	for _, c := range after {
		if before[c.Hash] {
			r.Unchanged++
		} else {
			r.Changed++
		}
	}
	return r
}
