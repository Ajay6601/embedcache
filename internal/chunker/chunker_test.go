package chunker

import (
	"bytes"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"testing"
)

func sha256Sum(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// realDoc is this repo's actual README at build time — real project prose,
// not generated filler. Using go:embed keeps the fixture honest: it changes
// exactly when the real document changes.
//
//go:embed testdata/sample.md
var realDoc []byte

func TestSplitDeterministic(t *testing.T) {
	a := Split(realDoc, Options{})
	b := Split(realDoc, Options{})
	if len(a) != len(b) {
		t.Fatalf("chunk count differs across runs: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i].Hash != b[i].Hash {
			t.Fatalf("chunk %d hash differs across runs", i)
		}
	}
}

func TestSplitReassemblesExactly(t *testing.T) {
	chunks := Split(realDoc, Options{})
	var buf bytes.Buffer
	for _, c := range chunks {
		buf.Write(c.Data)
	}
	if !bytes.Equal(buf.Bytes(), realDoc) {
		t.Fatalf("reassembled document does not match original: got %d bytes, want %d", buf.Len(), len(realDoc))
	}
}

func TestSplitRespectsBounds(t *testing.T) {
	opts := Options{Min: 100, Avg: 400, Max: 1000}
	chunks := Split(realDoc, opts)
	if len(chunks) < 2 {
		t.Fatalf("expected the real document to produce multiple chunks, got %d", len(chunks))
	}
	for i, c := range chunks {
		if len(c.Data) > opts.Max {
			t.Errorf("chunk %d exceeds Max: %d > %d", i, len(c.Data), opts.Max)
		}
		last := i == len(chunks)-1
		if !last && len(c.Data) < opts.Min {
			t.Errorf("non-final chunk %d is below Min: %d < %d", i, len(c.Data), opts.Min)
		}
	}
}

func TestSplitEmptyInput(t *testing.T) {
	if got := Split(nil, Options{}); len(got) != 0 {
		t.Fatalf("empty input must produce zero chunks, got %d", len(got))
	}
}

func TestSplitTinyInput(t *testing.T) {
	tiny := []byte("hello")
	chunks := Split(tiny, Options{})
	if len(chunks) != 1 || string(chunks[0].Data) != "hello" {
		t.Fatalf("input smaller than Min must be a single chunk, got %+v", chunks)
	}
}

// TestEditLocality is the core claim: inserting one realistic sentence into
// the middle of the real document changes only the chunks touching the
// edit. Chunks elsewhere in the document must come out byte-identical (same
// hash) to before the edit — that's what lets them hit embedcache's
// exact-match cache on re-ingestion instead of every downstream chunk
// missing, which is what fixed-size chunking does (see EXPERIMENTS.md E4).
func TestEditLocality(t *testing.T) {
	before := Split(realDoc, Options{})
	beforeHashes := HashSet(before)

	// insert one real, on-topic sentence at a paragraph boundary roughly a
	// third of the way through — a realistic single edit, not bulk filler
	insertAt := bytes.Index(realDoc[len(realDoc)/3:], []byte("\n\n"))
	if insertAt < 0 {
		t.Fatal("fixture has no paragraph break to edit at")
	}
	insertAt += len(realDoc) / 3
	edit := []byte("\n\nThis line was added by an editing pass to simulate a realistic documentation update.")
	edited := append(append(append([]byte{}, realDoc[:insertAt]...), edit...), realDoc[insertAt:]...)

	after := Split(edited, Options{})
	diff := Diff(beforeHashes, after)

	unchangedFrac := float64(diff.Unchanged) / float64(diff.Total)
	t.Logf("content-defined: %d/%d chunks unchanged (%.1f%%) after a one-sentence insertion", diff.Unchanged, diff.Total, unchangedFrac*100)
	if unchangedFrac < 0.5 {
		t.Fatalf("expected most chunks to survive a single localized edit, only %.1f%% did", unchangedFrac*100)
	}

	// contrast: naive fixed-size chunking on the SAME real edit should churn
	// almost everything downstream of the insertion point, reproducing the
	// 0%-absorbed result documented in EXPERIMENTS.md E4 pass 3
	fixedBefore := fixedSizeSplit(realDoc, 800)
	fixedAfter := fixedSizeSplit(edited, 800)
	fixedHashesBefore := HashSet(fixedBefore)
	fixedDiff := Diff(fixedHashesBefore, fixedAfter)
	fixedFrac := float64(fixedDiff.Unchanged) / float64(fixedDiff.Total)
	t.Logf("fixed-size:      %d/%d chunks unchanged (%.1f%%) after the same edit", fixedDiff.Unchanged, fixedDiff.Total, fixedFrac*100)

	if fixedFrac >= unchangedFrac {
		t.Fatalf("content-defined chunking should survive edits far better than fixed-size: cdc=%.1f%% fixed=%.1f%%", unchangedFrac*100, fixedFrac*100)
	}
}

// fixedSizeSplit is the naive baseline for comparison only — not the
// package's chunking strategy, just what a typical RAG pipeline does today.
func fixedSizeSplit(data []byte, size int) []Chunk {
	var out []Chunk
	for len(data) > 0 {
		n := size
		if n > len(data) {
			n = len(data)
		}
		sum := sha256Sum(data[:n])
		out = append(out, Chunk{Data: data[:n], Hash: sum})
		data = data[n:]
	}
	return out
}
