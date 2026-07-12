package semantic

import "testing"

func TestNearDuplicatesScoreHigh(t *testing.T) {
	ix := New(0)
	ix.Add("k1", "how do I reset my password")
	ix.Add("k2", "what are the pricing plans")

	// a punctuation/word-order near-duplicate of k1 should match k1 with high sim
	key, sim, ok := ix.Nearest("how do I reset my password?")
	if !ok || key != "k1" {
		t.Fatalf("expected k1, got key=%q ok=%v", key, ok)
	}
	if sim < 0.8 {
		t.Errorf("near-duplicate similarity too low: %.3f", sim)
	}
}

func TestUnrelatedScoresLow(t *testing.T) {
	ix := New(0)
	ix.Add("k1", "how do I reset my password")
	_, sim, ok := ix.Nearest("photosynthesis converts sunlight into energy")
	if ok && sim > 0.3 {
		t.Errorf("unrelated text should score low, got %.3f", sim)
	}
}

func TestExactTextIsNearlyOne(t *testing.T) {
	ix := New(0)
	ix.Add("k1", "the quarterly revenue report covers three product lines")
	_, sim, ok := ix.Nearest("the quarterly revenue report covers three product lines")
	if !ok || sim < 0.99 {
		t.Errorf("identical text should be ~1.0, got %.3f ok=%v", sim, ok)
	}
}

func TestEmptyIndex(t *testing.T) {
	ix := New(0)
	if _, _, ok := ix.Nearest("anything"); ok {
		t.Error("empty index must report no match")
	}
}

func TestMaxKeysBound(t *testing.T) {
	ix := New(2)
	ix.Add("k1", "alpha one two three")
	ix.Add("k2", "beta four five six")
	ix.Add("k3", "gamma seven eight nine") // dropped, over the bound
	if ix.Len() != 2 {
		t.Errorf("index should stop at maxKeys=2, got %d", ix.Len())
	}
}

func TestJaccardMonotonic(t *testing.T) {
	base := "how do I cancel my subscription"
	near := jaccard(base, "how do I cancel my subscription?")
	far := jaccard(base, "what time does the store open")
	if near <= far {
		t.Errorf("near-duplicate jaccard (%.3f) should exceed unrelated (%.3f)", near, far)
	}
	if got := len(sortedKeys(shingle(base))); got == 0 {
		t.Error("shingling produced no trigrams")
	}
}
