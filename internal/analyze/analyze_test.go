package analyze

import (
	"strings"
	"testing"

	"github.com/Ajay6601/embedcache/internal/fingerprint"
	"github.com/Ajay6601/embedcache/internal/pricing"
)

func TestDuplicateCounting(t *testing.T) {
	log := strings.Join([]string{
		`{"model":"m","input":"aaaa"}`,
		`{"model":"m","input":"bbbb"}`,
		`{"model":"m","input":"aaaa"}`,
		`{"model":"m","input":["aaaa","cccc"]}`,
	}, "\n")
	res, err := Run(strings.NewReader(log), Options{Pricing: pricing.Default()})
	if err != nil {
		t.Fatal(err)
	}
	if res.Requests != 4 || res.Items != 5 {
		t.Fatalf("requests=%d items=%d", res.Requests, res.Items)
	}
	if res.Unique != 3 || res.DupItems != 2 {
		t.Fatalf("unique=%d dup=%d", res.Unique, res.DupItems)
	}
	if len(res.TopDup) != 1 || res.TopDup[0].Count != 3 {
		t.Fatalf("topdup=%+v", res.TopDup)
	}
	if res.WastedTokens == 0 || res.WastedUSD <= 0 {
		t.Fatalf("wasted tokens=%d usd=%f", res.WastedTokens, res.WastedUSD)
	}
}

func TestModelsDoNotCrossMatch(t *testing.T) {
	log := `{"model":"m1","input":"same text"}
{"model":"m2","input":"same text"}`
	res, err := Run(strings.NewReader(log), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.DupItems != 0 {
		t.Fatal("same text under different models is not a duplicate")
	}
}

func TestWrappedAndJunkLines(t *testing.T) {
	log := strings.Join([]string{
		`{"ts":"2026-07-01","body":{"model":"m","input":"wrapped object"}}`,
		`{"request":"{\"model\":\"m\",\"input\":\"wrapped string\"}"}`,
		`not json at all`,
		`{"model":"m","messages":[{"role":"user"}]}`, // a chat request, not embeddings
		`{"model":"m","input":"wrapped object"}`,
	}, "\n")
	res, err := Run(strings.NewReader(log), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Requests != 3 {
		t.Fatalf("requests=%d want 3", res.Requests)
	}
	if res.Skipped != 2 {
		t.Fatalf("skipped=%d want 2", res.Skipped)
	}
	if res.DupItems != 1 {
		t.Fatalf("dup=%d want 1 (wrapped object seen twice)", res.DupItems)
	}
}

func TestNormalizationWidensMatching(t *testing.T) {
	log := `{"model":"m","input":"hello world"}
{"model":"m","input":"  hello   world "}`
	strict, _ := Run(strings.NewReader(log), Options{})
	if strict.DupItems != 0 {
		t.Fatal("byte-exact must not match differently-spaced text")
	}
	norm, _ := Run(strings.NewReader(log), Options{Norm: fingerprint.Normalizer{TrimSpace: true, CollapseWhitespace: true}})
	if norm.DupItems != 1 {
		t.Fatal("trim+collapse must match differently-spaced text")
	}
}
