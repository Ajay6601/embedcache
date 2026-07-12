package fingerprint

import (
	"encoding/json"
	"testing"

	"github.com/Ajay6601/embedcache/internal/api"
)

func text(s string) api.InputItem { return api.InputItem{Text: s} }

func TestNFCFoldsDecomposedForm(t *testing.T) {
	nfc := "café"  // café with precomposed é (U+00E9)
	nfd := "café" // café with e + combining acute (U+0301)
	if nfc == nfd {
		t.Fatal("test inputs should be different byte sequences")
	}
	norm := Normalizer{NFC: true}
	// without folding, they are different keys (the leak validation found)
	if Key("m", 0, "", "", text(nfc), Normalizer{}) == Key("m", 0, "", "", text(nfd), Normalizer{}) {
		t.Fatal("byte-exact mode should keep NFC and NFD distinct")
	}
	// with nfc folding, they share a key
	if Key("m", 0, "", "", text(nfc), norm) != Key("m", 0, "", "", text(nfd), norm) {
		t.Error("nfc normalization should fold NFC and NFD to one key")
	}
	// and folding a value that is already NFC is a no-op
	if foldNFC(nfc) != nfc {
		t.Error("already-composed text must pass through unchanged")
	}
	// unrelated combining sequences we don't cover pass through, never wrong
	other := "a̰" // a + combining tilde below (not in the Latin table)
	if foldNFC(other) != other {
		t.Error("uncovered combining sequences must pass through unchanged")
	}
}

func TestKeyDeterministic(t *testing.T) {
	a := Key("m", 0, "", "", text("hello"), Normalizer{})
	b := Key("m", 0, "", "", text("hello"), Normalizer{})
	if a != b {
		t.Fatal("same input must produce the same key")
	}
}

func TestKeySeparatesParameters(t *testing.T) {
	base := Key("m", 0, "", "", text("hello"), Normalizer{})
	if Key("m2", 0, "", "", text("hello"), Normalizer{}) == base {
		t.Error("different model must change the key")
	}
	if Key("m", 256, "", "", text("hello"), Normalizer{}) == base {
		t.Error("different dimensions must change the key")
	}
	if Key("m", 0, "base64", "", text("hello"), Normalizer{}) == base {
		t.Error("different encoding_format must change the key")
	}
	if Key("m", 0, "", "", text("hello!"), Normalizer{}) == base {
		t.Error("different text must change the key")
	}
}

func TestTokensVsTextNeverCollide(t *testing.T) {
	a := Key("m", 0, "", "", api.InputItem{Text: "1,2,3"}, Normalizer{})
	b := Key("m", 0, "", "", api.InputItem{Tokens: []int{1, 2, 3}, IsTokens: true}, Normalizer{})
	if a == b {
		t.Fatal("token input and text input must not share keys")
	}
}

func TestParamsDigestSeparatesEntries(t *testing.T) {
	query := ParamsDigest(map[string]json.RawMessage{"input_type": json.RawMessage(`"query"`)})
	doc := ParamsDigest(map[string]json.RawMessage{"input_type": json.RawMessage(`"document"`)})
	if query == doc {
		t.Fatal("different param values must digest differently")
	}
	if ParamsDigest(nil) != "" {
		t.Fatal("no params digests to empty string")
	}
	if Key("m", 0, "", query, text("t"), Normalizer{}) == Key("m", 0, "", doc, text("t"), Normalizer{}) {
		t.Fatal("same text with different provider params must not share a key")
	}
	// digest is order-independent (canonicalized)
	a := ParamsDigest(map[string]json.RawMessage{"a": json.RawMessage(`1`), "b": json.RawMessage(`2`)})
	b := ParamsDigest(map[string]json.RawMessage{"b": json.RawMessage(`2`), "a": json.RawMessage(`1`)})
	if a != b {
		t.Fatal("digest must not depend on map iteration order")
	}
}

func TestNormalization(t *testing.T) {
	exact := Normalizer{}
	if Key("m", 0, "", "", text(" hello "), exact) == Key("m", 0, "", "", text("hello"), exact) {
		t.Error("default must be byte-exact: whitespace matters")
	}
	trim := Normalizer{TrimSpace: true}
	if Key("m", 0, "", "", text(" hello "), trim) != Key("m", 0, "", "", text("hello"), trim) {
		t.Error("trim must make padded input match")
	}
	collapse := Normalizer{CollapseWhitespace: true}
	if Key("m", 0, "", "", text("a  b\n c"), collapse) != Key("m", 0, "", "", text("a b c"), collapse) {
		t.Error("collapse must unify internal whitespace")
	}
}

func TestParseNormalizer(t *testing.T) {
	n, err := ParseNormalizer("trim,collapse")
	if err != nil || !n.TrimSpace || !n.CollapseWhitespace || n.Lowercase {
		t.Fatalf("unexpected: %+v err=%v", n, err)
	}
	if _, err := ParseNormalizer("bogus"); err == nil {
		t.Fatal("unknown rule must error")
	}
}
