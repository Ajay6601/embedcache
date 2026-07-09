package fingerprint

import (
	"testing"

	"embedcache/internal/api"
)

func text(s string) api.InputItem { return api.InputItem{Text: s} }

func TestKeyDeterministic(t *testing.T) {
	a := Key("m", 0, "", text("hello"), Normalizer{})
	b := Key("m", 0, "", text("hello"), Normalizer{})
	if a != b {
		t.Fatal("same input must produce the same key")
	}
}

func TestKeySeparatesParameters(t *testing.T) {
	base := Key("m", 0, "", text("hello"), Normalizer{})
	if Key("m2", 0, "", text("hello"), Normalizer{}) == base {
		t.Error("different model must change the key")
	}
	if Key("m", 256, "", text("hello"), Normalizer{}) == base {
		t.Error("different dimensions must change the key")
	}
	if Key("m", 0, "base64", text("hello"), Normalizer{}) == base {
		t.Error("different encoding_format must change the key")
	}
	if Key("m", 0, "", text("hello!"), Normalizer{}) == base {
		t.Error("different text must change the key")
	}
}

func TestTokensVsTextNeverCollide(t *testing.T) {
	a := Key("m", 0, "", api.InputItem{Text: "1,2,3"}, Normalizer{})
	b := Key("m", 0, "", api.InputItem{Tokens: []int{1, 2, 3}, IsTokens: true}, Normalizer{})
	if a == b {
		t.Fatal("token input and text input must not share keys")
	}
}

func TestNormalization(t *testing.T) {
	exact := Normalizer{}
	if Key("m", 0, "", text(" hello "), exact) == Key("m", 0, "", text("hello"), exact) {
		t.Error("default must be byte-exact: whitespace matters")
	}
	trim := Normalizer{TrimSpace: true}
	if Key("m", 0, "", text(" hello "), trim) != Key("m", 0, "", text("hello"), trim) {
		t.Error("trim must make padded input match")
	}
	collapse := Normalizer{CollapseWhitespace: true}
	if Key("m", 0, "", text("a  b\n c"), collapse) != Key("m", 0, "", text("a b c"), collapse) {
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
