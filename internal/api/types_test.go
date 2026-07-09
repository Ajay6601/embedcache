package api

import (
	"encoding/json"
	"testing"
)

func TestSplitInputVariants(t *testing.T) {
	cases := []struct {
		raw      string
		n        int
		isTokens bool
		wantErr  bool
	}{
		{`"hello"`, 1, false, false},
		{`["a","b","c"]`, 3, false, false},
		{`[1,2,3]`, 1, true, false},
		{`[[1,2],[3]]`, 2, true, false},
		{`[]`, 0, false, true},
		{`123`, 0, false, true},
		{`{"x":1}`, 0, false, true},
	}
	for _, c := range cases {
		items, err := SplitInput(json.RawMessage(c.raw))
		if c.wantErr {
			if err == nil {
				t.Errorf("%s: expected error", c.raw)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", c.raw, err)
			continue
		}
		if len(items) != c.n {
			t.Errorf("%s: n=%d want %d", c.raw, len(items), c.n)
		}
		if len(items) > 0 && items[0].IsTokens != c.isTokens {
			t.Errorf("%s: isTokens=%v want %v", c.raw, items[0].IsTokens, c.isTokens)
		}
	}
}

func TestMarshalInputsRoundtrip(t *testing.T) {
	items, err := SplitInput(json.RawMessage(`["x","y"]`))
	if err != nil {
		t.Fatal(err)
	}
	out, err := MarshalInputs(items[1:])
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != `["y"]` {
		t.Fatalf("got %s", out)
	}
	tok, _ := SplitInput(json.RawMessage(`[[1,2],[3]]`))
	out, _ = MarshalInputs(tok)
	if string(out) != `[[1,2],[3]]` {
		t.Fatalf("got %s", out)
	}
}
