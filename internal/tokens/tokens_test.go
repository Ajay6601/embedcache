package tokens

import (
	"testing"

	"github.com/Ajay6601/embedcache/internal/api"
)

func TestEstimate(t *testing.T) {
	if Estimate(api.InputItem{Text: ""}) != 1 {
		t.Error("empty text estimates to at least 1")
	}
	if Estimate(api.InputItem{Text: "12345678"}) != 2 {
		t.Error("8 chars ~ 2 tokens")
	}
	if Estimate(api.InputItem{Tokens: []int{1, 2, 3}, IsTokens: true}) != 3 {
		t.Error("token inputs are exact")
	}
}

func TestCJKCountsPerCharNotPerByte(t *testing.T) {
	// 10 Chinese characters are ~30 UTF-8 bytes. The old bytes/4 rule estimated
	// ~8 tokens; real tokenizers bill closer to ~10 (about one per character).
	cjk := "机器学习模型训练数据" // 10 CJK characters
	got := EstimateText(cjk)
	if got < 9 || got > 12 {
		t.Errorf("CJK estimate %d not near one-token-per-char (~10)", got)
	}
	// the old byte rule would have been (30+3)/4 = 8; make sure we moved off it
	if got == (len(cjk)+3)/4 {
		t.Errorf("CJK still using the bytes/4 rule (%d)", got)
	}
}

func TestLatinUnchanged(t *testing.T) {
	// Latin text must still follow the ~4-bytes-per-token rule
	if EstimateText("12345678") != 2 {
		t.Errorf("Latin 8 chars should still be ~2 tokens, got %d", EstimateText("12345678"))
	}
}

func TestApportionSumsToTotal(t *testing.T) {
	items := []api.InputItem{
		{Text: "short"},
		{Text: "a considerably longer piece of text that should get more of the total"},
		{Text: "medium length text here"},
	}
	for _, total := range []int{1, 3, 17, 100, 12345} {
		got := Apportion(total, items)
		sum := 0
		for _, v := range got {
			sum += v
		}
		if sum != total {
			t.Errorf("total=%d apportioned sum=%d", total, sum)
		}
	}
}
