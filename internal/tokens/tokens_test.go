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
