// Package tokens estimates token counts for savings accounting.
//
// The heuristic (~4 chars per token for text) is deliberately rough: it is
// only used to apportion upstream-reported usage across batch items and to
// estimate waste in the offline analyzer, never for billing.
package tokens

import (
	"sort"

	"embedcache/internal/api"
)

func Estimate(item api.InputItem) int {
	if item.IsTokens {
		if len(item.Tokens) == 0 {
			return 1
		}
		return len(item.Tokens)
	}
	n := (len(item.Text) + 3) / 4
	if n < 1 {
		n = 1
	}
	return n
}

func EstimateText(text string) int {
	n := (len(text) + 3) / 4
	if n < 1 {
		n = 1
	}
	return n
}

// Apportion splits an exact total (upstream-reported prompt_tokens) across
// items proportionally to their estimates, so per-item attribution sums to
// the real billed amount.
func Apportion(total int, items []api.InputItem) []int {
	if len(items) == 0 {
		return nil
	}
	est := make([]int, len(items))
	sum := 0
	for i, it := range items {
		est[i] = Estimate(it)
		sum += est[i]
	}
	out := make([]int, len(items))
	if total <= 0 || sum == 0 {
		copy(out, est)
		return out
	}
	// largest-remainder allocation: floors first, then hand the leftover
	// tokens to the items with the biggest truncated fractions, so the
	// result always sums exactly to the billed total
	type frac struct{ idx, rem int }
	fracs := make([]frac, len(items))
	acc := 0
	for i := range items {
		out[i] = total * est[i] / sum
		fracs[i] = frac{idx: i, rem: total * est[i] % sum}
		acc += out[i]
	}
	sort.Slice(fracs, func(a, b int) bool { return fracs[a].rem > fracs[b].rem })
	for i := 0; i < total-acc; i++ {
		out[fracs[i%len(fracs)].idx]++
	}
	return out
}
