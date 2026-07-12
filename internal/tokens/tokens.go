// Package tokens estimates token counts for savings accounting.
//
// The heuristic is deliberately rough: it is only used to apportion
// upstream-reported usage across batch items and to estimate waste in the
// offline analyzer, never for billing. The bytes/4 rule is fine for Latin
// script but badly overcounts CJK and other wide scripts (a Chinese character
// is ~3 UTF-8 bytes but often ~1 token), which validation measured drifting by
// tens of percent. EstimateText counts CJK/wide runes at ~1 token each and the
// rest at ~4 bytes/token, which tracks real tokenizers much more closely.
package tokens

import (
	"sort"
	"unicode/utf8"

	"github.com/Ajay6601/embedcache/internal/api"
)

func Estimate(item api.InputItem) int {
	if item.IsTokens {
		if len(item.Tokens) == 0 {
			return 1
		}
		return len(item.Tokens)
	}
	return EstimateText(item.Text)
}

func EstimateText(text string) int {
	// Split the estimate by script: wide (CJK, Hiragana/Katakana, Hangul)
	// characters are roughly one token each; everything else follows the
	// ~4-bytes-per-token rule. This keeps Latin text unchanged while fixing the
	// large overcount on Chinese/Japanese/Korean the byte rule produced.
	wide := 0
	narrowBytes := 0
	for _, r := range text {
		if isWide(r) {
			wide++
		} else {
			narrowBytes += utf8.RuneLen(r)
		}
	}
	n := wide + (narrowBytes+3)/4
	if n < 1 {
		n = 1
	}
	return n
}

// isWide reports whether a rune is from a script that tokenizes at roughly one
// token per character rather than the Latin ~4-bytes-per-token rate.
func isWide(r rune) bool {
	switch {
	case r >= 0x3400 && r <= 0x9FFF: // CJK Unified Ideographs (+ Extension A)
		return true
	case r >= 0xF900 && r <= 0xFAFF: // CJK Compatibility Ideographs
		return true
	case r >= 0x20000 && r <= 0x2FA1F: // CJK Extension B+ (supplementary)
		return true
	case r >= 0x3040 && r <= 0x30FF: // Hiragana + Katakana
		return true
	case r >= 0xAC00 && r <= 0xD7A3: // Hangul syllables
		return true
	case r >= 0x3000 && r <= 0x303F: // CJK symbols and punctuation
		return true
	}
	return false
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
