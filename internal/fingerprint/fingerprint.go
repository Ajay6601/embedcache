// Package fingerprint computes deterministic, content-addressed cache keys
// for embedding inputs.
//
// Two inputs share a key only when the upstream response for them is
// guaranteed interchangeable: same model, same dimensions, same encoding
// format, same (optionally normalized) content. Normalization is off by
// default — the default posture is byte-exact.
package fingerprint

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strconv"
	"strings"

	"github.com/Ajay6601/embedcache/internal/api"
)

// Normalizer controls opt-in text canonicalization before hashing. Each rule
// widens what counts as "the same input", trading exactness for hit rate.
type Normalizer struct {
	TrimSpace          bool // strip leading/trailing whitespace
	CollapseWhitespace bool // collapse runs of whitespace to a single space
	Lowercase          bool // ASCII-insensitive matching; wrong for case-sensitive corpora
	NFC                bool // fold canonical Latin combining sequences (see foldNFC)
}

// ParseNormalizer builds a Normalizer from a comma-separated rule list, e.g.
// "trim,collapse". Empty string means no normalization.
func ParseNormalizer(spec string) (Normalizer, error) {
	var n Normalizer
	if spec == "" {
		return n, nil
	}
	for _, rule := range strings.Split(spec, ",") {
		switch strings.TrimSpace(rule) {
		case "trim":
			n.TrimSpace = true
		case "collapse":
			n.CollapseWhitespace = true
		case "lowercase":
			n.Lowercase = true
		case "nfc":
			n.NFC = true
		case "":
		default:
			return n, &UnknownRuleError{Rule: rule}
		}
	}
	return n, nil
}

type UnknownRuleError struct{ Rule string }

func (e *UnknownRuleError) Error() string {
	return "unknown normalization rule: " + e.Rule + " (valid: trim, collapse, lowercase, nfc)"
}

// Apply returns the canonical form of text under this normalizer.
func (n Normalizer) Apply(text string) string {
	if n.NFC {
		text = foldNFC(text)
	}
	if n.TrimSpace {
		text = strings.TrimSpace(text)
	}
	if n.CollapseWhitespace {
		text = strings.Join(strings.Fields(text), " ")
	}
	if n.Lowercase {
		text = strings.ToLower(text)
	}
	return text
}

// foldNFC composes a base letter followed by a canonical combining mark into
// its precomposed form, so text that arrives decomposed (NFD, e.g. "e"+U+0301)
// caches as the same entry as the precomposed form (NFC, U+00E9). Validation
// found this NFC/NFD split leaking as duplicate cache entries.
//
// Scope, stated honestly: this covers the common Latin combining sequences
// (accented a/e/i/o/u/y/n/c and their capitals with grave, acute, circumflex,
// tilde, diaeresis, ring, and cedilla) — which is where real-world NFC/NFD
// divergence overwhelmingly occurs. It is a deliberate zero-dependency subset,
// not full Unicode NFC (that would require golang.org/x/text). Anything not in
// the table passes through unchanged, so folding is never wrong, only partial.
func foldNFC(text string) string {
	// fast path: no combining marks present, nothing to do
	hasCombining := false
	for _, r := range text {
		if r >= 0x0300 && r <= 0x036F {
			hasCombining = true
			break
		}
	}
	if !hasCombining {
		return text
	}
	rs := []rune(text)
	out := make([]rune, 0, len(rs))
	for i := 0; i < len(rs); i++ {
		if i+1 < len(rs) {
			if composed, ok := nfcCompose[[2]rune{rs[i], rs[i+1]}]; ok {
				out = append(out, composed)
				i++ // consume the combining mark
				continue
			}
		}
		out = append(out, rs[i])
	}
	return string(out)
}

// nfcCompose maps (base rune, combining mark) -> precomposed rune for the common
// Latin combining sequences. Built once at init from a compact description so
// the table stays readable and correct.
var nfcCompose = buildNFCTable()

func buildNFCTable() map[[2]rune]rune {
	const (
		grave      = 0x0300
		acute      = 0x0301
		circumflex = 0x0302
		tilde      = 0x0303
		diaeresis  = 0x0308
		ring       = 0x030A
		cedilla    = 0x0327
	)
	m := map[[2]rune]rune{}
	add := func(base rune, mark rune, composed rune) { m[[2]rune{base, mark}] = composed }
	// lowercase
	add('a', grave, 'à')
	add('a', acute, 'á')
	add('a', circumflex, 'â')
	add('a', tilde, 'ã')
	add('a', diaeresis, 'ä')
	add('a', ring, 'å')
	add('e', grave, 'è')
	add('e', acute, 'é')
	add('e', circumflex, 'ê')
	add('e', diaeresis, 'ë')
	add('i', grave, 'ì')
	add('i', acute, 'í')
	add('i', circumflex, 'î')
	add('i', diaeresis, 'ï')
	add('o', grave, 'ò')
	add('o', acute, 'ó')
	add('o', circumflex, 'ô')
	add('o', tilde, 'õ')
	add('o', diaeresis, 'ö')
	add('u', grave, 'ù')
	add('u', acute, 'ú')
	add('u', circumflex, 'û')
	add('u', diaeresis, 'ü')
	add('y', acute, 'ý')
	add('y', diaeresis, 'ÿ')
	add('n', tilde, 'ñ')
	add('c', cedilla, 'ç')
	// uppercase
	add('A', grave, 'À')
	add('A', acute, 'Á')
	add('A', circumflex, 'Â')
	add('A', tilde, 'Ã')
	add('A', diaeresis, 'Ä')
	add('A', ring, 'Å')
	add('E', grave, 'È')
	add('E', acute, 'É')
	add('E', circumflex, 'Ê')
	add('E', diaeresis, 'Ë')
	add('I', grave, 'Ì')
	add('I', acute, 'Í')
	add('I', circumflex, 'Î')
	add('I', diaeresis, 'Ï')
	add('O', grave, 'Ò')
	add('O', acute, 'Ó')
	add('O', circumflex, 'Ô')
	add('O', tilde, 'Õ')
	add('O', diaeresis, 'Ö')
	add('U', grave, 'Ù')
	add('U', acute, 'Ú')
	add('U', circumflex, 'Û')
	add('U', diaeresis, 'Ü')
	add('Y', acute, 'Ý')
	add('N', tilde, 'Ñ')
	add('C', cedilla, 'Ç')
	return m
}

// ParamsDigest canonicalizes provider-specific request parameters (Voyage's
// input_type, output_dtype, ...) into a stable digest for the cache key. Two
// requests with different extra params must never share an entry — providers
// return different vectors for them. Empty input yields "".
func ParamsDigest(extra map[string]json.RawMessage) string {
	if len(extra) == 0 {
		return ""
	}
	keys := make([]string, 0, len(extra))
	for k := range extra {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	h := sha256.New()
	for _, k := range keys {
		h.Write([]byte(k))
		h.Write([]byte{0})
		h.Write(extra[k])
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

// Key computes the cache key for one input item. The version prefix guards
// against silently mixing entries across incompatible key schemas.
func Key(model string, dimensions int, encodingFormat string, paramsDigest string, item api.InputItem, norm Normalizer) string {
	h := sha256.New()
	h.Write([]byte("ec2\x00"))
	h.Write([]byte(model))
	h.Write([]byte{0})
	h.Write([]byte(strconv.Itoa(dimensions)))
	h.Write([]byte{0})
	h.Write([]byte(encodingFormat))
	h.Write([]byte{0})
	h.Write([]byte(paramsDigest))
	h.Write([]byte{0})
	if item.IsTokens {
		h.Write([]byte("t\x00"))
		var buf [8]byte
		for _, t := range item.Tokens {
			binary.LittleEndian.PutUint64(buf[:], uint64(int64(t)))
			h.Write(buf[:])
		}
	} else {
		h.Write([]byte("s\x00"))
		h.Write([]byte(norm.Apply(item.Text)))
	}
	return hex.EncodeToString(h.Sum(nil))
}
