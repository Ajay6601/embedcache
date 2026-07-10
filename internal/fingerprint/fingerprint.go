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
		case "":
		default:
			return n, &UnknownRuleError{Rule: rule}
		}
	}
	return n, nil
}

type UnknownRuleError struct{ Rule string }

func (e *UnknownRuleError) Error() string {
	return "unknown normalization rule: " + e.Rule + " (valid: trim, collapse, lowercase)"
}

// Apply returns the canonical form of text under this normalizer.
func (n Normalizer) Apply(text string) string {
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

// Key computes the cache key for one input item. The version prefix guards
// against silently mixing entries across incompatible key schemas.
func Key(model string, dimensions int, encodingFormat string, item api.InputItem, norm Normalizer) string {
	h := sha256.New()
	h.Write([]byte("ec1\x00"))
	h.Write([]byte(model))
	h.Write([]byte{0})
	h.Write([]byte(strconv.Itoa(dimensions)))
	h.Write([]byte{0})
	h.Write([]byte(encodingFormat))
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
