// Package api defines the OpenAI-compatible embeddings wire types.
//
// The embedding payload itself is carried as json.RawMessage end to end so
// that cached responses are byte-exact replicas of what the upstream
// returned — no float re-encoding, no precision drift.
package api

import (
	"encoding/json"
	"errors"
	"fmt"
)

// EmbeddingsRequest mirrors the OpenAI POST /v1/embeddings body. Input is kept
// raw because it may be a string, []string, []int (tokens), or [][]int.
type EmbeddingsRequest struct {
	Input          json.RawMessage `json:"input"`
	Model          string          `json:"model"`
	EncodingFormat string          `json:"encoding_format,omitempty"`
	Dimensions     int             `json:"dimensions,omitempty"`
	User           string          `json:"user,omitempty"`
}

// InputItem is one element of the request input after splitting.
type InputItem struct {
	Text     string
	Tokens   []int
	IsTokens bool
}

// EmbeddingData is one embedding in a response. Embedding is raw JSON: either
// a float array or a base64 string, depending on encoding_format.
type EmbeddingData struct {
	Object    string          `json:"object"`
	Index     int             `json:"index"`
	Embedding json.RawMessage `json:"embedding"`
}

type Usage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type EmbeddingsResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  Usage           `json:"usage"`
}

// ErrorResponse is the OpenAI-style error envelope.
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

// SplitInput expands the raw input field into individual items. All items in
// one request are homogeneous: either all text or all pre-tokenized.
func SplitInput(raw json.RawMessage) ([]InputItem, error) {
	if len(raw) == 0 {
		return nil, errors.New("missing input")
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return []InputItem{{Text: s}}, nil
	}
	var ss []string
	if err := json.Unmarshal(raw, &ss); err == nil {
		if len(ss) == 0 {
			return nil, errors.New("input array is empty")
		}
		items := make([]InputItem, len(ss))
		for i, t := range ss {
			items[i] = InputItem{Text: t}
		}
		return items, nil
	}
	var tt [][]int
	if err := json.Unmarshal(raw, &tt); err == nil {
		if len(tt) == 0 {
			return nil, errors.New("input array is empty")
		}
		items := make([]InputItem, len(tt))
		for i, t := range tt {
			items[i] = InputItem{Tokens: t, IsTokens: true}
		}
		return items, nil
	}
	var t []int
	if err := json.Unmarshal(raw, &t); err == nil {
		return []InputItem{{Tokens: t, IsTokens: true}}, nil
	}
	return nil, fmt.Errorf("input must be a string, array of strings, array of tokens, or array of token arrays")
}

// MarshalInputs re-encodes a subset of items as a valid input field for an
// upstream request, preserving order.
func MarshalInputs(items []InputItem) (json.RawMessage, error) {
	if len(items) == 0 {
		return nil, errors.New("no items")
	}
	if items[0].IsTokens {
		arrs := make([][]int, len(items))
		for i, it := range items {
			if it.Tokens == nil {
				arrs[i] = []int{}
			} else {
				arrs[i] = it.Tokens
			}
		}
		return json.Marshal(arrs)
	}
	strs := make([]string, len(items))
	for i, it := range items {
		strs[i] = it.Text
	}
	return json.Marshal(strs)
}
