// Package pricing maps embedding models to $ per million tokens so savings
// can be expressed in money, which is the entire point.
package pricing

import (
	"encoding/json"
	"os"
	"strings"
)

// Table maps model name -> USD per 1M tokens. DefaultPrice applies to models
// not in the table (self-hosted models have no list price; the default lets
// teams assign an amortized GPU cost per token).
type Table struct {
	PerMillion   map[string]float64
	DefaultPrice float64
}

func Default() *Table {
	return &Table{
		PerMillion: map[string]float64{
			"text-embedding-3-small": 0.02,
			"text-embedding-3-large": 0.13,
			"text-embedding-ada-002": 0.10,
		},
		DefaultPrice: 0.02,
	}
}

// LoadFile merges a JSON file of {"model": dollarsPerMillion} into the table.
// The special key "default" overrides DefaultPrice.
func (t *Table) LoadFile(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var m map[string]float64
	if err := json.Unmarshal(b, &m); err != nil {
		return err
	}
	for k, v := range m {
		if strings.EqualFold(k, "default") {
			t.DefaultPrice = v
			continue
		}
		t.PerMillion[k] = v
	}
	return nil
}

func (t *Table) Cost(model string, tokenCount int64) float64 {
	price, ok := t.PerMillion[model]
	if !ok {
		price = t.DefaultPrice
	}
	return float64(tokenCount) * price / 1e6
}
