// Package analyze is the offline waste analyzer: feed it a JSONL log of
// embedding requests and it reports how much of the spend was duplicate work
// — before anyone installs anything in the request path.
package analyze

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/Ajay6601/embedcache/internal/api"
	"github.com/Ajay6601/embedcache/internal/fingerprint"
	"github.com/Ajay6601/embedcache/internal/pricing"
	"github.com/Ajay6601/embedcache/internal/tokens"
)

type Options struct {
	Norm    fingerprint.Normalizer
	Pricing *pricing.Table
	TopN    int
}

type ModelAgg struct {
	Items        int     `json:"items"`
	Unique       int     `json:"unique"`
	DupItems     int     `json:"duplicate_items"`
	TotalTokens  int64   `json:"total_tokens_est"`
	WastedTokens int64   `json:"wasted_tokens_est"`
	WastedUSD    float64 `json:"wasted_usd_est"`
}

type DupEntry struct {
	Preview      string `json:"preview"`
	Model        string `json:"model"`
	Count        int    `json:"count"`
	Tokens       int    `json:"tokens_est"`
	WastedTokens int64  `json:"wasted_tokens_est"`
}

type Result struct {
	Requests     int                  `json:"requests"`
	Skipped      int                  `json:"skipped_lines"`
	Items        int                  `json:"items"`
	Unique       int                  `json:"unique_items"`
	DupItems     int                  `json:"duplicate_items"`
	DupRate      float64              `json:"duplicate_rate"`
	TotalTokens  int64                `json:"total_tokens_est"`
	WastedTokens int64                `json:"wasted_tokens_est"`
	TotalUSD     float64              `json:"total_usd_est"`
	WastedUSD    float64              `json:"wasted_usd_est"`
	PerModel     map[string]*ModelAgg `json:"per_model"`
	TopDup       []DupEntry           `json:"top_duplicates"`
}

type fpInfo struct {
	count   int
	tokens  int
	preview string
	model   string
}

// Run reads JSONL from r, one embedding request per line. Lines that are not
// embedding requests are skipped, so raw gateway/access logs work as long as
// each line embeds a {"model": ..., "input": ...} object somewhere at the
// top level or under "body"/"request"/"payload".
func Run(r io.Reader, opts Options) (*Result, error) {
	if opts.Pricing == nil {
		opts.Pricing = pricing.Default()
	}
	if opts.TopN <= 0 {
		opts.TopN = 10
	}
	res := &Result{PerModel: map[string]*ModelAgg{}}
	seen := map[string]*fpInfo{}

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 1<<20), 64<<20)
	for sc.Scan() {
		line := sc.Bytes()
		if len(strings.TrimSpace(string(line))) == 0 {
			continue
		}
		req, ok := extractRequest(line)
		if !ok {
			res.Skipped++
			continue
		}
		items, err := api.SplitInput(req.Input)
		if err != nil {
			res.Skipped++
			continue
		}
		res.Requests++
		agg := res.PerModel[req.Model]
		if agg == nil {
			agg = &ModelAgg{}
			res.PerModel[req.Model] = agg
		}
		paramsDigest := fingerprint.ParamsDigest(req.Extra)
		for _, item := range items {
			key := fingerprint.Key(req.Model, req.Dimensions, req.EncodingFormat, paramsDigest, item, opts.Norm)
			tok := tokens.Estimate(item)
			res.Items++
			agg.Items++
			res.TotalTokens += int64(tok)
			agg.TotalTokens += int64(tok)
			if info, dup := seen[key]; dup {
				info.count++
				res.DupItems++
				agg.DupItems++
				res.WastedTokens += int64(tok)
				agg.WastedTokens += int64(tok)
			} else {
				seen[key] = &fpInfo{count: 1, tokens: tok, preview: preview(item), model: req.Model}
				res.Unique++
				agg.Unique++
			}
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}

	if res.Items > 0 {
		res.DupRate = float64(res.DupItems) / float64(res.Items)
	}
	for model, agg := range res.PerModel {
		agg.WastedUSD = opts.Pricing.Cost(model, agg.WastedTokens)
		res.WastedUSD += agg.WastedUSD
		res.TotalUSD += opts.Pricing.Cost(model, agg.TotalTokens)
	}

	dups := make([]DupEntry, 0)
	for _, info := range seen {
		if info.count < 2 {
			continue
		}
		dups = append(dups, DupEntry{
			Preview:      info.preview,
			Model:        info.model,
			Count:        info.count,
			Tokens:       info.tokens,
			WastedTokens: int64(info.count-1) * int64(info.tokens),
		})
	}
	sort.Slice(dups, func(i, j int) bool { return dups[i].WastedTokens > dups[j].WastedTokens })
	if len(dups) > opts.TopN {
		dups = dups[:opts.TopN]
	}
	res.TopDup = dups
	return res, nil
}

// extractRequest digs an embeddings request out of a log line: either the
// line is the request itself, or it wraps one under body/request/payload
// (as an object or as a string of JSON).
func extractRequest(line []byte) (*api.EmbeddingsRequest, bool) {
	var direct api.EmbeddingsRequest
	if err := json.Unmarshal(line, &direct); err == nil && direct.Model != "" && len(direct.Input) > 0 {
		return &direct, true
	}
	var wrapper map[string]json.RawMessage
	if err := json.Unmarshal(line, &wrapper); err != nil {
		return nil, false
	}
	for _, field := range []string{"body", "request", "payload"} {
		raw, ok := wrapper[field]
		if !ok {
			continue
		}
		// the field may hold JSON-in-a-string
		var s string
		if err := json.Unmarshal(raw, &s); err == nil {
			raw = []byte(s)
		}
		var req api.EmbeddingsRequest
		if err := json.Unmarshal(raw, &req); err == nil && req.Model != "" && len(req.Input) > 0 {
			return &req, true
		}
	}
	return nil, false
}

func preview(item api.InputItem) string {
	if item.IsTokens {
		return fmt.Sprintf("<%d pre-tokenized tokens>", len(item.Tokens))
	}
	t := strings.Join(strings.Fields(item.Text), " ")
	if len(t) > 60 {
		t = t[:57] + "..."
	}
	return t
}

// RenderText prints the human-readable waste report.
func (res *Result) RenderText(w io.Writer) {
	fmt.Fprintf(w, "embedcache offline waste analysis\n")
	fmt.Fprintf(w, "=================================\n")
	fmt.Fprintf(w, "requests analyzed        %d", res.Requests)
	if res.Skipped > 0 {
		fmt.Fprintf(w, "   (skipped %d unrecognized lines)", res.Skipped)
	}
	fmt.Fprintf(w, "\n")
	fmt.Fprintf(w, "embedding items          %d\n", res.Items)
	fmt.Fprintf(w, "unique items             %d\n", res.Unique)
	fmt.Fprintf(w, "duplicate items          %d   (%.1f%% of all items)\n", res.DupItems, res.DupRate*100)
	fmt.Fprintf(w, "estimated tokens         %d   (~$%.2f)\n", res.TotalTokens, res.TotalUSD)
	fmt.Fprintf(w, "estimated wasted tokens  %d   (~$%.2f)\n", res.WastedTokens, res.WastedUSD)
	if res.TotalUSD > 0 {
		fmt.Fprintf(w, "\n>> %.1f%% of this embedding spend was duplicate work an exact-match\n", res.WastedUSD/res.TotalUSD*100)
		fmt.Fprintf(w, ">> cache would have absorbed.\n")
	}
	if len(res.PerModel) > 0 {
		fmt.Fprintf(w, "\nper model:\n")
		models := make([]string, 0, len(res.PerModel))
		for m := range res.PerModel {
			models = append(models, m)
		}
		sort.Strings(models)
		for _, m := range models {
			a := res.PerModel[m]
			fmt.Fprintf(w, "  %-30s items=%-8d dup=%-8d wasted_tokens=%-10d wasted=$%.2f\n",
				m, a.Items, a.DupItems, a.WastedTokens, a.WastedUSD)
		}
	}
	if len(res.TopDup) > 0 {
		fmt.Fprintf(w, "\ntop duplicated inputs:\n")
		for _, d := range res.TopDup {
			fmt.Fprintf(w, "  %4dx  %-8s tokens=%-6d  %q\n", d.Count, d.Model, d.Tokens, d.Preview)
		}
	}
}
