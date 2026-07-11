// embedcache is a caching, deduplicating, cost-accounting proxy for
// OpenAI-compatible embedding APIs — self-hosted engines (vLLM, Ollama, TEI)
// and hosted providers alike.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/Ajay6601/embedcache/internal/analyze"
	"github.com/Ajay6601/embedcache/internal/auth"
	"github.com/Ajay6601/embedcache/internal/breaker"
	"github.com/Ajay6601/embedcache/internal/budget"
	"github.com/Ajay6601/embedcache/internal/cache"
	"github.com/Ajay6601/embedcache/internal/chunker"
	"github.com/Ajay6601/embedcache/internal/fingerprint"
	"github.com/Ajay6601/embedcache/internal/pricing"
	"github.com/Ajay6601/embedcache/internal/proxy"
	"github.com/Ajay6601/embedcache/internal/server"
	"github.com/Ajay6601/embedcache/internal/stats"
	"github.com/Ajay6601/embedcache/internal/upstream"
)

const version = "0.2.0"

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "serve":
		err = runServe(os.Args[2:])
	case "analyze":
		err = runAnalyze(os.Args[2:])
	case "report":
		err = runReport(os.Args[2:])
	case "chunk":
		err = runChunk(os.Args[2:])
	case "version":
		fmt.Println("embedcache", version)
	case "help", "-h", "--help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n", os.Args[1])
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Print(`embedcache — the cost-control and dedupe layer for embedding APIs

usage:
  embedcache serve   -upstream URL [flags]   run the caching proxy
  embedcache analyze [flags] [file ...]      offline waste report from JSONL request logs (or stdin)
  embedcache report  [-addr URL]             fetch the live waste report from a running proxy
  embedcache chunk   [flags] [file]          split a file into content-defined chunks (stdin if no file)
  embedcache version

run "embedcache serve -h", "embedcache analyze -h", or "embedcache chunk -h" for flags.
`)
}

func runServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	listen := fs.String("listen", ":8090", "address to listen on")
	upstreamURL := fs.String("upstream", "", "base URL of the OpenAI-compatible upstream (required), e.g. https://api.openai.com or http://localhost:8000")
	upstreamKey := fs.String("upstream-api-key", "", "API key for the upstream; default forwards the client's Authorization header")
	upstreamTimeout := fs.Duration("upstream-timeout", 120*time.Second, "upstream request timeout")
	maxEntries := fs.Int("max-entries", 1_000_000, "max cached embeddings (0 = unlimited)")
	maxMemoryMB := fs.Int64("max-memory-mb", 1024, "max cache payload size in MB (0 = unlimited)")
	normalize := fs.String("normalize", "", "opt-in input normalization before matching: comma-separated trim,collapse,lowercase (default: byte-exact)")
	persist := fs.String("persist", "", "path to a cache snapshot file; loaded at start, saved on shutdown and every persist-interval")
	persistInterval := fs.Duration("persist-interval", 5*time.Minute, "how often to snapshot the cache when -persist is set")
	pricingFile := fs.String("pricing", "", "JSON file of {\"model\": dollarsPerMillionTokens}; key \"default\" sets the fallback price")
	requestLog := fs.String("request-log", "", "append every embedding request as JSONL (feed it to `embedcache analyze`)")
	adminToken := fs.String("admin-token", "", "bearer token required for stats/report/metrics/flush endpoints")
	authMode := fs.String("auth-mode", "off", "client key validation: off, allowlist, or verify (checks keys against the upstream)")
	apiKeys := fs.String("api-keys", "", "comma-separated client keys for -auth-mode allowlist")
	apiKeysFile := fs.String("api-keys-file", "", "file with one client key per line for -auth-mode allowlist")
	authCacheTTL := fs.Duration("auth-cache-ttl", 5*time.Minute, "how long a verified key stays trusted in -auth-mode verify")
	cacheTTL := fs.Duration("ttl", 0, "expire cached embeddings after this duration (0 = never; use when model weights change under the same name)")
	maxBatch := fs.Int("max-batch-items", 2048, "reject batches with more items than this (0 = unlimited)")
	maxBodyMB := fs.Int64("max-body-mb", 64, "reject request bodies larger than this")
	budgetTokens := fs.Int64("budget-tokens", 0, "default per-key upstream token budget per window; once spent, requests needing new computation get 429 while cache hits keep serving (0 = no budgets)")
	budgetsFile := fs.String("budgets-file", "", "JSON file of {\"<api key>\": tokensPerWindow}; key \"default\" overrides -budget-tokens, 0 makes a key unlimited")
	budgetWindow := fs.Duration("budget-window", 24*time.Hour, "budget window; spend counters reset at each window boundary (and on restart)")
	retries := fs.Int("upstream-retries", 2, "extra attempts for transient upstream failures (network errors, 5xx, 429)")
	breakerThreshold := fs.Int("breaker-threshold", 5, "consecutive upstream failures before the circuit opens (0 = disabled)")
	breakerCooldown := fs.Duration("breaker-cooldown", 10*time.Second, "how long the circuit stays open before probing the upstream")
	requestLogMaxMB := fs.Int64("request-log-max-mb", 512, "rotate the request log when it reaches this size (one .1 backup kept)")
	fs.Parse(args)

	if *upstreamURL == "" {
		return fmt.Errorf("-upstream is required")
	}
	norm, err := fingerprint.ParseNormalizer(*normalize)
	if err != nil {
		return err
	}
	up, err := upstream.New(*upstreamURL, *upstreamKey, *upstreamTimeout)
	if err != nil {
		return err
	}
	up.Retries = *retries
	up.Breaker = breaker.New(*breakerThreshold, *breakerCooldown)
	table := pricing.Default()
	if *pricingFile != "" {
		if err := table.LoadFile(*pricingFile); err != nil {
			return fmt.Errorf("loading pricing file: %w", err)
		}
	}

	c := cache.New(*maxEntries, *maxMemoryMB<<20)
	if *persist != "" {
		n, err := c.Load(*persist)
		if err != nil {
			return fmt.Errorf("loading cache snapshot: %w", err)
		}
		if n > 0 {
			log.Printf("loaded %d cached embeddings from %s", n, *persist)
		}
	}

	st := stats.New()
	p := proxy.New(c, up, st, norm)
	p.CacheTTL = *cacheTTL
	p.MaxBatchItems = *maxBatch
	p.MaxBody = *maxBodyMB << 20

	if auth.Mode(*authMode) != auth.ModeOff {
		keys := splitKeys(*apiKeys)
		if *apiKeysFile != "" {
			fileKeys, err := readKeysFile(*apiKeysFile)
			if err != nil {
				return err
			}
			keys = append(keys, fileKeys...)
		}
		authorizer, err := auth.New(auth.Mode(*authMode), keys, *upstreamURL, *authCacheTTL, nil)
		if err != nil {
			return err
		}
		p.Auth = authorizer
	}
	if *adminToken == "" {
		log.Printf("warning: admin endpoints (stats/report/flush) are unauthenticated; set -admin-token for anything beyond a trusted network")
	}
	if auth.Mode(*authMode) == auth.ModeOff {
		log.Printf("warning: cache hits are served without key validation (-auth-mode off); use allowlist or verify for untrusted callers")
	}

	if *budgetTokens > 0 || *budgetsFile != "" {
		defLimit := *budgetTokens
		perKey := map[string]int64{}
		if *budgetsFile != "" {
			b, err := os.ReadFile(*budgetsFile)
			if err != nil {
				return fmt.Errorf("reading budgets file: %w", err)
			}
			var m map[string]int64
			if err := json.Unmarshal(b, &m); err != nil {
				return fmt.Errorf("parsing budgets file: %w", err)
			}
			for k, v := range m {
				if strings.EqualFold(k, "default") {
					defLimit = v
				} else {
					perKey[k] = v
				}
			}
		}
		enforcer := budget.New(defLimit, *budgetWindow)
		for k, v := range perKey {
			enforcer.SetLimit(k, v)
		}
		p.Budget = enforcer
		log.Printf("budget enforcement on: window=%s default=%d tokens, %d per-key limits", *budgetWindow, defLimit, len(perKey))
	}

	up.OnRetry = st.RecordRetry

	if *requestLog != "" {
		if err := p.OpenRequestLog(*requestLog, *requestLogMaxMB<<20); err != nil {
			return fmt.Errorf("opening request log: %w", err)
		}
		defer p.CloseRequestLog()
	}

	websrv := server.New(p, st, table, up.Base)
	websrv.AdminToken = *adminToken
	websrv.SnapshotPath = *persist
	srv := &http.Server{
		Addr:    *listen,
		Handler: websrv.Handler(),
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if *persist != "" {
		go func() {
			t := time.NewTicker(*persistInterval)
			defer t.Stop()
			for {
				select {
				case <-t.C:
					if err := c.Snapshot(*persist); err != nil {
						log.Printf("cache snapshot failed: %v", err)
					}
				case <-ctx.Done():
					return
				}
			}
		}()
	}

	go func() {
		<-ctx.Done()
		log.Println("shutting down...")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		srv.Shutdown(shutdownCtx)
	}()

	log.Printf("embedcache %s listening on %s -> %s (normalize=%q)", version, *listen, *upstreamURL, *normalize)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return err
	}
	if *persist != "" {
		if err := c.Snapshot(*persist); err != nil {
			return fmt.Errorf("final cache snapshot: %w", err)
		}
		log.Printf("saved %d cached embeddings to %s", c.Len(), *persist)
	}
	return nil
}

func splitKeys(csv string) []string {
	var out []string
	for _, k := range strings.Split(csv, ",") {
		if k = strings.TrimSpace(k); k != "" {
			out = append(out, k)
		}
	}
	return out
}

func readKeysFile(path string) ([]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading api keys file: %w", err)
	}
	var out []string
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line != "" && !strings.HasPrefix(line, "#") {
			out = append(out, line)
		}
	}
	return out, nil
}

func runAnalyze(args []string) error {
	fs := flag.NewFlagSet("analyze", flag.ExitOnError)
	normalize := fs.String("normalize", "", "normalization to assume when matching: trim,collapse,lowercase")
	pricingFile := fs.String("pricing", "", "JSON pricing file, {\"model\": dollarsPerMillionTokens}")
	defaultPrice := fs.Float64("default-price", 0, "override $/1M tokens for models without a list price (e.g. amortized GPU cost)")
	topN := fs.Int("top", 10, "how many top duplicated inputs to show")
	asJSON := fs.Bool("json", false, "emit the result as JSON instead of a text report")
	fs.Parse(args)

	norm, err := fingerprint.ParseNormalizer(*normalize)
	if err != nil {
		return err
	}
	table := pricing.Default()
	if *pricingFile != "" {
		if err := table.LoadFile(*pricingFile); err != nil {
			return err
		}
	}
	if *defaultPrice > 0 {
		table.DefaultPrice = *defaultPrice
	}

	var in io.Reader = os.Stdin
	if fs.NArg() > 0 {
		readers := make([]io.Reader, 0, fs.NArg())
		for _, path := range fs.Args() {
			f, err := os.Open(path)
			if err != nil {
				return err
			}
			defer f.Close()
			readers = append(readers, f)
		}
		in = io.MultiReader(readers...)
	}

	res, err := analyze.Run(in, analyze.Options{Norm: norm, Pricing: table, TopN: *topN})
	if err != nil {
		return err
	}
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(res)
	}
	res.RenderText(os.Stdout)
	return nil
}

func runReport(args []string) error {
	fs := flag.NewFlagSet("report", flag.ExitOnError)
	addr := fs.String("addr", "http://localhost:8090", "address of a running embedcache proxy")
	fs.Parse(args)
	resp, err := http.Get(*addr + "/_ec/report")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	_, err = io.Copy(os.Stdout, resp.Body)
	return err
}

// runChunk splits a file into content-defined chunks so an ingestion
// pipeline in any language can pipe the output straight into batched
// /v1/embeddings calls. Because boundaries are content-defined, editing one
// part of a document only changes the chunks touching the edit — the rest
// re-chunk to byte-identical text and hit embedcache's cache on re-ingestion,
// closing the gap fixed-size chunking has (see EXPERIMENTS.md E4).
func runChunk(args []string) error {
	fs := flag.NewFlagSet("chunk", flag.ExitOnError)
	min := fs.Int("min", chunker.DefaultMin, "minimum chunk size in bytes")
	avg := fs.Int("avg", chunker.DefaultAvg, "target average chunk size in bytes")
	max := fs.Int("max", chunker.DefaultMax, "maximum chunk size in bytes")
	fs.Parse(args)

	var (
		data []byte
		err  error
	)
	if fs.NArg() > 0 {
		data, err = os.ReadFile(fs.Arg(0))
	} else {
		data, err = io.ReadAll(os.Stdin)
	}
	if err != nil {
		return err
	}

	chunks := chunker.Split(data, chunker.Options{Min: *min, Avg: *avg, Max: *max})
	enc := json.NewEncoder(os.Stdout)
	for _, c := range chunks {
		if err := enc.Encode(struct {
			Hash string `json:"hash"`
			Size int    `json:"size"`
			Text string `json:"text"`
		}{c.Hash, len(c.Data), string(c.Data)}); err != nil {
			return err
		}
	}
	return nil
}
