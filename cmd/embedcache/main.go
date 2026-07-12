// embedcache is a caching, deduplicating, cost-accounting proxy for
// OpenAI-compatible embedding APIs — self-hosted engines (vLLM, Ollama, TEI)
// and hosted providers alike.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
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
	"github.com/Ajay6601/embedcache/internal/mockllm"
	"github.com/Ajay6601/embedcache/internal/pricing"
	"github.com/Ajay6601/embedcache/internal/proxy"
	"github.com/Ajay6601/embedcache/internal/rediscache"
	"github.com/Ajay6601/embedcache/internal/semantic"
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
	case "demo":
		err = runDemo(os.Args[2:])
	case "check":
		err = runCheck(os.Args[2:])
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
  embedcache demo    [flags]                 zero-setup walkthrough against a built-in mock backend
  embedcache check   -upstream URL [flags]   probe a backend and print the flags to serve it
  embedcache analyze [flags] [file ...]      offline waste report from JSONL request logs (or stdin)
  embedcache report  [-addr URL]             fetch the live waste report from a running proxy
  embedcache chunk   [flags] [file]          split a file into content-defined chunks (stdin if no file)
  embedcache version

new here? run "embedcache demo" first — no upstream, no API key, no config needed.

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
	normalize := fs.String("normalize", "", "opt-in input normalization before matching: comma-separated trim,collapse,lowercase,nfc (default: byte-exact)")
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
	sharedRedis := fs.String("shared-redis", "", "host:port of a Redis/Valkey to use as a shared cache tier so instances share entries (empty = in-memory only)")
	sharedRedisPassword := fs.String("shared-redis-password", "", "password for -shared-redis (or set EMBEDCACHE_REDIS_PASSWORD)")
	sharedRedisDB := fs.Int("shared-redis-db", 0, "logical Redis db for -shared-redis")
	sharedRedisPrefix := fs.String("shared-redis-prefix", "ec:", "key prefix for shared cache entries")
	sharedRedisTTL := fs.Duration("shared-redis-ttl", 0, "expiry for shared cache entries (0 = no expiry)")
	semanticMode := fs.String("semantic", "off", "near-duplicate matching: off | shadow (measure only, serves nothing approximate) | active (serve a neighbor's vector above -semantic-threshold)")
	semanticThreshold := fs.Float64("semantic-threshold", 0.9, "text (trigram Jaccard) similarity in [0,1] above which two inputs are treated as near-duplicates")
	semanticMaxKeys := fs.Int("semantic-max-keys", 500_000, "max inputs kept in the near-duplicate index (0 = unbounded)")
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
	if *sharedRedis != "" {
		pw := *sharedRedisPassword
		if pw == "" {
			pw = os.Getenv("EMBEDCACHE_REDIS_PASSWORD")
		}
		shared, err := rediscache.New(rediscache.Options{
			Addr:     *sharedRedis,
			Password: pw,
			DB:       *sharedRedisDB,
			Prefix:   *sharedRedisPrefix,
			TTL:      *sharedRedisTTL,
		})
		if err != nil {
			return fmt.Errorf("connecting to shared Redis at %s: %w", *sharedRedis, err)
		}
		c.SetShared(shared)
		log.Printf("shared cache tier: Redis at %s (db %d, prefix %q, ttl %s)", *sharedRedis, *sharedRedisDB, *sharedRedisPrefix, *sharedRedisTTL)
	}

	st := stats.New()
	p := proxy.New(c, up, st, norm)
	p.CacheTTL = *cacheTTL
	p.MaxBatchItems = *maxBatch
	p.MaxBody = *maxBodyMB << 20

	switch *semanticMode {
	case "off":
	case "shadow", "active":
		p.Semantic = semantic.New(*semanticMaxKeys)
		p.SemanticMode = *semanticMode
		p.SemanticThreshold = *semanticThreshold
		if *semanticMode == "active" {
			log.Printf("warning: semantic caching is ACTIVE (threshold %.2f) — near-duplicate inputs are served an existing vector instead of being computed; run -semantic shadow first to confirm the cosine deltas are acceptable on your data", *semanticThreshold)
		} else {
			log.Printf("semantic caching in shadow mode (threshold %.2f): measuring near-neighbor cosine deltas, serving nothing approximate", *semanticThreshold)
		}
	default:
		return fmt.Errorf("-semantic must be off, shadow, or active (got %q)", *semanticMode)
	}

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

// runDemo runs embedcache against a built-in deterministic mock embedding
// backend (internal/mockllm) so a newcomer can see miss/hit/partial/waste-report
// behavior in one command, with zero external setup: no Ollama, no API key,
// no config file. The mock backend produces a real HTTP response over a real
// loopback connection through the real proxy code path — nothing here is
// faked at the proxy layer, only the "model" behind it is a stand-in.
func runDemo(args []string) error {
	fs := flag.NewFlagSet("demo", flag.ExitOnError)
	listen := fs.String("listen", "127.0.0.1:8090", "address for the demo proxy to listen on")
	fs.Parse(args)

	mock := mockllm.New(256)
	mockLn, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return err
	}
	go http.Serve(mockLn, mock.Handler())
	mockBase := "http://" + mockLn.Addr().String()

	up, err := upstream.New(mockBase, "", 30*time.Second)
	if err != nil {
		return err
	}
	c := cache.New(0, 0)
	st := stats.New()
	p := proxy.New(c, up, st, fingerprint.Normalizer{})
	websrv := server.New(p, st, pricing.Default(), up.Base)

	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		return fmt.Errorf("listening on %s: %w (pass -listen to use a different port)", *listen, err)
	}
	go http.Serve(ln, websrv.Handler())
	base := "http://" + ln.Addr().String()

	fmt.Println("embedcache demo — proxying a built-in mock embedding model, nothing external required")
	fmt.Println("proxy listening on", base)
	fmt.Println()

	fmt.Println("1. a new input — expect a cache MISS (real compute against the mock model):")
	demoEmbed(base, "the quick brown fox jumps over the lazy dog")
	fmt.Println()

	fmt.Println("2. the exact same input again — expect a cache HIT (no upstream call):")
	demoEmbed(base, "the quick brown fox jumps over the lazy dog")
	fmt.Println()

	fmt.Println("3. a batch of 3 inputs, one repeated + two new — expect PARTIAL:")
	demoEmbedBatch(base, []string{
		"the quick brown fox jumps over the lazy dog",
		"a sentence nobody has embedded before",
		"another brand new sentence for the cache",
	})
	fmt.Println()

	fmt.Println("4. the live waste report:")
	if resp, err := http.Get(base + "/_ec/report"); err == nil {
		io.Copy(os.Stdout, resp.Body)
		resp.Body.Close()
	}

	fmt.Println()
	fmt.Println("the demo proxy keeps running — try it yourself:")
	fmt.Printf("  curl %s/v1/embeddings -d '{\"model\":\"demo\",\"input\":\"hello world\"}'\n", base)
	fmt.Printf("  curl %s/_ec/stats\n", base)
	fmt.Println("press Ctrl+C to stop.")

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	<-ctx.Done()
	return nil
}

func demoEmbed(base, input string) {
	start := time.Now()
	body, _ := json.Marshal(map[string]any{"model": "demo", "input": input})
	req, _ := http.NewRequest(http.MethodPost, base+"/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Println("   error:", err)
		return
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	fmt.Printf("   -> %s in %s (X-Embedcache-Status: %s)\n", resp.Status, time.Since(start).Round(time.Millisecond), resp.Header.Get("X-Embedcache-Status"))
}

func demoEmbedBatch(base string, inputs []string) {
	start := time.Now()
	body, _ := json.Marshal(map[string]any{"model": "demo", "input": inputs})
	req, _ := http.NewRequest(http.MethodPost, base+"/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Println("   error:", err)
		return
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	fmt.Printf("   -> %s in %s (X-Embedcache-Status: %s)\n", resp.Status, time.Since(start).Round(time.Millisecond), resp.Header.Get("X-Embedcache-Status"))
}

// runCheck probes a real upstream and prints exactly what -upstream/-upstream-api-key
// (and, for path quirks like Gemini's OpenAI-compat endpoint, the client base_url)
// to use — the setup step most first-time backend integrations get wrong.
func runCheck(args []string) error {
	fs := flag.NewFlagSet("check", flag.ExitOnError)
	upstreamURL := fs.String("upstream", "", "base URL of the OpenAI-compatible upstream (required)")
	apiKey := fs.String("upstream-api-key", os.Getenv("EMBEDCACHE_UPSTREAM_KEY"), "API key for the upstream, if it requires one")
	model := fs.String("model", "", "model to probe with (required — any model name the upstream serves)")
	fs.Parse(args)

	if *upstreamURL == "" || *model == "" {
		return fmt.Errorf("-upstream and -model are required, e.g. embedcache check -upstream http://localhost:11434 -model all-minilm")
	}

	base := strings.TrimRight(*upstreamURL, "/")
	body, _ := json.Marshal(map[string]any{"model": *model, "input": "embedcache check probe"})
	req, _ := http.NewRequest(http.MethodPost, base+"/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if *apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+*apiKey)
	}
	client := &http.Client{Timeout: 30 * time.Second}
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		fmt.Printf("FAIL: could not reach %s: %v\n", base+"/v1/embeddings", err)
		fmt.Println("  - is the upstream running and reachable from this machine?")
		fmt.Println("  - if it's a hosted API, check the base URL doesn't already include /v1")
		return nil
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	elapsed := time.Since(start)

	if resp.StatusCode != 200 {
		fmt.Printf("FAIL: %s responded %s in %s\n", base+"/v1/embeddings", resp.Status, elapsed.Round(time.Millisecond))
		fmt.Printf("  body: %.300s\n", raw)
		if resp.StatusCode == 401 || resp.StatusCode == 403 {
			fmt.Println("  - looks like an auth problem: pass -upstream-api-key, or check the key has embeddings access")
		}
		if resp.StatusCode == 404 {
			if strings.Contains(strings.ToLower(string(raw)), "model") {
				fmt.Printf("  - the upstream doesn't recognize model %q — check the exact model name it serves\n", *model)
			} else {
				fmt.Println("  - some providers use a different path (e.g. Gemini's OpenAI-compat endpoint is <base>/v1beta/openai, not <base>/v1)")
			}
		}
		return nil
	}

	var parsed struct {
		Data []struct {
			Embedding json.RawMessage `json:"embedding"`
		} `json:"data"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil || len(parsed.Data) == 0 {
		fmt.Printf("FAIL: %s returned 200 but the body isn't a recognizable /v1/embeddings response\n", base+"/v1/embeddings")
		fmt.Printf("  body: %.300s\n", raw)
		return nil
	}
	var dims int
	var f []float64
	if json.Unmarshal(parsed.Data[0].Embedding, &f) == nil {
		dims = len(f)
	}

	fmt.Printf("OK: %s responded 200 in %s\n", base+"/v1/embeddings", elapsed.Round(time.Millisecond))
	fmt.Printf("  model %q -> %d-dim vectors, usage.total_tokens=%d\n", *model, dims, parsed.Usage.TotalTokens)
	fmt.Println()
	fmt.Println("start embedcache with:")
	if *apiKey != "" {
		fmt.Printf("  embedcache serve -upstream %s -upstream-api-key %s\n", *upstreamURL, *apiKey)
	} else {
		fmt.Printf("  embedcache serve -upstream %s\n", *upstreamURL)
	}
	return nil
}
