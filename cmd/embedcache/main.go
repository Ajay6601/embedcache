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
	"time"

	"embedcache/internal/analyze"
	"embedcache/internal/cache"
	"embedcache/internal/fingerprint"
	"embedcache/internal/pricing"
	"embedcache/internal/proxy"
	"embedcache/internal/server"
	"embedcache/internal/stats"
	"embedcache/internal/upstream"
)

const version = "0.1.0"

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
  embedcache version

run "embedcache serve -h" or "embedcache analyze -h" for flags.
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
	if *requestLog != "" {
		f, err := os.OpenFile(*requestLog, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return fmt.Errorf("opening request log: %w", err)
		}
		defer f.Close()
		p.SetRequestLog(f)
	}

	srv := &http.Server{
		Addr:    *listen,
		Handler: server.New(p, st, table, up.Base).Handler(),
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
