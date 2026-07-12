// semanticshadow exercises semantic (near-duplicate) caching in shadow mode
// against a REAL backend and reports what the shadow numbers actually look like:
// for real punctuation/word-order/casing variants of real questions, how close
// is the near-neighbor's cached vector to the freshly-computed one? That cosine
// distribution is exactly what tells you whether turning ACTIVE mode on would be
// safe on your data — measured here, never guessed.
//
// Usage:
//
//	go build -o embedcache.exe ./cmd/embedcache
//	go run ./experiments/semanticshadow -bin ./embedcache.exe -upstream http://localhost:11434 -model nomic-embed-text -out SEMANTIC.md
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"time"
)

var (
	binPath   = flag.String("bin", "./embedcache.exe", "compiled embedcache binary")
	upstream  = flag.String("upstream", "http://localhost:11434", "real embedding backend")
	model     = flag.String("model", "nomic-embed-text", "embedding model")
	threshold = flag.Float64("threshold", 0.8, "trigram-Jaccard threshold for a near-duplicate")
	outPath   = flag.String("out", "SEMANTIC.md", "results file")
)

var (
	rep    bytes.Buffer
	client = &http.Client{Timeout: 60 * time.Second}
)

func line(f string, a ...any) { fmt.Fprintf(&rep, f+"\n", a...) }

// base questions and, for each, real near-duplicates a user would actually type:
// punctuation, casing, filler words, mild word-order changes. No synthetic noise.
var pairs = []struct {
	base     string
	variants []string
}{
	{"how do I reset my password", []string{
		"how do I reset my password?", "How do I reset my password", "how do i reset my password please",
	}},
	{"what are the pricing plans", []string{
		"what are the pricing plans?", "what are your pricing plans", "what are the pricing plans exactly",
	}},
	{"how do I cancel my subscription", []string{
		"how do I cancel my subscription?", "how to cancel my subscription", "how do I cancel my subscription now",
	}},
	{"where can I download the invoice", []string{
		"where can I download the invoice?", "where do I download the invoice", "where can I download my invoice",
	}},
}

func freePort() string {
	l, _ := net.Listen("tcp", "127.0.0.1:0")
	a := l.Addr().String()
	l.Close()
	return a
}

func embed(base, input string) error {
	body, _ := json.Marshal(map[string]any{"model": *model, "input": input})
	resp, err := client.Post(base+"/v1/embeddings", "application/json", bytes.NewReader(body))
	if err != nil {
		return err
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	return nil
}

func stats(base string) map[string]any {
	r, err := client.Get(base + "/_ec/stats")
	if err != nil {
		return nil
	}
	defer r.Body.Close()
	var m map[string]any
	json.NewDecoder(r.Body).Decode(&m)
	return m
}

func main() {
	flag.Parse()

	addr := freePort()
	cmd := exec.Command(*binPath, "serve", "-listen", addr, "-upstream", *upstream,
		"-semantic", "shadow", "-semantic-threshold", fmt.Sprint(*threshold))
	cmd.Stdout, cmd.Stderr = io.Discard, io.Discard
	if err := cmd.Start(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer func() { cmd.Process.Kill(); cmd.Wait() }()
	base := "http://" + addr
	for i := 0; i < 100; i++ {
		if r, err := client.Get(base + "/healthz"); err == nil {
			r.Body.Close()
			break
		}
		time.Sleep(80 * time.Millisecond)
	}

	fmt.Fprintf(&rep, "# embedcache semantic caching: shadow-mode measurement (real backend)\n\n")
	fmt.Fprintf(&rep, "Generated %s. Semantic caching in **shadow** mode against real Ollama (`%s`), threshold %.2f.\n", time.Now().Format("2006-01-02 15:04 MST"), *model, *threshold)
	fmt.Fprintf(&rep, "Nothing approximate is served here — the proxy computes every input for real and only records how\n")
	fmt.Fprintf(&rep, "far each near-neighbor's cached vector is from the truth. That cosine is the number that decides\n")
	fmt.Fprintf(&rep, "whether **active** mode would be safe on this kind of traffic.\n\n")

	// warm the base questions first so they are in the cache/index
	for _, p := range pairs {
		if err := embed(base, p.base); err != nil {
			fmt.Fprintln(os.Stderr, "warm:", err)
		}
	}
	// then send the real near-duplicate variants; each triggers a shadow comparison
	variants := 0
	for _, p := range pairs {
		for _, v := range p.variants {
			if err := embed(base, v); err != nil {
				fmt.Fprintln(os.Stderr, "variant:", err)
				continue
			}
			variants++
		}
	}

	s := stats(base)
	samples := statFloat(s, "semantic_shadow_samples")
	mean := statFloat(s, "semantic_shadow_mean_cosine")
	min := statFloat(s, "semantic_shadow_min_cosine")

	line("Sent %d real near-duplicate variants of %d base questions.", variants, len(pairs))
	line("")
	line("| shadow observations | real-vs-neighbor cosine (mean) | worst case |")
	line("|---|---|---|")
	line("| %.0f | **%.4f** | %.4f |", samples, mean, min)
	line("")
	if samples == 0 {
		line("No near-duplicates crossed the %.2f threshold — nothing to measure at this setting.", *threshold)
	} else if min >= 0.95 {
		line("At this threshold, even the *worst* near-duplicate's cached vector was cosine **%.4f** from the", min)
		line("freshly-computed one. Serving it (active mode) would have changed retrieval negligibly, so active")
		line("mode looks safe for this traffic at threshold %.2f.", *threshold)
	} else {
		line("The worst near-duplicate was only cosine **%.4f** from the truth. That is enough drift that active", min)
		line("mode could shift some rankings — raise the threshold or keep it in shadow before enabling.")
	}
	line("")
	line("This is the whole point of shadow mode: the decision to enable approximate serving is made from")
	line("your own measured cosine distribution, not from a vendor's blanket claim.")

	if err := os.WriteFile(*outPath, rep.Bytes(), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("shadow-mode measurement written to %s: %.0f samples, mean cosine %.4f, worst %.4f\n", *outPath, samples, mean, min)
}

func statFloat(m map[string]any, key string) float64 {
	if m == nil {
		return 0
	}
	if v, ok := m[key].(float64); ok {
		return v
	}
	return 0
}
