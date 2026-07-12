// perf measures embedcache latency under REAL open-loop (constant-rate) load
// against REAL local backends, avoiding the coordinated-omission bias that a
// naive closed-loop worker pool (fire next request only after the previous
// one completes) produces: under a closed loop, a slow request simply delays
// the next dispatch instead of showing up as tail latency, which understates
// p99/p99.9 exactly when it matters most (during saturation).
//
// Method: for each scenario, first calibrate the backend's real sustainable
// closed-loop throughput, then replay traffic at a fixed target rate (a
// ticker fires dispatches on schedule regardless of completion) for a fixed
// duration, recording send-to-completion latency for every request — queueing
// delay under saturation is captured honestly, not hidden.
//
// Usage:
//
//	go build -o embedcache.exe ./cmd/embedcache
//	go run ./experiments/perf -bin ./embedcache.exe -out PERF.md
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
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

var (
	binPath   = flag.String("bin", "./embedcache.exe", "compiled embedcache binary")
	ollamaURL = flag.String("ollama", "http://localhost:11434", "local Ollama base URL")
	outPath   = flag.String("out", "PERF.md", "results file")
	modelsArg = flag.String("models", "all-minilm,nomic-embed-text,mxbai-embed-large", "comma-separated models spanning dims")
	duration  = flag.Duration("duration", 12*time.Second, "open-loop load duration per scenario")
)

var (
	rep    bytes.Buffer
	client = &http.Client{Timeout: 60 * time.Second}
)

func section(f string, a ...any) { fmt.Fprintf(&rep, "\n## "+f+"\n\n", a...) }
func line(f string, a ...any)    { fmt.Fprintf(&rep, f+"\n", a...) }
func logf(f string, a ...any)    { fmt.Printf(f+"\n", a...) }

func freePort() string {
	l, _ := net.Listen("tcp", "127.0.0.1:0")
	addr := l.Addr().String()
	l.Close()
	return addr
}

type proxy struct {
	base string
	stop func()
}

func startProxy(upstream string) (*proxy, error) {
	addr := freePort()
	cmd := exec.Command(*binPath, "serve", "-listen", addr, "-upstream", upstream)
	cmd.Stdout, cmd.Stderr = io.Discard, io.Discard
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	base := "http://" + addr
	for i := 0; i < 100; i++ {
		if resp, err := client.Get(base + "/healthz"); err == nil {
			resp.Body.Close()
			return &proxy{base: base, stop: func() { cmd.Process.Kill(); cmd.Wait() }}, nil
		}
		time.Sleep(80 * time.Millisecond)
	}
	cmd.Process.Kill()
	return nil, fmt.Errorf("proxy never healthy")
}

func embedOnce(base, model string, input any) (time.Duration, error) {
	body, _ := json.Marshal(map[string]any{"model": model, "input": input})
	req, _ := http.NewRequest(http.MethodPost, strings.TrimRight(base, "/")+"/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	d := time.Since(start)
	if resp.StatusCode != 200 {
		return d, fmt.Errorf("status %d", resp.StatusCode)
	}
	return d, nil
}

// calibrate measures real closed-loop sustainable throughput: N workers hammer
// fn continuously for a short window; rate = completions / elapsed.
func calibrate(fn func() (time.Duration, error), workers int, window time.Duration) float64 {
	var count atomic.Int64
	var wg sync.WaitGroup
	stop := make(chan struct{})
	start := time.Now()
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				if _, err := fn(); err == nil {
					count.Add(1)
				}
			}
		}()
	}
	time.Sleep(window)
	close(stop)
	wg.Wait()
	return float64(count.Load()) / time.Since(start).Seconds()
}

// openLoopLoad dispatches fn at a fixed target rate for the given duration —
// a ticker schedules sends independent of completion, so a slow request shows
// up as tail latency on THAT request rather than delaying the next dispatch.
func openLoopLoad(fn func() (time.Duration, error), rate float64, dur time.Duration) (lats []time.Duration, errs int64) {
	interval := time.Duration(float64(time.Second) / rate)
	if interval <= 0 {
		interval = time.Nanosecond
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	deadline := time.Now().Add(dur)
	var mu sync.Mutex
	var wg sync.WaitGroup
	var errCount atomic.Int64
	for now := range ticker.C {
		if now.After(deadline) {
			break
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			d, err := fn()
			if err != nil {
				errCount.Add(1)
				return
			}
			mu.Lock()
			lats = append(lats, d)
			mu.Unlock()
		}()
	}
	wg.Wait()
	return lats, errCount.Load()
}

func pctl(d []time.Duration, p float64) time.Duration {
	if len(d) == 0 {
		return 0
	}
	s := append([]time.Duration(nil), d...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	idx := int(p * float64(len(s)-1))
	return s[idx]
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

func main() {
	flag.Parse()
	fmt.Fprintf(&rep, "# embedcache open-loop latency benchmark\n\n")
	fmt.Fprintf(&rep, "Generated %s · %s/%s · %d CPUs · real local Ollama backend, no synthetic timing.\n\n",
		time.Now().Format("2006-01-02 15:04 MST"), runtime.GOOS, runtime.GOARCH, runtime.NumCPU())
	fmt.Fprintf(&rep, "Constant-rate (open-loop) load: a ticker schedules request dispatch on a fixed\n")
	fmt.Fprintf(&rep, "schedule regardless of completion, so tail latency under saturation is captured\n")
	fmt.Fprintf(&rep, "honestly instead of being hidden by a closed-loop worker pool (coordinated omission).\n")

	for _, model := range strings.Split(*modelsArg, ",") {
		model = strings.TrimSpace(model)
		logf("== %s ==", model)
		p, err := startProxy(*ollamaURL)
		if err != nil {
			line("- **%s**: could not start proxy: %v", model, err)
			continue
		}

		// probe real dims directly from the backend
		var dims int
		if resp, err := client.Post(*ollamaURL+"/v1/embeddings", "application/json",
			bytes.NewReader([]byte(fmt.Sprintf(`{"model":%q,"input":"dim probe"}`, model)))); err == nil {
			var parsed struct {
				Data []struct {
					Embedding []float32 `json:"embedding"`
				} `json:"data"`
			}
			raw, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			if json.Unmarshal(raw, &parsed) == nil && len(parsed.Data) > 0 {
				dims = len(parsed.Data[0].Embedding)
			}
		}

		section("%s (%d-dim)", model, dims)

		// --- MISS scenario: every request is a unique input, real upstream compute ---
		var missCounter atomic.Int64
		missFn := func() (time.Duration, error) {
			n := missCounter.Add(1)
			return embedOnce(p.base, model, fmt.Sprintf("open-loop perf miss probe %d %d", time.Now().UnixNano(), n))
		}
		missRate := calibrate(missFn, 6, 3*time.Second)
		targetMiss := missRate * 0.7
		logf("  miss: calibrated %.1f req/s sustainable, running open-loop at %.1f req/s", missRate, targetMiss)
		missLats, missErrs := openLoopLoad(missFn, targetMiss, *duration)

		// --- HIT scenario: same single warmed input, repeated ---
		warmInput := fmt.Sprintf("open-loop perf hit probe %d", time.Now().UnixNano())
		embedOnce(p.base, model, warmInput) // warm it
		hitFn := func() (time.Duration, error) { return embedOnce(p.base, model, warmInput) }
		hitRate := calibrate(hitFn, 16, 3*time.Second)
		targetHit := hitRate * 0.7
		if targetHit > 4000 {
			targetHit = 4000 // keep goroutine fan-out bounded on modest hardware
		}
		logf("  hit: calibrated %.1f req/s sustainable, running open-loop at %.1f req/s", hitRate, targetHit)
		hitLats, hitErrs := openLoopLoad(hitFn, targetHit, *duration)

		// --- DIRECT scenario: bypass the proxy entirely, same miss shape, for overhead comparison ---
		var directCounter atomic.Int64
		directFn := func() (time.Duration, error) {
			n := directCounter.Add(1)
			return embedOnce(*ollamaURL, model, fmt.Sprintf("open-loop perf direct probe %d %d", time.Now().UnixNano(), n))
		}
		directRate := calibrate(directFn, 6, 3*time.Second)
		targetDirect := directRate * 0.7
		logf("  direct: calibrated %.1f req/s sustainable, running open-loop at %.1f req/s", directRate, targetDirect)
		directLats, directErrs := openLoopLoad(directFn, targetDirect, *duration)

		p.stop()

		line("| scenario | target rate | completed | errors | p50 | p90 | p99 | p99.9 |")
		line("|---|---|---|---|---|---|---|---|")
		line("| miss (real compute, unique input) | %.0f/s | %d | %d | %.1fms | %.1fms | %.1fms | %.1fms |",
			targetMiss, len(missLats), missErrs, ms(pctl(missLats, .5)), ms(pctl(missLats, .9)), ms(pctl(missLats, .99)), ms(pctl(missLats, .999)))
		line("| hit (cached, repeated input) | %.0f/s | %d | %d | %.1fms | %.1fms | %.1fms | %.1fms |",
			targetHit, len(hitLats), hitErrs, ms(pctl(hitLats, .5)), ms(pctl(hitLats, .9)), ms(pctl(hitLats, .99)), ms(pctl(hitLats, .999)))
		line("| direct to backend (no proxy) | %.0f/s | %d | %d | %.1fms | %.1fms | %.1fms | %.1fms |",
			targetDirect, len(directLats), directErrs, ms(pctl(directLats, .5)), ms(pctl(directLats, .9)), ms(pctl(directLats, .99)), ms(pctl(directLats, .999)))
		if len(missLats) > 0 && len(directLats) > 0 {
			overhead := ms(pctl(missLats, .5)) - ms(pctl(directLats, .5))
			line("")
			line("- proxy overhead at p50 (miss vs direct, both real upstream compute): %.2fms", overhead)
		}
	}

	section("Method")
	line("- All traffic is open-loop (constant-rate via ticker), not closed-loop worker-pool polling —")
	line("  this avoids coordinated omission, which understates tail latency under saturation.")
	line("- Per-scenario rate is calibrated from real measured closed-loop throughput (70%% of sustained")
	line("  capacity), not a guessed constant, so the target rate is realistic for the actual hardware.")
	line("- Only local Ollama backends are covered here (no hosted API key in this environment); the")
	line("  proxy-overhead number isolates embedcache's own cost from backend inference time.")

	if err := os.WriteFile(*outPath, rep.Bytes(), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("\n%s written\n", *outPath)
}
