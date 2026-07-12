// sharedcache proves the distributed cache tier for real: two separate
// embedcache processes, each with its own in-memory cache, both pointed at one
// Redis, sitting in front of one backend. A vector computed by instance A is
// served by instance B without B ever calling the backend — the property that
// keeps the measured savings from eroding the moment you run more than one
// replica behind a load balancer.
//
// The backend here is the in-repo mock (internal/mockllm) so the test needs no
// GPU and can count upstream calls exactly; every cache decision is made by the
// real proxy binary against real Redis.
//
// Usage:
//
//	go build -o embedcache.exe ./cmd/embedcache
//	go run ./experiments/sharedcache -bin ./embedcache.exe -redis 127.0.0.1:6379 -out SHAREDCACHE.md
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

	"github.com/Ajay6601/embedcache/internal/mockllm"
)

var (
	binPath = flag.String("bin", "./embedcache.exe", "compiled embedcache binary")
	redis   = flag.String("redis", "127.0.0.1:6379", "Redis host:port for the shared tier")
	outPath = flag.String("out", "SHAREDCACHE.md", "results file")
)

var (
	rep    bytes.Buffer
	failed int
	client = &http.Client{Timeout: 30 * time.Second}
)

func check(name string, ok bool, detail string) {
	mark := "PASS"
	if !ok {
		mark = "FAIL"
		failed++
	}
	fmt.Fprintf(&rep, "- **%s** — %s: %s\n", mark, name, detail)
	fmt.Printf("[%s] %s: %s\n", mark, name, detail)
}

func line(f string, a ...any) { fmt.Fprintf(&rep, f+"\n", a...) }

func freePort() string {
	l, _ := net.Listen("tcp", "127.0.0.1:0")
	a := l.Addr().String()
	l.Close()
	return a
}

func startInstance(upstream, prefix string) (base string, stop func(), err error) {
	addr := freePort()
	cmd := exec.Command(*binPath, "serve", "-listen", addr, "-upstream", upstream,
		"-shared-redis", *redis, "-shared-redis-prefix", prefix)
	cmd.Stdout, cmd.Stderr = io.Discard, io.Discard
	if err = cmd.Start(); err != nil {
		return "", nil, err
	}
	base = "http://" + addr
	for i := 0; i < 100; i++ {
		if r, e := client.Get(base + "/healthz"); e == nil {
			r.Body.Close()
			return base, func() { cmd.Process.Kill(); cmd.Wait() }, nil
		}
		time.Sleep(80 * time.Millisecond)
	}
	cmd.Process.Kill()
	return "", nil, fmt.Errorf("instance never healthy")
}

func embed(base, input string) (status string, raw []byte, err error) {
	body, _ := json.Marshal(map[string]any{"model": "shared-demo", "input": input})
	req, _ := http.NewRequest(http.MethodPost, base+"/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return "", nil, err
	}
	defer resp.Body.Close()
	raw, _ = io.ReadAll(resp.Body)
	return resp.Header.Get("X-Embedcache-Status"), raw, nil
}

// embeddingBytes returns the raw bytes of data[0].embedding, the part the cache
// guarantees identical (the usage envelope legitimately differs per request).
func embeddingBytes(raw []byte) ([]byte, bool) {
	var parsed struct {
		Data []struct {
			Embedding json.RawMessage `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil || len(parsed.Data) == 0 {
		return nil, false
	}
	return parsed.Data[0].Embedding, true
}

func main() {
	flag.Parse()

	// one shared backend that counts exactly how many times it is actually called
	mock := mockllm.New(256)
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	go http.Serve(ln, mock.Handler())
	upstream := "http://" + ln.Addr().String()

	// a unique prefix so reruns don't collide in Redis
	prefix := fmt.Sprintf("ecshared:%d:", time.Now().UnixNano())

	fmt.Fprintf(&rep, "# embedcache shared cache: cross-instance reuse (real Redis)\n\n")
	fmt.Fprintf(&rep, "Generated %s. Two separate embedcache processes, each with its own in-memory\n", time.Now().Format("2006-01-02 15:04 MST"))
	fmt.Fprintf(&rep, "cache, both pointed at one Redis (`%s`), in front of one backend that counts its\n", *redis)
	fmt.Fprintf(&rep, "own calls. This is what keeps the savings from eroding across a multi-replica fleet.\n\n")

	instA, stopA, err := startInstance(upstream, prefix)
	if err != nil {
		fmt.Fprintln(os.Stderr, "instance A:", err)
		line("- **FAIL** — could not start instance A (is Redis running at %s?): %v", *redis, err)
		os.WriteFile(*outPath, rep.Bytes(), 0o644)
		os.Exit(1)
	}
	defer stopA()
	instB, stopB, err := startInstance(upstream, prefix)
	if err != nil {
		fmt.Fprintln(os.Stderr, "instance B:", err)
		os.Exit(1)
	}
	defer stopB()

	input := fmt.Sprintf("a real sentence embedded once by instance A at %d", time.Now().UnixNano())

	// 1. instance A embeds it: a miss, one real upstream call
	stA, rawA, err := embed(instA, input)
	if err != nil {
		check("instance A first request", false, err.Error())
		finish()
		return
	}
	callsAfterA := mock.Calls()
	check("instance A computes a new input (miss)", stA == "miss" && callsAfterA == 1,
		fmt.Sprintf("status=%s, backend calls so far=%d", stA, callsAfterA))

	// give the async write-through a moment to reach Redis
	time.Sleep(300 * time.Millisecond)

	// 2. instance B embeds the SAME input: its local cache is cold, but the
	//    shared tier has it, so it must serve a hit WITHOUT a new backend call
	stB, rawB, err := embed(instB, input)
	if err != nil {
		check("instance B second request", false, err.Error())
		finish()
		return
	}
	callsAfterB := mock.Calls()
	check("instance B reuses instance A's vector from the shared tier (hit, no new backend call)",
		stB == "hit" && callsAfterB == 1,
		fmt.Sprintf("status=%s, backend calls total=%d (still 1 = B did not recompute)", stB, callsAfterB))

	// 3. the VECTOR B served is byte-identical to what A produced. (The full
	//    envelopes differ by design: usage.prompt_tokens reflects what THIS
	//    request was billed — real tokens on A's miss, 0 on B's hit — so only the
	//    embedding itself is guaranteed identical, which is exactly the point.)
	embA, okA := embeddingBytes(rawA)
	embB, okB := embeddingBytes(rawB)
	check("the shared vector is byte-identical across instances",
		okA && okB && bytes.Equal(embA, embB),
		fmt.Sprintf("A embedding %d bytes, B embedding %d bytes, equal=%v", len(embA), len(embB), bytes.Equal(embA, embB)))

	// 4. a brand-new input on B is a genuine miss (one more real backend call),
	//    confirming the shared tier isn't just answering everything blindly
	other := fmt.Sprintf("a different sentence only instance B sees at %d", time.Now().UnixNano())
	stB2, _, _ := embed(instB, other)
	callsAfterOther := mock.Calls()
	check("a genuinely new input still reaches the backend",
		stB2 == "miss" && callsAfterOther == 2,
		fmt.Sprintf("status=%s, backend calls total=%d", stB2, callsAfterOther))

	line("")
	line("Net: across two independent instances, the backend computed each distinct input **once**.")
	line("A second replica does not restart cold; it inherits the whole fleet's cache through Redis.")
	finish()
}

func finish() {
	if err := os.WriteFile(*outPath, rep.Bytes(), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if failed > 0 {
		fmt.Fprintf(os.Stderr, "\n%d assertion(s) FAILED — see %s\n", failed, *outPath)
		os.Exit(1)
	}
	fmt.Printf("\nshared-cache cross-instance reuse verified — %s written\n", *outPath)
}
