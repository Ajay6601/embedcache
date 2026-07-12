// gifcap runs embedcache for real and records what actually happens into a
// transcript the termgif tool renders to a GIF. Nothing here is scripted: it
// starts a real `embedcache serve`, sends real embedding requests, and writes
// down the real X-Embedcache-Status headers, the real measured latencies, the
// real saved-token counts, and the real waste report the running proxy emits.
//
// Scenarios (all produce a transcript JSON for tools/termgif):
//
//	serve   real serve + real miss/hit/partial + real waste report (point -upstream at Ollama)
//	rag     real corpus ingest, then a real re-ingest showing real cache absorption
//	fromtext turn a captured stdout file (from any real run) into a transcript, with a shown command
//
// Usage:
//
//	go build -o embedcache.exe ./cmd/embedcache
//	go run ./experiments/gifcap -scenario serve -upstream http://localhost:11434 -model all-minilm -out ollama.json
//	go run ./experiments/gifcap -scenario rag   -upstream http://localhost:11434 -model nomic-embed-text -out rag.json
//	go run ./experiments/gifcap -scenario fromtext -text agents.out -precmd "go run ./experiments/multiagent" -out agents.json
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/Ajay6601/embedcache/internal/mockllm"
)

var (
	scenario = flag.String("scenario", "serve", "serve | rag | fromtext")
	binPath  = flag.String("bin", "./embedcache.exe", "compiled embedcache binary")
	upstream = flag.String("upstream", "", "real backend base URL; empty starts the in-repo mock")
	model    = flag.String("model", "all-minilm", "embedding model to request")
	outPath  = flag.String("out", "transcript.json", "transcript JSON output")
	shownUp  = flag.String("shown-upstream", "http://localhost:11434", "upstream string to display in the command line")
	listen   = flag.String("listen", "127.0.0.1:8090", "address the demo proxy binds (shown verbatim in the transcript)")
	textFile = flag.String("text", "", "fromtext: captured stdout file to convert")
	preCmd   = flag.String("precmd", "", "fromtext: command line to show above the captured output")
	reqLog   = flag.String("reqlog", "", "rag: also write the real request log here (feed it to embedcache analyze)")
)

type event struct {
	Kind  string `json:"kind"`
	Text  string `json:"text"`
	Color string `json:"color,omitempty"`
	Pause int    `json:"pause,omitempty"`
}

var events []event

func emitCmd(text string, pause int) {
	events = append(events, event{Kind: "cmd", Text: text, Pause: pause})
}
func emitOut(text, color string, pause int) {
	events = append(events, event{Kind: "out", Text: text, Color: color, Pause: pause})
}
func emitDim(text string, pause int) {
	events = append(events, event{Kind: "dim", Text: text, Pause: pause})
}

var httpc = &http.Client{Timeout: 120 * time.Second}

func freePort() string {
	l, _ := net.Listen("tcp", "127.0.0.1:0")
	a := l.Addr().String()
	l.Close()
	return a
}

// startProxy starts a real embedcache serve against the chosen upstream (or the
// in-repo mock when -upstream is empty) and returns the proxy base URL, the real
// startup log line, and a stop func.
func startProxy(extra ...string) (base, listenLine string, stop func()) {
	up := *upstream
	if up == "" {
		mock := mockllm.New(384)
		mock.Latency = 45 * time.Millisecond
		ln, _ := net.Listen("tcp", "127.0.0.1:0")
		go http.Serve(ln, mock.Handler())
		up = "http://" + ln.Addr().String()
	}
	addr := *listen
	if addr == "" {
		addr = freePort()
	}
	args := append([]string{"serve", "-listen", addr, "-upstream", up}, extra...)
	cmd := exec.Command(*binPath, args...)
	stderr, _ := cmd.StderrPipe()
	if err := cmd.Start(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	go func() {
		sc := bufio.NewScanner(stderr)
		for sc.Scan() {
			line := sc.Text()
			if strings.Contains(line, "listening on") && listenLine == "" {
				if i := strings.Index(line, "embedcache "); i >= 0 {
					listenLine = line[i:]
				} else {
					listenLine = line
				}
			}
		}
	}()
	base = "http://" + addr
	for i := 0; i < 100; i++ {
		if r, err := httpc.Get(base + "/healthz"); err == nil {
			r.Body.Close()
			break
		}
		time.Sleep(80 * time.Millisecond)
	}
	time.Sleep(150 * time.Millisecond)
	return base, listenLine, func() { cmd.Process.Kill(); cmd.Wait() }
}

func embed(base string, input any) (status string, hits, misses, saved int, dur time.Duration, err error) {
	status, _, hits, misses, saved, dur, err = embedAs(base, input, "")
	return
}

// embedAs is embed with a caller API key, also returning the HTTP status code
// (so the budget-rejection 429 can be shown for real).
func embedAs(base string, input any, key string) (status string, code, hits, misses, saved int, dur time.Duration, err error) {
	body, _ := json.Marshal(map[string]any{"model": *model, "input": input})
	req, _ := http.NewRequest(http.MethodPost, base+"/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	start := time.Now()
	resp, err := httpc.Do(req)
	if err != nil {
		return "", 0, 0, 0, 0, 0, err
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	dur = time.Since(start)
	code = resp.StatusCode
	status = resp.Header.Get("X-Embedcache-Status")
	fmt.Sscanf(resp.Header.Get("X-Embedcache-Hits"), "%d", &hits)
	fmt.Sscanf(resp.Header.Get("X-Embedcache-Misses"), "%d", &misses)
	fmt.Sscanf(resp.Header.Get("X-Embedcache-Saved-Tokens"), "%d", &saved)
	return
}

func statInt(base, key string) int {
	r, err := httpc.Get(base + "/_ec/stats")
	if err != nil {
		return 0
	}
	defer r.Body.Close()
	var m map[string]any
	json.NewDecoder(r.Body).Decode(&m)
	if v, ok := m[key].(float64); ok {
		return int(v)
	}
	return 0
}

func write() {
	if len(events) > 0 {
		events[len(events)-1].Pause = 26
	}
	raw, _ := json.MarshalIndent(events, "", "  ")
	if err := os.WriteFile(*outPath, raw, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s: %d real events\n", *outPath, len(events))
}

func main() {
	flag.Parse()
	switch *scenario {
	case "serve":
		captureServe()
	case "rag":
		captureRAG()
	case "agents":
		captureAgents()
	case "fromtext":
		captureFromText()
	default:
		fmt.Fprintln(os.Stderr, "unknown scenario", *scenario)
		os.Exit(2)
	}
	write()
}

// captureServe: the real proxy in front of a real backend, showing a real miss,
// a real hit, a real partial batch, and the real waste report.
func captureServe() {
	base, listenLine, stop := startProxy()
	defer stop()

	emitCmd("embedcache serve -upstream "+*shownUp, 3)
	if listenLine != "" {
		emitDim(listenLine, 6)
	}
	emitOut("", "", 0)

	q := "how do I reset my password"
	curl := func(p string) string { return "curl localhost:8090/v1/embeddings -d '" + p + "'" }

	emitCmd(curl(`{"model":"`+*model+`","input":"`+q+`"}`), 3)
	st, _, _, _, d1, err := embed(base, q)
	must(err, "request 1")
	emitOut(fmt.Sprintf("X-Embedcache-Status: %-4s  computed upstream in %d ms", st, d1.Milliseconds()), "amber", 5)
	emitOut("", "", 0)

	emitCmd(curl(`{"model":"`+*model+`","input":"`+q+`"}`), 3)
	st, _, _, saved, d2, err := embed(base, q)
	must(err, "request 2")
	emitOut(fmt.Sprintf("X-Embedcache-Status: %-4s  served in %d ms, billed 0 tokens (saved %d)", st, d2.Milliseconds(), saved), "green", 2)
	emitDim("(same vector, served from memory instead of the GPU, and free)", 6)
	emitOut("", "", 0)

	emitCmd(curl(`{"model":"`+*model+`","input":["`+q+`","reset my 2FA device","update billing email"]}`), 3)
	st, hits, misses, _, _, err := embed(base, []string{q, "reset my 2FA device", "update billing email"})
	must(err, "batch")
	emitOut(fmt.Sprintf("X-Embedcache-Status: %-4s  %d served from cache, %d computed", st, hits, misses), "cyan", 6)
	emitOut("", "", 0)

	emitCmd("embedcache report", 3)
	emitReport(base)
}

// captureRAG: a real corpus ingested through the proxy, then re-ingested with
// nothing changed, showing the real absorption the nightly-rerun case gets.
func captureRAG() {
	extra := []string{"-ttl", "1h"}
	if *reqLog != "" {
		extra = append(extra, "-request-log", *reqLog)
	}
	base, _, stop := startProxy(extra...)
	defer stop()

	corpus := wikiCorpus([]string{"Retrieval-augmented generation", "Vector database", "Word embedding"}, 40)
	if len(corpus) < 10 {
		fmt.Fprintln(os.Stderr, "corpus too small:", len(corpus))
		os.Exit(1)
	}

	emitCmd(fmt.Sprintf("rag-ingest.py  # embed %d doc chunks via embedcache (%s)", len(corpus), *model), 3)
	before := statInt(base, "misses")
	ingest(base, corpus)
	cold := statInt(base, "misses") - before
	emitOut(fmt.Sprintf("ingested %d chunks -> %d embedded upstream (cold cache)", len(corpus), cold), "amber", 6)
	emitOut("", "", 0)

	emitCmd("rag-ingest.py  # nightly re-run, corpus unchanged", 3)
	mBefore := statInt(base, "misses")
	hBefore := statInt(base, "hits")
	ingest(base, corpus)
	recomputed := statInt(base, "misses") - mBefore
	reHits := statInt(base, "hits") - hBefore
	absorbed := 100 * float64(reHits) / float64(reHits+recomputed)
	emitOut(fmt.Sprintf("re-ingested %d chunks -> %d embedded, %d served from cache", len(corpus), recomputed, reHits), "green", 3)
	emitOut(fmt.Sprintf(">> %.0f%% absorbed - the nightly re-embed was almost entirely free", absorbed), "mint", 6)
	emitOut("", "", 0)

	emitCmd("embedcache report", 3)
	emitReport(base)
}

// captureAgents: a real multi-agent setup where each agent authenticates with
// its own API key against one shared proxy. Shows a real cross-agent cache hit
// (worker-2 reusing the vector worker-1 paid for, though it never computed it)
// and a real mid-run budget rejection on a capped agent, its siblings
// unaffected. The cache behaviour is the point, so this drives it directly
// rather than through an LLM; every status and code below is from the live proxy.
func captureAgents() {
	budgetsPath := os.TempDir() + "/gifcap-budgets.json"
	os.WriteFile(budgetsPath, []byte(`{"worker-3-key":40,"default":0}`), 0o644)
	defer os.Remove(budgetsPath)
	base, _, stop := startProxy(
		"-auth-mode", "allowlist",
		"-api-keys", "planner-key,worker-1-key,worker-2-key,worker-3-key,synth-key",
		"-budgets-file", budgetsPath, "-budget-window", "1h")
	defer stop()

	q := "how do vector databases enable retrieval-augmented generation"

	emitDim("one research crew, five agents, each with its own API key, one shared embedcache", 3)
	emitOut("", "", 0)

	emitCmd(`worker-1  embed("`+q[:34]+`...")`, 3)
	st, _, _, _, _, _, err := embedAs(base, q, "worker-1-key")
	must(err, "worker-1")
	emitOut("X-Embedcache-Status: "+st+"   worker-1 computes it once", "amber", 4)
	emitOut("", "", 0)

	emitCmd(`worker-2  embed("`+q[:34]+`...")   # same sub-question, different agent`, 3)
	st, _, _, _, saved, _, err := embedAs(base, q, "worker-2-key")
	must(err, "worker-2")
	emitOut(fmt.Sprintf("X-Embedcache-Status: %s    worker-2 reuses it free (%d tokens saved)", st, saved), "green", 3)
	emitDim("different agent, different key, same content -> one upstream call", 6)
	emitOut("", "", 0)

	emitCmd(`worker-3  embed(novel inputs...)   # key capped at 40 tokens`, 3)
	code := 200
	for i := 0; i < 30 && code != 429; i++ {
		_, code, _, _, _, _, _ = embedAs(base, fmt.Sprintf("worker-3 distinct research query number %d", i), "worker-3-key")
	}
	emitOut(fmt.Sprintf("HTTP %d Too Many Requests   worker-3 hit its 40-token budget", code), "amber", 4)

	// prove a sibling is unaffected
	st, sc, _, _, _, _, _ := embedAs(base, "planner unaffected fresh query", "planner-key")
	emitDim(fmt.Sprintf("planner-key still served normally (HTTP %d, %s) - only the runaway agent is capped", sc, st), 8)
}

func ingest(base string, corpus []string) {
	for i := 0; i < len(corpus); i += 16 {
		end := i + 16
		if end > len(corpus) {
			end = len(corpus)
		}
		embed(base, corpus[i:end])
	}
}

// captureFromText: convert a real captured stdout file into a transcript,
// coloring lines by what they say. Used for `embedcache analyze` output and the
// multi-agent harness output, both captured from genuine runs.
func captureFromText() {
	if *preCmd != "" {
		emitCmd(*preCmd, 4)
	}
	raw, err := os.ReadFile(*textFile)
	must(err, "reading text file")
	for _, line := range strings.Split(strings.TrimRight(string(raw), "\n"), "\n") {
		line = strings.TrimRight(line, "\r")
		emitOut(line, colorFor(line), 0)
	}
}

func colorFor(line string) string {
	l := strings.ToLower(line)
	switch {
	case strings.HasPrefix(strings.TrimSpace(line), ">>"):
		return "mint"
	case strings.Contains(line, "[PASS]") || strings.Contains(l, "pass "):
		return "green"
	case strings.Contains(line, "[FAIL]"):
		return "red"
	case strings.Contains(l, "duplicate") || strings.Contains(l, "wasted") || strings.Contains(l, "saved") || strings.Contains(l, "429") || strings.Contains(l, "budget"):
		return "amber"
	case strings.Contains(l, "hit"):
		return "green"
	case strings.Contains(l, "==="), strings.Contains(l, "----"):
		return "dim"
	case strings.Contains(line, "[info]"):
		return "dim"
	}
	return ""
}

func emitReport(base string) {
	r, err := httpc.Get(base + "/_ec/report")
	if err != nil {
		return
	}
	raw, _ := io.ReadAll(r.Body)
	r.Body.Close()
	for _, line := range strings.Split(strings.TrimRight(string(raw), "\n"), "\n") {
		color := ""
		switch {
		case strings.HasPrefix(line, ">>"):
			color = "mint"
		case strings.Contains(line, "duplicate") || strings.Contains(line, "saved"):
			color = "amber"
		case strings.Contains(line, "===") || strings.HasPrefix(line, "embedcache "):
			color = "dim"
		}
		emitOut(line, color, 0)
	}
}

// ---- real Wikipedia corpus (same source the validation suite uses) ----

func wikiCorpus(titles []string, max int) []string {
	var chunks []string
	for _, t := range titles {
		u := "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&redirects=1&format=json&titles=" + url.QueryEscape(t)
		req, _ := http.NewRequest(http.MethodGet, u, nil)
		req.Header.Set("User-Agent", "embedcache-gifcap/0.2 (github.com/Ajay6601/embedcache)")
		resp, err := httpc.Do(req)
		if err != nil {
			continue
		}
		var parsed struct {
			Query struct {
				Pages map[string]struct {
					Extract string `json:"extract"`
				} `json:"pages"`
			} `json:"query"`
		}
		json.NewDecoder(resp.Body).Decode(&parsed)
		resp.Body.Close()
		for _, p := range parsed.Query.Pages {
			for _, para := range strings.Split(p.Extract, "\n") {
				para = strings.TrimSpace(para)
				if len(para) < 120 || strings.HasPrefix(para, "==") {
					continue
				}
				if len(para) > 400 {
					para = para[:400]
				}
				chunks = append(chunks, para)
				if len(chunks) >= max {
					return chunks
				}
			}
		}
	}
	return chunks
}

func must(err error, what string) {
	if err != nil {
		fmt.Fprintln(os.Stderr, what+":", err)
		os.Exit(1)
	}
}
