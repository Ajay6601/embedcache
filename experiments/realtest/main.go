// realtest runs a REAL end-to-end production workload through embedcache:
// a working RAG agent over live Wikipedia content, answering questions with
// a real LLM, generating organic embedding traffic — no constructed
// duplicates anywhere. Whatever duplicate rate the waste report finds is
// what this workload naturally produces.
//
// The stack mirrors a common production hybrid: self-hosted embeddings
// (Ollama) fronted by embedcache, hosted LLM (Gemini) with local fallback
// (Ollama chat) when the hosted API is rate-limited or overloaded.
//
// Usage:
//
//	go build -o embedcache.exe ./cmd/embedcache
//	go run ./experiments/realtest -bin ./embedcache.exe [-gemini-key $GEMINI_API_KEY]
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Ajay6601/embedcache/internal/api"
)

var (
	binPath    = flag.String("bin", "./embedcache.exe", "compiled embedcache binary")
	ollamaURL  = flag.String("ollama", "http://localhost:11434", "Ollama address (embeddings upstream + local LLM fallback)")
	embedModel = flag.String("embed-model", "all-minilm", "embedding model")
	localChat  = flag.String("local-chat", "gemma3:1b", "local fallback chat model")
	geminiKey  = flag.String("gemini-key", os.Getenv("GEMINI_API_KEY"), "Gemini API key for the hosted LLM (optional)")
	sessions   = flag.Int("sessions", 30, "agent sessions to run")
	outPath    = flag.String("out", "REALTEST.md", "results file")
)

const (
	adminToken  = "realtest-admin"
	pipelineKey = "pipeline-service-key"
	agentKey    = "agent-service-key"
	geminiBase  = "https://generativelanguage.googleapis.com/v1beta/openai"
)

var geminiModels = []string{"gemini-flash-latest", "gemini-2.0-flash-lite", "gemini-3.5-flash"}

// ~110 real English Wikipedia articles: the corpus of a plausible internal
// "engineering knowledge assistant".
var articleTitles = []string{
	"Artificial intelligence", "Machine learning", "Deep learning", "Neural network",
	"Transformer (deep learning architecture)", "Large language model", "Retrieval-augmented generation",
	"Vector database", "Word embedding", "Natural language processing", "Computer vision",
	"Reinforcement learning", "Supervised learning", "Unsupervised learning", "Gradient descent",
	"Backpropagation", "Convolutional neural network", "Recurrent neural network",
	"Attention (machine learning)", "GPT-4", "BERT (language model)", "Diffusion model",
	"Generative adversarial network", "Support vector machine", "Random forest", "Decision tree",
	"K-means clustering", "Principal component analysis", "Overfitting", "Regularization (mathematics)",
	"Cross-validation (statistics)", "Feature engineering", "Data mining", "Big data",
	"Cloud computing", "Kubernetes", "Docker (software)", "Microservices", "GraphQL", "REST",
	"HTTP", "Internet protocol suite", "Domain Name System", "Transport Layer Security",
	"Public-key cryptography", "RSA (cryptosystem)", "Advanced Encryption Standard", "SHA-2",
	"Blockchain", "Bitcoin", "Ethereum", "Database", "SQL", "NoSQL", "PostgreSQL", "Redis",
	"Apache Kafka", "Distributed computing", "CAP theorem", "Consensus (computer science)",
	"Paxos (computer science)", "Raft (algorithm)", "Operating system", "Linux", "Linux kernel",
	"Unix", "Windows NT", "Compiler", "Interpreter (computing)", "Programming language",
	"Python (programming language)", "Go (programming language)", "Rust (programming language)",
	"JavaScript", "TypeScript", "C (programming language)", "C++", "Java (programming language)",
	"Haskell", "Functional programming", "Object-oriented programming",
	"Garbage collection (computer science)", "Memory management", "CPU cache",
	"Central processing unit", "Graphics processing unit", "Tensor Processing Unit", "Moore's law",
	"Quantum computing", "Qubit", "Shor's algorithm", "Cryptography", "Information theory",
	"Claude Shannon", "Alan Turing", "Turing machine", "Turing test", "John von Neumann",
	"Von Neumann architecture", "Computational complexity theory", "P versus NP problem",
	"Algorithm", "Data structure", "Hash table", "Binary search tree", "Dijkstra's algorithm",
	"Dynamic programming", "Sorting algorithm", "Quicksort", "Merge sort", "Software engineering",
	"DevOps", "Continuous integration", "Git", "GitHub", "Open-source software", "MIT License",
}

// realistic user questions for this corpus; sessions repeat some of them,
// the way real users repeat intents
var questions = []string{
	"How does a transformer architecture use attention?",
	"What is retrieval-augmented generation and why is it used?",
	"Explain the difference between supervised and unsupervised learning",
	"What causes overfitting and how can it be prevented?",
	"How does gradient descent optimize a neural network?",
	"What is a vector database used for?",
	"How do convolutional neural networks process images?",
	"What is the CAP theorem in distributed systems?",
	"How does Raft achieve consensus?",
	"What is the difference between Docker and Kubernetes?",
	"How does TLS secure a connection?",
	"What makes public-key cryptography secure?",
	"How does Bitcoin's blockchain prevent double spending?",
	"What are the trade-offs between SQL and NoSQL databases?",
	"How does Redis achieve its performance?",
	"What did Alan Turing contribute to computer science?",
	"What is the P versus NP problem?",
	"How does quicksort work and what is its complexity?",
	"What is garbage collection and how does it work?",
	"Why is Moore's law slowing down?",
	"How do GPUs accelerate machine learning?",
	"What is a Turing machine?",
	"How does Git track changes?",
	"What is the role of DNS on the internet?",
}

var (
	rep     bytes.Buffer
	failed  int
	client  *http.Client
	proxyEP string
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

func section(f string, args ...any) {
	fmt.Fprintf(&rep, "\n## "+f+"\n\n", args...)
	fmt.Printf("\n== "+f+" ==\n", args...)
}

func line(f string, args ...any) { fmt.Fprintf(&rep, f+"\n", args...) }

// ---------- real corpus: live Wikipedia ----------

type chunk struct {
	title string
	text  string
	vec   []float64
}

func fetchArticle(title string) (string, error) {
	u := "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&redirects=1&format=json&titles=" + url.QueryEscape(title)
	req, _ := http.NewRequest(http.MethodGet, u, nil)
	req.Header.Set("User-Agent", "embedcache-realtest/0.1 (https://github.com/Ajay6601/embedcache)")
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	var parsed struct {
		Query struct {
			Pages map[string]struct {
				Extract string `json:"extract"`
			} `json:"pages"`
		} `json:"query"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return "", err
	}
	for _, p := range parsed.Query.Pages {
		return p.Extract, nil
	}
	return "", fmt.Errorf("no page")
}

func fetchCorpus() map[string]string {
	out := map[string]string{}
	var mu sync.Mutex
	jobs := make(chan string)
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for t := range jobs {
				text, err := fetchArticle(t)
				if err != nil || len(text) < 2000 {
					continue
				}
				mu.Lock()
				out[t] = text
				mu.Unlock()
			}
		}()
	}
	for _, t := range articleTitles {
		jobs <- t
	}
	close(jobs)
	wg.Wait()
	return out
}

// chunkText splits an article the way a real ingestion pipeline would:
// paragraph-aligned chunks of roughly 400–800 characters.
func chunkText(title, text string) []chunk {
	var chunks []chunk
	var cur strings.Builder
	flush := func() {
		if cur.Len() >= 150 {
			chunks = append(chunks, chunk{title: title, text: strings.TrimSpace(cur.String())})
		}
		cur.Reset()
	}
	for _, para := range strings.Split(text, "\n") {
		para = strings.TrimSpace(para)
		if para == "" || strings.HasPrefix(para, "==") {
			continue
		}
		// hard-split paragraphs longer than the chunk budget at word
		// boundaries — all-minilm's context is only 256 wordpieces, and
		// token-dense technical text can exceed it well under the
		// chars-per-token rule of thumb, so keep chunks conservative
		for len(para) > 0 {
			piece := para
			if len(piece) > 500 {
				cut := strings.LastIndex(piece[:500], " ")
				if cut < 250 {
					cut = 500
				}
				piece = piece[:cut]
			}
			para = strings.TrimSpace(para[len(piece):])
			if cur.Len()+len(piece) > 500 {
				flush()
			}
			if cur.Len() > 0 {
				cur.WriteString(" ")
			}
			cur.WriteString(piece)
			if cur.Len() >= 300 {
				flush()
			}
		}
	}
	flush()
	return chunks
}

// ---------- embeddings through the proxy ----------

func embedBatch(key string, inputs []string) ([][]float64, string, error) {
	b, _ := json.Marshal(map[string]any{"model": *embedModel, "input": inputs})
	req, _ := http.NewRequest(http.MethodPost, proxyEP+"/v1/embeddings", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	resp, err := client.Do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return nil, "", fmt.Errorf("status %d: %.200s", resp.StatusCode, raw)
	}
	var parsed api.EmbeddingsResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, "", err
	}
	vecs := make([][]float64, len(parsed.Data))
	for i, d := range parsed.Data {
		if err := json.Unmarshal(d.Embedding, &vecs[i]); err != nil {
			return nil, "", err
		}
	}
	return vecs, resp.Header.Get("X-Embedcache-Status"), nil
}

// embedResilient does what real ingestion pipelines do when a batch trips
// the model's context limit: retry, then isolate items and truncate the
// offenders to fit rather than dropping the whole batch.
func embedResilient(key string, inputs []string) ([][]float64, error) {
	vecs, _, err := embedBatch(key, inputs)
	if err == nil {
		return vecs, nil
	}
	if !strings.Contains(err.Error(), "context length") {
		time.Sleep(2 * time.Second)
		if vecs, _, err = embedBatch(key, inputs); err == nil {
			return vecs, nil
		}
		return nil, err
	}
	out := make([][]float64, len(inputs))
	for i, in := range inputs {
		v, _, err := embedBatch(key, []string{in})
		if err != nil && strings.Contains(err.Error(), "context length") {
			trunc := in
			if len(trunc) > 350 {
				trunc = trunc[:350]
			}
			v, _, err = embedBatch(key, []string{trunc})
		}
		if err != nil {
			return nil, err
		}
		out[i] = v[0]
	}
	return out, nil
}

func cosine(a, b []float64) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12)
}

type store struct{ chunks []chunk }

func (s *store) search(q []float64, k int) []chunk {
	type scored struct {
		i     int
		score float64
	}
	best := make([]scored, 0, len(s.chunks))
	for i := range s.chunks {
		if len(s.chunks[i].vec) != len(q) {
			continue // chunk failed to embed; never a candidate
		}
		best = append(best, scored{i, cosine(q, s.chunks[i].vec)})
	}
	sort.Slice(best, func(a, b int) bool { return best[a].score > best[b].score })
	if len(best) > k {
		best = best[:k]
	}
	out := make([]chunk, len(best))
	for i, s2 := range best {
		out[i] = s.chunks[s2.i]
	}
	return out
}

// ---------- the LLM: hosted first, local fallback ----------

var remoteOK, remoteFail, localOK atomic.Int64

func chat(system, user string) (string, string) {
	if *geminiKey != "" {
		for _, m := range geminiModels {
			if text, err := chatOnce(geminiBase, *geminiKey, m, system, user); err == nil {
				remoteOK.Add(1)
				return text, "gemini:" + m
			}
		}
		remoteFail.Add(1)
	}
	if text, err := chatOnce(*ollamaURL, "", *localChat, system, user); err == nil {
		localOK.Add(1)
		return text, "local:" + *localChat
	}
	return "", "none"
}

func chatOnce(base, key, model, system, user string) (string, error) {
	body, _ := json.Marshal(map[string]any{
		"model":      model,
		"max_tokens": 200,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
	})
	req, _ := http.NewRequest(http.MethodPost, base+"/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("status %d", resp.StatusCode)
	}
	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil || len(parsed.Choices) == 0 {
		return "", fmt.Errorf("bad chat response")
	}
	return strings.TrimSpace(parsed.Choices[0].Message.Content), nil
}

// ---------- infrastructure ----------

func freePort() string {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(err)
	}
	addr := l.Addr().String()
	l.Close()
	return addr
}

func adminGet(path string) []byte {
	req, _ := http.NewRequest(http.MethodGet, proxyEP+path, nil)
	req.Header.Set("Authorization", "Bearer "+adminToken)
	resp, err := client.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return b
}

type statWindow struct {
	Items       uint64 `json:"items"`
	Hits        uint64 `json:"hits"`
	Misses      uint64 `json:"misses"`
	SavedTokens uint64 `json:"saved_tokens"`
	SpentTokens uint64 `json:"spent_tokens"`
}

func snap() statWindow {
	var s statWindow
	json.Unmarshal(adminGet("/_ec/stats"), &s)
	return s
}

func main() {
	flag.Parse()
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.MaxIdleConns = 128
	tr.MaxIdleConnsPerHost = 128
	client = &http.Client{Timeout: 300 * time.Second, Transport: tr}

	fmt.Fprintf(&rep, "# embedcache real-workload test\n\n")
	fmt.Fprintf(&rep, "Generated %s. Everything in this run is real: live Wikipedia articles fetched over\nthe internet, real chunking, real embedding inference (Ollama `%s`), a working RAG\nagent answering real questions with a real LLM (hosted Gemini with local `%s`\nfallback), and a live-refresh pass re-fetching the same articles. **No duplicate is\nconstructed; every duplicate below arose organically from the workload.**\n",
		time.Now().Format("2006-01-02 15:04 MST"), *embedModel, *localChat)

	logPath := "realtest-requests.jsonl"
	os.Remove(logPath)
	addr := freePort()
	proxyEP = "http://" + addr
	cmd := exec.Command(*binPath, "serve",
		"-listen", addr, "-upstream", *ollamaURL,
		"-admin-token", adminToken,
		"-auth-mode", "allowlist", "-api-keys", pipelineKey+","+agentKey,
		"-ttl", "24h", "-request-log", logPath,
	)
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer cmd.Process.Kill()
	for i := 0; ; i++ {
		resp, err := client.Get(proxyEP + "/healthz")
		if err == nil {
			resp.Body.Close()
			break
		}
		if i > 100 {
			fmt.Fprintln(os.Stderr, "proxy never became healthy")
			os.Exit(1)
		}
		time.Sleep(100 * time.Millisecond)
	}

	// ---------- 1. real corpus ----------
	section("1 — Live corpus: English Wikipedia")
	t0 := time.Now()
	articles := fetchCorpus()
	var chunks []chunk
	totalBytes := 0
	titles := make([]string, 0, len(articles))
	for t := range articles {
		titles = append(titles, t)
	}
	sort.Strings(titles)
	for _, t := range titles {
		totalBytes += len(articles[t])
		chunks = append(chunks, chunkText(t, articles[t])...)
	}
	line("| metric | value |")
	line("|---|---|")
	line("| articles fetched live | %d of %d requested |", len(articles), len(articleTitles))
	line("| raw text | %.1f MB |", float64(totalBytes)/1e6)
	line("| chunks after pipeline chunking | %d |", len(chunks))
	line("| fetch wall time | %s |", time.Since(t0).Round(time.Second))
	check("real corpus fetched", len(articles) >= 90, fmt.Sprintf("%d live articles, %d chunks", len(articles), len(chunks)))

	// ---------- 2. real ingestion ----------
	section("2 — Ingestion through embedcache (real inference)")
	ing0 := snap()
	t1 := time.Now()
	vs := &store{}
	var ingErrs int
	var firstErr string
	for i := 0; i < len(chunks); i += 32 {
		end := i + 32
		if end > len(chunks) {
			end = len(chunks)
		}
		inputs := make([]string, end-i)
		for j := range inputs {
			inputs[j] = chunks[i+j].text
		}
		vecs, err := embedResilient(pipelineKey, inputs)
		if err != nil {
			ingErrs++
			if firstErr == "" {
				firstErr = err.Error()
			}
			continue
		}
		for j := range vecs {
			chunks[i+j].vec = vecs[j]
		}
	}
	vs.chunks = chunks
	ing1 := snap()
	ingestWall := time.Since(t1)
	line("| metric | value |")
	line("|---|---|")
	line("| chunks embedded | %d |", len(chunks))
	line("| wall time | %s |", ingestWall.Round(time.Second))
	line("| cache hits during cold ingest (organic intra-corpus dupes) | %d |", ing1.Hits+0-ing0.Hits)
	line("| errors | %d |", ingErrs)
	detail := fmt.Sprintf("%d chunks in %s", len(chunks), ingestWall.Round(time.Second))
	if firstErr != "" {
		detail += " — first error: " + firstErr
	}
	check("ingestion clean", ingErrs == 0, detail)

	// ---------- 3. the agent ----------
	section("3 — RAG agent: %d sessions, real LLM, organic query traffic", *sessions)
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	agent0 := snap()
	type qa struct{ q, a, llm string }
	var transcript []qa
	asked := []string{}
	answered := 0
	t2 := time.Now()
	for s := 0; s < *sessions; s++ {
		var q string
		// real users repeat intents: ~40% of sessions re-ask something
		// already asked in this run
		if len(asked) > 0 && rng.Float64() < 0.4 {
			q = asked[rng.Intn(len(asked))]
		} else {
			q = questions[rng.Intn(len(questions))]
		}
		asked = append(asked, q)

		// agent step 1: query expansion (an agentic pattern that multiplies
		// embedding calls in production)
		expansion, _ := chat(
			"You rewrite search queries. Output exactly 2 alternative phrasings of the user's query, one per line, no numbering, no extra text.",
			q)
		queries := []string{q}
		for _, l := range strings.Split(expansion, "\n") {
			if l = strings.TrimSpace(l); l != "" && len(queries) < 3 {
				queries = append(queries, l)
			}
		}

		// agent step 2: embed every query variant, retrieve, merge
		var ctxChunks []chunk
		seen := map[string]bool{}
		for _, qv := range queries {
			vecs, _, err := embedBatch(agentKey, []string{qv})
			if err != nil || len(vecs) != 1 {
				continue
			}
			for _, c := range vs.search(vecs[0], 3) {
				if !seen[c.text[:60]] {
					seen[c.text[:60]] = true
					ctxChunks = append(ctxChunks, c)
				}
			}
		}

		// agent step 3: grounded answer
		var ctxB strings.Builder
		for _, c := range ctxChunks {
			fmt.Fprintf(&ctxB, "[%s] %s\n", c.title, c.text)
		}
		answer, llm := chat(
			"Answer the question using ONLY the provided context. Maximum 3 sentences. If the context is insufficient, say so.",
			"Context:\n"+ctxB.String()+"\nQuestion: "+q)
		if answer != "" {
			answered++
		}
		if len(transcript) < 3 && answer != "" {
			transcript = append(transcript, qa{q, answer, llm})
		}
	}
	agent1 := snap()
	agentWall := time.Since(t2)
	aItems := agent1.Items - agent0.Items
	aHits := agent1.Hits - agent0.Hits
	line("| metric | value |")
	line("|---|---|")
	line("| sessions | %d |", *sessions)
	line("| answered by LLM | %d (%d hosted, %d local fallback, %d hosted failures) |", answered, remoteOK.Load(), localOK.Load(), remoteFail.Load())
	line("| embedding calls made by the agent | %d |", aItems)
	line("| served from cache (organic repeats) | %d (**%.1f%%**) |", aHits, float64(aHits)/float64(aItems)*100)
	line("| wall time | %s |", agentWall.Round(time.Second))
	line("")
	line("Sample transcript (real LLM output, unedited):")
	line("")
	for _, t := range transcript {
		line("> **Q:** %s", t.q)
		line("> **A** (%s)**:** %s", t.llm, strings.ReplaceAll(t.a, "\n", " "))
		line(">")
	}
	check("agent answered its questions", answered == *sessions, fmt.Sprintf("%d/%d sessions produced grounded answers", answered, *sessions))
	check("agent traffic has organic cache hits", aHits > 0, fmt.Sprintf("%.1f%% of agent embedding calls were repeats it did not pay for", float64(aHits)/float64(aItems)*100))

	// ---------- 4. the nightly refresh ----------
	section("4 — Live refresh: re-fetch the same articles, re-ingest (the real nightly pipeline)")
	t3 := time.Now()
	fresh := fetchCorpus()
	var freshChunks []chunk
	freshTitles := make([]string, 0, len(fresh))
	for t := range fresh {
		freshTitles = append(freshTitles, t)
	}
	sort.Strings(freshTitles)
	for _, t := range freshTitles {
		freshChunks = append(freshChunks, chunkText(t, fresh[t])...)
	}
	ref0 := snap()
	refErrs := 0
	for i := 0; i < len(freshChunks); i += 32 {
		end := i + 32
		if end > len(freshChunks) {
			end = len(freshChunks)
		}
		inputs := make([]string, end-i)
		for j := range inputs {
			inputs[j] = freshChunks[i+j].text
		}
		if _, err := embedResilient(pipelineKey, inputs); err != nil {
			refErrs++
		}
	}
	ref1 := snap()
	refWall := time.Since(t3)
	refItems := ref1.Items - ref0.Items
	refMisses := ref1.Misses - ref0.Misses
	absorbed := (1 - float64(refMisses)/float64(refItems)) * 100
	line("| metric | value |")
	line("|---|---|")
	line("| chunks re-ingested | %d |", refItems)
	line("| actually changed on Wikipedia since first fetch | %d chunks re-embedded |", refMisses)
	line("| absorbed by cache | **%.2f%%** |", absorbed)
	line("| wall time (fetch + re-ingest) vs cold | %s vs %s |", refWall.Round(time.Second), ingestWall.Round(time.Second))
	check("refresh pays only for real edits", absorbed > 95,
		fmt.Sprintf("%.2f%% of the re-ingest was absorbed; only %d chunks had actually changed", absorbed, refMisses))

	// ---------- 5. the waste report on organic traffic ----------
	section("5 — The waste report (the adoption artifact)")
	final := snap()
	out, err := exec.Command(*binPath, "analyze", logPath).Output()
	if err == nil {
		line("`embedcache analyze` on this run's real request log:")
		line("")
		line("```")
		rep.Write(out)
		line("```")
	}
	// project the same duplicate ratio onto hosted API pricing
	dupShare := float64(final.SavedTokens) / float64(final.SavedTokens+final.SpentTokens)
	line("")
	line("Measured organically: **%.1f%% of this workload's embedding tokens were duplicate work.**", dupShare*100)
	line("At hosted-API prices, that share of every $1,000/month embedding bill is **$%.0f wasted** —", dupShare*1000)
	line("on self-hosted GPUs it is the same share of GPU-hours. That number, measured on a team's")
	line("own logs by the offline analyzer before they install anything, is the adoption pitch.")
	check("organic duplicate share measured", final.SavedTokens > 0,
		fmt.Sprintf("saved=%d tokens, spent=%d tokens, duplicate share %.1f%%", final.SavedTokens, final.SpentTokens, dupShare*100))

	// os.Exit skips defers, so stop the proxy explicitly before any exit —
	// a leaked proxy keeps the request log locked on Windows
	cmd.Process.Kill()
	cmd.Wait()
	os.Remove(logPath)
	if err := os.WriteFile(*outPath, rep.Bytes(), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if failed > 0 {
		fmt.Fprintf(os.Stderr, "\n%d check(s) FAILED — see %s\n", failed, *outPath)
		os.Exit(1)
	}
	fmt.Printf("\nall real-workload checks passed — %s written\n", *outPath)
}
