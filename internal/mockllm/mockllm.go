// Package mockllm is a deterministic OpenAI-compatible embeddings backend
// for tests and experiments. The embedding for an input is a pure function of
// (model, input), so any two responses for the same input are guaranteed
// identical — which is what lets the experiments assert byte-exactness and
// count real upstream work.
package mockllm

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Ajay6601/embedcache/internal/api"
	"github.com/Ajay6601/embedcache/internal/tokens"
)

type Server struct {
	Dim     int
	Latency time.Duration

	mu       sync.Mutex
	calls    int
	items    int
	perInput map[string]int
}

func New(dim int) *Server {
	return &Server{Dim: dim, perInput: map[string]int{}}
}

func (s *Server) Calls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls
}

func (s *Server) Items() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.items
}

// CountFor reports how many times an input text reached the backend.
func (s *Server) CountFor(text string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.perInput[text]
}

func (s *Server) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls, s.items = 0, 0
	s.perInput = map[string]int{}
}

func (s *Server) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/embeddings") {
			http.NotFound(w, r)
			return
		}
		var req api.EmbeddingsRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		items, err := api.SplitInput(req.Input)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if s.Latency > 0 {
			time.Sleep(s.Latency)
		}

		s.mu.Lock()
		s.calls++
		s.items += len(items)
		for _, it := range items {
			s.perInput[seedFor(it)]++
		}
		s.mu.Unlock()

		dim := s.Dim
		if req.Dimensions > 0 && req.Dimensions < dim {
			dim = req.Dimensions
		}
		promptTokens := 0
		data := make([]api.EmbeddingData, len(items))
		for i, it := range items {
			promptTokens += tokens.Estimate(it)
			vec := Vector(req.Model, seedFor(it), dim)
			var raw json.RawMessage
			if req.EncodingFormat == "base64" {
				raw, _ = json.Marshal(encodeBase64(vec))
			} else {
				raw, _ = json.Marshal(vec)
			}
			data[i] = api.EmbeddingData{Object: "embedding", Index: i, Embedding: raw}
		}
		resp := api.EmbeddingsResponse{
			Object: "list",
			Data:   data,
			Model:  req.Model,
			Usage:  api.Usage{PromptTokens: promptTokens, TotalTokens: promptTokens},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})
}

func seedFor(it api.InputItem) string {
	if it.IsTokens {
		return fmt.Sprintf("tokens:%v", it.Tokens)
	}
	return it.Text
}

// Vector derives a deterministic unit-ish vector from (model, seed).
func Vector(model, seed string, dim int) []float32 {
	out := make([]float32, dim)
	state := sha256.Sum256([]byte(model + "\x00" + seed))
	for i := 0; i < dim; i++ {
		if i%8 == 0 && i > 0 {
			state = sha256.Sum256(state[:])
		}
		u := binary.LittleEndian.Uint32(state[(i%8)*4 : (i%8)*4+4])
		out[i] = float32(u)/float32(math.MaxUint32)*2 - 1
	}
	return out
}

func encodeBase64(vec []float32) string {
	buf := make([]byte, 4*len(vec))
	for i, f := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return base64.StdEncoding.EncodeToString(buf)
}
