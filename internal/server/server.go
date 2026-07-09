// Package server wires the data plane and the admin endpoints onto one
// listener. Anything that is not an embeddings POST or an admin path is
// reverse-proxied to the upstream untouched, so embedcache can sit in front
// of a full OpenAI-compatible server.
package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"

	"embedcache/internal/pricing"
	"embedcache/internal/proxy"
	"embedcache/internal/stats"
)

type Server struct {
	Proxy   *proxy.Proxy
	Stats   *stats.Collector
	Pricing *pricing.Table
	rp      *httputil.ReverseProxy
}

func New(p *proxy.Proxy, st *stats.Collector, table *pricing.Table, upstreamBase *url.URL) *Server {
	return &Server{
		Proxy:   p,
		Stats:   st,
		Pricing: table,
		rp:      httputil.NewSingleHostReverseProxy(upstreamBase),
	}
}

func (s *Server) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(path, "/embeddings"):
			s.Proxy.ServeEmbeddings(w, r)
		case path == "/_ec/stats" || path == "/stats":
			s.serveStats(w)
		case path == "/_ec/report" || path == "/report":
			s.serveReport(w)
		case path == "/_ec/flush" && r.Method == http.MethodPost:
			n := s.Proxy.Cache.Flush()
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `{"flushed":%d}`, n)
		case path == "/metrics":
			s.serveMetrics(w)
		case path == "/healthz":
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("ok"))
		default:
			s.Stats.RecordPassthrough()
			s.rp.ServeHTTP(w, r)
		}
	})
}

func (s *Server) snapshot() stats.Report {
	return s.Stats.Snapshot(s.Pricing, s.Proxy.Cache.Len(), s.Proxy.Cache.Bytes())
}

func (s *Server) serveStats(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	enc.Encode(s.snapshot())
}

func (s *Server) serveReport(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	s.snapshot().RenderText(w)
}

func (s *Server) serveMetrics(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	s.snapshot().RenderPrometheus(w)
}
