# embedcache vs BEIR SciFact: retrieval-quality and eval-loop-cost benchmark

Generated 2026-07-12 18:03 EDT. Dataset: BEIR SciFact (Thakur et al. 2021) — real published
scientific-claim queries, real paper abstracts, real expert relevance judgments.
5183 real documents, 300 real test queries with relevance judgments.

**Claim under test:** routing embedding calls through embedcache changes nothing about
retrieval quality (it is a transparent cache, not a lossy approximation), and re-running
the same eval a second time — the normal workflow when tuning a RAG pipeline — costs
almost nothing once the corpus is warm.

## Model: nomic-embed-text

| pass | nDCG@10 | Recall@100 |
|---|---|---|
| direct to backend (ground truth) | 0.5221 | 0.8297 |
| via embedcache, cold cache | 0.5221 | 0.8297 |
| via embedcache, warm cache (re-run) | 0.5221 | 0.8297 |

- **PASS** — zero retrieval-quality loss: top-10 rankings identical (cold vs direct)=true (0 differ), (warm vs direct)=true (0 differ); metrics equal at 1e-4; warm re-run replays cold byte-exact=true
- **eval-loop cost:** re-running the identical 5483-item eval (300 queries + 5183 docs) recomputed only
  0 items (100.0% absorbed from cache) — 5483 served instantly from the first pass
- **eval-loop wall clock:** the identical warm re-run took **2.064s** (100% cache hits). The cold-pass duration is not reported because the run was interrupted (machine sleep / backend stall) and its wall clock would include idle time, not compute.

## Method

- Dataset: BEIR SciFact test split, downloaded from the public BEIR host, used unmodified.
- Metrics: nDCG@10 and Recall@100, standard TREC/BEIR formulas, computed in this harness
  directly from cosine similarity over the real embedding vectors (no external eval library).
- Three passes per model: direct-to-backend (ground truth, bypasses embedcache entirely),
  cold-cache-via-proxy (first exposure, should match direct exactly), warm-cache-via-proxy
  (second exposure to the identical corpus+queries — the real 'run my eval again' workflow).
- Only real backends (local Ollama) are used; no synthetic vectors, no mocked scoring.
