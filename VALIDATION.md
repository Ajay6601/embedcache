# embedcache multi-scenario validation

Generated 2026-07-11 14:59 EDT. Every result below is produced by the real compiled binary against
**real embedding backends** with **real data** — this repository's own Go source as the
code corpus, live Wikipedia articles as the prose corpus, and real edits driving
re-ingestion. No generated placeholder corpora. Where a capability cannot be
exercised for real on the available backends, the report says so.

## Discovered real backends

- _info_ — backend Ollama all-minilm: live, 384-dim
- _info_ — backend Ollama nomic-embed-text: live, 768-dim
- _info_ — backend Ollama mxbai-embed-large: live, 1024-dim
- _info_ — backend Ollama bge-m3: live, 1024-dim
- _info_ — backend Ollama snowflake-arctic-embed2: live, 1024-dim
- _info_ — backend Ollama granite-embedding: live, 384-dim

6 real backend(s) live; every scenario below runs against each where applicable.
- _info_ — code corpus: 119 real chunks from this repo's own Go source
- _info_ — prose corpus: 120 real chunks from 10 live Wikipedia articles

## Scenario 1 — Correctness across every real model

Byte-exact replay (miss→hit and proxy-vs-direct) and mixed-batch index mapping, per model.
Different models = different real dimensions and behaviors; the cache guarantee must hold for all.

- **PASS** — Ollama all-minilm cache hit is byte-exact replay: 384-dim, miss=miss hit=hit, 4709 bytes identical
- _info_ — Ollama all-minilm backend determinism: backend returns identical bytes across calls
- **PASS** — Ollama all-minilm mixed-batch index mapping: partial=true, cachedA-exact=true, B/C-distinct=true, B-slot-consistent=true
- **PASS** — Ollama nomic-embed-text cache hit is byte-exact replay: 768-dim, miss=miss hit=hit, 9500 bytes identical
- _info_ — Ollama nomic-embed-text backend determinism: backend returns identical bytes across calls
- **PASS** — Ollama nomic-embed-text mixed-batch index mapping: partial=true, cachedA-exact=true, B/C-distinct=true, B-slot-consistent=true
- **PASS** — Ollama mxbai-embed-large cache hit is byte-exact replay: 1024-dim, miss=miss hit=hit, 12719 bytes identical
- _info_ — Ollama mxbai-embed-large backend determinism: backend returns identical bytes across calls
- **PASS** — Ollama mxbai-embed-large mixed-batch index mapping: partial=true, cachedA-exact=true, B/C-distinct=true, B-slot-consistent=true
- **PASS** — Ollama bge-m3 cache hit is byte-exact replay: 1024-dim, miss=miss hit=hit, 12756 bytes identical
- _info_ — Ollama bge-m3 backend determinism: backend returns identical bytes across calls
- **PASS** — Ollama bge-m3 mixed-batch index mapping: partial=true, cachedA-exact=true, B/C-distinct=true, B-slot-consistent=true
- **PASS** — Ollama snowflake-arctic-embed2 cache hit is byte-exact replay: 1024-dim, miss=miss hit=hit, 12734 bytes identical
- _info_ — Ollama snowflake-arctic-embed2 backend determinism: backend returns identical bytes across calls
- **PASS** — Ollama snowflake-arctic-embed2 mixed-batch index mapping: partial=true, cachedA-exact=true, B/C-distinct=true, B-slot-consistent=true
- **PASS** — Ollama granite-embedding cache hit is byte-exact replay: 384-dim, miss=miss hit=hit, 4740 bytes identical
- _info_ — Ollama granite-embedding backend determinism: backend returns identical bytes across calls
- **PASS** — Ollama granite-embedding mixed-batch index mapping: partial=true, cachedA-exact=true, B/C-distinct=true, B-slot-consistent=true

## Scenario 2 — RAG re-ingest on real code and real prose

Cold-ingest a real corpus, edit ~5% of chunks, re-ingest. Absorbed % = the dedup win.

| backend · corpus | size | cold misses | re-ingest recomputed | absorbed |
|---|---|---|---|---|
| Ollama all-minilm · code | 119 chunks (64 natural dupes) | cold 55 | re-ingest 1 | **99.2% absorbed** |
- _info_ — Ollama all-minilm code incremental re-ingest: backend rejected 4 cold + 4 re-ingest batches (rate limit or input exceeding model context); absorption (99.2%) not reliable under those errors
| Ollama all-minilm · prose | 120 chunks (0 natural dupes) | cold 120 | re-ingest 6 | **95.0% absorbed** |
- **PASS** — Ollama all-minilm prose incremental re-ingest: 120 chunks (0 natural dupes on cold ingest), 6 edited, 95.0% of re-ingest absorbed
| Ollama nomic-embed-text · code | 119 chunks (0 natural dupes) | cold 119 | re-ingest 5 | **95.8% absorbed** |
- **PASS** — Ollama nomic-embed-text code incremental re-ingest: 119 chunks (0 natural dupes on cold ingest), 5 edited, 95.8% of re-ingest absorbed
| Ollama nomic-embed-text · prose | 120 chunks (0 natural dupes) | cold 120 | re-ingest 6 | **95.0% absorbed** |
- **PASS** — Ollama nomic-embed-text prose incremental re-ingest: 120 chunks (0 natural dupes on cold ingest), 6 edited, 95.0% of re-ingest absorbed
| Ollama mxbai-embed-large · code | 119 chunks (0 natural dupes) | cold 119 | re-ingest 5 | **95.8% absorbed** |
- **PASS** — Ollama mxbai-embed-large code incremental re-ingest: 119 chunks (0 natural dupes on cold ingest), 5 edited, 95.8% of re-ingest absorbed
| Ollama mxbai-embed-large · prose | 120 chunks (0 natural dupes) | cold 120 | re-ingest 6 | **95.0% absorbed** |
- **PASS** — Ollama mxbai-embed-large prose incremental re-ingest: 120 chunks (0 natural dupes on cold ingest), 6 edited, 95.0% of re-ingest absorbed
| Ollama bge-m3 · code | 119 chunks (0 natural dupes) | cold 119 | re-ingest 5 | **95.8% absorbed** |
- **PASS** — Ollama bge-m3 code incremental re-ingest: 119 chunks (0 natural dupes on cold ingest), 5 edited, 95.8% of re-ingest absorbed
| Ollama bge-m3 · prose | 120 chunks (0 natural dupes) | cold 120 | re-ingest 6 | **95.0% absorbed** |
- **PASS** — Ollama bge-m3 prose incremental re-ingest: 120 chunks (0 natural dupes on cold ingest), 6 edited, 95.0% of re-ingest absorbed
| Ollama snowflake-arctic-embed2 · code | 119 chunks (0 natural dupes) | cold 119 | re-ingest 5 | **95.8% absorbed** |
- **PASS** — Ollama snowflake-arctic-embed2 code incremental re-ingest: 119 chunks (0 natural dupes on cold ingest), 5 edited, 95.8% of re-ingest absorbed
| Ollama snowflake-arctic-embed2 · prose | 120 chunks (0 natural dupes) | cold 120 | re-ingest 6 | **95.0% absorbed** |
- **PASS** — Ollama snowflake-arctic-embed2 prose incremental re-ingest: 120 chunks (0 natural dupes on cold ingest), 6 edited, 95.0% of re-ingest absorbed
| Ollama granite-embedding · code | 119 chunks (0 natural dupes) | cold 119 | re-ingest 5 | **95.8% absorbed** |
- **PASS** — Ollama granite-embedding code incremental re-ingest: 119 chunks (0 natural dupes on cold ingest), 5 edited, 95.8% of re-ingest absorbed
| Ollama granite-embedding · prose | 120 chunks (0 natural dupes) | cold 120 | re-ingest 6 | **95.0% absorbed** |
- **PASS** — Ollama granite-embedding prose incremental re-ingest: 120 chunks (0 natural dupes on cold ingest), 6 edited, 95.0% of re-ingest absorbed

## Scenario 3 — Agentic query traffic (real query-expansion loops)

- **PASS** — Ollama all-minilm agentic query traffic has organic hits: 57/72 agent embeds served from cache (79.2%) with real repeated intents
- **PASS** — Ollama nomic-embed-text agentic query traffic has organic hits: 57/72 agent embeds served from cache (79.2%) with real repeated intents
- **PASS** — Ollama mxbai-embed-large agentic query traffic has organic hits: 57/72 agent embeds served from cache (79.2%) with real repeated intents
- **PASS** — Ollama bge-m3 agentic query traffic has organic hits: 57/72 agent embeds served from cache (79.2%) with real repeated intents
- **PASS** — Ollama snowflake-arctic-embed2 agentic query traffic has organic hits: 57/72 agent embeds served from cache (79.2%) with real repeated intents
- **PASS** — Ollama granite-embedding agentic query traffic has organic hits: 57/72 agent embeds served from cache (79.2%) with real repeated intents

## Scenario 4 — Semantic-search economics (Zipf traffic on a real corpus)

- **PASS** — Ollama all-minilm search-shaped traffic cache economics: 400 Zipf queries over 120 docs → 79.8% hit rate among 400 completed
- **PASS** — Ollama nomic-embed-text search-shaped traffic cache economics: 400 Zipf queries over 120 docs → 80.0% hit rate among 400 completed
- **PASS** — Ollama mxbai-embed-large search-shaped traffic cache economics: 400 Zipf queries over 120 docs → 82.0% hit rate among 400 completed
- **PASS** — Ollama bge-m3 search-shaped traffic cache economics: 400 Zipf queries over 120 docs → 80.0% hit rate among 400 completed
- **PASS** — Ollama snowflake-arctic-embed2 search-shaped traffic cache economics: 400 Zipf queries over 120 docs → 80.0% hit rate among 400 completed
- **PASS** — Ollama granite-embedding search-shaped traffic cache economics: 400 Zipf queries over 120 docs → 83.2% hit rate among 400 completed

## Scenario 5 — Multilingual real text: cost-estimator drift, mixed-batch attribution, Unicode normalization

Real (randomly-selected) live Wikipedia articles in Chinese, Hindi, Arabic and Spanish —
no translation, no hand-picked titles. Measures where the internal bytes/4 token estimator
(used only for apportionment and the offline waste analyzer, never for billing) drifts from
each backend's real reported usage, and whether visually-identical text in different Unicode
normalization forms (NFC vs NFD) leaks as a cache duplicate.

- _info_ — Chinese corpus: 20 real chunks fetched from live Chinese Wikipedia
- _info_ — Hindi corpus: 6 real chunks fetched from live Hindi Wikipedia
- _info_ — Arabic corpus: 19 real chunks fetched from live Arabic Wikipedia
- _info_ — Spanish corpus: 6 real chunks fetched from live Spanish Wikipedia
- _info_ — Ollama all-minilm unicode normalization duplicate-leak: visually-identical "café" in NFC vs NFD form: first=miss second=miss (leaks=true) — real backends/tools mix normalization forms; embedcache does not fold them today
- _info_ — Ollama all-minilm multilingual Chinese cost-estimator drift: 20 real Chinese paragraphs: backend billed 1364 tokens, bytes/4 estimate 847 (-37.9% drift)
- _info_ — Ollama all-minilm multilingual Hindi cost-estimator drift: 6 real Hindi paragraphs: backend billed 108 tokens, bytes/4 estimate 201 (86.1% drift)
- _info_ — Ollama all-minilm multilingual Arabic: backend rejected real Arabic batch: status 400: {"error":{"message":"the input length exceeds the context length","type":"invalid_request_error","param":null,"code":null}}

- _info_ — Ollama all-minilm multilingual Spanish cost-estimator drift: 6 real Spanish paragraphs: backend billed 458 tokens, bytes/4 estimate 395 (-13.8% drift)
- **PASS** — Ollama all-minilm mixed-language batch is billed as one exact total: 4-item EN+ZH batch billed 27 total_tokens by backend
- _info_ — Ollama nomic-embed-text unicode normalization duplicate-leak: visually-identical "café" in NFC vs NFD form: first=miss second=miss (leaks=true) — real backends/tools mix normalization forms; embedcache does not fold them today
- _info_ — Ollama nomic-embed-text multilingual Arabic cost-estimator drift: 19 real Arabic paragraphs: backend billed 1875 tokens, bytes/4 estimate 1141 (-39.1% drift)
- _info_ — Ollama nomic-embed-text multilingual Spanish cost-estimator drift: 6 real Spanish paragraphs: backend billed 421 tokens, bytes/4 estimate 395 (-6.2% drift)
- _info_ — Ollama nomic-embed-text multilingual Chinese cost-estimator drift: 20 real Chinese paragraphs: backend billed 1318 tokens, bytes/4 estimate 847 (-35.7% drift)
- _info_ — Ollama nomic-embed-text multilingual Hindi cost-estimator drift: 6 real Hindi paragraphs: backend billed 108 tokens, bytes/4 estimate 201 (86.1% drift)
- **PASS** — Ollama nomic-embed-text mixed-language batch is billed as one exact total: 4-item EN+ZH batch billed 27 total_tokens by backend
- _info_ — Ollama mxbai-embed-large unicode normalization duplicate-leak: visually-identical "café" in NFC vs NFD form: first=miss second=miss (leaks=true) — real backends/tools mix normalization forms; embedcache does not fold them today
- _info_ — Ollama mxbai-embed-large multilingual Spanish cost-estimator drift: 6 real Spanish paragraphs: backend billed 458 tokens, bytes/4 estimate 395 (-13.8% drift)
- _info_ — Ollama mxbai-embed-large multilingual Chinese cost-estimator drift: 20 real Chinese paragraphs: backend billed 1364 tokens, bytes/4 estimate 847 (-37.9% drift)
- _info_ — Ollama mxbai-embed-large multilingual Hindi cost-estimator drift: 6 real Hindi paragraphs: backend billed 108 tokens, bytes/4 estimate 201 (86.1% drift)
- _info_ — Ollama mxbai-embed-large multilingual Arabic cost-estimator drift: 19 real Arabic paragraphs: backend billed 1889 tokens, bytes/4 estimate 1141 (-39.6% drift)
- **PASS** — Ollama mxbai-embed-large mixed-language batch is billed as one exact total: 4-item EN+ZH batch billed 27 total_tokens by backend
- _info_ — Ollama bge-m3 unicode normalization duplicate-leak: visually-identical "café" in NFC vs NFD form: first=miss second=miss (leaks=true) — real backends/tools mix normalization forms; embedcache does not fold them today
- _info_ — Ollama bge-m3 multilingual Hindi cost-estimator drift: 6 real Hindi paragraphs: backend billed 83 tokens, bytes/4 estimate 201 (142.2% drift)
- _info_ — Ollama bge-m3 multilingual Arabic cost-estimator drift: 19 real Arabic paragraphs: backend billed 876 tokens, bytes/4 estimate 1141 (30.3% drift)
- _info_ — Ollama bge-m3 multilingual Spanish cost-estimator drift: 6 real Spanish paragraphs: backend billed 377 tokens, bytes/4 estimate 395 (4.8% drift)
- _info_ — Ollama bge-m3 multilingual Chinese cost-estimator drift: 20 real Chinese paragraphs: backend billed 1041 tokens, bytes/4 estimate 847 (-18.6% drift)
- **PASS** — Ollama bge-m3 mixed-language batch is billed as one exact total: 4-item EN+ZH batch billed 32 total_tokens by backend
- _info_ — Ollama snowflake-arctic-embed2 unicode normalization duplicate-leak: visually-identical "café" in NFC vs NFD form: first=miss second=miss (leaks=true) — real backends/tools mix normalization forms; embedcache does not fold them today
- _info_ — Ollama snowflake-arctic-embed2 multilingual Chinese cost-estimator drift: 20 real Chinese paragraphs: backend billed 1042 tokens, bytes/4 estimate 847 (-18.7% drift)
- _info_ — Ollama snowflake-arctic-embed2 multilingual Hindi cost-estimator drift: 6 real Hindi paragraphs: backend billed 83 tokens, bytes/4 estimate 201 (142.2% drift)
- _info_ — Ollama snowflake-arctic-embed2 multilingual Arabic cost-estimator drift: 19 real Arabic paragraphs: backend billed 878 tokens, bytes/4 estimate 1141 (30.0% drift)
- _info_ — Ollama snowflake-arctic-embed2 multilingual Spanish cost-estimator drift: 6 real Spanish paragraphs: backend billed 377 tokens, bytes/4 estimate 395 (4.8% drift)
- **PASS** — Ollama snowflake-arctic-embed2 mixed-language batch is billed as one exact total: 4-item EN+ZH batch billed 32 total_tokens by backend
- _info_ — Ollama granite-embedding unicode normalization duplicate-leak: visually-identical "café" in NFC vs NFD form: first=miss second=miss (leaks=true) — real backends/tools mix normalization forms; embedcache does not fold them today
- _info_ — Ollama granite-embedding multilingual Chinese cost-estimator drift: 20 real Chinese paragraphs: backend billed 2403 tokens, bytes/4 estimate 847 (-64.8% drift)
- _info_ — Ollama granite-embedding multilingual Hindi cost-estimator drift: 6 real Hindi paragraphs: backend billed 489 tokens, bytes/4 estimate 201 (-58.9% drift)
- _info_ — Ollama granite-embedding multilingual Arabic cost-estimator drift: 19 real Arabic paragraphs: backend billed 2538 tokens, bytes/4 estimate 1141 (-55.0% drift)
- _info_ — Ollama granite-embedding multilingual Spanish cost-estimator drift: 6 real Spanish paragraphs: backend billed 542 tokens, bytes/4 estimate 395 (-27.1% drift)
- **PASS** — Ollama granite-embedding mixed-language batch is billed as one exact total: 4-item EN+ZH batch billed 29 total_tokens by backend

## Scenario 6 — Long-context real payloads

Real concatenated Wikipedia prose sized to ~4000 tokens, through bge-m3 (8k real context
window) — the corpus size our other tested models (all-minilm at 256 tokens) cannot accept.

- **PASS** — Ollama bge-m3 long-context real payload caches correctly: ~4126-token payload (16501 real chars): miss=miss hit=hit

## Scenario 7 — Multi-tenant cost control with per-key budgets

Two real tenants share one proxy; one has a tiny token budget. Proves per-tenant
enforcement and that an exhausted tenant still gets cache hits.

- **PASS** — small tenant hits its budget and gets 429: new computation rejected after budget spent
- **PASS** — over-budget tenant still served from cache: cached input returns 200 hit while budget is exhausted
- **PASS** — unlimited tenant unaffected by other tenant's cap: tenant-big served normally

## Scenario 8 — Multimodal / image embeddings (honest boundary)

The OpenAI `/v1/embeddings` contract embedcache proxies is text-only. Real image
embeddings require a multimodal endpoint with a different request shape (Voyage
multimodal-3, or Gemini's native multimodal endpoint) — none of which is an
OpenAI-`/v1/embeddings`-compatible backend reachable here (Gemini's compat endpoint
exposes only text models; no local vision-embed model serves that route).

- **PASS** — content-agnostic fingerprint (image-shaped input caches correctly): a 4422-byte base64 image blob: first=miss second=hit

**Honest verdict:** text embeddings are validated across 6 real models; multimodal
image caching is *architecturally* supported (proven above by the content-agnostic
fingerprint test) but **not live-tested end-to-end** for lack of an OpenAI-compatible
image-embedding backend. It's a v0.3 item pending a Voyage multimodal key.

## Bottom line

- real backends exercised: Ollama all-minilm (384d), Ollama bge-m3 (1024d), Ollama granite-embedding (384d), Ollama mxbai-embed-large (1024d), Ollama nomic-embed-text (768d), Ollama snowflake-arctic-embed2 (1024d)
- real corpora: this repo's Go source (code), live Wikipedia (prose, and zh/hi/ar/es for multilingual)
- scenarios: correctness · RAG re-ingest · agentic loops · semantic search · multilingual/unicode · long-context · multi-tenant budgets
- multimodal: architecturally supported, live test deferred (documented, not faked)
