# embedcache experiments

Generated 2026-07-09 08:32 EDT · windows/amd64 · 16 CPUs · Go go1.26.5 · mock upstream on loopback

Every experiment runs the real `embedcache serve` binary as a subprocess against
a deterministic mock OpenAI-compatible backend: the embedding for an input is a pure
function of (model, input), so byte-level comparisons against ground truth are exact,
and the mock counts every request/item that reaches it.

## E1 — Byte-exact correctness under randomized batching

**Claim tested:** a response served through embedcache — any mix of cache hits,
misses, and intra-batch duplicates, in float or base64 encoding — is byte-identical
to what the upstream would have returned, with correct index mapping.

**Method:** 400 randomized requests (batch size 1–16, drawn with replacement from a
300-input pool, ~20% base64), each embedding compared byte-for-byte against a direct
call to the mock upstream. This also covers the failure class of LiteLLM issue
[#22659](https://github.com/BerriAI/litellm/issues/22659) (mixed cached/uncached batches
returning wrong vectors) continuously, since the cache fills as the fuzz runs.

| metric | value |
|---|---|
| batches sent | 400 |
| embeddings verified byte-exact | 3453 |
| mismatches | 0 |

- **PASS** — byte-exact fuzz: 3453 embeddings verified, 0 mismatches

## E2 — In-flight coalescing

**Claim tested:** concurrent requests for the same input trigger exactly one
upstream computation; overlapping batches compute each unique input once.

**Scenario A:** 200 concurrent identical requests (upstream latency 50ms)

| metric | value |
|---|---|
| client requests | 200 |
| request errors | 0 |
| upstream computations of the input | 1 |

- **PASS** — singleton coalescing: 200 concurrent requests -> 1 upstream computation(s)

**Scenario B:** 100 concurrent batches of 8, drawn from 20 unique inputs (800 items total)

| metric | value |
|---|---|
| items requested | 800 |
| unique inputs | 20 |
| items computed upstream | 20 |
| request errors | 0 |

- **PASS** — batch coalescing: 800 requested items -> 20 upstream computations (ideal: 20)

## E3 — Proxy overhead and throughput

**Claim tested:** the proxy's added latency is negligible next to a real embedding
call (typically 10–100ms upstream).

**Method:** sequential single-input requests on loopback with a zero-latency mock;
direct-to-mock is the baseline. Numbers below are wall-clock per request on this
machine (Windows loopback stack) — treat them as upper bounds, not benchmarks.

| path (1500 sequential reqs) | p50 ms | p95 ms | p99 ms |
|---|---|---|---|
| direct to mock upstream | 0.00 | 0.62 | 4.31 |
| proxy, cache hit | 0.00 | 0.55 | 0.64 |
| proxy, cache miss (adds one upstream hop) | 0.53 | 0.64 | 1.08 |

Added p50 latency: **0.00 ms on a hit**, **0.53 ms on a miss** (miss includes a second
loopback round-trip to the upstream, which a real deployment pays anyway).

Sustained cache-hit throughput, 32 concurrent clients, 5s: **65241 req/s**.

- **PASS** — hit overhead under 5ms p50: hit adds 0.00 ms at p50
- **PASS** — throughput sane: 65241 cache-hit req/s on loopback

## E4 — RAG re-ingestion: where dedupe saves money (and where it honestly doesn't)

**Claim tested:** re-ingesting a corpus after incremental document changes only pays
for what changed. **Honest counter-test:** changing the chunking configuration itself
produces different chunk text, so an exact-match cache saves ~nothing — that scenario
needs the roadmap's chunk-diff engine, and we say so rather than hide it.

| pass | items ingested | items paid for upstream | saved |
|---|---|---|---|
| 1 — cold ingest | 10000 | 10000 | 0% |
| 2 — 5% of docs edited, full re-ingest | 10000 | 475 | **95.2%** |
| 3 — chunking config changed | 10000 | 10000 | 0.0% |

Pass 3 is the documented limitation: exact-match dedupe cannot absorb a re-chunk;
that is the Phase-2 chunk-diff problem, not a cache problem.

- **PASS** — cold ingest pays full price: 10000/10000 items upstream
- **PASS** — incremental re-ingest pays only for changes: re-ingest of 10000 items sent only 475 upstream (95.2% saved; 500 items belong to edited docs)
- **PASS** — re-chunk honestly saves nothing: 10000/10000 items had to be recomputed after re-chunking

## E5 — Query workload (Zipf) + offline analyzer cross-check

**Claim tested:** on a skewed query distribution (few hot queries, long tail — the
shape of real search/RAG traffic), exact-match caching absorbs a large share of
embedding calls; and `embedcache analyze` on the request log reports the same waste
the live proxy actually avoided.

| metric | value |
|---|---|
| queries | 20000 |
| unique pool | 3000 |
| upstream computations | 2052 |
| cache hit rate | **89.7%** |

- **PASS** — skewed workload hit rate: 89.7% of queries served from cache; upstream saw 2052 of 20000 items
- **PASS** — upstream items equal unique queries seen: upstream 2052 == queries 20000 - hits 17948

Offline analyzer on the same request log: 20000 items, 2052 unique, 17948 duplicates (89.7%).

- **PASS** — analyzer agrees with live proxy: analyzer found 17948 duplicates; live cache served 17948 hits
