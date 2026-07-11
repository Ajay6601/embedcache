# embedcache production simulation

Generated 2026-07-09 19:47 EDT · windows/amd64 · 16 CPUs · backend http://localhost:11434 (`all-minilm`, real inference)

Security fully enabled throughout: `-auth-mode allowlist`, `-admin-token`, `-ttl 24h`,
persistence on. Corpus 50000 chunks (2500 docs × 20), query storm 300000 requests / 64 clients.

## Phase 1 - Cold corpus ingestion (50000 chunks, real inference)

| metric | value |
|---|---|
| chunks ingested | 50000 |
| wall time | 7m16s |
| embeddings/sec (real inference) | 115 |
| batch latency p50 / p95 | 8813 ms / 9463 ms |
| request errors | 0 |
- **PASS** - cold ingest completes with zero errors: 50000 chunks in 7m16s
- **PASS** - cold ingest is all misses: misses=50000

## Phase 2 - Query storm (300000 requests, 64 concurrent clients, Zipf)

| metric | value |
|---|---|
| requests | 299968 |
| wall time | 8s |
| sustained throughput | **39184 req/s** |
| hit rate | **100.00%** |
| latency p50 / p95 / p99 | 1.25 / 4.06 / 6.93 ms |
| errors | 0 |
- **PASS** - query storm zero errors: 299968 requests
- **PASS** - query storm hit rate > 95%%: 100.00% (pool drawn from ingested corpus)

## Phase 3 - Nightly re-ingestion (5% of documents edited)

| metric | value |
|---|---|
| chunks re-ingested | 50000 |
| recomputed upstream | 2500 |
| absorbed by cache | **95.0%** |
| wall time vs cold ingest | 23s vs 7m16s (**18.7x faster**) |
- **PASS** - re-ingest zero errors: 50000 chunks
- **PASS** - re-ingest recomputes only edited docs: 95.0% absorbed; 125 edited docs = 2500 chunks expected

## Phase 4 - Security under load (25% hostile keys mixed into live traffic)

| metric | value |
|---|---|
| requests | 29984 (40364/s) |
| hostile rejected with 401 | 7561 |
| legitimate served | 22423 |
| legit hit latency p50 under attack | 0.63 ms |
- **PASS** - every hostile request rejected, every legit served: 401=7561 200=22423 other=0

## Phase 5 - Byte-exact correctness after 429968 requests

- **PASS** - 500 sampled embeddings still byte-exact: 500 verified against ground truth captured at ingest, 0 mismatches

## Phase 6 - Snapshot, hard kill, restart, cache survives

- **PASS** - snapshot endpoint works under admin token: {"snapshotted":52500} (251 MB on disk)
- **PASS** - cache survives a hard crash: 500/500 sampled entries served as byte-exact hits after restart

## Bottom line

Across the whole simulation:

- total embedding items served: **399968+** (ingest ×2 + storm + attack traffic)
- final hit rate in the restarted proxy's window: 100.0%, saved tokens: 20850
- zero errors, zero wrong bytes, zero hostile requests served, across ~429k requests
