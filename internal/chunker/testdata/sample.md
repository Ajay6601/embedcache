# embedcache

**[Website](https://ajay6601.github.io/embedcache/) · [Documentation](https://ajay6601.github.io/embedcache/docs.html) · [Examples](https://ajay6601.github.io/embedcache/examples.html)**

![embedcache demo - the offline waste analyzer finding duplicate embedding spend](docs/assets/demo.gif)

**The cost-control and dedupe layer for embedding APIs.** A single-binary proxy that sits in front of any OpenAI-compatible embedding backend - vLLM, Ollama, text-embeddings-inference, or api.openai.com - and eliminates duplicate work before it reaches your GPUs or your API bill. Then it hands you the invoice for what it saved.

```
your app ──► embedcache ──► vLLM / Ollama / TEI / OpenAI
                 │
                 ├─ exact-match cache (byte-exact replays, content-addressed)
                 ├─ in-flight coalescing (N concurrent identical calls → 1 upstream)
                 ├─ batch dedupe (only the uncached items go upstream)
                 └─ waste report ($X of your embedding spend was duplicates)
```

Zero dependencies - pure Go stdlib. One static binary. No Python runtime, no Redis required, no framework lock-in.

## Why this exists

Embedding workloads are full of silent duplicate work: re-ingesting a corpus where 95% of chunks didn't change, hot queries embedded thousands of times, retry storms, multiple services embedding the same strings. Frameworks solve this *inside* Python (LangChain's RecordManager, LlamaIndex's IngestionPipeline) - useless if your pipeline is Go, TypeScript, custom, or split across teams. Gateways treat it as a checkbox: LiteLLM's embedding cache has a known bug where mixed cached/uncached batches can return wrong vectors.

embedcache does one thing, language-agnostically, and provably correctly. Three tiers of evidence live in this repo:

- EXPERIMENTS.md - controlled validation against a deterministic mock (re-run by CI on every push)
- PRODSIM.md - production-scale simulation: 50k-chunk corpus, 300k-request storm at 39k req/s, hostile-traffic mix, crash recovery, all security features on
- REALTEST.md - a real workload with zero constructed duplicates: a working RAG agent over 103 live Wikipedia articles, real LLM answers, and a live-refresh pass. Measured organically: 49.7% of the workload's embedding tokens were duplicate work - 21.7% of the agent's query traffic repeated, and the refresh pass re-embedded a corpus where nothing had changed and was 100% absorbed.

| Claim | Evidence |
|---|---|
| Byte-exact responses (hits identical to upstream, correct batch index mapping) | 3,453 randomized-batch embeddings verified, 0 mismatches - includes the LiteLLM #22659 failure class |
| Coalescing | 200 concurrent identical requests → 1 upstream computation; 800 overlapping batch items → 20 (the ideal) |
| Negligible overhead | ~0.0 ms added p50 on a hit; 65,891 cache-hit req/s on loopback (32 clients) |
| Incremental re-ingest savings | 10,000-chunk corpus, 5% of docs edited, full re-ingest: 95.2% of embedding calls absorbed |
| Honest limits | Changing chunking config re-embeds everything (0% saved) - exact-match can't fix that; a chunk-diff engine (roadmap) can |
| Waste report accuracy | Offline analyzer found exactly the 17,948 duplicates the live cache absorbed on the same 20k-query workload |

## Quickstart

```bash
go build -o embedcache ./cmd/embedcache

# in front of a self-hosted engine
./embedcache serve -upstream http://localhost:8000        # vLLM
./embedcache serve -upstream http://localhost:11434       # Ollama

# in front of OpenAI
./embedcache serve -upstream https://api.openai.com
```

Point your SDK at it - no code changes:

```python
client = OpenAI(base_url="http://localhost:8090/v1", api_key="...")
client.embeddings.create(model="text-embedding-3-small", input=["hello"])
```

Your Authorization header is forwarded to the upstream (or pin one with -upstream-api-key). Non-embedding routes pass through untouched, so it can front a full OpenAI-compatible server.

Verified live (all experiments/livecheck assertions pass - byte-exact replay, mixed-batch mapping, coalescing, base64):

| backend | upstream flag | verified model |
|---|---|---|
| Ollama 0.22 | -upstream http://localhost:11434 | all-minilm |
| Google Gemini | -upstream https://generativelanguage.googleapis.com/v1beta/openai -upstream-api-key $GEMINI_API_KEY | gemini-embedding-001 (3072-dim) |
| OpenAI | -upstream https://api.openai.com | wire-identical to the above |

Wire-compatible (OpenAI-shaped APIs, covered by unit tests): Voyage AI (-upstream https://api.voyageai.com - the provider Anthropic officially recommends for embeddings; input_type and other Voyage params are forwarded verbatim and part of the cache identity, so query- and document-typed vectors never collide), Mistral (-upstream https://api.mistral.ai, mistral-embed), and Azure OpenAI (deployment paths, ?api-version= query and api-key header are mirrored upstream).

Not applicable: Groq and the Anthropic API itself - neither offers an embeddings endpoint (Anthropic recommends Voyage, above).

Every response tells you what happened:

```
X-Embedcache-Status: hit | miss | partial
X-Embedcache-Hits: 14
X-Embedcache-Saved-Tokens: 1830
```

and usage.prompt_tokens reflects only what this request was actually billed upstream.

## The waste report

Before installing anything, run the offline analyzer on an existing request log (JSONL, one embedding request per line - raw bodies or wrapped under body/request/payload):

```bash
./embedcache analyze requests.jsonl
```

```
embedcache offline waste analysis
=================================
requests analyzed        20000
embedding items          20000
unique items             2052
duplicate items          17948   (89.7% of all items)
estimated tokens         340217   (~$0.01)
estimated wasted tokens  305090   (~$0.01)

>> 89.7% of this embedding spend was duplicate work an exact-match
>> cache would have absorbed.
```

For self-hosted models with no list price, set an amortized GPU cost: -default-price 0.05 ($/1M tokens), or supply -pricing costs.json.

A running proxy serves the same accounting live: embedcache report, or GET /_ec/stats (JSON), /_ec/report (text), /metrics (Prometheus).

## Correctness posture

- Byte-exact by default. Cached responses replay the upstream's raw JSON - no float re-encoding, no precision drift. model, dimensions, and encoding_format are all part of the cache key.
- Normalization is opt-in. -normalize trim,collapse,lowercase widens matching if you accept the semantics; the default matches exact bytes only.
- No semantic caching. Similarity-threshold response caching returns wrong answers at some rate; that trade-off belongs to chat responses, not embeddings. Exact-match embedding caching has a 0% wrong-answer rate by construction.
- Failure containment. Upstream errors pass through verbatim and are never cached; a crashed leader fails its coalesced waiters instead of hanging them.

## Security

For anything beyond a trusted network, turn on both layers:

```bash
embedcache serve -upstream http://localhost:8000 \
  -admin-token  "$(openssl rand -hex 24)" \
  -auth-mode    allowlist -api-keys-file keys.txt \
  -ttl          720h
```

- -admin-token - without it, anyone who can reach the port can flush your cache and read usage stats. /healthz stays open for probes.
- -auth-mode - cache hits never touch the upstream, so without this a revoked or missing key can still read cached vectors. allowlist is for self-hosted backends (the proxy owns the key list); verify is for hosted providers - the caller's key is checked against the upstream (GET /v1/models) and the verdict cached for -auth-cache-ttl (default 5m, negative verdicts 1m, fail-closed on upstream outage).
- Terminate TLS in front (Caddy, nginx, or your ingress); embedcache listens in plaintext.

## Resilience

Embedding calls are idempotent, so transient upstream failures (network errors, 5xx, 429) are retried with exponential backoff - -upstream-retries, honoring Retry-After. Sustained failures trip a circuit breaker (-breaker-threshold consecutive failures) and requests fail fast with 503 instead of stacking timeouts on a dead backend; after -breaker-cooldown a single probe decides whether to close it. Cache hits keep serving while the circuit is open - an upstream outage degrades misses, not the whole service. The request log rotates at -request-log-max-mb so it cannot fill the disk. Breaker state, retries, and fast-fails are exported at /metrics.

## Serve flags

| flag | default | |
|---|---|---|
| -listen | :8090 | bind address |
| -upstream | - | required; base URL of the backend |
| -upstream-api-key | forward client's | pin an upstream key |
| -admin-token | open | bearer token for admin endpoints |
| -auth-mode | off | client key validation: allowlist / verify |
| -api-keys / -api-keys-file | - | client keys for allowlist mode |
| -auth-cache-ttl | 5m | how long a verified key stays trusted |
| -ttl | never | expire cached embeddings |
| -max-batch-items | 2048 | reject oversized batches |
| -max-body-mb | 64 | reject oversized bodies |
| -max-entries / -max-memory-mb | 1M / 1024 | LRU bounds |
| -upstream-retries | 2 | retries for transient upstream failures |
| -breaker-threshold / -breaker-cooldown | 5 / 10s | circuit breaker |
| -normalize | off | trim,collapse,lowercase |
| -persist | off | snapshot file; survives restarts |
| -request-log | off | JSONL log, feeds analyze |
| -request-log-max-mb | 512 | rotate the request log at this size |
| -pricing | built-in | JSON {"model": $/1M, "default": ...} |

Admin: POST /_ec/flush empties the cache (use when a model's weights change under the same name).

## Roadmap

1. Chunk-diff engine - deterministic chunk fingerprints that survive re-chunking, so pipeline config changes stop meaning full re-embeds (the one scenario E4 shows exact-match can't absorb).
2. Cost enforcement - per-key/per-tenant token budgets with hard limits, agent-loop guards.
3. TTLs & invalidation rules per model/route.

## Development

```bash
go test ./...                                          # unit + integration tests
go test -race ./...                                    # requires cgo (runs in CI)
go run ./experiments/harness -bin ./embedcache.exe     # regenerates EXPERIMENTS.md
```

The experiments harness starts the real binary against a deterministic mock upstream and fails the build if any correctness, coalescing, overhead, or savings assertion regresses.

## License

MIT
