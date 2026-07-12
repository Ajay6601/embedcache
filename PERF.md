# embedcache open-loop latency benchmark

Generated 2026-07-12 08:17 EDT · windows/amd64 · 16 CPUs · real local Ollama backend, no synthetic timing.

Constant-rate (open-loop) load: a ticker schedules request dispatch on a fixed
schedule regardless of completion, so tail latency under saturation is captured
honestly instead of being hidden by a closed-loop worker pool (coordinated omission).

## all-minilm (384-dim)

| scenario | target rate | completed | errors | p50 | p90 | p99 | p99.9 |
|---|---|---|---|---|---|---|---|
| miss (real compute, unique input) | 93/s | 1121 | 0 | 35.5ms | 40.3ms | 96.0ms | 108.8ms |
| hit (cached, repeated input) | 4000/s | 20755 | 0 | 0.0ms | 0.0ms | 0.8ms | 2.6ms |
| direct to backend (no proxy) | 93/s | 1118 | 0 | 34.5ms | 38.0ms | 67.7ms | 90.9ms |

- proxy overhead at p50 (miss vs direct, both real upstream compute): 1.08ms

## nomic-embed-text (768-dim)

| scenario | target rate | completed | errors | p50 | p90 | p99 | p99.9 |
|---|---|---|---|---|---|---|---|
| miss (real compute, unique input) | 32/s | 378 | 0 | 48.2ms | 52.1ms | 72.6ms | 90.7ms |
| hit (cached, repeated input) | 3793/s | 21399 | 0 | 0.0ms | 0.0ms | 1.1ms | 2.6ms |
| direct to backend (no proxy) | 32/s | 382 | 1 | 48.0ms | 53.7ms | 87.5ms | 91.2ms |

- proxy overhead at p50 (miss vs direct, both real upstream compute): 0.19ms

## mxbai-embed-large (1024-dim)

| scenario | target rate | completed | errors | p50 | p90 | p99 | p99.9 |
|---|---|---|---|---|---|---|---|
| miss (real compute, unique input) | 12/s | 138 | 0 | 79.3ms | 87.8ms | 100.6ms | 102.3ms |
| hit (cached, repeated input) | 1701/s | 19472 | 0 | 0.0ms | 0.0ms | 1.2ms | 4.3ms |
| direct to backend (no proxy) | 13/s | 150 | 0 | 79.1ms | 102.6ms | 283.0ms | 286.2ms |

- proxy overhead at p50 (miss vs direct, both real upstream compute): 0.28ms

## Method

- All traffic is open-loop (constant-rate via ticker), not closed-loop worker-pool polling —
  this avoids coordinated omission, which understates tail latency under saturation.
- Per-scenario rate is calibrated from real measured closed-loop throughput (70% of sustained
  capacity), not a guessed constant, so the target rate is realistic for the actual hardware.
- Only local Ollama backends are covered here (no hosted API key in this environment); the
  proxy-overhead number isolates embedcache's own cost from backend inference time.
