# embedcache open-loop latency benchmark

Generated 2026-07-12 08:49 EDT · windows/amd64 · 16 CPUs · real local Ollama backend, no synthetic timing.

Constant-rate (open-loop) load: a ticker schedules request dispatch on a fixed
schedule regardless of completion, so tail latency under saturation is captured
honestly instead of being hidden by a closed-loop worker pool (coordinated omission).

## all-minilm (384-dim)

| scenario | target rate | completed | errors | p50 | p90 | p99 | p99.9 |
|---|---|---|---|---|---|---|---|
| miss (real compute, unique input) | 95/s | 1138 | 0 | 34.2ms | 38.5ms | 83.1ms | 91.1ms |
| hit (cached, repeated input) | 4000/s | 21007 | 0 | 0.0ms | 0.0ms | 0.6ms | 2.4ms |
| direct to backend (no proxy) | 96/s | 1153 | 0 | 34.6ms | 39.4ms | 76.2ms | 88.3ms |

- proxy overhead at p50 (miss vs direct, both real upstream compute): -0.44ms

| sustainable throughput (calibrated, closed-loop) | req/s |
|---|---|
| backend miss (real compute) | 136 |
| direct to backend (no proxy) | 137 |
| **cache hit** | **9234** |

- a cache hit serves **68×** faster than the backend computes a miss

## nomic-embed-text (768-dim)

| scenario | target rate | completed | errors | p50 | p90 | p99 | p99.9 |
|---|---|---|---|---|---|---|---|
| miss (real compute, unique input) | 34/s | 405 | 0 | 47.2ms | 50.0ms | 61.7ms | 83.8ms |
| hit (cached, repeated input) | 3403/s | 20698 | 0 | 0.0ms | 0.0ms | 1.2ms | 2.9ms |
| direct to backend (no proxy) | 33/s | 397 | 3 | 46.2ms | 50.0ms | 68.4ms | 85.3ms |

- proxy overhead at p50 (miss vs direct, both real upstream compute): 1.08ms

| sustainable throughput (calibrated, closed-loop) | req/s |
|---|---|
| backend miss (real compute) | 48 |
| direct to backend (no proxy) | 48 |
| **cache hit** | **4861** |

- a cache hit serves **101×** faster than the backend computes a miss

## mxbai-embed-large (1024-dim)

| scenario | target rate | completed | errors | p50 | p90 | p99 | p99.9 |
|---|---|---|---|---|---|---|---|
| miss (real compute, unique input) | 12/s | 138 | 0 | 79.6ms | 86.8ms | 104.3ms | 110.4ms |
| hit (cached, repeated input) | 1409/s | 16406 | 0 | 0.0ms | 0.0ms | 1.1ms | 2.8ms |
| direct to backend (no proxy) | 13/s | 146 | 0 | 79.1ms | 89.8ms | 293.2ms | 402.8ms |

- proxy overhead at p50 (miss vs direct, both real upstream compute): 0.50ms

| sustainable throughput (calibrated, closed-loop) | req/s |
|---|---|
| backend miss (real compute) | 17 |
| direct to backend (no proxy) | 19 |
| **cache hit** | **2013** |

- a cache hit serves **122×** faster than the backend computes a miss

## Method

- All traffic is open-loop (constant-rate via ticker), not closed-loop worker-pool polling —
  this avoids coordinated omission, which understates tail latency under saturation.
- Per-scenario rate is calibrated from real measured closed-loop throughput (70% of sustained
  capacity), not a guessed constant, so the target rate is realistic for the actual hardware.
- Only local Ollama backends are covered here (no hosted API key in this environment); the
  proxy-overhead number isolates embedcache's own cost from backend inference time.
