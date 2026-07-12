# embedcache semantic caching: shadow-mode measurement (real backend)

Generated 2026-07-12 23:22 EDT. Semantic caching in **shadow** mode against real Ollama (`nomic-embed-text`), threshold 0.80.
Nothing approximate is served here — the proxy computes every input for real and only records how
far each near-neighbor's cached vector is from the truth. That cosine is the number that decides
whether **active** mode would be safe on this kind of traffic.

Sent 12 real near-duplicate variants of 4 base questions.

| shadow observations | real-vs-neighbor cosine (mean) | worst case |
|---|---|---|
| 6 | **0.9824** | 0.9386 |

The worst near-duplicate was only cosine **0.9386** from the truth. That is enough drift that active
mode could shift some rankings — raise the threshold or keep it in shadow before enabling.

This is the whole point of shadow mode: the decision to enable approximate serving is made from
your own measured cosine distribution, not from a vendor's blanket claim.
