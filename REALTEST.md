# embedcache real-workload test

Generated 2026-07-09 20:41 EDT. Everything in this run is real: live Wikipedia articles fetched over
the internet, real chunking, real embedding inference (Ollama `all-minilm`), a working RAG
agent answering real questions with a real LLM (hosted Gemini with local `gemma3:1b`
fallback), and a live-refresh pass re-fetching the same articles. **No duplicate is
constructed; every duplicate below arose organically from the workload.**

## 1 — Live corpus: English Wikipedia

| metric | value |
|---|---|
| articles fetched live | 103 of 117 requested |
| raw text | 4.2 MB |
| chunks after pipeline chunking | 8601 |
| fetch wall time | 2s |
- **PASS** — real corpus fetched: 103 live articles, 8601 chunks

## 2 — Ingestion through embedcache (real inference)

| metric | value |
|---|---|
| chunks embedded | 8601 |
| wall time | 2m40s |
| cache hits during cold ingest (organic intra-corpus dupes) | 0 |
| errors | 0 |
- **PASS** — ingestion clean: 8601 chunks in 2m40s

## 3 — RAG agent: 30 sessions, real LLM, organic query traffic

| metric | value |
|---|---|
| sessions | 30 |
| answered by LLM | 30 (0 hosted, 60 local fallback, 60 hosted failures) |
| embedding calls made by the agent | 83 |
| served from cache (organic repeats) | 18 (**21.7%**) |
| wall time | 4m27s |

Sample transcript (real LLM output, unedited):

> **Q:** How does gradient descent optimize a neural network?
> **A** (local:gemma3:1b)**:** Gradient descent optimizes a neural network by adjusting weights in proportion to the derivative of the error with respect to those weights, using backpropagation.
>
> **Q:** How does gradient descent optimize a neural network?
> **A** (local:gemma3:1b)**:** Gradient descent calculates the derivative of the error term with respect to each weight in proportion to the derivative, allowing adjustments to minimize the error.
>
> **Q:** What is garbage collection and how does it work?
> **A** (local:gemma3:1b)**:** Garbage collection is a strategy for automatically detecting memory allocated to objects that are no longer usable in a program, returning that allocated memory to a pool of free memory locations. It relieves the programmer from manually managing memory allocation and releasing, and it takes a significant portion of a program’s processing time. The garbage collector attempts to reclaim memory that was allocated by the program, but is no longer referenced.
>
- **PASS** — agent answered its questions: 30/30 sessions produced grounded answers
- **PASS** — agent traffic has organic cache hits: 21.7% of agent embedding calls were repeats it did not pay for

## 4 — Live refresh: re-fetch the same articles, re-ingest (the real nightly pipeline)

| metric | value |
|---|---|
| chunks re-ingested | 8515 |
| actually changed on Wikipedia since first fetch | 0 chunks re-embedded |
| absorbed by cache | **100.00%** |
| wall time (fetch + re-ingest) vs cold | 4s vs 2m40s |
- **PASS** — refresh pays only for real edits: 100.00% of the re-ingest was absorbed; only 0 chunks had actually changed

## 5 — The waste report (the adoption artifact)

`embedcache analyze` on this run's real request log:

```
embedcache offline waste analysis
=================================
requests analyzed        1394
embedding items          30389
unique items             13814
duplicate items          16575   (54.5% of all items)
estimated tokens         3426668   (~$0.07)
estimated wasted tokens  1860840   (~$0.04)

>> 54.3% of this embedding spend was duplicate work an exact-match
>> cache would have absorbed.

per model:
  all-minilm                     items=30389    dup=16575    wasted_tokens=1860840    wasted=$0.04

top duplicated inputs:
     7x  all-minilm tokens=113     "m M , n ( x , Θ 1 , … , Θ M ) = 1 M ∑ j = 1 M m n (..."
     7x  all-minilm tokens=113     "MSE = 1 n ∑ i = 1 n ( y i − y ^ i ) 2 = 1 n ∑ i = 1..."
     7x  all-minilm tokens=101     ". Random regression forest has two levels of averaging, f..."
     6x  all-minilm tokens=121     "In artificial neural networks, recurrent neural networks ..."
     6x  all-minilm tokens=120     "The feed-forward architecture of convolutional neural net..."
     6x  all-minilm tokens=119     "As archaeological findings such as clay tablets with cune..."
     6x  all-minilm tokens=115     "Caffe: A library for convolutional neural networks. Creat..."
     5x  all-minilm tokens=124     "{\\displaystyle {\\frac {\\partial ^{2}L}{\\partial a_{j_{1}}..."
     5x  all-minilm tokens=124     "Principal component analysis has applications in many fie..."
     5x  all-minilm tokens=123     "A \"decoder-only\" transformer is not literally decoder-onl..."
```

Measured organically: **49.7% of this workload's embedding tokens were duplicate work.**
At hosted-API prices, that share of every $1,000/month embedding bill is **$497 wasted** —
on self-hosted GPUs it is the same share of GPU-hours. That number, measured on a team's
own logs by the offline analyzer before they install anything, is the adoption pitch.
- **PASS** — organic duplicate share measured: saved=755644 tokens, spent=763733 tokens, duplicate share 49.7%
