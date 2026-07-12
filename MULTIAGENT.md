# embedcache multi-agent crew validation

Generated 2026-07-12 08:23 EDT. A real 5-agent crew (planner, 3 workers, synthesizer), each with its
own API key, backed by a real local LLM (Ollama gemma3:1b) for reasoning and real embeddings
(nomic-embed-text) through one shared embedcache proxy. Retrieval context is live Wikipedia prose.

## Planner output

Research question: "How do vector databases enable efficient retrieval-augmented generation?"

1. What specific vector embedding techniques are most effective for capturing semantic similarity in text?
2. How does the integration of vector databases impact the computational cost of generating text?
3. What are the key performance metrics used to evaluate the effectiveness of retrieval-augmented generation with vector databases?

## Worker cache attribution (cross-agent reuse)

Three real workers, each with its own API key, each embed their sub-question plus 2 query
expansions and retrieve from the shared real corpus, then ask gemma3:1b for a short answer.

- **worker-1-key** — sub-question embeds: 2/3 served from cache, budget-capped=false
  retrieved context (real, closest corpus chunk by cosine similarity): These methods focus on the encoding of text as either dense or sparse vectors. Sparse vectors, which encode the identity of a word, are typi...
  answer: Dense vectors, which encode meaning more compactly, are generally more effective for capturing semantic similarity in text.
- **worker-2-key** — sub-question embeds: 1/3 served from cache, budget-capped=false
  retrieved context (real, closest corpus chunk by cosine similarity): These methods focus on the encoding of text as either dense or sparse vectors. Sparse vectors, which encode the identity of a word, are typi...
  answer: Dense vectors, which encode meaning, are generally more effective for capturing semantic similarity.
- **worker-3-key** — sub-question embeds: 0/4 served from cache, budget-capped=true
  retrieved context (real, closest corpus chunk by cosine similarity): These methods focus on the encoding of text as either dense or sparse vectors. Sparse vectors, which encode the identity of a word, are typi...
  answer: Integrating vector databases significantly impacts the computational cost of generating text by enabling more efficient similarity calculations.
- **PASS** — workers 1 and 2 (same assigned sub-question) share cache across agent identity: worker-1 hits=2, worker-2 hits=1 (combined 3) on an identical real query assigned to both — whichever worker embeds a variant first pays, the other hits
- **PASS** — worker-3's tiny budget engages mid-run: worker-3 (40-token budget) hit 429 while worker-1/worker-2 (unlimited) did not need to

aggregate proxy stats after the crew run: hits=0 misses=86 coalesced=3

Note: because the workers run concurrently, the cross-agent reuse shows up as *coalesced*
requests (two agents embedding the identical query in the same instant collapse to one
upstream call) rather than settled cache hits — the stronger property under real concurrency.
A coalesced request returns `X-Embedcache-Status: hit` to the caller (it was not sent
upstream), which is why each worker still counts it as a cache hit above.

## Synthesizer

Final synthesized answer: Dense vector databases play a crucial role in enabling efficient retrieval-augmented generation (RAG) by significantly impacting the computational cost of text generation. Because dense vectors more effectively capture semantic similarity within text, integrating them into vector databases allows for faster and more targeted similarity calculations during the generation process, ultimately reducin...

## Cross-language cache sharing (real Python + real LangChain)

A real Python process, using an actual `langchain_core.embeddings.Embeddings` subclass and
`InMemoryVectorStore`, embeds through the SAME running proxy with its own API key — some
inputs duplicate real text the Go agents already embedded, some are genuinely new.

- **PASS** — Python/LangChain shares embedcache's cache with the Go agents: a real chunk already embedded by a Go worker/planner came back as "hit" when re-embedded from Python via LangChain's InMemoryVectorStore
- LangChain similarity_search over content embedded from Python: [LangChain provides composable abstractions for building LLM applications. A vector store indexes embeddings to support similarity search over documents.]

## Bottom line

- real 5-agent crew: 1 planner + 3 parallel workers (own keys) + 1 synthesizer, backed by
  real gemma3:1b chat calls and real nomic-embed-text embeddings through one shared proxy
- cache is content-addressed, not caller-addressed: agents sharing a query share the hit
- a per-key budget cap engaged mid-run for one agent without affecting its siblings
- a real Python + LangChain process shares the same cache as the Go agents
