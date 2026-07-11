# chunk-diff engine: real-data validation

Generated 2026-07-10 18:46 EDT. Corpus: one live Wikipedia article ("Retrieval-augmented generation"). Edit: one realistic
single-sentence insertion - no synthetic or bulk-generated text. All embeddings
computed by real Ollama (`all-minilm`) through the real embedcache proxy.

Article length: 10903 bytes. Edit: +68 bytes at one point.

## Fixed-size chunking (naive baseline, 800-byte blocks)

| metric | value |
|---|---|
| v1 chunks (cold ingest) | 14 |
| v2 chunks (after the edit) | 14 |
| v2 cache hits | 4 |
| **hit rate on re-ingest after one edit** | **28.6%** |

## Content-defined chunking (embedcache's chunker)

| metric | value |
|---|---|
| v1 chunks (cold ingest) | 16 |
| v2 chunks (after the edit) | 16 |
| v2 cache hits | 15 |
| **hit rate on re-ingest after one edit** | **93.8%** |

## Bottom line

Same real article, same single realistic edit: content-defined chunking absorbed
**93.8%** of the re-ingest from cache; fixed-size chunking absorbed only **28.6%**.
- **PASS** - content-defined chunking beats fixed-size on a real edit: cdc=93.8% fixed=28.6% on real article "Retrieval-augmented generation"
- **PASS** - content-defined chunking absorbs most of the re-ingest: 93.8% hit rate after a single-sentence edit
