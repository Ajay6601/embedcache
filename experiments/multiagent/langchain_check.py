"""Real cross-language proof: a genuine langchain_core.embeddings.Embeddings
subclass, used with a real InMemoryVectorStore, calling embedcache's
/v1/embeddings over HTTP with its own API key. Proves the cache is shared
across languages/frameworks, not just across Go callers.

Not a mock: this makes actual HTTP requests to a running embedcache proxy
(started by experiments/multiagent's Go harness) and uses LangChain's own
vector store implementation for the similarity search.
"""
import argparse
import json
import sys

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore


class EmbedcacheEmbeddings(Embeddings):
    """A real LangChain Embeddings implementation backed by embedcache."""

    def __init__(self, base, key, model):
        self.base = base
        self.key = key
        self.model = model
        self.last_status = None  # X-Embedcache-Status of the most recent call

    def _call(self, inputs):
        resp = requests.post(
            f"{self.base}/v1/embeddings",
            json={"model": self.model, "input": inputs},
            headers={"Authorization": f"Bearer {self.key}"},
            timeout=60,
        )
        resp.raise_for_status()
        self.last_status = resp.headers.get("X-Embedcache-Status")
        data = resp.json()["data"]
        return [d["embedding"] for d in data]

    def embed_documents(self, texts):
        return self._call(texts)

    def embed_query(self, text):
        return self._call([text])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--duplicate", required=True, help="real text already embedded by a Go agent")
    args = ap.parse_args()

    emb = EmbedcacheEmbeddings(args.base, args.key, args.model)

    # 1. re-embed the duplicate (a real chunk the Go crew already embedded
    #    under different API keys) — this is the cross-language/cross-agent
    #    cache-sharing proof
    emb.embed_query(args.duplicate)
    duplicate_was_hit = emb.last_status == "hit"

    # 2. genuinely new content, added via a real LangChain vector store
    new_docs = [
        "LangChain provides composable abstractions for building LLM applications.",
        "A vector store indexes embeddings to support similarity search over documents.",
        args.duplicate,  # also included in the corpus so similarity_search can find it
    ]
    store = InMemoryVectorStore(emb)
    store.add_texts(new_docs)
    results = store.similarity_search("composable LLM application building blocks", k=2)

    print(json.dumps({
        "duplicate_was_hit": duplicate_was_hit,
        "search_results": [r.page_content[:80] for r in results],
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"duplicate_was_hit": False, "search_results": [], "error": str(e)}))
        sys.exit(1)
