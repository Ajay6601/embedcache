This directory holds the BEIR SciFact dataset used by `experiments/benchmark`.
It is not checked into git (see `.gitignore`) — fetch it yourself:

```
curl -o scifact.zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
unzip scifact.zip
```

Expected layout after unzip: `scifact/corpus.jsonl`, `scifact/queries.jsonl`,
`scifact/qrels/test.tsv`. Dataset: Thakur et al., "BEIR: A Heterogeneous
Benchmark for Zero-shot Evaluation of Information Retrieval Models" (2021).
