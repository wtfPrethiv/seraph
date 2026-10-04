| component | dev NDCG@10 | dev R@100 | dev_stdin NDCG@10 | dev_stdin R@100 | p50 latency |
|---|---|---|---|---|---|
| BM25 (code-aware tokenizer) | 0.3714 | 0.6120 | 0.0941 | 0.2844 | 0.6 ms |
| dense only (Qwen3-Embedding-0.6B, cached vectors) | 0.8367 (+0.4653) | 0.9800 | 0.7220 (+0.6279) | 0.9701 | 0.3 ms |
| + fusion (weighted minmax, lexical 0.2 / semantic 0.8) | 0.8473 (+0.0106) | 0.9810 | 0.7343 (+0.0123) | 0.9731 | 1.9 ms |
| + structural gamma view (0.2) | 0.8530 (+0.0057) | 0.9830 | 0.7397 (+0.0054) | 0.9731 | 5.2 ms |
| + adaptive weights (rule-based) | 0.8498 (-0.0032) | 0.9820 | 0.7420 (+0.0023) | 0.9731 | 5.7 ms |
| + MMR (lambda 0.85), on the structural config | 0.8541 (+0.0011) | 0.9830 | 0.7400 (+0.0003) | 0.9731 | 7.3 ms |
| + Gemini listwise rerank (depth 20), on the structural config | n/a | | 0.8060 (+0.0664) | 0.9731 | 4288.3 ms |
| + graph expansion (delta) | n/a | | n/a | | AppsRetrieval snippets have no call graph; evaluated on SeraphBench |
| + lineage dedup / evolution view (epsilon) | n/a | | n/a | | AppsRetrieval has a single version per snippet; evaluated on SeraphBench |
