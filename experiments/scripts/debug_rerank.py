"""Sanity-check a reranker: device, throughput, and gold-vs-random score separation on dev_stdin."""

import random
import sys
import time

from seraph.evaluation.coir import load_apps
from seraph.retrieval.reranker import head_tail, load_reranker

name = sys.argv[1] if len(sys.argv) > 1 else "bge-reranker-v2-m3"
data = load_apps()
s = data.split("dev_stdin")
r = load_reranker(name)
model = getattr(r, "model", None)
print("device:", getattr(model, "device", None), "dtype:", next(model.parameters()).dtype if hasattr(model, "parameters") else getattr(getattr(model, "model", None), "dtype", None))

rng = random.Random(0)
docs = list(data.corpus)
pairs, labels = [], []
for q in list(s.queries)[:16]:
    gold = next(iter(s.qrels[q]))
    for d, lab in [(gold, 1)] + [(rng.choice(docs), 0) for _ in range(3)]:
        pairs.append((head_tail(s.queries[q], 3000), head_tail(data.corpus[d], 3000)))
        labels.append(lab)
t = time.perf_counter()
scores = r.score_pairs(pairs)
dt = time.perf_counter() - t
print(f"{len(pairs) / dt:.1f} pairs/s")
gold_s = [x for x, y in zip(scores, labels, strict=True) if y]
neg_s = [x for x, y in zip(scores, labels, strict=True) if not y]
print("mean gold", sum(gold_s) / len(gold_s), "mean neg", sum(neg_s) / len(neg_s))
wins = sum(scores[i * 4] > max(scores[i * 4 + 1 : i * 4 + 4]) for i in range(16))
print("gold ranked first among 4:", wins, "/ 16")
