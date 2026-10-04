# seraph
Seraph - a query-adaptive code intelligence engine that combines semantic, lexical, structural, dependency, and Git-aware retrieval to help coding agents understand large, evolving codebases.

## How retrieval works

```
query ─► analyzer (parse, classify, decompose) ─► views ─► fusion ─► rerank ─► dedup / MMR ─► results
                                                  │
        α lexical    BM25 with a code-aware tokenizer (camelCase/snake_case splitting)
        β semantic   dense embeddings (Gemini API, or cached local vectors)
        γ structural tree-sitter traits of each snippet vs. traits predicted from the query
        δ graph      expansion over the call/import graph from the top seeds
        ε evolution  lineage matches against commit messages and version refs
```

Each view returns scored chunks, and fusion is either RRF or a weighted sum of per-query min-max scores. Weights are static, rule-based (from query type and features) or learned per query (`fusion: adaptive`). Every fused hit keeps the raw score from each view in `retrieval_scores`.

| Module | Contents |
|---|---|
| `seraph.retrieval.pipeline` | `Pipeline`, plus the public `search(query, repo, top_k, version, include_history)` |
| `seraph.retrieval.{lexical,semantic,structural,fusion,reranker,dedup}` | the views, fusion, reranking, lineage dedup and MMR |
| `seraph.query.{parser,classifier,decomposition,analyzer,llm,weights}` | the query layer and the adaptive weights |
| `seraph.graph.traversal` | graph expansion and dependency chains over any `CodeGraph` |
| `seraph.evaluation.*` | AppsRetrieval harness, metrics (cross-checked against MTEB), experiments, ablations |

Interfaces shared with the indexing side (`ChunkStore`, `CodeGraph`, `VersionStore`) live in `seraph.protocols`. In-memory versions for tests and fixtures are in `seraph.memory`.

```python
from seraph.retrieval.pipeline import register_repo, search
register_repo("myrepo", store, graph=graph, versions=versions)   # protocol implementations
search("where is the config parsed", repo="myrepo", top_k=10, include_history=True).to_dict()
```

## Versioned index, CLI and MCP server

`seraph.index.VersionedIndex` stores Python function/class chunks (plus file chunks for JavaScript, TypeScript and Go) per Git commit in SQLite. Files unchanged since an already-indexed parent commit are reused instead of re-parsed. Search currently uses the index's built-in lexical baseline; the `Pipeline` above is not yet connected to it.

```bash
uv sync
seraph --repo /path/to/repo index --ref HEAD~1
seraph --repo /path/to/repo index --ref HEAD
seraph --repo /path/to/repo search "where is the input normalized" --ref HEAD
seraph --repo /path/to/repo versions
```

The database defaults to `REPO/.seraph/index.sqlite`, which Git ignores; `--db PATH` moves it. `index` reports how many files it parsed and how many chunks it reused, so index a parent commit before its child to get reuse. `search --history` searches every indexed commit. Output is JSON with the commit, path, line range, code and score.

```python
from seraph import VersionedIndex

with VersionedIndex("/path/to/repo") as index:
    index.index_commit("HEAD")
    chunks = list(index.iter_chunks("HEAD"))
```

Each `seraph.index.Chunk` (distinct from `seraph.types.Chunk`) has an `occurrence_id`, `commit`, `path`, `symbol`, line range, `text`, `language` and `content_hash`. Use `occurrence_id` to identify results, since the same text in two files or commits gets two occurrence IDs, and `content_hash` to cache embeddings across versions.

### MCP server (Morpheus and other agents)

```bash
uv sync --extra mcp
```

Merge this entry into the existing `mcpServers` object of `~/.morpheus/config.json` (do not replace other servers):

```json
{
  "mcpServers": {
    "seraph": {
      "command": "/absolute/path/to/python3",
      "args": ["-m", "seraph.server"],
      "env": {"SERAPH_REPO": "/absolute/path/to/repository"}
    }
  }
}
```

The tools appear as `mcp_seraph_search_code`, `mcp_seraph_search_at_version` and `mcp_seraph_index_repository`. A search indexes the requested version on first use and returns path, commit, line range, snippet and score. `SERAPH_DB` optionally overrides the database path.

## Models: API first, local optional

| Role | Default | Where it runs |
|---|---|---|
| Dense embeddings | `gemini-embedding-001` (768-d) | Gemini API |
| Reranker | `gemini-3.5-flash-lite`, listwise over a window of 20 | Gemini API |
| LLM query analyzer (`analyzer: llm`) | `gemini-3.5-flash-lite` via the OpenAI-compatible endpoint | Gemini API |
| Local alternatives | Qwen3-Embedding-0.6B, CodeSage, bge/jina/Qwen3 rerankers | GPU, only with `allow_local_models: true` |

Local model loading is off by default (`LocalModelsDisabledError`). The AppsRetrieval experiments below use Qwen3-Embedding-0.6B vectors that were computed once and cached in `.seraph_cache/embeddings/`. They are read from disk without loading the model, because embedding the 8,765-document corpus on the Gemini free tier (1,000 texts per day) would take about nine days. All API responses (embeddings, rerank orders, LLM analyses) are cached in SQLite or JSON under `.seraph_cache/`, so reruns are free.

## Evaluation protocol (AppsRetrieval, CoIR)

| Split | Queries | Use |
|---|---|---|
| `fit` | 4,000 | train-split queries minus dev; trains the structural trait model |
| `dev` | 1,000 | 20% of train (seed 13); all tuning |
| `dev_stdin` | 334 | dev queries whose gold solution reads stdin |
| `test` | 3,765 | guarded; needs `--final` and is meant to run once |

The corpus has 8,765 Python solutions. **Distribution shift:** most train solutions are LeetCode-style functions, while the test split is almost entirely stdin-style competitive programs. `dev_stdin` mimics the test distribution, so tuning decisions weigh it at least as heavily as `dev`. Our metrics use trec_eval tie-breaking and agree with MTEB's evaluator to within 2e-6 (`seraph-eval crosscheck`). Every result JSON records the git SHA, config hash, model revisions, seed and dataset revision.

### Ablation (one component at a time)

From `seraph-eval ablate` (`experiments/configs/ablation.yaml`); the full table is in `experiments/results/ablation.md`. Deltas are against the previous row, except the MMR and rerank rows, which are compared with the structural row.

| component | dev NDCG@10 | dev_stdin NDCG@10 | dev_stdin R@100 |
|---|---|---|---|
| BM25 (code-aware tokenizer) | 0.3714 | 0.0941 | 0.2844 |
| dense only (Qwen3-Embedding-0.6B) | 0.8367 | 0.7220 | 0.9701 |
| + weighted fusion (lexical 0.2 / semantic 0.8) | 0.8473 (+0.0106) | 0.7343 (+0.0123) | 0.9731 |
| + structural γ view (weight 0.2) | 0.8530 (+0.0057) | 0.7397 (+0.0054) | 0.9731 |
| + rule-based adaptive weights | 0.8498 (−0.0032) | 0.7420 (+0.0023) | 0.9731 |
| + MMR (λ 0.85) on the structural config | 0.8541 (+0.0011) | 0.7400 (+0.0003) | 0.9731 |
| + Gemini listwise rerank (depth 20) on the structural config | – | **0.8060** (+0.0664) | 0.9731 |
| + graph expansion / lineage dedup / evolution view | n/a | n/a | n/a |

Listwise reranking is the largest single gain. The rerank row's p50 latency (4.3 s per query) is dominated by free-tier pacing at 14 requests per minute; the other rows run in single-digit milliseconds. To save quota, the rerank rung runs only on dev_stdin, the split closest to test.

The graph, dedup and evolution components cannot show gains on AppsRetrieval: its snippets have no call graph and only one version each. They are covered by unit tests on fixture repos and are meant for SeraphBench.

### Dense model sweep

On dev / dev_stdin NDCG@10: Qwen3-Embedding-0.6B scores 0.837 / 0.722, gte-modernbert 0.705 / 0.505, CodeSage-small 0.680 / 0.319, CodeRankEmbed 0.650 / 0.206 and jina-code-v2 0.601 / 0.160. CodeBERT, GraphCodeBERT and UniXcoder used without retrieval fine-tuning score under 0.11. The stdin shift costs every model, but much less for the instruction-tuned Qwen3 (`experiments/results/dense_sweep.md`).

### Adaptive weights versus the oracle

From `seraph-eval tune-adaptive` on dev (`experiments/results/adaptive_weights.md`). Each view is retrieved once; every query is scored at all 66 cells of a (lexical, semantic, structural) weight simplex with step 0.1; learned models are evaluated with 5-fold cross-validation.

| weighting | dev NDCG@10 | dev_stdin NDCG@10 |
|---|---|---|
| static tuned (0.16 / 0.64 / 0.2) | 0.8531 | 0.7396 |
| best single grid cell, 5-fold CV | 0.8518 | 0.7418 |
| rule-based adaptive | 0.8505 | 0.7420 |
| learned (LightGBM, top-3 cells, 5-fold CV) | 0.8531 | 0.7395 |
| learned (LightGBM, all 66 cells, 5-fold CV) | 0.8451 | 0.7289 |
| per-query oracle (upper bound) | 0.8967 | 0.7966 |

Per-query weighting has 4.4 to 5.7 NDCG points of headroom, but neither the rules nor the learned model capture it on AppsRetrieval. Every query here is a competitive-programming statement, so the query features barely vary, and choosing freely among all 66 cells overfits noise. Restricting the choice to a few strong cells keeps the learned model level with static weights. We expect adaptive weighting to matter on mixed query types (identifier lookups, dependency and history questions), which is what SeraphBench measures.

### Structural γ view

Each snippet gets a sparse binary vector of 62 tree-sitter traits: input style (`input()`, `sys.stdin`, multi-test loops, grid reads), loop depth, recursion, `heapq`/`deque`/`bisect`/sorting, mod constants, bit operations and imports. A per-trait logistic regression on query TF-IDF, trained on `fit` (query, gold solution) pairs, predicts the traits the answer should have. A candidate's score is the Bernoulli log-likelihood ratio of its traits against the training prior, computed over the fused candidate pool. Weight 0.2 was chosen on dev, and larger weights hurt (0.5 drops dev_stdin to 0.718).

## Reproducing

```bash
uv sync --extra eval --extra api           # add --extra gpu only for local models
seraph-eval crosscheck                     # metrics vs MTEB
seraph-eval train-structural               # fit split -> experiments/models/structural.pkl
seraph-eval tune-structural                # gamma grid on dev / dev_stdin
seraph-eval tune-adaptive                  # adaptive weights table + learned model
seraph-eval ablate [--skip-api]            # ablation ladder -> experiments/results/ablation.md
seraph-eval run experiments/configs/b9_structural.yaml --split dev_stdin
seraph-eval ablate --final                 # once, at the very end (test split)
```

The Gemini key is read from `GEMINI_API_KEY`. On the free tier, a daily-quota error stops the affected rung and marks it `skipped (quota)` without failing the run.

## Limitations

- Dense numbers come from cached local Qwen3 vectors; the Gemini embedder works but has not been benchmarked on the full corpus because of free-tier quota.
- The test split has not been run. It needs Qwen3 query vectors for the 3,765 test queries (one local encoding pass) or a full Gemini re-embedding, and enough reranker quota for 3,765 listwise calls.
- Dense query decomposition was not ablated: the sub-queries have no cached embeddings. BM25 decomposition is in `b7_bm25_decomp`.
- Measured latency excludes query encoding when vectors come from the cache.
- Graph, evolution and dedup gains need SeraphBench, which is not yet available.
