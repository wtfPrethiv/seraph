# seraph
Seraph - a query-adaptive code intelligence engine that combines semantic, lexical, structural, dependency, and Git-aware retrieval to help coding agents understand large, evolving codebases.

## Submission

Team **floppydisk**, SRM Institute of Science and Technology · Samsung PRISM Generative AI Hackathon 2026, Theme 1: Agentic Code Intelligence

| Item | Link |
|---|---|
| Demo video | [Google Drive](https://drive.google.com/drive/folders/1TbShN44FSyemdtEXjca11KWpaW50liuU?usp=sharing) |
| Presentation | [docs/SRM_floppydisk_Submission.pptx](docs/SRM_floppydisk_Submission.pptx) ([PDF](docs/SRM_floppydisk_Submission.pdf)) |
| AI usage disclosure | [docs/floppydisk_AI_Disclosure.docx](docs/floppydisk_AI_Disclosure.docx) |
| AppsRetrieval result JSON | [release `PRISM_GENAI_HACKATHON_Y2026`](https://github.com/wtfPrethiv/seraph/releases/tag/PRISM_GENAI_HACKATHON_Y2026) |
| Agent integration | [Morpheus, `seraph-test` branch](https://github.com/projectakshith/morpheus/tree/seraph-test) |

## For judges

Everything below runs locally with no API keys. Commands are run from the repository root.

**1. Setup**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # only if uv is missing
git clone https://github.com/wtfPrethiv/seraph && cd seraph
uv sync --extra eval --extra cpu --extra mcp            # use --extra gpu instead of cpu on a CUDA machine
source .venv/bin/activate
# or, without uv: python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
seraph-eval train-structural                            # trains the structural (γ) model on the train split, about a minute
```

On macOS, if a run stops with `OMP: Error #15`, see the OpenMP note under [Limitations](#limitations).

**2. P0: the screening score** (already attached to the [PRISM_GENAI_HACKATHON_Y2026 release](https://github.com/wtfPrethiv/seraph/releases/tag/PRISM_GENAI_HACKATHON_Y2026): nDCG@10 0.9770, MRR@10 0.9702)

```bash
seraph-eval submit --final --out appsretrieval_results.json
```

The first run downloads BGE-Code-v1 (about 6 GB) and embeds the 8,765 corpus solutions once; on an Apple M5 GPU the whole run took about an hour. Embeddings are cached in `.seraph_cache/`, so later runs and step 3 reuse them.

**3. Hands-on: ask your own questions** with the submitted pipeline (BM25 + BGE-Code-v1 + structural γ)

```bash
seraph-eval ask "Read an integer n and print the sum of the digits of n factorial."
seraph-eval ask --file problem.txt --top-k 10
seraph-eval ask        # paste problems one by one; end each with a line holding only "."
```

Loading takes about 20 s once the corpus is embedded; each question then takes 15 to 600 ms. Results show the code, the score of each view and the search time. A question taken from the dataset gets its known correct solution marked.

**4. P1: retrieval across versions** on any Git repository (this one works)

```bash
export SERAPH_CONFIG=experiments/configs/submission.yaml    # BGE pipeline; unset it for the instant keyword-only mode
seraph --repo . index --ref HEAD~1
seraph --repo . index --ref HEAD                            # re-parses only changed files: 1 file, 350 of 367 chunks reused, 0.18 s
seraph --repo . search "where is the BM25 index built"
seraph --repo . search "where is the BM25 index built" --ref HEAD~1
```

With `SERAPH_CONFIG` set, the first search of a version embeds its chunks (about 45 s for this repository on an M5); later searches take under a second.

**5. Bonus: evolutionary retrieval**

```bash
seraph --repo . search "how are identical chunks merged across versions" --history   # every indexed version, identical code shown once
seraph --repo . compare HEAD~1 HEAD                                                   # changed symbols, dependencies and commits
seraph --repo . symbol search_index
seraph --repo . deps search_index --direction out
```

Symbols are followed across commits through renames and moves, so history results stay linked when code is renamed.

**6. In a coding agent (optional).** [Morpheus](https://github.com/projectakshith/morpheus/tree/seraph-test) calls Seraph over MCP; see [MCP server](#mcp-server-morpheus-and-other-agents).

## Results

CoIR `AppsRetrieval`, test split (3,765 queries over 8,765 Python solutions), scored through `mteb.evaluate`. The submitted JSON is attached to the [PRISM_GENAI_HACKATHON_Y2026 release](https://github.com/wtfPrethiv/seraph/releases/tag/PRISM_GENAI_HACKATHON_Y2026).

Both rows are the same pipeline (BM25 + dense view + structural γ, weighted fusion 0.16 / 0.64 / 0.2) with a different embedder in the dense view:

| embedder | nDCG@10 | MRR@10 | Recall@10 | Recall@100 |
|---|---|---|---|---|
| **BGE-Code-v1** (1.5B), submitted | **0.9770** | **0.9702** | 0.9971 | 0.9992 |
| Qwen3-Embedding-0.6B | 0.7487 | 0.7039 | 0.8882 | 0.9835 |

Everything runs locally with no API calls. Reproduce the submission:

```bash
uv sync --extra eval --extra cpu                 # --extra gpu on a CUDA machine
seraph-eval train-structural                     # fit split -> experiments/models/structural.pkl
seraph-eval submit --config experiments/configs/submission.yaml --final --out appsretrieval_results.json
```

Add `--set retrieval.dense_model=qwen3-emb-0.6b` for the Qwen3 row. The first run downloads the model (about 6 GB) and embeds the corpus once; embeddings are cached under `.seraph_cache/`, so later runs only encode queries. On Apple silicon the model runs on the GPU (MPS).

**How the test split was used.** The Qwen3 row was the first, frozen submission. We then compared embedders on dev, switched the dense view to BGE-Code-v1 and ran the test split again with the fusion weights unchanged (they were tuned for Qwen3, never on test). An intermediate run of BGE-Code-v1 alone, without BM25 or γ, scored 0.9795, so on this benchmark the extra views add nothing on top of a strong embedder. BGE-Code-v1 was trained on public code-retrieval data that likely overlaps the APPS train queries (dev_stdin 0.989); its test score matches the 98.08 its authors report.

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
from seraph.retrieval.pipeline import Pipeline, register_repo, search
pipe = Pipeline.from_config(cfg, store, graph=graph, versions=versions)   # protocol implementations
register_repo("myrepo", pipe)
search("where is the config parsed", repo="myrepo", top_k=10, include_history=True).to_dict()
```

## Versioned index, CLI and MCP server

`seraph.index.VersionedIndex` stores symbol-level chunks per Git commit in SQLite: functions and classes for Python, and functions, classes, methods, interfaces and types for TypeScript, JavaScript and Go (tree-sitter). Files unchanged since an already-indexed parent commit are reused instead of re-parsed.

Indexing also records each file's calls, imports, base classes and name references. `seraph.service.repo_graph(index, ref)` resolves them across files into a `CodeGraph` whose nodes are `path::symbol` (and files by path), with `calls`, `imports`, `inherits`, `references` and `defines` edges. Python imports resolve through packages and relative imports, TypeScript/JavaScript through relative paths, and Go through package directories; calls into the standard library or third-party packages are left out. A call on an unknown receiver (`obj.save()`) links only when exactly one method has that name. Set `retrieval.use_graph_expansion: true` to add the graph (δ) view to repository search.

`seraph.service` connects the two halves. `IndexChunkStore` exposes an index snapshot as a `ChunkStore`, and `search_index()` ranks it with the `Pipeline`, caching one built pipeline per repo, commit set and config. The CLI and MCP server both go through it. By default the pipeline is BM25 with the code-aware tokenizer and makes no API calls; set `SERAPH_CONFIG=configs/service_hybrid.yaml` to add Gemini embeddings. `seraph search --engine baseline` runs the index's original keyword search for comparison.

Each indexed commit is matched against its nearest indexed first-parent ancestor, so a symbol keeps one `lineage_id` while it is modified, renamed or moved. Matching goes by path and name first, then identical content under another path or name (moved/renamed), then body similarity with the symbol's own name masked out (token 3-gram Jaccard ≥ 0.5), so a function that was renamed and edited in the same commit still links. Every occurrence is recorded with its change type (`added`, `modified`, `renamed`, `moved`, `deleted`, `unchanged`). Indexing an older commit after its descendants replays lineage in commit order. `seraph.lineage.IndexVersionStore` exposes this as the pipeline's `VersionStore`, which drives lineage dedup, the evolution (ε) view, and the per-result `history` in history search.

```bash
uv sync
seraph --repo /path/to/repo index --ref HEAD~1
seraph --repo /path/to/repo index --ref HEAD
seraph --repo /path/to/repo search "where is the input normalized" --ref HEAD
seraph --repo /path/to/repo symbol parse_config
seraph --repo /path/to/repo deps main --direction out
seraph --repo /path/to/repo compare HEAD~1 HEAD
seraph --repo /path/to/repo versions
```

The database defaults to `REPO/.seraph/index.sqlite`, which Git ignores; `--db PATH` moves it. `index` reports how many files it parsed and how many chunks it reused, so index a parent commit before its child to get reuse. `search --history` searches every indexed commit. `search` prints a ranked list with index and search timings; `--json` prints the raw result. With `--history`, identical code found in several versions is returned once, with every version it appears in listed under `versions`; code that changed stays separate. `symbol` looks up a definition by name or `path::symbol`. `deps` walks the graph from that symbol (`--direction out|in|both`). `compare` reports symbol, dependency and commit differences between two refs.

```python
from seraph import VersionedIndex

with VersionedIndex("/path/to/repo") as index:
    index.index_commit("HEAD")
    chunks = list(index.iter_chunks("HEAD"))
```

Each `seraph.index.Chunk` (distinct from `seraph.types.Chunk`) has an `occurrence_id`, `commit`, `path`, `symbol`, line range, `text`, `language` and `content_hash`. Use `occurrence_id` to identify results, since the same text in two files or commits gets two occurrence IDs, and `content_hash` to cache embeddings across versions.

### MCP server (Morpheus and other agents)

[Morpheus](https://github.com/projectakshith/morpheus/tree/seraph-test) is our terminal coding agent, and its [`seraph-test`](https://github.com/projectakshith/morpheus/tree/seraph-test) branch is built around Seraph:

- `/seraph setup|status|enable|disable` connects and checks the server.
- The agent searches with Seraph first when it looks for code by behavior, and keeps grep for exact names.
- Seraph calls render as ranked cards: relevance bars, a highlighted code preview, index reuse and timings, and version tags for history searches.
- `/versus <question>` (or `/vs`) runs one question through Seraph and grep side by side and shows where Seraph's top hit lands in grep's list.

```bash
git clone -b seraph-test https://github.com/projectakshith/morpheus && cd morpheus
npm install && npm run dev
# inside Morpheus
/seraph setup /path/to/seraph
/vs where are mcp server processes started
```

```bash
uv sync --extra mcp
```

In Morpheus, run `/seraph setup /path/to/seraph` once; it writes the server entry to `~/.morpheus/config.json` next to your other servers. Any other MCP client can use the same entry:

```json
{
  "mcpServers": {
    "seraph": {
      "command": "/path/to/seraph/.venv/bin/python",
      "args": ["-m", "seraph.server"]
    }
  }
}
```

The server searches the Git repository it is started in (`SERAPH_REPO` overrides that, `SERAPH_DB` the database path). Tools: `search_code`, `search_at_version`, `search_history`, `find_symbol`, `find_dependencies`, `compare_versions` and `index_repository`; a search indexes the requested version on first use and returns path, commit, line range, snippet, score and timings. Morpheus renders the results as ranked cards, and `/versus <question>` runs the same question through Seraph and grep side by side.

## Models: API first, local optional

| Role | Default | Where it runs |
|---|---|---|
| Dense embeddings | `gemini-embedding-001` (768-d) | Gemini API |
| Reranker | `gemini-3.5-flash-lite`, listwise over a window of 20 | Gemini API |
| LLM query analyzer (`analyzer: llm`) | `gemini-3.5-flash-lite` via the OpenAI-compatible endpoint | Gemini API |
| Local embedders | BGE-Code-v1, EmbeddingGemma-300M, Qwen3-Embedding-0.6B/4B, CodeSage, and others | CPU, CUDA or Apple MPS, with `allow_local_models: true` |
| Local rerankers | bge/jina/Qwen3 rerankers | same |

Local model loading is off by default (`LocalModelsDisabledError`); the submission configs turn it on. The AppsRetrieval experiments below use Qwen3-Embedding-0.6B vectors that were computed once and cached in `.seraph_cache/embeddings/`. They are read from disk without loading the model, because embedding the 8,765-document corpus on the Gemini free tier (1,000 texts per day) would take about nine days. All API responses (embeddings, rerank orders, LLM analyses) are cached in SQLite or JSON under `.seraph_cache/`, so reruns are free.

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

On dev / dev_stdin NDCG@10, BGE-Code-v1 scores 0.992 / 0.989 and EmbeddingGemma-300M 0.744 / 0.782. BGE-Code-v1's dev numbers overstate it, since it was likely trained on these train queries; its test score is in Results. Qwen3-Embedding-0.6B scores 0.837 / 0.722, gte-modernbert 0.705 / 0.505, CodeSage-small 0.680 / 0.319, CodeRankEmbed 0.650 / 0.206 and jina-code-v2 0.601 / 0.160. CodeBERT, GraphCodeBERT and UniXcoder used without retrieval fine-tuning score under 0.11. The stdin shift costs every model, but much less for the instruction-tuned Qwen3 (`experiments/results/dense_sweep.md`).

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
seraph-eval submit --final                 # upload JSON (experiments/configs/submission.yaml)
```

The Gemini key is read from `GEMINI_API_KEY`. On the free tier, a daily-quota error stops the affected rung and marks it `skipped (quota)` without failing the run.

## Limitations

- The Gemini embedder works but has not been benchmarked on the full corpus because of free-tier quota.
- The test split was run without reranking: 3,765 listwise Gemini calls do not fit the free tier, and no local reranker has been benchmarked yet.
- On macOS, `torch`, `faiss-cpu` and `scikit-learn` each ship their own OpenMP runtime. If a run aborts with `OMP: Error #15`, point the `libomp.dylib` copies under `faiss/.dylibs/` and `sklearn/.dylibs/` at the one in `torch/lib/`.
- Dense query decomposition was not ablated: the sub-queries have no cached embeddings. BM25 decomposition is in `b7_bm25_decomp`.
- Measured latency excludes query encoding when vectors come from the cache.
- Graph, evolution and dedup gains need SeraphBench, which is not yet available.
