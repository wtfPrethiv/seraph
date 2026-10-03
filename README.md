# Seraph

Seraph is a code-search engine for changing repositories. This branch contains
the first versioned indexing slice: Python function/class chunks, file chunks
for JavaScript/TypeScript/Go, content reuse across Git commits, and a lexical
search fallback. The ranking teammate can read chunks with `iter_chunks()` and
replace the fallback ranking without changing Git indexing.

## Quick start

Use Python 3.12 or newer. This slice has no runtime package dependencies.

```bash
python3 -m pip install -e .
seraph --repo /path/to/repo index --ref HEAD~1
seraph --repo /path/to/repo index --ref HEAD
seraph --repo /path/to/repo search "where is the input normalized" --ref HEAD
seraph --repo /path/to/repo search "where is the input normalized" --ref HEAD~1
```

The database defaults to `REPO/.seraph/index.sqlite`, which Git ignores.
Pass `--db /path/to/index.sqlite` to keep it elsewhere. `index` reports how many
files it parsed and how many chunks it reused. Index a parent commit before its
child to get incremental reuse. `search --history` includes all indexed commits.

The CLI prints JSON containing the commit, path, line range, code, and score.
Current search is deliberately a small lexical baseline; APPS ranking should
use the retrieval model developed by the scoring side of the team.

## Retrieval handoff

```python
from seraph import VersionedIndex

with VersionedIndex("/path/to/repo") as index:
    index.index_commit("HEAD")
    chunks = list(index.iter_chunks("HEAD"))
    # Chunk has occurrence_id, commit, path, symbol, line range, text,
    # language, and content_hash. Use occurrence_id for search results;
    # use content_hash to cache embeddings across versions.
```

The same text in multiple files or commits gets separate occurrence IDs. This
matters when showing the correct file and version in results.

## Morpheus tool server

Install the optional MCP dependency, then add a `seraph` server to Morpheus's
`~/.morpheus/config.json` under `mcpServers`. Morpheus uses stdio MCP and makes
the tools available to its main agent as `mcp_seraph_search_code`,
`mcp_seraph_search_at_version`, and `mcp_seraph_index_repository`.

```bash
python3 -m pip install -e '.[mcp]'
```

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

Merge this entry into an existing `mcpServers` object; do not replace other
servers. A search will index the requested version on first use. The response
contains the matching path, commit, line range, code snippet, and score. This
server currently uses the lexical fallback until the scoring team's ranker is
connected. Morpheus's README confirms stdio MCP configuration and tool naming.

## Test

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
