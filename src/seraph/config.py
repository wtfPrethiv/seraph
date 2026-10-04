"""Configuration and model registry. Every component is toggled by a flag here."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    name: str
    hf_id: str
    revision: str = "main"
    backend: Literal[
        "st", "hf", "codet5p", "cross_encoder", "causal_reranker", "gemini_embed", "gemini_rerank"
    ] = "st"
    pooling: Literal["mean", "cls", "last"] = "mean"
    query_prefix: str = ""
    doc_prefix: str = ""
    max_len: int = 512
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    trust_remote_code: bool = False
    batch_size: int = 32
    params_m: int = 0
    options: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_api(self) -> bool:
        return self.backend.startswith("gemini")


_QWEN_EMB_INSTRUCT = (
    "Instruct: Given a programming problem description, retrieve Python code that solves it\n"
    "Query: "
)

EMBEDDERS: dict[str, ModelSpec] = {
    m.name: m
    for m in [
        ModelSpec(
            name="gemini-embedding-001",
            hf_id="gemini-embedding-001",
            backend="gemini_embed",
            options={"output_dim": 768, "query_task": "CODE_RETRIEVAL_QUERY", "doc_task": "RETRIEVAL_DOCUMENT", "rpm": 90, "batch": 100},
        ),
        ModelSpec(name="codebert", hf_id="microsoft/codebert-base", backend="hf", params_m=125),
        ModelSpec(name="graphcodebert", hf_id="microsoft/graphcodebert-base", backend="hf", params_m=125),
        ModelSpec(name="unixcoder", hf_id="microsoft/unixcoder-base", backend="hf", params_m=125),
        ModelSpec(
            name="codet5p-110m",
            hf_id="Salesforce/codet5p-110m-embedding",
            backend="codet5p",
            trust_remote_code=True,
            dtype="float32",
            params_m=110,
        ),
        ModelSpec(
            name="jina-code-v2",
            hf_id="jinaai/jina-embeddings-v2-base-code",
            trust_remote_code=True,
            max_len=1024,
            params_m=161,
        ),
        ModelSpec(
            name="codesage-small",
            hf_id="codesage/codesage-small-v2",
            trust_remote_code=True,
            max_len=1024,
            params_m=130,
        ),
        ModelSpec(
            name="coderankembed",
            hf_id="nomic-ai/CodeRankEmbed",
            trust_remote_code=True,
            query_prefix="Represent this query for searching relevant code: ",
            max_len=1024,
            params_m=137,
        ),
        ModelSpec(
            name="gte-modernbert",
            hf_id="Alibaba-NLP/gte-modernbert-base",
            max_len=1024,
            params_m=149,
        ),
        ModelSpec(
            name="qwen3-emb-0.6b",
            hf_id="Qwen/Qwen3-Embedding-0.6B",
            query_prefix=_QWEN_EMB_INSTRUCT,
            max_len=1024,
            batch_size=8,
            params_m=600,
        ),
    ]
}

RERANKERS: dict[str, ModelSpec] = {
    m.name: m
    for m in [
        ModelSpec(
            name="gemini-flash-lite-listwise",
            hf_id="gemini-2.5-flash-lite",
            backend="gemini_rerank",
            options={"window": 20, "rpm": 14, "doc_chars": 1200, "query_chars": 3000},
        ),
        ModelSpec(
            name="gemini-flash-listwise",
            hf_id="gemini-2.5-flash",
            backend="gemini_rerank",
            options={"window": 20, "rpm": 9, "doc_chars": 1200, "query_chars": 3000},
        ),
        ModelSpec(
            name="bge-reranker-v2-m3",
            hf_id="BAAI/bge-reranker-v2-m3",
            backend="cross_encoder",
            max_len=1024,
            batch_size=16,
            params_m=568,
        ),
        ModelSpec(
            name="jina-reranker-v2",
            hf_id="jinaai/jina-reranker-v2-base-multilingual",
            backend="cross_encoder",
            trust_remote_code=True,
            max_len=1024,
            batch_size=16,
            params_m=278,
        ),
        ModelSpec(
            name="qwen3-reranker-0.6b",
            hf_id="Qwen/Qwen3-Reranker-0.6B",
            backend="causal_reranker",
            max_len=1536,
            batch_size=8,
            params_m=600,
        ),
    ]
}


class RetrievalConfig(BaseModel):
    first_stage_k: int = 100
    use_bm25: bool = True
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    bm25_stem: bool = True
    use_dense: bool = False
    dense_model: str = "gemini-embedding-001"
    dense_fallback: str | None = None  # e.g. qwen3-emb-0.6b to fall back to a local model
    fusion: Literal["none", "rrf", "weighted", "adaptive"] = "rrf"
    rrf_k: int = 60
    fusion_norm: Literal["minmax", "zscore"] = "minmax"
    static_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "lexical": 0.3,
            "semantic": 0.7,
            "structural": 0.0,
            "graph": 0.0,
            "evolution": 0.0,
        }
    )
    use_reranker: bool = False
    reranker_model: str = "gemini-flash-lite-listwise"
    reranker_fallback: str | None = None
    rerank_depth: int = 20
    rerank_blend: float = 0.0
    use_query_analyzer: bool = False
    analyzer: Literal["rules", "llm"] = "rules"
    use_decomposition: bool = False
    adaptive_weights: Literal["off", "rules", "learned"] = "off"
    weight_model_path: str | None = None
    use_structural: bool = False
    structural_model_path: str | None = None
    use_graph_expansion: bool = False
    graph_hops: int = 1
    graph_decay: float = 0.5
    use_evolution: bool = False
    use_dedup: bool = False
    use_mmr: bool = False
    mmr_lambda: float = 0.7


class LLMConfig(BaseModel):
    """Any OpenAI-compatible endpoint; defaults to Gemini's."""

    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    model: str = "gemini-2.5-flash-lite"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.0


class SeraphConfig(BaseModel):
    name: str = "default"
    seed: int = 42
    device: str = "auto"
    cache_dir: str = ".seraph_cache"
    cache_query_embeddings: bool = True  # turn off when measuring latency
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @property
    def cache_path(self) -> Path:
        p = Path(self.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> SeraphConfig:
    """Load a YAML config. A top-level `extends:` key merges a parent config first."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    parent = data.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent).model_dump()
        data = _deep_merge(base, data)
    if overrides:
        data = _deep_merge(data, overrides)
    return SeraphConfig.model_validate(data)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
