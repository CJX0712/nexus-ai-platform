"""融合检索器：BM25（稀疏、关键词精确）+ 向量（稠密、语义近似）→ RRF 融合。

为什么必须融合：
- 纯向量检索对专有名词、型号、参数这类"词的精确匹配"不敏感
- 纯 BM25 无法处理同义改写，且完全错过无字面重叠的相关内容

为什么不直接加权求和：两路分数的量纲完全不同（BM25 无上界、余弦在 [-1,1]），
加权前必须归一化，而归一化系数会随语料漂移。RRF 只用名次信息，天然免疫量纲问题：
    rrf_score(d) = Σ_r 1 / (k + rank_r(d))
这也是业界混合检索的默认做法。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from nexus.protocols import Embedder, VectorStore
from nexus.retriever.bm25 import IncrementalBM25
from nexus.types import Chunk, Hit, RetrieveMode
from nexus.util.text import tokenize

_RRF_K = 60


def _reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[str]], k: int = _RRF_K) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """把任意量纲的分数线性归一化到 [0,1]，仅用于展示与解释。"""
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    span = hi - lo
    if span <= 1e-9:
        return {key: 1.0 for key in scores}
    return {key: (v - lo) / span for key, v in scores.items()}


class HybridRetriever:
    """统一的检索入口。三种模式：hybrid / vector / bm25。"""

    name = "hybrid-rrf"

    def __init__(
        self,
        embedder: Embedder,
        vectorstore: VectorStore,
        *,
        bm25: IncrementalBM25 | None = None,
        mode: RetrieveMode = "hybrid",
        rrf_k: int = _RRF_K,
        candidate_ratio: int = 4,
    ) -> None:
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.bm25 = bm25 or IncrementalBM25()
        self.mode: RetrieveMode = mode
        self.rrf_k = rrf_k
        self.candidate_ratio = candidate_ratio
        self._chunks: dict[str, Chunk] = {}

    def count(self) -> int:
        return len(self._chunks)

    def add(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        for c in chunks:
            self._chunks[c.chunk_id] = c
        self.embedder.observe(texts)
        vectors = self.embedder.embed(texts)
        self.vectorstore.upsert(ids, vectors)
        self.bm25.add_texts(ids, texts)

    def delete(self, chunk_ids: Sequence[str]) -> int:
        ids = [i for i in chunk_ids if i in self._chunks]
        if not ids:
            return 0
        for i in ids:
            self._chunks.pop(i, None)
        self.vectorstore.delete(ids)
        self.bm25.delete(ids)
        return len(ids)

    def delete_document(self, doc_id: str) -> int:
        ids = [c.chunk_id for c in self._chunks.values() if c.doc_id == doc_id]
        return self.delete(ids)

    def documents(self) -> list[str]:
        return sorted({c.doc_id for c in self._chunks.values()})

    def _bm25_pass(self, query: str, top_n: int) -> tuple[list[str], dict[str, float]]:
        pairs = self.bm25.search(query, top_k=top_n)
        return [i for i, _ in pairs], {i: s for i, s in pairs}

    def _vector_pass(self, query: str, top_n: int) -> tuple[list[str], dict[str, float]]:
        vec = self.embedder.embed([query])
        q = np.asarray(vec[0], dtype=np.float32)
        pairs = self.vectorstore.search(q, top_n)
        return [i for i, _ in pairs], {i: s for i, s in pairs}

    def retrieve(self, query: str, top_k: int = 5, mode: RetrieveMode | None = None) -> list[Hit]:
        m: RetrieveMode = mode or self.mode
        if not query.strip() or self.count() == 0:
            return []
        top_n = max(top_k * self.candidate_ratio, top_k + 1)

        bm25_ids: list[str] = []
        vec_ids: list[str] = []
        bm25_scores: dict[str, float] = {}
        vec_scores: dict[str, float] = {}

        if m in ("hybrid", "bm25"):
            bm25_ids, bm25_scores = self._bm25_pass(query, top_n)
        if m in ("hybrid", "vector"):
            vec_ids, vec_scores = self._vector_pass(query, top_n)

        if m == "hybrid":
            fused = _reciprocal_rank_fusion([bm25_ids, vec_ids], self.rrf_k)
            ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        elif m == "vector":
            ordered = [(i, s) for i, s in vec_scores.items()][:top_k]
        else:
            ordered = [(i, s) for i, s in bm25_scores.items()][:top_k]

        norm_b = _normalize(bm25_scores)
        norm_v = _normalize(vec_scores)
        hits: list[Hit] = []
        for rank, (chunk_id, score) in enumerate(ordered, start=1):
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                continue
            hits.append(
                Hit(
                    chunk=chunk,
                    score=float(score),
                    rank=rank,
                    vector_score=norm_v.get(chunk_id),
                    bm25_score=norm_b.get(chunk_id),
                    stage=m,
                )
            )
        return hits

    def explain(self, query: str, top_k: int = 5) -> dict[str, object]:
        """返回融合前后的完整分数明细，供控制台解释"为什么命中这条"。"""
        hits = self.retrieve(query, top_k)
        return {
            "query_tokens": tokenize(query)[:32],
            "mode": self.mode,
            "hits": [h.to_dict() for h in hits],
        }
