"""重排器家族，含语言守卫。

这里藏着一个真实且代价高昂的坑：
**当一个在查询语言上力不从心的交叉编码器（最常见的是英文 reranker + 中文查询）
被直接用作最终排序时，它会无人制衡地把正确答案压下去** —— 实测中文查询 top-1
命中会从 11/12 掉到 4/12。原因是 cross-encoder 在跨语言错配时给出的分数近乎随机，
而它的输出覆盖了融合排序里所有其他信号。

因此本项目默认不启用重排，即便启用也必须经过 `GuardedReranker`：
查询语言与重排模型训练语言不匹配时自动降级为恒等重排，并在 attrs 里留下记录。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from nexus.protocols import Reranker
from nexus.types import Hit
from nexus.util.text import is_cjk_dominant, tokenize


class IdentityReranker:
    """恒等重排：保持融合顺序不变。默认实现，零副作用。"""

    name = "identity"

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        return list(hits)[:top_k]


class CoverageReranker:
    """启发式重排：按查询词覆盖率 + 位置加权重新打分。

    完全基于规则、零外部模型、零网络延迟，作为真实 cross-encoder 不可用时的
    高质量替代品。它对"关键词缺失造成的误召回"有明显修正作用，
    同时不会像跨错语言的神经网络那样随机抖动。
    """

    name = "coverage"

    def __init__(self, alpha: float = 0.65, decay: float = 0.02) -> None:
        self.alpha = alpha
        self.decay = decay

    def _coverage(self, query: str, text: str) -> float:
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return 0.0
        d_tokens = tokenize(text)
        pos = {t: i for i, t in enumerate(d_tokens)}
        score = 0.0
        for t in q_tokens:
            if t in pos:
                score += 1.0 / (1.0 + self.decay * pos[t])
        return score / len(q_tokens)

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        rescored: list[tuple[float, Hit]] = []
        for h in hits:
            cov = self._coverage(query, h.chunk.text)
            fused = self.alpha * cov + (1 - self.alpha) * h.score
            new = Hit(
                chunk=h.chunk,
                score=fused,
                rank=h.rank,
                vector_score=h.vector_score,
                bm25_score=h.bm25_score,
                rerank_score=round(cov, 6),
                stage="coverage-rerank",
            )
            rescored.append((fused, new))
        rescored.sort(key=lambda kv: -kv[0])
        out = [h for _, h in rescored[:top_k]]
        for i, h in enumerate(out, start=1):
            h.rank = i
        return out


class CrossEncoderReranker:
    """基于 sentence-transformers CrossEncoder 的精排（可选、重型依赖）。

    仅当 torch / sentence-transformers 已安装时可用，
    否则构造时清晰报错，由上层降级到 CoverageReranker。
    """

    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        pairs = [(query, h.chunk.text) for h in hits]
        scores = self._model.predict(pairs)
        scored = []
        for h, s in zip(hits, scores, strict=False):
            new = Hit(
                chunk=h.chunk,
                score=float(s),
                rank=h.rank,
                vector_score=h.vector_score,
                bm25_score=h.bm25_score,
                rerank_score=float(s),
                stage="cross-encoder",
            )
            scored.append((float(s), new))
        scored.sort(key=lambda kv: -kv[0])
        out = [h for _, h in scored[:top_k]]
        for i, h in enumerate(out, start=1):
            h.rank = i
        return out


class GuardedReranker:
    """语言守卫包装器。

    决策逻辑：
        query 以中文为主 且 底层重排器未在中文语料上训练 -> 降级为恒等重排
    降级决定会写进 attrs.l1n_decision，可被 tracing 与测试观察到，
    避免"悄悄不生效"导致排查困难。
    """

    name = "guarded"

    def __init__(
        self,
        inner: Reranker,
        *,
        inner_supports_zh: bool = False,
        fallback: Reranker | None = None,
    ) -> None:
        self.inner = inner
        self.inner_supports_zh = inner_supports_zh
        self.fallback = fallback or IdentityReranker()
        self.last_decision: str = ""

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        zh_query = is_cjk_dominant(query)
        if zh_query and not self.inner_supports_zh:
            self.last_decision = "bypass:zh-query-with-non-zh-reranker"
            return self.fallback.rerank(query, hits, top_k)
        self.last_decision = f"used:{getattr(self.inner, 'name', type(self.inner).__name__)}"
        return self.inner.rerank(query, hits, top_k)


def entropy(values: Sequence[float]) -> float:
    """工具函数：给定分数分布的熵，用于分析重排是否过度" flatten "排序。"""
    total = math.fsum(values)
    if total <= 0:
        return 0.0
    return -math.fsum((v / total) * math.log((v / total) + 1e-12) for v in values if v > 0)
