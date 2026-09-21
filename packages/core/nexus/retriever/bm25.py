"""可增量维护的 BM25。

不使用第三方实现的原因：主流 BM25 库要求一次性构造语料，
每次新增文档都得全量重建索引。这里维护 tf / df / 文档长度增量统计，
支持单条增删，更适合动态知识库。

公式：
    score(q, d) = Σ_{t ∈ q} idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·dl/avgdl))
    idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))     （Robertson-Spärck Jones 平滑式）

不变量：
    1. 文档从未被查询词命中时得分为 0
    2. 删除文档后 df 必须同步下降，否则 idf 漂移
    3. 同一份语料重复 add 同一 id 不会导致重复计数
"""

from __future__ import annotations

from collections.abc import Sequence

from nexus.util.text import tokenize


class IncrementalBM25:
    """支持增量 add / delete 的 BM25 倒排检索。

    Args:
        k1: 词频饱和参数，越大则重复词贡献越高
        b: 长度归一强度，0 表示完全不惩罚长文档
    """

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._tf: dict[str, dict[str, int]] = {}
        self._df: dict[str, int] = {}
        self._len: dict[str, int] = {}
        self._total_len = 0

    @property
    def n_docs(self) -> int:
        return len(self._tf)

    @property
    def avgdl(self) -> float:
        return self._total_len / len(self._tf) if self._tf else 0.0

    def idf(self, token: str) -> float:
        n = len(self._tf)
        if n == 0:
            return 0.0
        df = self._df.get(token, 0)
        import math

        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def add_texts(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        if len(ids) != len(texts):
            raise ValueError("ids 与 texts 长度不一致")
        for sid, text in zip(ids, texts, strict=False):
            toks = tokenize(text)
            self._add(sid, toks)

    def _add(self, sid: str, toks: Sequence[str]) -> None:
        if sid in self._tf:
            self._remove_tokens(sid)
            del self._tf[sid]
        counts: dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        self._tf[sid] = counts
        self._len[sid] = len(toks)
        self._total_len += len(toks)
        for t in counts:
            self._df[t] = self._df.get(t, 0) + 1

    def _remove_tokens(self, sid: str) -> None:
        counts = self._tf.get(sid, {})
        for t in counts:
            self._df[t] = max(0, self._df.get(t, 0) - 1)
            if self._df[t] == 0:
                del self._df[t]
        self._total_len -= self._len.get(sid, 0)

    def delete(self, ids: Sequence[str]) -> int:
        removed = 0
        for sid in ids:
            if sid not in self._tf:
                continue
            self._remove_tokens(sid)
            del self._tf[sid]
            del self._len[sid]
            removed += 1
        return removed

    def score(self, query_tokens: Sequence[str], sid: str) -> float:
        counts = self._tf.get(sid)
        if not counts:
            return 0.0
        dl = self._len.get(sid, 0)
        avgdl = self.avgdl or 1.0
        total = 0.0
        for t in set(query_tokens):
            tf = counts.get(t, 0)
            if tf == 0:
                continue
            denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
            total += self.idf(t) * (tf * (self.k1 + 1.0)) / denom
        return total

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        q_tokens = tokenize(query)
        if not q_tokens or not self._tf:
            return []
        scored = [
            (sid, self.score(q_tokens, sid)) for sid in self._tf if self.score(q_tokens, sid) > 0.0
        ]
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:top_k]

    def clear(self) -> None:
        self._tf.clear()
        self._df.clear()
        self._len.clear()
        self._total_len = 0
