"""零依赖 TF-IDF 哈希嵌入器。

用 hashing trick 把「子词 TF-IDF」投影到固定维度，得到稠密单位向量。
相比纯哈希词袋，IDF 加权让高频停用词的贡献被压低、
低频特征词的区分度被抬高 —— 这是中文场景下向量通道质量的关键。

不变量：
    1. 输出每行 L2 范数 == 1（零向量文本除外，返回全 0）
    2. 相同输入必然产出相同向量（无随机初始化）
    3. observe() 只更新统计，不会让已有向量突然改变语义方向
"""

from __future__ import annotations

import asyncio
import math
import threading
from collections import Counter
from collections.abc import Sequence

import numpy as np
from nexus.util.mathx import l2_normalize, signed_hash
from nexus.util.text import char_ngrams, tokenize


class TfidfHashingEmbedder:
    """子词 TF-IDF + hashing trick。

    Args:
        dim: 目标维度，默认 384（兼顾精度与内存）
        use_idf: 是否启用 IDF 加权，关闭后退化为加权词袋
        ngram: 额外加入的字符 n-gram 长度，提升对拼写变体的鲁棒性
    """

    name = "tfidf-hashing"

    def __init__(self, dim: int = 384, use_idf: bool = True, ngram: int = 2) -> None:
        if dim <= 0:
            raise ValueError("dim 必须为正整数")
        self.dim = dim
        self.use_idf = use_idf
        self.ngram = ngram
        self._df: dict[str, int] = {}
        self._n_docs = 0
        self._lock = threading.Lock()

    @property
    def n_docs(self) -> int:
        return self._n_docs

    def _features(self, text: str) -> Counter[str]:
        c: Counter[str] = Counter(tokenize(text))
        if self.ngram > 1:
            for g in char_ngrams(text, self.ngram):
                c[g] += 1
        return c

    def observe(self, texts: Sequence[str]) -> None:
        """增量更新文档频率统计。摄入新文档时必须调用，否则 IDF 失效。"""
        with self._lock:
            for t in texts:
                self._n_docs += 1
                for tok in self._features(t):
                    self._df[tok] = self._df.get(tok, 0) + 1

    def idf(self, token: str) -> float:
        if not self.use_idf or self._n_docs == 0:
            return 1.0
        df = self._df.get(token, 0)
        return math.log((1.0 + self._n_docs) / (1.0 + df)) + 1.0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        with self._lock:
            for r, text in enumerate(texts):
                feats = self._features(text)
                for tok, cnt in feats.items():
                    idx, sign = signed_hash(tok, self.dim)
                    tf = 1.0 + math.log(cnt)
                    out[r, idx] += tf * self.idf(tok) * sign
        return l2_normalize(out)

    async def aembed(self, texts: Sequence[str]) -> np.ndarray:
        """CPU 密集，卸载到线程执行，避免阻塞事件循环。"""
        return await asyncio.to_thread(self.embed, list(texts))
