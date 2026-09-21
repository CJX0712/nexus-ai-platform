"""基于本地 Ollama 的真实语义嵌入。

相比哈希/TF-IDF 这类「词面相似」，稠密嵌入能处理同义改写：
用户问"怎么给 Rerank 加旁路"，证据写"重排器语言不匹配时降级"，
字面几乎不重叠，但语义几乎一致 —— 这正是稀疏检索补不上的那一半。

推荐模型：
    bge-m3        1024 维，中英双语强，支持长文本
    nomic-embed-text  768 维，体积小、速度快
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np
from nexus.util.mathx import l2_normalize


class OllamaEmbedder:
    """调用 Ollama /api/embeddings 生成稠密向量。

    Args:
        base_url: Ollama 服务地址
        model: 嵌入模型名
        probe: 初始化时是否探测一次以确认维度与服务可用性
    """

    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "bge-m3",
        timeout: float = 60.0,
        probe: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.dim = 0
        self._client: httpx.Client | None = None
        if probe:
            vec = self._post_one("维度探测")
            self.dim = len(vec)

    @property
    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout, limits=httpx.Limits(max_connections=8)
            )
        return self._client

    def _post_one(self, text: str) -> list[float]:
        resp = self._http.post(
            f"{self.base_url}/api/embeddings", json={"model": self.model, "prompt": text}
        )
        resp.raise_for_status()
        body: dict[str, Any] = resp.json()
        vec = body.get("embedding")
        if vec is None:
            # 新版 API 可能返回 {"embeddings": [[...]]}
            emb = body.get("embeddings")
            if emb and isinstance(emb[0], list):
                vec = emb[0]
        if not vec:
            raise RuntimeError(f"[{self.model}] 未返回嵌入向量")
        return [float(x) for x in vec]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = [self._post_one(t if t.strip() else " ") for t in texts]
        mat = np.asarray(rows, dtype=np.float32)
        if self.dim == 0:
            self.dim = mat.shape[1]
        return l2_normalize(mat)

    async def aembed(self, texts: Sequence[str]) -> np.ndarray:
        return await asyncio.to_thread(self.embed, list(texts))

    def observe(self, texts: Sequence[str]) -> None:
        """稠密嵌入没有需要维护的语料统计，这里保持空实现以满足协议。"""

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
