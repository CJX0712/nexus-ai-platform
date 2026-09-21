"""进程内精确向量库。

零依赖、零网络。用于本地运行、单元测试与 CI；生产可替换为
Qdrant / pgvector 实现，因为它们满足同一个 VectorStore 协议。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from nexus.util.mathx import l2_normalize, top_k_indices


class InMemoryVectorStore:
    """基于 numpy 矩阵的精确余弦检索（输入向量必须是单位向量）。

    不变量：
        1. 同一 id 重复 upsert 不会新增行，而是覆盖
        2. 检索自身必然返回 (自身, score=1.0)，误差 <= 1e-5
        3. delete 不存在的 id 返回 0，不影响矩阵
    """

    name = "memory"
    _GROW_MIN = 64

    def __init__(self, dim: int, path: str | None = None) -> None:
        self.dim = dim
        self._path = Path(path) if path else None
        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._mat = np.zeros((0, dim), dtype=np.float32)
        if self._path and self._path.with_suffix(".npz").exists():
            self.load()

    def _ensure_capacity(self, extra: int) -> None:
        need = len(self._ids) + extra
        cap = self._mat.shape[0]
        if need <= cap:
            return
        new_cap = max(self._GROW_MIN, cap * 2, need)
        pad = np.zeros((new_cap - cap, self.dim), dtype=np.float32)
        self._mat = np.vstack([self._mat, pad])

    def upsert(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        vec = np.asarray(vectors, dtype=np.float32)
        if vec.ndim != 2 or vec.shape[1] != self.dim:
            raise ValueError(f"期望 (n,{self.dim}) 矩阵，实际 {vec.shape}")
        if len(ids) != vec.shape[0]:
            raise ValueError("ids 与向量行数不一致")
        vec = l2_normalize(vec)
        self._ensure_capacity(len(ids))
        for i, sid in enumerate(ids):
            row = self._index.get(sid)
            if row is None:
                row = len(self._ids)
                self._index[sid] = row
                self._ids.append(sid)
            self._mat[row] = vec[i]
        self._flush()

    def delete(self, ids: Sequence[str]) -> int:
        rows = [self._index.pop(i) for i in ids if i in self._index]
        if not rows:
            return 0
        keep_mask = np.ones(len(self._ids), dtype=bool)
        for r in rows:
            keep_mask[r] = False
        kept_ids = [i for i, k in zip(self._ids, keep_mask, strict=False) if k]
        self._mat = np.ascontiguousarray(self._mat[: len(self._ids)][keep_mask])
        self._ids = kept_ids
        self._index = {sid: i for i, sid in enumerate(self._ids)}
        self._flush()
        return len(rows)

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if not self._ids or top_k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        q = l2_normalize(q)
        sims = np.asarray(self._mat[: len(self._ids)] @ q.T, dtype=np.float32).ravel()
        out: list[tuple[str, float]] = []
        for idx in top_k_indices(sims, top_k):
            score = float(sims[int(idx)])
            out.append((self._ids[int(idx)], max(-1.0, min(1.0, score))))
        return out

    def count(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self._ids.clear()
        self._index.clear()
        self._mat = np.zeros((0, self.dim), dtype=np.float32)
        self._flush()

    def ids(self) -> list[str]:
        return list(self._ids)

    def save(self, path: str | None = None) -> None:
        target = Path(path) if path else self._path
        if target is None:
            return
        np.savez(target.with_suffix(".npz"), mat=self._mat[: len(self._ids)])
        meta: dict[str, Any] = {"ids": self._ids, "dim": self.dim}
        target.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")

    def load(self, path: str | None = None) -> None:
        target = Path(path) if path else self._path
        if target is None:
            return
        data = np.load(target.with_suffix(".npz"))
        meta = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
        self.dim = int(meta["dim"])
        self._ids = list(meta["ids"])
        self._index = {sid: i for i, sid in enumerate(self._ids)}
        self._mat = np.zeros((max(self._GROW_MIN, len(self._ids)), self.dim), dtype=np.float32)
        if len(self._ids):
            self._mat[: len(self._ids)] = data["mat"]

    def _flush(self) -> None:
        if self._path:
            self.save()
