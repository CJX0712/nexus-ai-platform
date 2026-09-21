"""数值工具。向量化计算全部走 numpy，避免 Python 层循环成为瓶颈。"""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """按行 L2 归一化。零向量保持不变（不会除零）。"""
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x / max(n, eps)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a(n,d) 与 b(m,d) 的余弦相似度矩阵 (n,m)。要求输入已归一化。"""
    if b.size == 0:
        return np.zeros((a.shape[0], 0), dtype=np.float32)
    return np.asarray(a @ b.T, dtype=np.float32)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """返回分数最高的前 k 个下标，按分数降序。"""
    if scores.size == 0 or k <= 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, scores.size)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def stable_hash(token: str) -> int:
    """稳定整数哈希：同一 token 在任何进程/平台上结果一致。"""
    h = 2166136261
    for ch in token.encode("utf-8"):
        h = (h ^ ch) * 16777619
        h &= 0xFFFFFFFF
    return h


def signed_hash(token: str, dim: int) -> tuple[int, int]:
    """返回 (位置下标, 符号)。用于 hashing trick 的降维投影。"""
    h = stable_hash(token)
    return h % dim, 1 if (h >> 31) & 1 else -1


def percentile(values: list[float], p: float) -> float:
    """最近秩百分位。p 取 0~100。"""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round((p / 100.0) * (len(s) - 1))))
    return s[k]
