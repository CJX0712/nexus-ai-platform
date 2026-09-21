"""评测指标。

指标必须能回答三个不同层面的问题：
- 检索层：有没有把正确的东西找回来（recall / mrr / ndcg）
- 生成层：答案有没有胡说（忠实度：答案中的字符是否来自证据）
- 性能层：快不快、稳不稳（p50 / p95 延迟）

忠实度这里用「字符 n-gram 覆盖率」做近似：把答案切成字符二元组，
统计有多少比例能在被检索到的证据里找到。它不完美，但在没有裁判模型
（LLM-as-judge）的环境下，是一个可复现、可与人类直觉对齐的下界估计。
"""

from __future__ import annotations

from collections.abc import Sequence

from nexus.util.mathx import percentile


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """前 k 条结果里召回了多少比例的必需文档。"""
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & set(relevant)) / len(set(relevant))


def mrr_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """第一个正确答案出现在第几位（倒数）。"""
    rel = set(relevant)
    for i, item in enumerate(retrieved[:k], start=1):
        if item in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """文档粒度的 nDCG。二元相关性下 nDCG 必然落在 [0,1]，
    若传入含重复 id 的列表会算出大于 1 的值，因此这里强制保序去重。
    """
    import math

    rel = set(relevant)
    retrieved = list(dict.fromkeys(retrieved))
    dcg = sum(1.0 / math.log2(i + 1) for i, item in enumerate(retrieved[:k], start=1) if item in rel)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


def char_ngrams(text: str, n: int = 2) -> set[str]:
    s = "".join(ch for ch in text if not ch.isspace())
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def faithfulness(answer: str, evidence: Sequence[str], n: int = 2) -> float:
    """答案里有多少比例的字符二元组能在证据中找到。

    这是「没有编造」的必要不充分条件：高覆盖率不代表推理正确，
    但低覆盖率基本可以断定模型在自由发挥。
    """
    ans_grams = char_ngrams(answer, n)
    if not ans_grams:
        return 0.0
    gold: set[str] = set()
    for ev in evidence:
        gold |= char_ngrams(ev, n)
    if not gold:
        return 0.0
    return len(ans_grams & gold) / len(ans_grams)


def citation_precision(cited_ids: Sequence[str], relevant: Sequence[str]) -> float:
    """引用的证据里有多少是真正常用的。"""
    if not cited_ids:
        return 0.0
    rel = set(relevant)
    return sum(1 for c in cited_ids if c in rel) / len(cited_ids)


def latency_stats(samples: Sequence[float]) -> dict[str, float]:
    return {
        "n": float(len(samples)),
        "p50_ms": round(percentile(list(samples), 50), 2),
        "p95_ms": round(percentile(list(samples), 95), 2),
        "max_ms": round(max(samples), 2) if samples else 0.0,
    }
