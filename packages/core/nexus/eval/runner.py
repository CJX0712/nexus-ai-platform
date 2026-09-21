"""评测数据集加载与批量运行。

评测不是"跑一遍看感觉"，而是回归门禁：
每次改动检索或生成链路，都必须能给出与上次可比较的数字。
因此数据集用 JSONL 固化、指标公式固定、阈值写在 `thresholds.json` 里。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nexus.eval.metrics import (
    citation_precision,
    faithfulness,
    latency_stats,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


@dataclass
class EvalCase:
    """一条评测样例。

    relevant 用文档 id 而非分片 id：同一篇文档的不同分片命中都应算命中，
    用分片 id 会把「切片策略调整」误判成「检索失败」。
    """

    query: str
    relevant_doc_ids: list[str]
    reference_answer: str = ""
    notes: str = ""


@dataclass
class CaseResult:
    query: str
    recall_at_5: float
    mrr_at_5: float
    ndcg_at_5: float
    faithfulness: float
    citation_precision: float
    latency_ms: float
    top_doc_id: str = ""
    answer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "recall@5": round(self.recall_at_5, 4),
            "mrr@5": round(self.mrr_at_5, 4),
            "ndcg@5": round(self.ndcg_at_5, 4),
            "faithfulness": round(self.faithfulness, 4),
            "citation_precision": round(self.citation_precision, 4),
            "latency_ms": round(self.latency_ms, 2),
            "top_doc_id": self.top_doc_id,
        }


@dataclass
class EvalReport:
    cases: list[CaseResult] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate": self.aggregate,
            "cases": [c.to_dict() for c in self.cases],
        }


def load_dataset(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        obj = json.loads(line)
        cases.append(
            EvalCase(
                query=obj["query"],
                relevant_doc_ids=list(obj.get("relevant_doc_ids", [])),
                reference_answer=obj.get("reference_answer", ""),
                notes=obj.get("notes", ""),
            )
        )
    return cases


async def run_retrieval_eval(system: Any, cases: list[EvalCase], top_k: int = 5) -> EvalReport:
    """跑一轮检索 + 生成评测。返回逐条结果 + 汇总。"""
    report = EvalReport()
    latencies: list[float] = []

    for case in cases:
        t0 = time.perf_counter()
        hits = system.pipeline.retrieve(case.query, top_k)
        # 同一文档可能有多个分片命中，评测在文档粒度进行，必须先保序去重
        retrieved_ids = list(dict.fromkeys(h.chunk.doc_id for h in hits))
        evidence = [h.chunk.text for h in hits]

        answer = ""
        cited: list[str] = []
        if hasattr(system.pipeline, "answer"):
            ans = await system.pipeline.answer(case.query, top_k=top_k)
            answer = ans.answer
            cited = [c.chunk.doc_id for c in ans.citations]

        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        report.cases.append(
            CaseResult(
                query=case.query,
                recall_at_5=recall_at_k(retrieved_ids, case.relevant_doc_ids, top_k),
                mrr_at_5=mrr_at_k(retrieved_ids, case.relevant_doc_ids, top_k),
                ndcg_at_5=ndcg_at_k(retrieved_ids, case.relevant_doc_ids, top_k),
                faithfulness=faithfulness(answer, evidence),
                citation_precision=citation_precision(cited, case.relevant_doc_ids),
                latency_ms=latency,
                top_doc_id=hits[0].chunk.doc_id if hits else "",
                answer=answer,
            )
        )

    n = len(report.cases) or 1
    report.aggregate = {
        "n_cases": float(len(report.cases)),
        "recall@5": round(sum(c.recall_at_5 for c in report.cases) / n, 4),
        "mrr@5": round(sum(c.mrr_at_5 for c in report.cases) / n, 4),
        "ndcg@5": round(sum(c.ndcg_at_5 for c in report.cases) / n, 4),
        "faithfulness": round(sum(c.faithfulness for c in report.cases) / n, 4),
        "citation_precision": round(sum(c.citation_precision for c in report.cases) / n, 4),
    }
    report.aggregate.update({k: v for k, v in latency_stats(latencies).items()})
    return report
