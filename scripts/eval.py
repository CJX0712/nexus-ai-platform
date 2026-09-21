"""回归评测入口：加载固定语料 -> 跑样例集 -> 与阈值比较 -> 给出退出码。

用法：
    python scripts/eval.py                     # 默认 mock provider，离线可跑
    python scripts/eval.py --provider ollama   # 用本地真实模型
    python scripts/eval.py --mode bm25         # 只测稀疏通道
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "core"))

from nexus.config import Settings  # noqa: E402
from nexus.container import build_system  # noqa: E402
from nexus.eval.runner import load_dataset, run_retrieval_eval  # noqa: E402

DATA = ROOT / "eval" / "datasets"


def load_corpus() -> list[dict]:
    rows = []
    for line in (DATA / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock")
    ap.add_argument("--mode", default=None, choices=[None, "hybrid", "vector", "bm25"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument(
        "--embed-provider", default="hash", choices=["hash", "ollama", "openai"],
        help="hash=离线零依赖；ollama=本地稠密嵌入 bge-m3",
    )
    ap.add_argument("--embed-model", default="bge-m3")
    ap.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = ap.parse_args()

    s = Settings(
        provider=args.provider,
        data_dir=str(ROOT / "data"),
        top_k=args.top_k,
        embed_provider=args.embed_provider,
        ollama_embed_model=args.embed_model,
    )
    if args.mode:
        s.rag_mode = args.mode  # type: ignore[assignment]
    system = build_system(s)

    corpus = load_corpus()
    for row in corpus:
        system.pipeline.add_document(row["title"], row["text"], row["doc_id"])
    cases = load_dataset(DATA / "queries.jsonl")

    report = await run_retrieval_eval(system, cases, args.top_k)
    raw = json.loads((DATA / "thresholds.json").read_text(encoding="utf-8"))
    override = dict(raw.get("overrides", {}).get(args.provider, {}))
    override.pop("note", None)
    thresholds = {**{k: v for k, v in raw.items() if k not in ("overrides", "note")}, **override}
    agg = report.aggregate

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"\n评测 provider={args.provider} mode={s.rag_mode} 样例数={len(cases)}")
        print("-" * 92)
        print(f"{'查询':<34}{'recall@5':>9}{'mrr@5':>8}{'faith':>8}{'延迟ms':>9}  命中文档")
        print("-" * 92)
        for c in report.cases:
            q = c.query[:32] + ".." if len(c.query) > 34 else c.query
            print(
                f"{q:<34}{c.recall_at_5:>9.2f}{c.mrr_at_5:>8.2f}"
                f"{c.faithfulness:>8.2f}{c.latency_ms:>9.1f}  {c.top_doc_id}"
            )
        print("-" * 92)
        for k in ("recall@5", "mrr@5", "ndcg@5", "citation_precision", "p50_ms", "p95_ms"):
            print(f"  {k:<20}{agg.get(k, 0):>10.3f}")

    # 质量类指标要求「不低于」阈值，延迟类指标要求「不高于」阈值
    higher_is_better = ("recall@5", "mrr@5", "ndcg@5", "citation_precision")
    failures = []
    for k, v in thresholds.items():
        if k == "note":
            continue
        actual = agg.get(k)
        if actual is None:
            continue
        bad = actual < v if k in higher_is_better else actual > v
        if bad:
            failures.append(f"{k}: {actual:.3f} 未达阈值 {v}")

    if failures:
        print("\n未达标：")
        for f in failures:
            print(f"  FAIL  {f}")
        return 1
    print("\n所有指标达标 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
