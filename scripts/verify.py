"""一键验证：把「文档里写的那些命令」串成一条，任何一个红灯就整体失败。

    python scripts/verify.py
    python scripts/verify.py --skip-e2e     # 跳过需要起进程的端到端
    python scripts/verify.py --provider ollama

步骤：
    lint -> type-check -> unit test -> e2e -> eval -> P0 扫描（emoji 门禁）
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_E = "=+"

STEPS = [
    ("lint", [sys.executable, "-m", "ruff", "check", "packages/core"]),
    ("type-check", [sys.executable, "-m", "mypy", "--ignore-missing-imports", "packages/core/nexus"]),
    ("unit-test", [sys.executable, "-m", "pytest", "-q"]),
    ("smoke", [sys.executable, "scripts/smoke.py"]),
    ("eval", [sys.executable, "scripts/eval.py"]),
    ("emoji-scan", [sys.executable, "scripts/scan_emoji.py"]),
]


def run(name: str, cmd: list[str], cwd: Path) -> tuple[bool, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    ok = proc.returncode == 0
    tail = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(tail.splitlines()[-12:])
    return ok, tail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--provider", default="mock")
    args = ap.parse_args()

    plan = list(STEPS)
    if not args.skip_e2e:
        plan.insert(4, ("e2e", [sys.executable, "scripts/e2e.py", "--provider", args.provider]))
    if args.skip_eval:
        plan = [s for s in plan if s[0] != "eval"]

    print(f"{'=' * 72}\nnexus-ai-platform 验证\n{'=' * 72}")
    results: list[tuple[str, bool]] = []
    for name, cmd in plan:
        print(f"\n[{name}] {' '.join(cmd)}")
        ok, tail = run(name, cmd, ROOT)
        print(tail)
        print(f"  -> {'PASS' if ok else 'FAIL'}")
        results.append((name, ok))

    print(f"\n{'=' * 72}\n汇总")
    failed = [n for n, ok in results if not ok]
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 72)
    if failed:
        print(f"未通过: {', '.join(failed)}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
