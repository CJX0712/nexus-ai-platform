"""P0 门禁：扫描交付物，禁止用 emoji 充当功能图标。

为什么禁止：
    emoji 在不同操作系统与字体下渲染差异极大（Windows 上是彩色表情、
    部分 Linux 上是黑白线条），会破坏界面的一致性，
    而且无法控制描边粗细与尺寸，缩放后会糊。

允许的例外：用户生成内容（UGC）与聊天消息本身。
因此本脚本只扫描源码、文档和界面代码，不扫描运行时数据。

用法：
    python scripts/scan_emoji.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f9ff"
    "\U00002600-\U000026ff"
    "\U00002700-\U000027bf"
    "\U0000fe00-\U0000fe0f"
    "\U0001f000-\U0001f02f"
    "\U0001f0a0-\U0001f0ff"
    "\U0001f100-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\U0000200d"
    "\U000020e3"
    "\U000e0020-\U000e007f"
    "]+",
    flags=re.UNICODE,
)

SCAN_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".html", ".css", ".md", ".json", ".yml", ".yaml"}
SKIP_DIRS = {"node_modules", "dist", ".git", "__pycache__", ".venv", "data", ".pytest_cache", ".mypy_cache"}
SKIP_FILES = {"scan_emoji.py"}


def scan() -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in SKIP_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if EMOJI_RE.search(line):
                rel = str(path.relative_to(ROOT))
                hits.append((rel, lineno, line.strip()[:90]))
    return hits


def main() -> int:
    hits = scan()
    if not hits:
        print("emoji 扫描通过：未发现用作图标的 emoji")
        return 0
    print(f"发现 {len(hits)} 处 emoji：")
    for rel, lineno, line in hits[:40]:
        print(f"  {rel}:{lineno}  {line}")
    if len(hits) > 40:
        print(f"  ... 另有 {len(hits) - 40} 处")
    return 1


if __name__ == "__main__":
    sys.exit(main())
