"""内置工具集：全部零依赖、可离线、执行时间可控。

安全约定：calculator 走 AST 白名单求值，不 eval 任意字符串；
http_fetch 默认禁用，需显式开启且限制响应体大小，避免 SSRF 与内存放大。
"""

from __future__ import annotations

import ast
import json
import operator
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

from nexus.types import ToolResult

_OPS: dict[type, Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_MAX_POW = 64
_CALC_RE = re.compile(r"^[\d\s+\-*/().%]+$")


def safe_eval(expr: str) -> float:
    """受限数学表达式求值。任意非白名单 AST 节点都会抛出 ValueError。"""
    src = expr.strip()
    if not src or len(src) > 200:
        raise ValueError("表达式过长或为空")
    tree = ast.parse(src, mode="eval")

    def _walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int | float):
                raise ValueError("只允许数字常量")
            return float(node.value)
        if isinstance(node, ast.BinOp):
            op = _OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
            if isinstance(node.op, ast.Pow):
                raise ValueError("不支持幂运算")
            return op(_walk(node.left), _walk(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _OPS.get(type(node.op))
            if op is None:
                raise ValueError("不支持的一元运算")
            return op(_walk(node.operand))
        raise ValueError(f"不允许的表达式节点: {type(node).__name__}")

    return _walk(tree)


class CalculatorTool:
    name = "calculator"
    description = "计算数学表达式，支持 + - * / % 与括号"

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "数学表达式"}},
            "required": ["expression"],
        }

    def run(self, **kwargs: Any) -> ToolResult:
        expr = str(kwargs.get("expression", ""))
        if not _CALC_RE.match(expr):
            return ToolResult(ok=False, output="", error="表达式含非法字符")
        try:
            value = safe_eval(expr)
        except ZeroDivisionError:
            return ToolResult(ok=False, output="", error="除数为零")
        except Exception as e:
            return ToolResult(ok=False, output="", error=f"求值失败: {e}")
        if isinstance(value, float) and value.is_integer():
            return ToolResult(ok=True, output=str(int(value)))
        return ToolResult(ok=True, output=f"{value:.6g}")


class NowTool:
    name = "now"
    description = "获取当前本地时间"

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs: Any) -> ToolResult:
        return ToolResult(ok=True, output=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class TextStatsTool:
    name = "text_stats"
    description = "统计文本的字符数、中文字符数、词数与估算 token 数"

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "待统计文本"}},
            "required": ["text"],
        }

    def run(self, **kwargs: Any) -> ToolResult:
        from nexus.util.text import estimate_tokens, tokenize

        text = str(kwargs.get("text", ""))
        cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
        stats = {
            "chars": len(text),
            "cjk_chars": cjk,
            "tokens": len(tokenize(text)),
            "est_tokens": estimate_tokens(text),
        }
        return ToolResult(ok=True, output=json.dumps(stats, ensure_ascii=False))


class KnowledgeSearchTool:
    """把检索能力暴露成工具，让 Agent 能自主决定何时去查知识库。"""

    name = "kb_search"
    description = "在知识库中检索与给定查询相关的片段"

    def __init__(self, retriever: Any, top_k: int = 3) -> None:
        self._retriever = retriever
        self.top_k = top_k

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "检索关键词"}},
            "required": ["query"],
        }

    def run(self, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query", ""))
        hits = self._retriever.retrieve(query, self.top_k)
        if not hits:
            return ToolResult(ok=True, output="知识库中没有命中内容")
        lines = [f"[{i}] {h.chunk.text[:200]}" for i, h in enumerate(hits, start=1)]
        return ToolResult(ok=True, output="\n".join(lines))


def builtin_tools(retriever: Any | None = None) -> list[Any]:
    tools: list[Any] = [CalculatorTool(), NowTool(), TextStatsTool()]
    if retriever is not None:
        tools.append(KnowledgeSearchTool(retriever))
    return tools
