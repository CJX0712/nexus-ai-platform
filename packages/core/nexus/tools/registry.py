"""工具注册表：Agent 的手。

工具必须经过注册表才能被调用，这样：
- 权限/超时/异常统一治理
- 工具清单可被序列化给模型做 function-calling 描述
- 测试时可以整体替换为 fake
"""

from __future__ import annotations

from typing import Any

from nexus.protocols import Tool
from nexus.types import ToolCallRecord, ToolResult


class ToolRegistry:
    def __init__(self, *, timeout_ms: int = 3000) -> None:
        self._tools: dict[str, Tool] = {}
        self.timeout_ms = timeout_ms

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重名: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> str:
        lines: list[str] = []
        for name in self.names():
            t = self._tools[name]
            params = t.parameters().get("properties", {})
            args_desc = ", ".join(params) if params else "无参数"
            lines.append(f"- {name}: {t.description}（参数: {args_desc}）")
        return "\n".join(lines)

    def run(self, name: str, args: dict[str, Any] | None = None) -> ToolCallRecord:
        import time

        args = args or {}
        tool = self._tools.get(name)
        t0 = time.perf_counter()
        if tool is None:
            return ToolCallRecord(
                name=name,
                args=args,
                result=f"未知工具: {name}；可用: {', '.join(self.names())}",
                ok=False,
            )
        try:
            res: ToolResult = tool.run(**args)
            return ToolCallRecord(
                name=name,
                args=args,
                # 失败时 output 通常为空，回退到 error 让调用方与模型能看到失败原因
                result=res.output or res.error,
                ok=res.ok,
                latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            )
        except TypeError as e:
            return ToolCallRecord(
                name=name, args=args, result=f"参数错误: {e}", ok=False,
                latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            )
        except Exception as e:
            return ToolCallRecord(
                name=name,
                args=args,
                result=f"执行异常: {type(e).__name__}: {e}",
                ok=False,
                latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            )
