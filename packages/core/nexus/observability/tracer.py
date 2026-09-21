"""轻量链路追踪。

记录每个请求的 span 树（检索 / 重排 / 生成等），
Web 控制台的火焰图直接消费这里的数据。
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

_MAX_TRACES = 200


@dataclass
class SpanRecord:
    span_id: str
    trace_id: str
    name: str
    parent_id: str | None
    start_ms: float
    dur_ms: float = 0.0
    attrs: dict[str, Any] = field(default_factory=dict)

    def set_attr(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def finish(self) -> None:
        if self.dur_ms == 0.0:
            self.dur_ms = round((time.perf_counter() - self.start_ms) * 1000, 3)

    def to_dict(self) -> dict[str, Any]:
        self.finish()
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_ms": round(self.start_ms, 3),
            "dur_ms": self.dur_ms,
            "attrs": self.attrs,
        }


class InMemoryTracer:
    """单进程内存追踪器，接口与 OpenTelemetry 风格一致，可平滑替换为 Jaeger / OTLP 导出。"""

    def __init__(self, max_traces: int = _MAX_TRACES) -> None:
        self.max_traces = max_traces
        self._traces: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []

    def start_trace(self, name: str, **attrs: Any) -> str:
        trace_id = uuid.uuid4().hex[:16]
        now = time.perf_counter()
        self._traces[trace_id] = {
            "trace_id": trace_id,
            "name": name,
            "_t0": now,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "attrs": dict(attrs),
            "spans": [],
        }
        self._order.append(trace_id)
        if len(self._order) > self.max_traces:
            dropped = self._order.pop(0)
            self._traces.pop(dropped, None)
        return trace_id

    @contextmanager
    def span(
        self,
        trace_id: str,
        name: str,
        parent_id: str | None = None,
        **attrs: Any,
    ):
        rec = SpanRecord(
            span_id=uuid.uuid4().hex[:12],
            trace_id=trace_id,
            name=name,
            parent_id=parent_id,
            start_ms=time.perf_counter(),
            attrs=dict(attrs),
        )
        trace = self._traces.get(trace_id)
        try:
            yield rec
        finally:
            rec.finish()
            t0 = float(trace["_t0"]) if trace else rec.start_ms
            # 统一为「相对 trace 起点的毫秒」，供前端火焰图直接使用
            rec.start_ms = max(0.0, (rec.start_ms - t0) * 1000)
            rec.dur_ms = max(rec.dur_ms, 0.0)
            if trace is not None:
                trace["spans"].append(rec)

    def finish_trace(self, trace_id: str) -> None:
        trace = self._traces.get(trace_id)
        if not trace:
            return
        trace["total_ms"] = round((time.perf_counter() - float(trace["_t0"])) * 1000, 3)
        trace["spans"].sort(key=lambda s: s.start_ms)

    def get(self, trace_id: str) -> dict[str, Any]:
        trace = self._traces.get(trace_id)
        if not trace:
            return {}
        return {
            "trace_id": trace["trace_id"],
            "name": trace["name"],
            "ts": trace["ts"],
            "total_ms": trace.get("total_ms", 0.0),
            "attrs": trace["attrs"],
            "spans": [s.to_dict() for s in trace["spans"]],
        }

    def list_traces(self, limit: int = 50) -> list[dict[str, Any]]:
        items = []
        for tid in reversed(self._order[-limit:]):
            t = self._traces[tid]
            items.append(
                {
                    "trace_id": tid,
                    "query": str(t["attrs"].get("query", ""))[:120],
                    "total_ms": t.get("total_ms", 0.0),
                    "n_spans": len(t["spans"]),
                    "ts": t["ts"],
                }
            )
        return items

    def clear(self) -> None:
        self._traces.clear()
        self._order.clear()
