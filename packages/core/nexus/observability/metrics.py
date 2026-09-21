"""运行时指标。

只统计在这 falsy 会被 GC 的位置：所有 counter 都是进程内的，
重启即清零，够用就不引入 Prometheus 客户端依赖。
"""

from __future__ import annotations

import threading
from collections import deque

from nexus.util.mathx import percentile


class Metrics:
    def __init__(self, window: int = 500) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.tokens = 0
        self._latencies: deque[float] = deque(maxlen=window)

    def record(self, latency_ms: float, tokens: int = 0, error: bool = False) -> None:
        with self._lock:
            self.requests += 1
            self.tokens += tokens
            if error:
                self.errors += 1
            self._latencies.append(latency_ms)

    def snapshot(self, uptime_s: float) -> dict[str, float | int]:
        with self._lock:
            lat = list(self._latencies)
            return {
                "requests": self.requests,
                "errors": self.errors,
                "p50_ms": round(percentile(lat, 50), 2),
                "p95_ms": round(percentile(lat, 95), 2),
                "p99_ms": round(percentile(lat, 99), 2),
                "tokens_total": self.tokens,
                "uptime_s": round(uptime_s, 2),
            }
