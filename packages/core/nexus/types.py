"""跨模块共享的数据模型。

这一层只有数据、没有行为，任何模块都可以 import 而不会产生循环依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
RetrieveMode = Literal["hybrid", "vector", "bm25"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    name: str | None = None

    def as_dict(self) -> dict[str, str]:
        d: dict[str, str] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
    created_at: str = ""


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    index: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Hit:
    """一次检索返回的命中结果，保留各阶段原始分数以便解释融合行为。"""

    chunk: Chunk
    score: float
    rank: int = 0
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    stage: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.title,
            "text": self.chunk.text,
            "score": round(self.score, 6),
            "rank": self.rank,
            "vector_score": None if self.vector_score is None else round(self.vector_score, 6),
            "bm25_score": None if self.bm25_score is None else round(self.bm25_score, 6),
            "stage": self.stage,
        }


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_ms: float = 0.0
    ttft_ms: float = 0.0


@dataclass
class LLMResponse:
    text: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str = "stop"


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    result: str
    ok: bool
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": self.args,
            "result": self.result,
            "ok": self.ok,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class ToolResult:
    ok: bool
    output: str
    error: str = ""


@dataclass
class Answer:
    session_id: str
    answer: str
    citations: list[Hit] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    trace_id: str = ""
    usage: Usage = field(default_factory=Usage)
