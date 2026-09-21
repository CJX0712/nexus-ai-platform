"""所有模块的接口契约（端口）。

本文件是系统的"宪法"：任何实现都必须满足这里的签名，
任何调用方都只能通过 Protocol 依赖具体实现。
这样每个模块都能被 fake 替换，从而在没有网络、没有 API Key、
没有数据库的环境下完成单元测试与端到端验证。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from nexus.types import Chunk, Document, Hit, LLMResponse, Message, RetrieveMode, ToolResult

__all__ = [
    "Chunker",
    "Embedder",
    "LLMProvider",
    "MemoryStore",
    "Reranker",
    "Retriever",
    "Span",
    "Tool",
    "Tracer",
    "VectorStore",
]


@runtime_checkable
class LLMProvider(Protocol):
    """大模型提供方统一接口。所有厂商差异被吸收在这里。"""

    name: str

    @property
    def model(self) -> str: ...

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> LLMResponse: ...

    def astream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        """异步生成器：实现用 `async def` + `yield` 即可满足本契约。"""
        ...

    def count_tokens(self, text: str) -> int: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class Embedder(Protocol):
    """文本向量化。CPU 密集，因此同步实现 + 异步包装。"""

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """返回 shape=(n, dim) 的 float32 矩阵，每行必须是 L2 单位向量。"""
        ...

    async def aembed(self, texts: Sequence[str]) -> np.ndarray: ...

    def observe(self, texts: Sequence[str]) -> None:
        """增量更新统计信息（如 IDF）。增量摄入语料时调用。"""
        ...


@runtime_checkable
class VectorStore(Protocol):
    name: str

    def upsert(self, ids: Sequence[str], vectors: np.ndarray) -> None: ...
    def delete(self, ids: Sequence[str]) -> int: ...
    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    name: str

    def add(self, chunks: Sequence[Chunk]) -> None: ...
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: RetrieveMode | None = None,
    ) -> list[Hit]: ...
    def delete_document(self, doc_id: str) -> int: ...
    def count(self) -> int: ...


@runtime_checkable
class Reranker(Protocol):
    """精排。实现必须自行决定是否适用（例如查询语言与模型语言不匹配时降级）。"""

    name: str

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        ...


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str

    def parameters(self) -> dict[str, Any]: ...
    def run(self, **kwargs: Any) -> ToolResult: ...


@runtime_checkable
class MemoryStore(Protocol):
    def append(self, session_id: str, message: Message) -> None: ...
    def window(self, session_id: str, n: int = 8) -> list[Message]: ...
    def summary(self, session_id: str) -> str: ...
    def reset(self, session_id: str) -> None: ...


class Span(Protocol):
    span_id: str

    def set_attr(self, key: str, value: Any) -> None: ...
    def finish(self) -> None: ...


@runtime_checkable
class Tracer(Protocol):
    def start_trace(self, name: str, **attrs: Any) -> str: ...
    def span(self, trace_id: str, name: str, parent_id: str | None = None, **attrs: Any) -> Any: ...
    def finish_trace(self, trace_id: str) -> None: ...
    def get(self, trace_id: str) -> dict[str, Any]: ...
    def list_traces(self, limit: int = 50) -> list[dict[str, Any]]: ...


@runtime_checkable
class Chunker(Protocol):
    def split(self, doc: Document) -> list[Chunk]: ...
