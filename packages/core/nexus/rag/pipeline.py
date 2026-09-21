"""RAG 编排：摄入 -> 检索 -> 精排 -> 生成 -> 引用回填。

这是系统中唯一知道"RAG 全流程"如何串起来的模块，
但它对具体的嵌入器、向量库、重排器、模型全部无感知 —— 它们都是注入进来的协议对象。
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import AsyncIterator, Sequence

from nexus.protocols import Embedder, LLMProvider, Reranker, Retriever, Tracer
from nexus.rag.chunker import RecursiveChunker, new_doc_id
from nexus.rag.prompt import build_messages
from nexus.types import Answer, Chunk, Document, Hit, Message, RetrieveMode, Usage

_CITE_RE = re.compile(r"\[(\d{1,2})\]")


async def _complete(llm: LLMProvider, messages: list[Message], provider: str | None):
    """调用模型。provider 参数仅路由层支持，直通单个 provider 时自动降级调用。"""
    if provider is None:
        return await llm.acomplete(messages)
    try:
        return await llm.acomplete(messages, provider=provider)  # type: ignore[call-arg]
    except TypeError:
        return await llm.acomplete(messages)


def _open_stream(llm: LLMProvider, messages: list[Message], provider: str | None):
    if provider is None:
        return llm.astream(messages)
    try:
        return llm.astream(messages, provider=provider)  # type: ignore[call-arg]
    except TypeError:
        return llm.astream(messages)


class RAGPipeline:
    """检索增强生成的编排器。

    Args:
        retriever: 融合检索器
        llm: 通常是 LLMRouter
        reranker: 可选精排；默认恒等
        tracer: 可选追踪
        cite_missing_policy: 模型未标注引用时，是否回填 top-1 命中作为兜底引用
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        tracer: Tracer | None = None,
        chunker: RecursiveChunker | None = None,
        top_k: int = 5,
        cite_missing_policy: bool = True,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.embedder = embedder
        self.reranker = reranker
        self.tracer = tracer
        self.chunker = chunker or RecursiveChunker()
        self.top_k = top_k
        self.cite_missing_policy = cite_missing_policy
        self._docs: dict[str, Document] = {}

    def add_document(
        self,
        title: str,
        text: str,
        doc_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> tuple[str, int]:
        """摄入一篇文档，返回 (doc_id, 切片数)。"""
        did = doc_id or new_doc_id()
        doc = Document(
            doc_id=did,
            title=title or "未命名文档",
            text=text,
            metadata=metadata or {},
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._docs[did] = doc
        chunks = self.chunker.split(doc)
        self.retriever.add(chunks)
        return did, len(chunks)

    def delete_document(self, doc_id: str) -> bool:
        self._docs.pop(doc_id, None)
        return self.retriever.delete_document(doc_id) > 0

    def documents(self) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for did, doc in self._docs.items():
            n = sum(
                1
                for c in getattr(self.retriever, "_chunks", {}).values()
                if c.doc_id == did
            )
            items.append(
                {
                    "doc_id": did,
                    "title": doc.title,
                    "n_chunks": n,
                    "n_chars": len(doc.text),
                    "created_at": doc.created_at,
                }
            )
        items.sort(key=lambda d: str(d["created_at"]), reverse=True)
        return items

    def chunk_preview(self, doc_id: str, limit: int = 8) -> list[Chunk]:
        chunks = [
            c
            for c in getattr(self.retriever, "_chunks", {}).values()
            if c.doc_id == doc_id
        ]
        chunks.sort(key=lambda c: c.index)
        return chunks[:limit]

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: RetrieveMode | None = None,
        trace_id: str | None = None,
    ) -> list[Hit]:
        """检索 + 精排。返回带 rank 与各阶段原始分数的命中列表。"""
        k = top_k or self.top_k
        ctx = None
        if self.tracer and trace_id:
            mode_name = str(mode or getattr(self.retriever, "mode", "hybrid"))
            ctx = self.tracer.span(trace_id, "retrieve", mode=mode_name, top_k=k)
            ctx.__enter__()
        try:
            hits = self.retriever.retrieve(query, k, mode)
            if self.reranker and hits:
                hits = self.reranker.rerank(query, hits, k)
            return hits
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)

    def _collect_citations(self, answer_text: str, hits: Sequence[Hit]) -> list[Hit]:
        """按模型标注的 [k] 回填命中；未标注则按策略兜底。"""
        cited: list[Hit] = []
        seen: set[str] = set()
        for m in _CITE_RE.finditer(answer_text):
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(hits):
                hit = hits[idx]
                if hit.chunk.chunk_id not in seen:
                    seen.add(hit.chunk.chunk_id)
                    cited.append(hit)
        if not cited and self.cite_missing_policy and hits:
            cited = [hits[0]]
        return cited

    async def answer(
        self,
        query: str,
        *,
        session_id: str = "",
        use_rag: bool = True,
        top_k: int | None = None,
        provider: str | None = None,
        history: Sequence[Message] | None = None,
        trace_id: str | None = None,
    ) -> Answer:
        sid = session_id or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        hits: list[Hit] = []
        if use_rag:
            hits = self.retrieve(query, top_k, trace_id=trace_id)

        gen_ctx = None
        if self.tracer and trace_id:
            gen_ctx = self.tracer.span(trace_id, "llm.generate", n_hits=len(hits))
            gen_ctx.__enter__()
        try:
            messages = build_messages(query, hits, history)
            resp = await _complete(self.llm, messages, provider)
        finally:
            if gen_ctx is not None:
                gen_ctx.__exit__(None, None, None)

        total_ms = (time.perf_counter() - t0) * 1000
        usage = Usage(
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            total_ms=round(total_ms, 2),
            ttft_ms=resp.usage.ttft_ms,
        )
        return Answer(
            session_id=sid,
            answer=resp.text,
            citations=self._collect_citations(resp.text, hits),
            tool_calls=[],
            trace_id=trace_id or "",
            usage=usage,
        )

    async def stream(
        self,
        query: str,
        *,
        session_id: str = "",
        use_rag: bool = True,
        top_k: int | None = None,
        provider: str | None = None,
        history: Sequence[Message] | None = None,
        trace_id: str | None = None,
    ) -> AsyncIterator[tuple[str, object]]:
        """流式生成。产出 (event, payload) 事件对，由网关序列化为 SSE。"""
        sid = session_id or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        hits: list[Hit] = []
        if use_rag:
            hits = self.retrieve(query, top_k, trace_id=trace_id)
            for h in hits:
                yield ("citation", h.to_dict())

        messages = build_messages(query, hits, history)
        buf: list[str] = []
        ttft_ms = 0.0
        astream = getattr(self.llm, "astream", None)
        if astream is None:
            raise ValueError("llm 不支持流式输出")
        async for piece in _open_stream(self.llm, messages, provider):
            if not buf:
                ttft_ms = (time.perf_counter() - t0) * 1000
            buf.append(piece)
            yield ("token", piece)

        text = "".join(buf)
        yield (
            "done",
            {
                "session_id": sid,
                "answer": text,
                "citations": [h.to_dict() for h in self._collect_citations(text, hits)],
                "trace_id": trace_id or "",
                "usage": {
                    "prompt_tokens": self.llm.count_tokens(query),
                    "completion_tokens": self.llm.count_tokens(text),
                    "total_ms": round((time.perf_counter() - t0) * 1000, 2),
                    "ttft_ms": round(ttft_ms, 2),
                },
            },
        )
