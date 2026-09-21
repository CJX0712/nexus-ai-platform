"""HTTP 网关：REST + SSE 的唯一对外入口。

职责边界：网关只做「协议转换 + 参数校验 + 追踪/指标埋点」，
不含任何业务逻辑；所有业务都委托给 RAGPipeline / ReactAgent。
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from nexus.container import System
from nexus.gateway.sse import sse_frame
from nexus.types import RetrieveMode
from nexus.version import __version__
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)
    doc_id: str | None = None
    metadata: dict[str, str] = {}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: RetrieveMode | None = None


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    use_rag: bool = True
    use_agent: bool | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    provider: str | None = None


def register_routes(app: FastAPI, sys_: System) -> FastAPI:
    started = time.time()
    app.state.system = sys_
    app.state.started = started

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "version": __version__,
            "provider": sys_.llm.default,
            "providers": sys_.llm.available,
            "model": sys_.llm.model,
            "uptime_s": round(time.time() - started, 2),
            "documents": len(sys_.pipeline.documents()),
            "chunks": sys_.retriever.count(),
        }

    @app.post("/api/v1/ingest")
    async def ingest(req: IngestRequest) -> dict[str, object]:
        doc_id, n = sys_.pipeline.add_document(req.title, req.text, req.doc_id, req.metadata)
        return {"doc_id": doc_id, "n_chunks": n, "n_chars": len(req.text)}

    @app.get("/api/v1/documents")
    async def list_documents() -> dict[str, object]:
        items = sys_.pipeline.documents()
        return {"items": items, "total": len(items)}

    @app.get("/api/v1/documents/{doc_id}/chunks")
    async def document_chunks(doc_id: str, limit: int = 8) -> dict[str, object]:
        chunks = sys_.pipeline.chunk_preview(doc_id, limit)
        return {
            "items": [
                {"chunk_id": c.chunk_id, "index": c.index, "text": c.text} for c in chunks
            ],
            "total": len(chunks),
        }

    @app.delete("/api/v1/documents/{doc_id}")
    async def delete_document(doc_id: str) -> dict[str, object]:
        ok = sys_.pipeline.delete_document(doc_id)
        if not ok:
            raise HTTPException(status_code=404, detail="文档不存在")
        return {"deleted": True, "doc_id": doc_id}

    @app.post("/api/v1/search")
    async def search(req: SearchRequest) -> dict[str, object]:
        trace_id = sys_.tracer.start_trace("search", query=req.query)
        t0 = time.perf_counter()
        try:
            hits = sys_.pipeline.retrieve(req.query, req.top_k, req.mode, trace_id=trace_id)
            return {
                "hits": [h.to_dict() for h in hits],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "trace_id": trace_id,
            }
        finally:
            sys_.tracer.finish_trace(trace_id)

    @app.post("/api/v1/chat")
    async def chat(req: ChatRequest) -> dict[str, object]:
        t0 = time.perf_counter()
        trace_id = sys_.tracer.start_trace("chat", query=req.query, use_agent=bool(req.use_agent))
        try:
            history = sys_.memory.context(req.session_id or "") if req.session_id else []
            use_agent = req.use_agent if req.use_agent is not None else sys_.settings.auto_agent
            if use_agent:
                with sys_.tracer.span(trace_id, "agent.react"):
                    ans, _ = await sys_.agent.run(
                        req.query, session_id=req.session_id or "", history=history
                    )
            else:
                ans = await sys_.pipeline.answer(
                    req.query,
                    session_id=req.session_id or "",
                    use_rag=req.use_rag,
                    top_k=req.top_k,
                    provider=req.provider,
                    history=history,
                    trace_id=trace_id,
                )
            if req.session_id:
                sys_.memory.append(req.session_id, ans_to_user_message(req.query))
                sys_.memory.append(req.session_id, ans_to_assistant_message(ans.answer))
            sys_.metrics.record((time.perf_counter() - t0) * 1000, ans.usage.completion_tokens)
            return {
                "session_id": ans.session_id,
                "answer": ans.answer,
                "citations": [c.to_dict() for c in ans.citations],
                "tool_calls": [t.to_dict() for t in ans.tool_calls],
                "trace_id": trace_id,
                "usage": ans.usage.__dict__,
            }
        except Exception:
            sys_.metrics.record((time.perf_counter() - t0) * 1000, 0, error=True)
            raise
        finally:
            sys_.tracer.finish_trace(trace_id)

    @app.post("/api/v1/chat/stream")
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        history = sys_.memory.context(req.session_id or "") if req.session_id else []
        trace_id = sys_.tracer.start_trace("chat.stream", query=req.query)
        sid = req.session_id or ""

        async def gen():
            buf: list[str] = []
            try:
                async for event, payload in sys_.pipeline.stream(
                    req.query,
                    session_id=sid,
                    use_rag=req.use_rag,
                    top_k=req.top_k,
                    provider=req.provider,
                    history=history,
                    trace_id=trace_id,
                ):
                    if event == "token":
                        buf.append(str(payload))
                    yield sse_frame(event, payload)
            except Exception as e:
                yield sse_frame("error", {"message": f"{type(e).__name__}: {e}"})
            finally:
                sys_.tracer.finish_trace(trace_id)
                if sid:
                    sys_.memory.append(sid, ans_to_assistant_message("".join(buf)))

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/v1/traces")
    async def list_traces(limit: int = 50) -> dict[str, object]:
        items = sys_.tracer.list_traces(limit)
        return {"items": items, "total": len(items)}

    @app.get("/api/v1/traces/{trace_id}")
    async def get_trace(trace_id: str) -> dict[str, object]:
        trace = sys_.tracer.get(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="trace 不存在")
        return trace

    @app.get("/api/v1/metrics")
    async def get_metrics() -> dict[str, object]:
        return sys_.metrics.snapshot(time.time() - started)

    return app


def ans_to_user_message(query: str):
    from nexus.types import Message

    return Message(role="user", content=query)


def ans_to_assistant_message(answer: str):
    from nexus.types import Message

    return Message(role="assistant", content=answer)
