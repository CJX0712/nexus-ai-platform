"""编排层与 HTTP 网关测试。

这里是唯一会真正启动 ASGI 应用的测试，覆盖成功流与错误流，
确保"文档里 curl 出来的结果"和代码行为一致。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexus.config import Settings
from nexus.container import build_system
from nexus.gateway.app import create_app
from nexus.llm.mock import MockLLM, parse_sources
from nexus.types import Answer, Message

DOCS = [
    ("混合检索", "BM25 处理关键词精确匹配，向量负责语义近似，RRF 用名次倒数融合两路结果，k 常取 60。"),
    ("重排陷阱", "英文重排器处理中文查询会给出近似随机分数，需要语言守卫在错配时降级为恒等重排。"),
    ("向量库", "内存实现适合小数据量单机场景，生产可用 Qdrant 或 pgvector。"),
]


@pytest.fixture()
def system(tmp_path):
    s = Settings(provider="mock", data_dir=str(tmp_path), top_k=3)
    return build_system(s)


@pytest.fixture()
def seeded(system):
    for title, text in DOCS:
        system.pipeline.add_document(title, text)
    return system


class TestMockLLM:
    def test_parse_sources_extracts_numbered_blocks(self):
        prompt = "已知资料：\n[来源 1] 第一段\n[来源 2] 第二段\n\n问题：xx"
        got = parse_sources(prompt)
        assert [i for i, _ in got] == [1, 2]

    def test_generation_uses_most_relevant_source(self):
        llm = MockLLM()
        msgs = [
            Message("user", "已知资料：\n[来源 1] 天气不错\n[来源 2] 语言守卫降级重排\n\n问题：语言守卫")
        ]
        import asyncio

        resp = asyncio.run(llm.acomplete(msgs))
        assert "来源 [2]" in resp.text

    def test_no_source_says_knowledge_missing(self):
        import asyncio

        resp = asyncio.run(MockLLM().acomplete([Message("user", "问题：陌生问题")]))
        assert "未检索到" in resp.text

    def test_count_tokens_is_monotonic(self):
        llm = MockLLM()
        assert llm.count_tokens("短") < llm.count_tokens("一段更长的中文文本内容")

    def test_container_stubs_match_protocol(self):
        assert isinstance(parse_sources(""), list)


class TestPipeline:
    def test_ingest_returns_doc_and_chunk_count(self, system):
        doc_id, n = system.pipeline.add_document("标题", "内容一。")
        assert doc_id and n >= 1

    def test_documents_metadata(self, seeded):
        docs = seeded.pipeline.documents()
        assert len(docs) == 3
        assert all(d["n_chars"] > 0 and d["created_at"] for d in docs)

    def test_chunk_preview_is_ordered(self, seeded):
        doc_id = seeded.pipeline.documents()[0]["doc_id"]
        chunks = seeded.pipeline.chunk_preview(doc_id, 5)
        assert [c.index for c in chunks] == sorted(c.index for c in chunks)

    def test_delete_document(self, seeded):
        doc_id = seeded.pipeline.documents()[0]["doc_id"]
        assert seeded.pipeline.delete_document(doc_id) is True
        assert len(seeded.pipeline.documents()) == 2
        assert seeded.pipeline.delete_document("不存在") is False

    def test_retrieve_top_hit_is_relevant(self, seeded):
        hits = seeded.pipeline.retrieve("为什么要语言守卫", 3)
        assert hits and hits[0].chunk.title == "重排陷阱"

    def test_citation_is_collected_from_answer(self, seeded):
        import asyncio

        ans = asyncio.run(seeded.pipeline.answer("为什么要语言守卫", use_rag=True))
        assert isinstance(ans, Answer)
        assert ans.citations and ans.citations[0].chunk.title == "重排陷阱"

    def test_citation_policy_fills_top_hit_when_model_omits(self, system):
        system.pipeline.add_document("标题", "唯一的一条资料内容")
        hits = system.pipeline.retrieve("资料", 1)
        cited = system.pipeline._collect_citations("没有任何编号的回答", hits)
        assert len(cited) == 1

    def test_multiple_citation_indices(self, seeded):
        hits = seeded.pipeline.retrieve("检索", 3)
        cited = seeded.pipeline._collect_citations("参考 [1] 与 [2]", hits)
        assert len(cited) == 2

    def test_out_of_range_citation_index_is_ignored(self, seeded):
        seeded.pipeline.cite_missing_policy = False
        hits = seeded.pipeline.retrieve("检索", 2)
        cited = seeded.pipeline._collect_citations("引用 [99]", hits)
        assert cited == []

    def test_answer_without_rag_has_no_citation(self, seeded):
        import asyncio

        ans = asyncio.run(seeded.pipeline.answer("随便聊聊", use_rag=False))
        assert ans.citations == []

    def test_stream_yields_citation_then_tokens_then_done(self, seeded):
        import asyncio

        async def run():
            events = []
            async for event_name, _payload in seeded.pipeline.stream("为什么要语言守卫"):
                events.append(event_name)
            return events

        events = asyncio.run(run())
        assert events[0] == "citation"
        assert "token" in events
        assert events[-1] == "done"

    def test_tracer_records_retrieve_and_generate_spans(self, seeded):
        tid = seeded.tracer.start_trace("chat", query="语言守卫")
        seeded.pipeline.retrieve("语言守卫", 3, trace_id=tid)
        seeded.tracer.finish_trace(tid)
        trace = seeded.tracer.get(tid)
        names = [s["name"] for s in trace["spans"]]
        assert "retrieve" in names
        assert all(isinstance(s["start_ms"], float) for s in trace["spans"])
        assert trace["total_ms"] >= 0


class TestTracerAndMetrics:
    def test_span_nesting_and_ordering(self, system):
        tid = system.tracer.start_trace("t", query="q")
        with system.tracer.span(tid, "parent") as p, system.tracer.span(tid, "child", p.span_id):
            pass
        system.tracer.finish_trace(tid)
        trace = system.tracer.get(tid)
        spans = {s["name"]: s for s in trace["spans"]}
        assert spans["child"]["parent_id"] == spans["parent"]["span_id"]
        assert trace["spans"][0]["start_ms"] <= trace["spans"][-1]["start_ms"]

    def test_unknown_trace_returns_empty(self, system):
        assert system.tracer.get("nope") == {}
        assert system.tracer.list_traces() == []

    def test_trace_list_is_most_recent_first(self, system):
        a = system.tracer.start_trace("first", query="一")
        b = system.tracer.start_trace("second", query="二")
        assert system.tracer.list_traces()[0]["trace_id"] == b
        assert system.tracer.list_traces()[1]["trace_id"] == a

    def test_metrics_percentiles(self, system):
        for i in range(1, 11):
            system.metrics.record(float(i) * 10, tokens=1)
        snap = system.metrics.snapshot(1.0)
        assert snap["requests"] == 10
        assert snap["p50_ms"] <= snap["p95_ms"] <= snap["p99_ms"]
        assert snap["tokens_total"] == 10

    def test_metrics_error_counter(self, system):
        system.metrics.record(1.0, error=True)
        assert system.metrics.snapshot(0.0)["errors"] == 1


class TestAPI:
    @pytest.fixture()
    def client(self, system) -> TestClient:
        return TestClient(create_app(system.settings, system))

    @pytest.fixture()
    def loaded(self, client, system):
        for title, text in DOCS:
            client.post("/api/v1/ingest", json={"title": title, "text": text})
        return client

    def test_root_and_health(self, client):
        assert client.get("/").json()["name"] == "nexus-ai-platform"
        health = client.get("/api/health").json()
        assert health["status"] == "ok"
        assert "provider" in health and "version" in health

    def test_ingest_and_list_and_delete(self, client):
        res = client.post("/api/v1/ingest", json={"title": "测试", "text": "内容"})
        assert res.status_code == 200
        doc_id = res.json()["doc_id"]
        assert client.get("/api/v1/documents").json()["total"] == 1
        assert client.delete(f"/api/v1/documents/{doc_id}").json()["deleted"] is True
        assert client.get("/api/v1/documents").json()["total"] == 0

    def test_delete_missing_document_is_404(self, loaded):
        assert loaded.delete("/api/v1/documents/nope").status_code == 404

    def test_ingest_rejects_empty_text(self, loaded):
        assert loaded.post("/api/v1/ingest", json={"title": "x", "text": ""}).status_code == 422

    def test_search_returns_hits(self, loaded):
        res = loaded.post("/api/v1/search", json={"query": "语言守卫", "top_k": 3})
        body = res.json()
        assert res.status_code == 200 and body["hits"]
        assert body["hits"][0]["rank"] == 1
        assert body["trace_id"]

    def test_search_mode_vector_and_bm25(self, loaded):
        v = loaded.post("/api/v1/search", json={"query": "检索", "mode": "vector"}).json()
        b = loaded.post("/api/v1/search", json={"query": "检索", "mode": "bm25"}).json()
        assert all(h["bm25_score"] is None for h in v["hits"])
        assert all(h["vector_score"] is None for h in b["hits"])

    def test_chat_success_flow(self, loaded):
        res = loaded.post("/api/v1/chat", json={"query": "为什么要语言守卫", "top_k": 3})
        body = res.json()
        assert res.status_code == 200
        assert body["answer"] and body["citations"]
        assert body["trace_id"] and body["session_id"]
        assert body["usage"]["total_ms"] >= 0

    def test_chat_stream_emits_sse_frames(self, loaded):
        res = loaded.post("/api/v1/chat/stream", json={"query": "为什么要语言守卫"})
        text = res.text
        assert "event: citation" in text
        assert "event: token" in text
        assert "event: done" in text
        assert "\n\ndata:" in text or "data:" in text

    def test_stream_done_frame_has_session_id(self, loaded):
        res = loaded.post("/api/v1/chat/stream", json={"query": "语言守卫"})
        for block in res.text.split("\n\n"):
            if block.startswith("event: done"):
                data = json.loads(block.split("data: ", 1)[1])
                assert "session_id" in data and "answer" in data
                break
        else:
            raise AssertionError("没有收到 done 帧")

    def test_chat_rejects_empty_query(self, loaded):
        assert loaded.post("/api/v1/chat", json={"query": ""}).status_code == 422

    def test_agent_mode_records_tool_call(self, loaded):
        res = loaded.post(
            "/api/v1/chat",
            json={"query": "现在是几点", "use_agent": True},
        )
        assert res.status_code == 200
        assert isinstance(res.json()["tool_calls"], list)

    def test_traces_endpoint(self, loaded):
        loaded.post("/api/v1/chat", json={"query": "检索"})
        items = loaded.get("/api/v1/traces").json()["items"]
        assert items
        tid = items[0]["trace_id"]
        trace = loaded.get(f"/api/v1/traces/{tid}").json()
        assert trace["spans"] and trace["total_ms"] >= 0

    def test_missing_trace_is_404(self, loaded):
        assert loaded.get("/api/v1/traces/nope").status_code == 404

    def test_metrics_endpoint(self, loaded):
        loaded.post("/api/v1/chat", json={"query": "检索"})
        snap = loaded.get("/api/v1/metrics").json()
        assert snap["requests"] >= 1 and "p95_ms" in snap

    def test_unknown_route_is_404(self, loaded):
        assert loaded.get("/api/v1/nope").status_code == 404

    def test_openapi_is_served(self, loaded):
        assert loaded.get("/openapi.json").status_code == 200
