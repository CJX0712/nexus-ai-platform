"""组合根：唯一知道"哪个接口用哪个实现"的地方。

业务模块之间只依赖 Protocol，互不感知具体实现；
要换掉任何一层（比如把内存向量库换成 Qdrant），只需要改这一个文件。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nexus.agent.react import ReactAgent
from nexus.config import Settings
from nexus.embedding.hashing import TfidfHashingEmbedder
from nexus.llm.mock import MockLLM
from nexus.llm.ollama import OllamaLLM
from nexus.llm.openai_compat import OpenAICompatLLM
from nexus.llm.router import LLMRouter
from nexus.memory.store import WindowMemory
from nexus.observability.metrics import Metrics
from nexus.observability.tracer import InMemoryTracer
from nexus.rag.chunker import RecursiveChunker
from nexus.rag.pipeline import RAGPipeline
from nexus.reranker.family import CoverageReranker, GuardedReranker, IdentityReranker
from nexus.retriever.bm25 import IncrementalBM25
from nexus.retriever.hybrid import HybridRetriever
from nexus.tools.builtin import builtin_tools
from nexus.tools.registry import ToolRegistry
from nexus.vectorstore.memory import InMemoryVectorStore


@dataclass
class System:
    """系统所有组件的持有者。"""

    settings: Settings
    llm: Any
    embedder: Any
    vectorstore: Any
    retriever: Any
    reranker: Any
    pipeline: Any
    agent: Any
    tools: Any
    memory: Any
    tracer: Any
    metrics: Any


def build_llm(s: Settings) -> LLMRouter:
    """按配置组装模型路由。mock 永远兜底，保证无网络也能完成一次完整链路。"""
    providers: dict[str, Any] = {}
    if s.provider in ("ollama", "auto"):
        providers["ollama"] = OllamaLLM(
            base_url=s.ollama_base_url, model=s.ollama_model, keep_alive=s.ollama_keep_alive
        )
    if s.provider in ("openai", "auto") and s.openai_api_key:
        providers["openai"] = OpenAICompatLLM(
            base_url=s.openai_base_url, model=s.openai_model, api_key=s.openai_api_key
        )
    providers["mock"] = MockLLM()
    default = s.provider if s.provider in providers else "mock"
    return LLMRouter(providers, default=default)


def build_reranker(s: Settings) -> Any:
    if s.rerank == "coverage":
        return GuardedReranker(CoverageReranker(), inner_supports_zh=True)
    if s.rerank == "cross":
        try:
            from nexus.reranker.family import CrossEncoderReranker

            inner = CrossEncoderReranker()
            return GuardedReranker(inner, inner_supports_zh=False, fallback=CoverageReranker())
        except Exception:
            return GuardedReranker(CoverageReranker(), inner_supports_zh=True)
    return IdentityReranker()


def build_embedder(s: Settings) -> Any:
    """选择嵌入实现。

    默认 hash（零依赖、离线可跑）；指定 ollama 时尝试真实稠密嵌入，
    服务不可用则回退而不是让整个系统启动失败。
    """
    if s.embed_provider == "ollama":
        try:
            from nexus.embedding.ollama import OllamaEmbedder

            return OllamaEmbedder(base_url=s.ollama_base_url, model=s.ollama_embed_model)
        except Exception as e:
            print(f"[warn] 稠密嵌入不可用({e})，回退到零依赖哈希嵌入", file=sys.stderr)
    return TfidfHashingEmbedder(dim=s.embed_dim)


def build_system(s: Settings) -> System:
    data_dir = Path(s.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    llm = build_llm(s)
    embedder = build_embedder(s)
    vectorstore = InMemoryVectorStore(
        dim=embedder.dim, path=str(data_dir / "index") if s.persist else None
    )
    retriever = HybridRetriever(embedder, vectorstore, bm25=IncrementalBM25(), mode=s.rag_mode)
    tracer = InMemoryTracer()
    metrics = Metrics()
    pipeline = RAGPipeline(
        retriever,
        llm,
        embedder=embedder,
        reranker=build_reranker(s),
        tracer=tracer,
        chunker=RecursiveChunker(chunk_size=s.chunk_size, overlap=s.chunk_overlap),
        top_k=s.top_k,
    )
    tools = ToolRegistry()
    for t in builtin_tools(retriever):
        tools.register(t)
    agent = ReactAgent(llm, tools, max_steps=s.agent_max_steps)
    return System(
        settings=s,
        llm=llm,
        embedder=embedder,
        vectorstore=vectorstore,
        retriever=retriever,
        reranker=build_reranker(s),
        pipeline=pipeline,
        agent=agent,
        tools=tools,
        memory=WindowMemory(window=8),
        tracer=tracer,
        metrics=metrics,
    )
