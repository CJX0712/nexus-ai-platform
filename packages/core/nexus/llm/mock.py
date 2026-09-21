"""确定性 Mock 大模型 —— 让系统在无网络、无 Key 的环境下依然端到端可运行。

它不是简单地返回固定字符串，而是做「抽取式生成」：
从 prompt 中解析出带编号的知识库片段，按与问题的词汇重叠度选出最相关的一条，
拼出答案并标注来源编号。这样即使在 Mock 模式下，
RAG 链路检索是否被真正用到，依然是可以被机械断言的。
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Sequence

from nexus.types import LLMResponse, Message, Usage
from nexus.util.text import estimate_tokens, tokenize

_SOURCE_RE = re.compile(r"^\s*\[(?:来源|source)?\s*(\d+)\]\s*(.+)$", re.IGNORECASE)
_QUERY_RE = re.compile(r"问题[:：]\s*(.+?)(?:\n|$)")


def parse_sources(prompt: str) -> list[tuple[int, str]]:
    """从 prompt 中解析 `[来源 k] text` 形式的上下文块。"""
    out: list[tuple[int, str]] = []
    for line in prompt.splitlines():
        m = _SOURCE_RE.match(line)
        if m:
            out.append((int(m.group(1)), m.group(2).strip()))
    return out


def _overlap(query: str, passage: str) -> float:
    """词汇重叠率（Jaccard 的加权变体），用于 Mock 的"相关性判断"。"""
    q = set(tokenize(query))
    p = set(tokenize(passage))
    if not q or not p:
        return 0.0
    return len(q & p) / (len(q) + 1e-9)


class MockLLM:
    """抽取式 Mock 模型。

    Args:
        delay_ms: 模拟网络延迟，用于验证流式与非流式的时序逻辑
    """

    name = "mock"
    model = "mock-extractive-v1"

    def __init__(self, delay_ms: int = 0) -> None:
        self.delay_ms = delay_ms

    def _question(self, messages: Sequence[Message]) -> str:
        for m in reversed(messages):
            if m.role == "user":
                mq = _QUERY_RE.search(m.content)
                if mq:
                    return mq.group(1).strip()
                tail = [ln for ln in m.content.splitlines() if ln.strip()]
                return tail[-1].strip() if tail else m.content
        return ""

    def _generate(self, messages: Sequence[Message]) -> str:
        joined = "\n".join(m.content for m in messages)
        sources = parse_sources(joined)
        question = self._question(messages)
        if not question:
            return "[mock] 未收到有效问题。"
        if sources:
            best_k, best_text = max(
                sources, key=lambda kv: _overlap(question, kv[1])
            )
            snippet = best_text[:160]
            return f"[mock] {snippet}（来源 [{best_k}]）"
        return f"[mock] 针对「{question[:60]}」：当前未检索到相关知识库内容。"

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> LLMResponse:
        if self.delay_ms:
            await asyncio.sleep(self.delay_ms / 1000.0)
        text = self._generate(messages)
        prompt_len = sum(self.count_tokens(m.content) for m in messages)
        return LLMResponse(
            text=text,
            usage=Usage(
                prompt_tokens=prompt_len,
                completion_tokens=self.count_tokens(text),
                total_ms=float(self.delay_ms),
                ttft_ms=float(self.delay_ms),
            ),
            model=self.model,
        )

    async def astream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        text = self._generate(messages)
        step = 8
        for i in range(0, len(text), step):
            if self.delay_ms:
                await asyncio.sleep(self.delay_ms / 1000.0 / max(1, len(text) / step))
            yield text[i : i + step]

    async def aclose(self) -> None:
        return None
