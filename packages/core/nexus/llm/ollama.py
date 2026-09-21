"""Ollama 原生接口适配层（llama.cpp 内核的本地推理服务）。

Ollama 的 /api/chat 是逐行 NDJSON 协议，与 OpenAI 协议不同，
因此单独实现一个 Provider；两者对上层而言都满足 LLMProvider 协议。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
from nexus.types import LLMResponse, Message, Usage
from nexus.util.text import estimate_tokens


class OllamaLLM:
    """Ollama 本地推理服务。

    Args:
        base_url: 形如 http://localhost:11434
        model: 例如 qwen2.5:1.5b-instruct
        keep_alive: 模型在内存中的保活时长，默认 5m；压测时建议设为 -1 常驻
    """

    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:1.5b-instruct",
        timeout: float = 120.0,
        keep_alive: str = "5m",
        max_retries: int = 1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=5.0),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )

    def _body(
        self,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    async def _post_json(self, body: dict[str, Any]) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post(f"{self.base_url}/api/chat", json=body)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPError as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
        raise RuntimeError(f"[ollama] 调用失败（模型是否已拉取？）: {last_err}")

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> LLMResponse:
        data = await self._post_json(self._body(messages, temperature, max_tokens, False))
        text = (data.get("message") or {}).get("content", "")
        total_ns = float(data.get("total_duration", 0) or 0)
        eval_ns = float(data.get("load_duration", 0) or 0)
        return LLMResponse(
            text=text,
            usage=Usage(
                prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
                completion_tokens=int(data.get("eval_count", 0) or 0),
                total_ms=round(total_ns / 1e6, 2),
                ttft_ms=round(eval_ns / 1e6, 2),
            ),
            model=data.get("model", self.model),
            finish_reason="stop" if data.get("done") else "length",
        )

    async def astream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        import orjson

        async with self._client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=self._body(messages, temperature, max_tokens, True),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    obj = orjson.loads(line)
                except Exception:
                    continue
                piece = (obj.get("message") or {}).get("content", "")
                if piece:
                    yield piece
                if obj.get("done"):
                    break

    async def health(self) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[str]:
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            return [m.get("name", "") for m in resp.json().get("models", [])]
        except httpx.HTTPError:
            return []

    async def aclose(self) -> None:
        await self._client.aclose()
