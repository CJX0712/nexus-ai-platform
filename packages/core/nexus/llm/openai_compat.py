"""OpenAI 兼容接口的大模型适配层。

一套实现覆盖 vLLM、DeepSeek、通义、OpenRouter、Ollama 的 /v1 网关等
所有遵循 OpenAI Chat Completions 协议的服务，避免为每个厂商各写一份。
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
from nexus.types import LLMResponse, Message, Usage
from nexus.util.text import estimate_tokens

_FINISH = "stop"


class OpenAICompatLLM:
    """通过 OpenAI 兼容协议调用远端大模型。

    Args:
        base_url: 形如 http://localhost:11434/v1（Ollama）或 https://api.deepseek.com/v1
        model: 模型名
        api_key: 未提供时读取环境变量 OPENAI_API_KEY，本地推理服务可传 "ollama"
    """

    name = "openai-compat"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

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
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post(
                    url,
                    json=self._body(messages, temperature, max_tokens, False),
                    headers=self._headers(),
                )
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                text = choice.get("message", {}).get("content", "") or ""
                usage_raw = data.get("usage", {}) or {}
                return LLMResponse(
                    text=text,
                    usage=Usage(
                        prompt_tokens=int(usage_raw.get("prompt_tokens", 0)),
                        completion_tokens=int(usage_raw.get("completion_tokens", 0)),
                        total_ms=float(resp.elapsed.total_seconds() * 1000),
                        ttft_ms=float(resp.elapsed.total_seconds() * 1000),
                    ),
                    model=data.get("model", self.model),
                    finish_reason=choice.get("finish_reason", _FINISH),
                )
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
        raise RuntimeError(f"[{self.name}] 调用失败: {last_err}")

    async def astream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        url = f"{self.base_url}/chat/completions"
        async with self._client.stream(
            "POST",
            url,
            json=self._body(messages, temperature, max_tokens, True),
            headers=self._headers(),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    import orjson

                    chunk = orjson.loads(payload)
                except Exception:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {}) or {}
                piece = delta.get("content")
                if piece:
                    yield piece

    async def aclose(self) -> None:
        await self._client.aclose()
