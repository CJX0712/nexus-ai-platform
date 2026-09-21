"""多模型路由：主备切换 + 失败降级。

业务代码永远只依赖 LLMProvider 协议，路由负责把「用哪个模型」的决策收敛到一处。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from nexus.types import LLMResponse, Message
from nexus.util.text import estimate_tokens

_FALLBACK_HINT = "[模型降级]"


class LLMRouter:
    """按优先级选择可用的 LLM Provider，失败自动降级到下一个。

    Args:
        providers: 有序优先级字典，name -> provider
        default: 默认使用的 provider 名；缺省取第一个
    """

    name = "router"

    def __init__(self, providers: dict[str, Any], default: str | None = None) -> None:
        if not providers:
            raise ValueError("至少需要一个 provider")
        self._providers = dict(providers)
        self.default = default or next(iter(self._providers))
        if self.default not in self._providers:
            raise KeyError(f"默认 provider {self.default!r} 不存在")

    @property
    def model(self) -> str:
        return getattr(self._providers[self.default], "model", self.default)

    @property
    def available(self) -> list[str]:
        return list(self._providers)

    def get(self, name: str | None = None) -> Any:
        key = name or self.default
        if key not in self._providers:
            raise KeyError(f"未知 provider: {key}，可选: {list(self._providers)}")
        return self._providers[key]

    def _order(self, preferred: str | None) -> list[str]:
        first = preferred or self.default
        rest = [k for k in self._providers if k != first]
        return [first, *rest]

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        provider: str | None = None,
    ) -> LLMResponse:
        errors: list[str] = []
        for key in self._order(provider):
            p = self._providers[key]
            try:
                resp = await p.acomplete(messages, temperature=temperature, max_tokens=max_tokens)
                if errors:
                    resp.text = f"{_FALLBACK_HINT} {resp.text}"
                return resp
            except Exception as e:
                errors.append(f"{key}: {type(e).__name__}")
                continue
        raise RuntimeError("所有 provider 均不可用 -> " + "; ".join(errors))

    async def astream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        provider: str | None = None,
    ) -> AsyncIterator[str]:
        errors: list[str] = []
        for key in self._order(provider):
            try:
                first = True
                async for piece in self._providers[key].astream(
                    messages, temperature=temperature, max_tokens=max_tokens
                ):
                    if first and errors:
                        yield f"{_FALLBACK_HINT} "
                        first = False
                    yield piece
                return
            except Exception as e:
                errors.append(f"{key}: {type(e).__name__}")
                continue
        raise RuntimeError("所有 provider 均不可用 -> " + "; ".join(errors))

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    async def aclose(self) -> None:
        for p in self._providers.values():
            close = getattr(p, "aclose", None)
            if close:
                await close()
