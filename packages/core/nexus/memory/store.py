"""会话记忆：滑动窗口 + 概要压缩。

长对话里把全部历史塞进 prompt 会导致：token 成本线性上涨、模型注意力被稀释、
且超出上下文上限直接报错。这里的策略是双层的：
- 短期：保留最近 n 轮原文（保证指代和上下文连贯）
- 长期：更旧的轮次压缩为一句话概要（保留事实，丢弃客套话）

压缩默认用确定性启发式（不额外调用模型）——这样离线环境也能跑、且行为可预测；
若注入了 LLM 则可切换为模型摘要。
"""

from __future__ import annotations

from threading import Lock

from nexus.types import Message


class WindowMemory:
    """进程内会话记忆。

    Args:
        window: 短期窗口保留的消息条数
        summarize: 是否对滑出窗口的消息做概要压缩
    """

    def __init__(self, window: int = 8, summarize: bool = True) -> None:
        if window < 1:
            raise ValueError("window 必须 >= 1")
        self.window = window
        self.summarize = summarize
        self._hist: dict[str, list[Message]] = {}
        self._summary: dict[str, list[str]] = {}
        self._lock = Lock()

    def append(self, session_id: str, message: Message) -> None:
        with self._lock:
            hist = self._hist.setdefault(session_id, [])
            hist.append(message)
            overflow = len(hist) - self.window
            if overflow > 0:
                old = hist[:overflow]
                del hist[:overflow]
                if self.summarize:
                    self._summary.setdefault(session_id, []).extend(
                        m for m in (self._condense(x) for x in old) if m
                    )

    @staticmethod
    def _condense(message: Message, limit: int = 140) -> str:
        text = message.content.replace("\n", " ").strip()
        if not text:
            return ""
        head = text[:limit]
        return f"{message.role}: {head}"

    def window_messages(self, session_id: str, n: int | None = None) -> list[Message]:
        with self._lock:
            hist = list(self._hist.get(session_id, []))
        return hist[-n:] if n else hist

    def summary(self, session_id: str, sep: str = " | ") -> str:
        with self._lock:
            parts = list(self._summary.get(session_id, []))
        return sep.join(parts[-6:])

    def context(self, session_id: str, n: int | None = None) -> list[Message]:
        """给 prompt 用的上下文：概要一句 + 近期窗口。"""
        summary = self.summary(session_id)
        msgs: list[Message] = []
        if summary:
            msgs.append(Message(role="system", content=f"较早对话概要：{summary}"))
        msgs.extend(self.window_messages(session_id, n))
        return msgs

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._hist.pop(session_id, None)
            self._summary.pop(session_id, None)

    def stats(self, session_id: str) -> dict[str, int]:
        with self._lock:
            return {
                "active": len(self._hist.get(session_id, [])),
                "summarized": len(self._summary.get(session_id, [])),
            }
