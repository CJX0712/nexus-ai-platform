"""SSE 帧编码与流式响应。

严格要求 `event: <name>\ndata: <json>\n\n` 格式（以 LF 换行 + 空行结尾），
这是 EventSource 的解析规范；用 \r\n 或漏结尾空行会导致部分客户端收不到事件。
"""

from __future__ import annotations

from typing import Any

import orjson


def sse_frame(event: str, data: Any) -> bytes:
    payload = orjson.dumps(data, option=orjson.OPT_NON_STR_KEYS).decode("utf-8")
    return f"event: {event}\ndata: {payload}\n\n".encode()


def sse_comment(text: str = "keep-alive") -> bytes:
    return f": {text}\n\n".encode()
