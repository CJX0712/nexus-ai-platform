"""ReAct 风格的 Agent 运行时。

设计要点：
1. **步数有硬上限**，绝不允许无限循环烧掉用户的 token
2. **解析失败即收敛**：模型输出不符合 Thought/Action 格式时，
   把它当作最终答案返回并结束，而不是抛异常 —— 这是真实环境里最重要的鲁棒性
3. 工具执行统一走注册表，任何工具异常都被收敛成"Observation: 失败原因"
4. 支持上下文精简：scratchpad 超过预算时保留最近若干轮
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from nexus.protocols import LLMProvider
from nexus.rag.prompt import build_agent_messages
from nexus.tools.registry import ToolRegistry
from nexus.types import Answer, Message, ToolCallRecord, Usage

_ACTION_RE = re.compile(r"行动\s*[:：]\s*(\S+)")
_INPUT_RE = re.compile(r"输入\s*[:：]\s*(.+)")
_FINAL_RE = re.compile(r"最终答案\s*[:：]\s*(.+)", re.DOTALL)
_MAX_SCRATCHPAD = 2400


def parse_action(text: str) -> tuple[str, dict]:
    """从模型输出里解析 (工具名, 参数)。解析不出返回 (", {})。"""
    m = _ACTION_RE.search(text)
    if not m:
        return "", {}
    name = m.group(1).strip().strip("`.")
    raw = _INPUT_RE.search(text)
    args: dict = {}
    if raw:
        candidate = raw.group(1).strip().strip("`")
        try:
            loaded = json.loads(candidate)
            args = loaded if isinstance(loaded, dict) else {"value": loaded}
        except json.JSONDecodeError:
            args = {"query": candidate}
    return name, args


class ReactAgent:
    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        *,
        max_steps: int = 4,
        max_tokens: int = 400,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.max_tokens = max_tokens

    def _trim(self, scratchpad: str) -> str:
        if len(scratchpad) <= _MAX_SCRATCHPAD:
            return scratchpad
        blocks = scratchpad.split("\n\n")
        keep: list[str] = []
        used = 0
        for b in reversed(blocks):
            if used + len(b) > _MAX_SCRATCHPAD:
                break
            keep.append(b)
            used += len(b)
        return "\n\n".join(reversed(keep))

    async def run(
        self,
        query: str,
        *,
        session_id: str = "",
        history: Sequence[Message] | None = None,
    ) -> tuple[Answer, list[ToolCallRecord]]:
        """执行 ReAct 循环，返回最终答案与工具调用轨迹。"""
        import time

        t0 = time.perf_counter()
        scratchpad = ""
        records: list[ToolCallRecord] = []
        tools_desc = self.tools.describe()
        final = ""

        for _step in range(self.max_steps):
            messages = build_agent_messages(query, tools_desc, scratchpad)
            resp = await self.llm.acomplete(messages, max_tokens=self.max_tokens)
            text = resp.text.strip()

            answer_match = _FINAL_RE.search(text)
            if answer_match:
                final = answer_match.group(1).strip()
                break

            name, args = parse_action(text)
            if not name:
                # 模型没按格式输出：视为直接作答，收敛而不是崩溃
                final = text
                break

            record = self.tools.run(name, args)
            records.append(record)
            observation = record.result if record.ok else f"失败：{record.result}"
            scratchpad = self._trim(
                f"{scratchpad}\n\n{text}\n观察: {observation[:600]}"
            )

        if not final:
            final = scratchpad.strip() or "（Agent 未在限定步数内得出结论）"

        usage = Usage(
            prompt_tokens=self.llm.count_tokens(query),
            completion_tokens=self.llm.count_tokens(final),
            total_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        return (
            Answer(
                session_id=session_id,
                answer=final,
                citations=[],
                tool_calls=records,
                usage=usage,
            ),
            records,
        )
