"""Prompt 构造。

Prompt 是唯一而非拷贝：所有调用模型的路径都必须经过这里，
保证 Mock 模型能稳定解析出 `[来源 k]` 结构，也让引用可追溯。
"""

from __future__ import annotations

from collections.abc import Sequence

from nexus.types import Hit, Message

SYSTEM_PROMPT = (
    "你是 nexus-ai-platform 的知识助理。回答必须严格基于给定的资料。"
    "若资料不足以回答，直接说明资料不足，不要编造。"
    "引用时在句末标注来源编号，例如 [1]。回答使用简体中文。"
)

_USER_TEMPLATE = """已知资料：
{context}

问题：{query}

请依据上述资料回答，并在句末标注来源编号。"""


def build_context(hits: Sequence[Hit], max_chars: int = 2400) -> str:
    """把命中片段拼成带编号的上下文块，按预算截断。"""
    lines: list[str] = []
    used = 0
    for i, h in enumerate(hits, start=1):
        piece = h.chunk.text.replace("\n", " ").strip()
        if used + len(piece) > max_chars and lines:
            break
        lines.append(f"[来源 {i}] {piece}")
        used += len(piece)
    return "\n".join(lines)


def build_messages(
    query: str,
    hits: Sequence[Hit],
    history: Sequence[Message] | None = None,
    max_chars: int = 2400,
) -> list[Message]:
    msgs: list[Message] = [Message(role="system", content=SYSTEM_PROMPT)]
    if history:
        msgs.extend(history[-6:])
    if hits:
        context = build_context(hits, max_chars)
        content = _USER_TEMPLATE.format(context=context, query=query)
    else:
        content = f"问题：{query}\n\n（知识库中没有检索到相关资料）"
    msgs.append(Message(role="user", content=content))
    return msgs


def build_agent_messages(
    query: str,
    tool_descriptions: str,
    scratchpad: str,
) -> list[Message]:
    """ReAct 循环的单轮 prompt。"""
    guide = (
        "你是一个可以调用工具的助理。可用工具：\n"
        f"{tool_descriptions}\n\n"
        "按以下格式思考与行动，最后一轮给出答案：\n"
        "思考: <你的推理>\n"
        "行动: <工具名>\n"
        "输入: <JSON 参数>\n"
        "最终答案: <给用户的回复>\n"
    )
    body = f"{guide}\n问题：{query}\n\n{scratchpad}" if scratchpad else f"{guide}\n问题：{query}"
    return [Message(role="system", content=guide), Message(role="user", content=body)]
