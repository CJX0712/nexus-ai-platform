"""工具 / Agent / 记忆测试。

安全是工具层的硬要求：这里专门验证了表达式求值无法逃逸到 Python 语言层。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexus.agent.react import ReactAgent, parse_action
from nexus.memory.store import WindowMemory
from nexus.tools.builtin import (
    CalculatorTool,
    KnowledgeSearchTool,
    NowTool,
    TextStatsTool,
    safe_eval,
)
from nexus.tools.registry import ToolRegistry
from nexus.types import LLMResponse, Message


class TestSafeEval:
    @pytest.mark.parametrize(
        ("expr", "expected"),
        [("1+2", 3.0), ("10/4", 2.5), ("(3+4)*2", 14.0), ("7%3", 1.0), ("-5+3", -2.0)],
    )
    def test_arithmetic(self, expr, expected):
        assert safe_eval(expr) == expected

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            safe_eval("1/0")

    def test_rejects_attribute_access_escape(self):
        with pytest.raises(ValueError):
            safe_eval("().__class__")

    def test_rejects_function_call(self):
        with pytest.raises(ValueError):
            safe_eval("__import__('os')")

    def test_rejects_power_operator(self):
        with pytest.raises(ValueError):
            safe_eval("2**9999")

    def test_rejects_names(self):
        with pytest.raises(ValueError):
            safe_eval("os")

    def test_rejects_boolean_literal(self):
        with pytest.raises(ValueError):
            safe_eval("True")

    def test_rejects_empty_and_too_long(self):
        with pytest.raises(ValueError):
            safe_eval("")
        with pytest.raises(ValueError):
            safe_eval("1+" * 300)


class TestTools:
    def test_calculator_ok(self):
        res = CalculatorTool().run(expression="12*(3+4)")
        assert res.ok and res.output == "84"

    def test_calculator_float_formatting(self):
        res = CalculatorTool().run(expression="10/3")
        assert res.ok and float(res.output) > 3.3

    def test_calculator_rejects_illegal_chars(self):
        res = CalculatorTool().run(expression="import os")
        assert res.ok is False and res.error

    def test_calculator_handles_zero_division(self):
        res = CalculatorTool().run(expression="5/0")
        assert res.ok is False and "除数为零" in res.error

    def test_calculator_missing_argument(self):
        res = CalculatorTool().run()
        assert res.ok is False

    def test_now_returns_timestamp_format(self):
        res = NowTool().run()
        assert res.ok and len(res.output) == 19

    def test_text_stats_counts(self):
        res = TextStatsTool().run(text="混合检索 abc")
        assert res.ok
        data = json.loads(res.output)
        assert data["chars"] == len("混合检索 abc")
        assert data["cjk_chars"] == 4
        assert data["est_tokens"] >= 1

    def test_kb_search_without_hit(self):
        class EmptyRetriever:
            def retrieve(self, q, k):
                return []

        res = KnowledgeSearchTool(EmptyRetriever()).run(query="anything")
        assert res.ok and "没有命中" in res.output

    def test_kb_search_formats_hits(self):
        class FakeRetriever:
            def retrieve(self, q, k):
                from nexus.types import Chunk, Hit

                return [
                    Hit(Chunk("c1", "d1", "标题", "片段内容", 0), 1.0, 1),
                ]

        res = KnowledgeSearchTool(FakeRetriever()).run(query="片段")
        assert res.ok and "片段内容" in res.output


class TestRegistry:
    def test_duplicate_registration_raises(self):
        reg = ToolRegistry()
        reg.register(NowTool())
        with pytest.raises(ValueError):
            reg.register(NowTool())

    def test_unknown_tool_returns_failed_record(self):
        rec = ToolRegistry().run("nope", {})
        assert rec.ok is False and "未知工具" in rec.result

    def test_tool_exception_is_contained(self):
        class Boom:
            name = "boom"
            description = "always fails"

            def parameters(self):
                return {}

            def run(self, **kwargs):
                raise RuntimeError("explode")

        reg = ToolRegistry()
        reg.register(Boom())
        rec = reg.run("boom")
        assert rec.ok is False and "RuntimeError" in rec.result

    def test_bad_arguments_are_reported_not_raised(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        rec = reg.run("calculator", {"typo": 1})
        assert rec.ok is False and rec.result

    def test_describe_lists_all_tools(self):
        reg = ToolRegistry()
        reg.register(NowTool())
        reg.register(CalculatorTool())
        desc = reg.describe()
        assert "now" in desc and "calculator" in desc

    def test_names_sorted(self):
        reg = ToolRegistry()
        reg.register(NowTool())
        reg.register(CalculatorTool())
        assert reg.names() == ["calculator", "now"]


class TestParseAction:
    def test_parses_action_and_json_input(self):
        blob = "思考: 需要计算\n行动: calculator\n输入: {\"expression\": \"1+1\"}"
        name, args = parse_action(blob)
        assert name == "calculator"
        assert args == {"expression": "1+1"}

    def test_non_json_input_falls_back_to_query(self):
        name, args = parse_action("行动: kb_search\n输入: 混合检索怎么融合")
        assert name == "kb_search"
        assert args == {"query": "混合检索怎么融合"}

    def test_no_action_returns_empty(self):
        assert parse_action("这只是一段普通回答") == ("", {})

    def test_scalar_json_wrapped_into_value(self):
        _name, args = parse_action("行动: kb_search\n输入: 42")
        assert args == {"value": 42}


class FakeLLM:
    """可编程的假模型，用于测试 Agent 的决策路径而不依赖任何真实推理服务。"""

    name = "fake"
    model = "fake-1"

    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.seen_prompts: list[str] = []

    def _next(self, messages) -> str:
        self.seen_prompts.append(messages[-1].content)
        return self.script.pop(0) if self.script else "最终答案: 兜底答案"

    async def acomplete(self, messages, *, temperature=0.2, max_tokens=400, **kw):
        return LLMResponse(text=self._next(messages), model=self.model)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 2)


class TestReactAgent:
    async def test_final_answer_stops_immediately(self):
        llm = FakeLLM(["最终答案: 直接给结论"])
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        ans, records = await ReactAgent(llm, reg).run("问题")
        assert ans.answer == "直接给结论"
        assert records == []

    async def test_tool_call_then_final_answer(self):
        llm = FakeLLM(
            [
                "思考: 需要计算\n行动: calculator\n输入: {\"expression\": \"6*7\"}",
                "最终答案: 结果是 42",
            ]
        )
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        ans, records = await ReactAgent(llm, reg).run("六乘七等于多少")
        assert len(records) == 1 and records[0].ok and records[0].result == "42"
        assert ans.answer == "结果是 42"
        assert ans.tool_calls == records

    async def test_unparsable_output_converges_instead_of_crashing(self):
        llm = FakeLLM(["这段输出完全没有遵循格式要求"])
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        ans, records = await ReactAgent(llm, reg).run("问题")
        assert ans.answer == "这段输出完全没有遵循格式要求"
        assert records == []

    async def test_step_limit_is_enforced(self):
        llm = FakeLLM(["行动: calculator\n输入: {\"expression\": \"1+1\"}"] * 10)
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        _ans, records = await ReactAgent(llm, reg, max_steps=3).run("问题")
        assert len(records) == 3

    async def test_failed_tool_observation_does_not_break_loop(self):
        llm = FakeLLM(
            [
                "行动: calculator\n输入: {\"expression\": \"bad\"}",
                "最终答案: 计算失败，我换了一种说法",
            ]
        )
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        ans, records = await ReactAgent(llm, reg).run("算一下")
        assert records[0].ok is False
        assert "换了一种说法" in ans.answer

    async def test_scratchpad_grows_across_steps(self):
        llm = FakeLLM(
            [
                "行动: now\n输入: {}",
                "行动: now\n输入: {}",
                "最终答案: 完成",
            ]
        )
        reg = ToolRegistry()
        reg.register(NowTool())
        await ReactAgent(llm, reg).run("现在几点")
        assert len(llm.seen_prompts) == 3
        assert "观察" in llm.seen_prompts[1]


class TestMemory:
    def test_window_keeps_recent_messages(self):
        mem = WindowMemory(window=2)
        for i in range(5):
            mem.append("s1", Message(role="user", content=f"m{i}"))
        got = mem.window_messages("s1")
        assert [m.content for m in got] == ["m3", "m4"]

    def test_summarized_content_is_reachable(self):
        mem = WindowMemory(window=1)
        mem.append("s1", Message(role="user", content="旧话第一轮"))
        mem.append("s1", Message(role="user", content="第二轮"))
        assert "旧话第一轮" in mem.summary("s1")

    def test_context_includes_summary_and_window(self):
        mem = WindowMemory(window=2)
        for i in range(4):
            mem.append("s1", Message(role="user", content=f"第{i}轮内容"))
        ctx = mem.context("s1")
        assert any(m.role == "system" for m in ctx)
        assert ctx[-1].content == "第3轮内容"

    def test_reset_clears_everything(self):
        mem = WindowMemory(window=2)
        mem.append("s1", Message(role="user", content="x"))
        mem.reset("s1")
        assert mem.window_messages("s1") == []
        assert mem.stats("s1") == {"active": 0, "summarized": 0}

    def test_sessions_are_isolated(self):
        mem = WindowMemory(window=5)
        mem.append("a", Message(role="user", content="会话A"))
        mem.append("b", Message(role="user", content="会话B"))
        assert mem.window_messages("a")[-1].content == "会话A"
        assert mem.window_messages("b")[-1].content == "会话B"

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            WindowMemory(window=0)
