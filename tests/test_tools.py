"""Built-in tools and the tool registry: calculator, clock, call parsing and repair, timeouts, results."""

from __future__ import annotations

import asyncio
import json
from datetime import date

import pytest

from ai_buddy.config import load_config
from ai_buddy.skills import discover
from ai_buddy.tools import RESULTS_PREFIX, Tool, ToolCall, ToolContext, ToolRegistry, ToolResult

SKILLS = discover()
CALC = SKILLS["calculator"].modules["tools"]
CLOCK = SKILLS["clock"].modules["tools"]
CTX = ToolContext(load_config(), None, None, "America/Edmonton")


def tool(name: str) -> Tool:
    return next(t for s in SKILLS.values() for t in s.tools if t.name == name)


# ---------- calculator ----------

@pytest.mark.parametrize("expr,value", [("137.50 * 1.15 / 4", 39.53125), ("18 / 100 * 240", 43.2), ("sqrt(729 * 3)",
                                        46.76537180435969), ("2^10", 1024), ("1,250 + 3", 1253), ("7 × 6", 42),
                                        ("round(2.675, 2)", 2.67), ("-(3 - 5) ** 2", -4), ("pi * 2", 6.283185307),
                                        ("max(3, 1,250)", 1250), ("min(4,5)", 4)])
def test_calculator_evaluates(expr, value):
    assert CALC.tidy(CALC.evaluate(expr)) == pytest.approx(value)


@pytest.mark.parametrize("expr", ["__import__('os')", "open('x')", "2 ** 100000", "1 / 0", "factorial(1000)",
                                  "a + 1", "x" * 301])
def test_calculator_refuses(expr):
    with pytest.raises(CALC.CalcError):
        CALC.evaluate(expr)


async def test_calculator_tool_reports_errors():
    calc = tool("calculate")
    ok = await calc.run({"expression": "6 * 7"}, CTX)
    assert ok.ok and ok.content["result"] == 42 and ok.summary == "6 * 7 = 42"
    bad = await calc.run({"expression": "1/0"}, CTX)
    assert not bad.ok and "division by zero" in bad.content["error"]


# ---------- clock ----------

@pytest.mark.parametrize("place,zone", [("Asia/Tokyo", "Asia/Tokyo"), ("Tokyo", "Asia/Tokyo"), ("calgary", "America/Edmonton"),
                                        ("New York", "America/New_York"), ("UK", "Europe/London"), ("Atlantis", None)])
def test_resolve_zone(place, zone):
    z = CLOCK.resolve_zone(place)
    assert (str(z) if z else None) == zone


def test_target_date():
    today = date(2026, 10, 14)
    assert CLOCK.target_date("Christmas", today) == date(2026, 12, 25)
    assert CLOCK.target_date("the new year's", today) == date(2027, 1, 1)
    assert CLOCK.target_date("October 3rd", today) == date(2027, 10, 3)      # passed: next year
    assert CLOCK.target_date("my birthday", today) is None
    assert CLOCK.target_date("February 30", today) is None


async def test_clock_tool():
    clock = tool("get_datetime")
    r = await clock.run({"timezone": "Tokyo", "until": "2026-12-25"}, CTX)
    assert r.ok and r.content["requested"]["timezone"] == "Asia/Tokyo" and "days_until" in r.content
    bad = await clock.run({"timezone": "Atlantis"}, CTX)
    assert not bad.ok and "Unknown timezone" in bad.content["error"]


# ---------- registry ----------

class Slow(Tool):
    name = "slow"
    parameters = {"seconds": {"type": "number"}}
    timeout_s = 0.05

    async def run(self, args, ctx):
        await asyncio.sleep(args.get("seconds", 1))
        return ToolResult({"ok": True})


class Broken(Tool):
    name = "broken"
    parameters = {"why": {"type": "string"}}

    async def run(self, args, ctx):
        raise RuntimeError("it broke")


def registry() -> ToolRegistry:
    return ToolRegistry([tool("web_search"), tool("get_datetime"), tool("calculate"), Slow(), Broken()])


def test_parse_valid_and_fenced_arguments():
    reg = registry()
    c = reg.parse({"id": "1", "name": "calculate", "arguments": '```json\n{"expression": "1+1"}\n```'})
    assert c.error is None and c.args == {"expression": "1+1"} and not c.repaired


def test_parse_repairs_a_blank_name_from_the_arguments():
    c = registry().parse({"id": "1", "name": "", "arguments": '{"query": "Calgary weather"}'})
    assert c.name == "web_search" and c.repaired and c.error is None


@pytest.mark.parametrize("call,error", [
    ({"name": "teleport", "arguments": '{"to": "Mars"}'}, "unknown tool 'teleport'"),
    ({"name": "calculate", "arguments": "{not json"}, "not valid JSON"),
    ({"name": "calculate", "arguments": "[1, 2]"}, "must be a JSON object"),
    ({"name": "calculate", "arguments": '{"expression": " "}'}, "missing required argument(s): expression"),
])
def test_parse_errors(call, error):
    assert error in registry().parse(call).error


async def test_execute_handles_errors_timeouts_and_failures():
    reg = registry()
    unknown = await reg.execute(reg.parse({"name": "nope", "arguments": "{}"}), CTX)
    assert not unknown.ok and "unknown tool" in unknown.content["error"]
    slow = await reg.execute(ToolCall("1", "slow", {"seconds": 1}, ""), CTX)
    assert not slow.ok and "timed out" in slow.content["error"]
    broken = await reg.execute(ToolCall("2", "broken", {}, ""), CTX)
    assert not broken.ok and "it broke" in broken.content["error"]


def test_results_message_is_one_capped_user_message():
    calls = [ToolCall("1", "calculate", {"expression": "1+1"}, ""), ToolCall("2", "web_search", {"query": "q"}, "")]
    results = [ToolResult({"result": 2}), ToolResult({"text": "x" * 10000})]
    msg = ToolRegistry.results_message(calls, results)
    assert msg["role"] == "user" and msg["content"].startswith(RESULTS_PREFIX)
    assert '<tool_response>\ncalculate: {"result": 2}\n</tool_response>' in msg["content"]
    assert "…[truncated]" in msg["content"] and len(msg["content"]) < 7000


def test_tool_calls_keep_arguments_as_json_strings():
    """Dict arguments render differently in the chat template and miss FLM's prompt cache."""
    entry = ToolCall("1", "calculate", {"expression": "1+1"}, "").for_history()
    assert isinstance(entry["function"]["arguments"], str)
    assert json.loads(entry["function"]["arguments"]) == {"expression": "1+1"}


def test_default_confirm_text_names_the_action():
    class Save(Tool):
        name = "save_note"
        label = "Saving a note"
        confirm = True

        def describe(self, args):
            return args["file"]
    assert Save().confirm_text({"file": "ideas.md"}) == "Should I go ahead with saving a note (ideas.md)?"
