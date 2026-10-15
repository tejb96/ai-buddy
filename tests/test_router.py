"""The router: tool calls the app makes itself when a request clearly needs them, and nothing otherwise."""

from __future__ import annotations

import json
from datetime import date

import pytest

from ai_buddy.config import load_config
from ai_buddy.router import figures, questions_a_fact, recheck, recheck_note, route
from ai_buddy.skills import SkillSet, discover
from ai_buddy.tools import RECHECK_SUFFIX, RESULTS_PREFIX, Tool, ToolCall, ToolContext, ToolRegistry, ToolResult

REG = SkillSet(discover(), ["web-search", "notes", "clock", "calculator"]).registry()


class Ctx(ToolContext):
    def today(self) -> date:
        return date(2026, 10, 14)


CTX = Ctx(load_config(), None, object(), "America/Edmonton")    # object(): a knowledge base is configured


def routed(text: str) -> dict[str, dict]:
    return {c.name: c.args for c in route(text, REG, CTX)}


@pytest.mark.parametrize("text,calls", [
    ("What time is it in Tokyo right now?", {"get_datetime": {"timezone": "Asia/Tokyo"}}),
    ("Nice. And what time is it in Tokyo right now?", {"get_datetime": {"timezone": "Asia/Tokyo"}}),
    ("what time is it", {"get_datetime": {}}),
    ("What's the time in New York?", {"get_datetime": {"timezone": "America/New_York"}}),
    ("What time is it in the UK?", {"get_datetime": {"timezone": "Europe/London"}}),
    ("How many days until Christmas?", {"get_datetime": {"until": "2026-12-25"}}),
    ("What's the weather in Paris, and what time is it there?",
     {"get_datetime": {"timezone": "Europe/Paris"}, "web_search": {"query": "Paris weather today"}}),
    ("What's the weather going to be like in Calgary tomorrow?", {"web_search": {"query": "Calgary weather tomorrow"}}),
    ("Search the web for the latest version of Python and tell me what's new.",
     {"web_search": {"query": "latest version of Python"}}),
    ("Can you look up the opening hours of the Royal Tyrrell Museum?",
     {"web_search": {"query": "opening hours of the Royal Tyrrell Museum"}}),
    ("Okay. Quick one: what's 18% of 240?", {"calculate": {"expression": "18 / 100 * 240"}}),
    ("Um, what is 12 times 37?", {"calculate": {"expression": "12 * 37"}}),
    ("What does my resume say about my Python experience?",
     {"search_notes": {"topic": "What does my resume say about my Python experience"}}),
    ("Remind me what I wrote about the budget in my notes.",
     {"search_notes": {"topic": "Remind me what I wrote about the budget in my notes"}}),
])
def test_clear_requests_are_routed(text, calls):
    assert routed(text) == calls


@pytest.mark.parametrize("text", [
    "I had a great time in Tokyo.",
    "What time is the meeting tomorrow?",
    "What's the time difference between Tokyo and Calgary?",
    "What time is it in Banff?",                 # a place we can't place: the model decides (or asks)
    "how many days till my birthday",
    "I got 18% of the votes.",
    "What's the square root of 729 times 3?",   # ambiguous grouping: left to the model
    "I lost my notes yesterday.",
    "The weather was lovely.",
    "Who won the Stanley Cup this year?",        # current facts are the model's call (it's told to search)
    "Tell me a joke about cats.",
    "I need to look up to my mentors more.",
])
def test_everything_else_is_left_to_the_model(text):
    assert routed(text) == {}


def test_notes_need_a_knowledge_base():
    ctx = Ctx(load_config(), None, None, "UTC")
    assert route("What's in my notes about the budget?", REG, ctx) == []


def test_routes_only_use_enabled_tools():
    reg = SkillSet(discover(), ["web-search"]).registry()
    assert route("What time is it in Tokyo?", reg, CTX) == []


@pytest.mark.parametrize("text,questioned", [
    ("Isn't it 16 degrees and feels like 12?", True),
    ("Are you sure?", True),
    ("Hmm, that doesn't sound right.", True),
    ("My weather app says 16.", True),
    ("I thought it was half past three.", True),
    ("Can you double-check that?", True),
    ("Isn't it lovely out there?", False),       # no figure, no doubt: just chat
    ("I thought so too!", False),
    ("Thanks, that's helpful.", False),
])
def test_questions_a_fact(text, questioned):
    assert questions_a_fact(text) is questioned


def _turn(*calls):
    return {"role": "assistant", "content": "", "tool_calls": [ToolCall(f"c{i}", n, a, json.dumps(a)).for_history()
                                                                for i, (n, a) in enumerate(calls)]}


def test_recheck_repeats_the_last_exchanges_read_only_calls():
    reg = SkillSet(discover(), ["web-search", "clock"]).registry()
    history = [{"role": "system", "content": "..."},
               {"role": "user", "content": "What's the weather in Calgary right now?"},
               _turn(("web_search", {"query": "Calgary weather now"})),
               {"role": "user", "content": RESULTS_PREFIX + " ..."},
               {"role": "assistant", "content": "It's nine degrees and cloudy."}]
    calls = recheck("Isn't it 16 degrees and feels like 12?", history, reg)
    assert [(c.name, c.args) for c in calls] == [("web_search", {"query": "Calgary weather now", "double_check": True})]
    assert calls[0].routed
    assert recheck("Thanks!", history, reg) == []
    earlier = history + [{"role": "user", "content": "Tell me a joke."}, {"role": "assistant", "content": "Ha."}]
    assert recheck("Are you sure?", earlier, reg) == []          # the last answer didn't come from a tool


def test_recheck_never_repeats_a_tool_that_changes_something():
    class Save(Tool):
        name, description, confirm = "save_note", "Save a note.", True
        parameters, required = {"text": {"type": "string"}}, ["text"]
    reg = ToolRegistry([SkillSet(discover(), ["clock"]).registry().tools["get_datetime"], Save()])
    history = [{"role": "user", "content": "Save 'buy milk' and tell me the time."},
               _turn(("save_note", {"text": "buy milk"}), ("get_datetime", {})),
               {"role": "user", "content": RESULTS_PREFIX + " ..."},
               {"role": "assistant", "content": "Saved, and it's 3 PM."}]
    assert [c.name for c in recheck("Are you sure?", history, reg)] == ["get_datetime"]


def test_figures():
    assert figures("Isn't it 16 degrees and feels like 12?") == ["16", "12"]
    assert figures("I thought it was sixteen, or twenty-one at 3:30") == ["16", "21", "3:30"]
    assert figures("about 1,200 people, 4.5 stars") == ["1200", "4.5"]


def test_recheck_note_names_the_users_figures_the_results_lack():
    results = [ToolResult({"results": [{"title": "Environment Canada", "text": "13°C · Observed 6:00 PM · High 12."}]})]
    note = recheck_note("Isn't it 16 degrees and feels like 12?", results)
    assert note.startswith(RECHECK_SUFFIX)
    assert "The user's figure 16 is not in any of these results: don't say a source shows it." in note
    assert recheck_note("Isn't it 13?", results) == RECHECK_SUFFIX
    dates = [ToolResult({"text": "Fri Oct 16 · Feels 3 · 12am · Feels -2 · 16 October: sunny"})]
    assert "figures 16, 12 are not in" in recheck_note("Isn't it 16 and feels like 12?", dates)
    assert "figures 45, 7" in recheck_note("Isn't it 45, or 7?", [ToolResult({"result": 43.2})])
