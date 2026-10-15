"""The router: tool calls the app makes itself when a request clearly needs them, and nothing otherwise."""

from __future__ import annotations

from datetime import date

import pytest

from ai_buddy.config import load_config
from ai_buddy.router import route
from ai_buddy.skills import SkillSet, discover
from ai_buddy.tools import ToolCall, ToolContext

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
