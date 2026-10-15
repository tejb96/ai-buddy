"""Stand-ins for Lemonade, the model manager and the browser, to test the session's agent loop without models.

FakeLM plays back scripted model replies (text pieces and tool calls) and records every request, so tests
can check what the model would have been sent; FakeModels "transcribes" with scripted text.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import numpy as np

from ai_buddy.config import load_config
from ai_buddy.session import Services, Session
from ai_buddy.skills import Skill, SkillSet, discover
from ai_buddy.storage import Store
from ai_buddy.tools import Tool, ToolContext, ToolResult


def speech(seconds: float = 1.2) -> bytes:
    """Loud tone between quiet stretches: enough for the app's speech detector."""
    t = np.arange(int(seconds * 16000)) / 16000
    tone = 6000 * np.sin(2 * np.pi * 220 * t)
    quiet = 30 * np.random.default_rng(1).standard_normal(4000)
    return np.concatenate([quiet, tone, quiet]).astype("<i2").tobytes()


def say(text: str) -> list[dict]:
    """A scripted reply: the text in a few pieces, as a stream would deliver it."""
    words = text.split(" ")
    return [{"type": "content", "text": (" " if i else "") + w} for i, w in enumerate(words)]


def call(name: str, **args) -> dict:
    return {"type": "tool_calls", "calls": [{"id": f"c_{name}", "name": name, "arguments": json.dumps(args)}]}


class FakeLM:
    def __init__(self, replies: list[list[dict]] | None = None):
        self.replies = list(replies or [])
        self.requests: list[dict] = []
        self.spoken: list[str] = []

    async def chat_events(self, model, messages, *, tools=None, think=False, temperature=0.7, max_tokens=300):
        self.requests.append({"messages": copy.deepcopy(messages), "tools": [t["function"]["name"] for t in tools or []]})
        reply = self.replies.pop(0) if self.replies else say("(no more scripted replies)")
        for ev in reply:
            await asyncio.sleep(0)
            yield ev

    async def chat_stream(self, model, messages, **kw):
        async for ev in self.chat_events(model, messages, **kw):
            if ev["type"] == "content":
                yield ev["text"]

    async def tts_stream(self, text, model, voice, speed=1.0, chunk_bytes=4800):
        self.spoken.append(text)
        yield b"\x00\x00" * 240


class FakeModels:
    """Like ModelManager: the LLM and Whisper share the NPU, so `overlaps` counts Whisper calls made while an
    LLM stream was open (there must be none)."""

    def __init__(self, lm: FakeLM, transcripts: list[str] | None = None, delay: float = 0.01):
        self.lm = lm
        self.names = {"stt": "whisper", "llm": "gemma", "tts": "kokoro"}
        self.state = {r: "ready" for r in self.names}
        self.voice = "af_heart"
        self.transcripts = list(transcripts or [])
        self.delay = delay
        self.transcribed = 0
        self.streaming = 0
        self.overlaps = 0

    async def ensure(self, *roles):
        pass

    def ensure_in_background(self, *roles):
        return None

    async def llm_events(self, messages, **kw):
        self.streaming += 1
        try:
            async for ev in self.lm.chat_events(self.names["llm"], messages, **kw):
                yield ev
        finally:
            self.streaming -= 1

    async def llm_text(self, messages, **kw):
        async for ev in self.llm_events(messages, **kw):
            if ev["type"] == "content":
                yield ev["text"]

    async def transcribe(self, pcm: bytes) -> str:
        self.overlaps += self.streaming > 0
        self.transcribed += 1
        await asyncio.sleep(self.delay)
        return self.transcripts.pop(0) if self.transcripts else ""

    def subscribe(self, fn):
        pass

    def unsubscribe(self, fn):
        pass

    def snapshot(self):
        return {}


class FakeWS:
    pass


_REAL_SEARCH = next(t for t in discover()["web-search"].tools if t.name == "web_search")


class FakeSearch(Tool):
    """web_search without the web (it routes like the real one)."""
    name = "web_search"
    description = "Search the web."
    parameters = {"query": {"type": "string"}}
    required = ["query"]
    label = "Searching the web"
    filler = "Let me look that up."

    def __init__(self):
        self.queries: list[str] = []

    async def run(self, args, ctx):
        self.queries.append(args["query"])
        return ToolResult({"query": args["query"], "results": [{"title": "Forecast", "text": "Sunny, high 14."}]},
                          summary="Read 1 of 1 results", sources=[{"title": "Forecast", "url": "https://example.com"}])

    def route(self, text, ctx):
        return _REAL_SEARCH.route(text, ctx)


class SaveNote(Tool):
    """A tool that changes something, so Kira asks first."""
    name = "save_note"
    description = "Save a note."
    parameters = {"text": {"type": "string"}}
    required = ["text"]
    label = "Saving a note"
    confirm = True

    def __init__(self):
        self.saved: list[str] = []

    def describe(self, args):
        return args.get("text", "")

    async def run(self, args, ctx):
        self.saved.append(args["text"])
        return ToolResult({"saved": args["text"]}, summary="Saved")


def make_session(tmp_path: Path, replies=None, transcripts=None, skills=("clock", "calculator", "speaking-coach"),
                 extra_tools=()) -> tuple[Session, FakeLM, FakeModels]:
    cfg = load_config()
    cfg.chat.greet = False
    cfg.skills.confirm_timeout_s = 2
    found = discover()
    for tool in extra_tools:
        found[f"test-{tool.name.replace('_', '-')}"] = Skill(f"test-{tool.name.replace('_', '-')}", "Test tool.",
                                                             f"- {tool.name} for tests.", tools=[tool])
    names = list(skills) + [f"test-{t.name.replace('_', '-')}" for t in extra_tools]
    skillset = SkillSet(found, names)
    lm = FakeLM(replies)
    models = FakeModels(lm, transcripts)
    store = Store(tmp_path, "Bella", found["speaking-coach"].modules["modes"].SCENARIOS)
    svc = Services(cfg, lm, models, store, skillset, ToolContext(cfg, None, None, "America/Edmonton"), None)
    skillset.on_change(svc.skills_changed)
    return Session(FakeWS(), svc), lm, models


def events(session: Session, kind: str | None = None) -> list[dict]:
    """Everything the session sent so far (JSON events only), optionally of one type."""
    out = []
    while not session._out.empty():
        msg = session._out.get_nowait()
        if isinstance(msg, dict):
            out.append(msg)
    session._sent = getattr(session, "_sent", []) + out
    return [e for e in session._sent if kind is None or e["type"] == kind]
