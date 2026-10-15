"""One browser connection = one Session. Drives chat and the practice modes of the speaking-coach skill.

Wire protocol (WebSocket /ws):
  browser -> server  binary: 16 kHz mono PCM16 microphone audio (only while recording)
                     JSON:   start{mode, scenario?, details?} | rec_start | rec_stop | end | cancel | speak{text} | leave
                             | confirm{call_id, allow}
  server -> browser  binary: 4-byte little-endian reply id + 24 kHz mono PCM16 speech
                     JSON:   models, state, session, user_partial, user_text, reply_start/delta/end,
                             tool_start/tool_end, confirm_request/confirm_end, caption, metrics,
                             feedback_status/delta/done/cancelled, notice, error

In chat, Whisper (FLM, on the NPU) transcribes what the user said and the LLM gets that text: the same words
go to the chat bubble, the history and the router (ai_buddy.router), which makes the tool calls a request
clearly needs before the model runs. (Until 2026-10-14 Gemma 4 heard the recording itself; in real use it
misheard short phrases that Whisper got right, so the app went back to text.)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import struct
import time
import wave
from dataclasses import dataclass, field
from types import ModuleType

from fastapi import WebSocket

from .audio import PresentationChunker, analyze_speech, has_speech, pcm_seconds
from .config import Config
from .knowledge import KnowledgeBase, keywords
from .lemonade import Lemonade, LemonadeError
from .metrics import Answer, compute_metrics, words
from .models import ModelManager
from .prompts import CHAT_KICKOFF, CLAIM_NUDGE, NUDGE, Persona, chat_system_prompt, notes_block
from .router import recheck, recheck_note, route, said_by_user
from .skills import SkillSet
from .storage import Store
from .text import SpeechChunker, clean_for_speech, strip_emoji
from .tools import ToolCall, ToolContext, ToolRegistry, ToolResult

log = logging.getLogger(__name__)

MODES = ("chat", "presentation", "interview")
# "Let me look that up." with no tool call: the model announced a lookup instead of doing it.
_ANNOUNCE = re.compile(r"\b(let me|i'?ll|i will|one (sec|second|moment)|give me a (sec|second|moment)|hang on|"
                       r"hold on)\b[^.!?]{0,60}\b(look|check|search|find|pull|see|get)", re.I)
# "Got it, buy bread has been added": a claim that something was done. Without a tool call that's not true (seen
# with qwen3.5-9b in 1 of 4 runs of "please add buy bread to my to-do list").
_CLAIMED = re.compile(r"\b(?:i'?ve|i have|i|it'?s|it has|has|have|is|are)\s+(?:now\s+|just\s+|already\s+)?(?:been\s+)?"
                      r"(?:added|saved|created|written|wrote|deleted|removed|moved|renamed|sent|scheduled|booked|"
                      r"updated|set up|put|noted|stored|recorded|marked)\b|\bis now (?:on|in) your\b", re.I)
# A request to change something ("please add...", "can you create...").
_ACTION_REQUEST = re.compile(r"^\W*(?:(?:um|uh|so|okay|ok|hey|please|can you|could you|would you|will you)\b[,\s]*)*"
                             r"(?:add|put|save|create|make|write|delete|remove|move|rename|send|schedule|book|remind|"
                             r"set|update|edit|change|mark|note|append|clear)\b", re.I)
_NO = re.compile(r"\b(no|nope|nah|don'?t|do not|stop|cancel|never ?mind|wait|hold on|not now|not yet)\b", re.I)
_YES = re.compile(r"^\W*(?:(?:um|uh|ok|okay|so|yeah)\W+)*(?:yes|yeah|yep|yup|sure|go ahead|do it|please do|okay|ok|"
                  r"sounds good|absolutely|of course|correct|confirm(?:ed)?|alright|all right|definitely|why not)\b",
                  re.I)


def yes_or_no(text: str) -> bool | None:
    """A spoken answer to "Should I...?": True, False, or None when it's neither."""
    if _NO.search(text):
        return False
    return True if _YES.search(text) else None


@dataclass
class Services:
    """Everything a session needs, shared across connections."""
    cfg: Config
    lm: Lemonade
    models: ModelManager
    store: Store
    skills: SkillSet
    tool_ctx: ToolContext
    knowledge: KnowledgeBase | None
    fillers: dict[tuple[str, str], bytes] = field(default_factory=dict)   # (voice, text) -> PCM
    _registry: ToolRegistry | None = None

    @property
    def tools(self) -> ToolRegistry:
        """The tools of the skills that are on (rebuilt after a skill is switched)."""
        if self._registry is None:
            self._registry = self.skills.registry()
        return self._registry

    def skills_changed(self) -> None:
        self._registry = None

    @property
    def persona(self) -> Persona:
        """The assistant, named after the voice in use."""
        return Persona.of(self.cfg, self.models.voice)


@dataclass
class Filler:
    """A canned line spoken while a tool runs (synthesized once, then replayed from memory)."""
    text: str


@dataclass
class Round:
    """One LLM request: what the model wrote (for the history), what was shown and spoken, its tool calls."""
    raw: str = ""
    text: str = ""
    calls: list[dict] = field(default_factory=list)
    false_claim: bool = False        # said it did something without calling a tool (held back, never shown)


class LiveTranscript:
    """Transcribes a long push-to-talk turn in chunks, cut in pauses, while the user is still talking,
    so only the last chunk is left when they let go."""

    def __init__(self, session: Session, chunker: PresentationChunker):
        self.session, self.chunker = session, chunker
        self.queue: asyncio.Queue = asyncio.Queue()
        self.texts: list[str] = []
        self.failed = False
        self.task = asyncio.create_task(self._worker())

    def feed(self, pcm: bytes) -> None:
        for chunk in self.chunker.feed(pcm):
            self.queue.put_nowait(chunk)

    async def _worker(self) -> None:
        while (item := await self.queue.get()) is not None:
            _, pcm = item
            if self.failed or not has_speech(pcm):
                continue
            try:
                text = await self.session.models.transcribe(pcm)
            except LemonadeError as e:
                log.warning("live transcription of a chunk failed: %s", e)
                self.failed = True
                continue
            if words(text):
                self.texts.append(text)
                self.session.send({"type": "user_partial", "text": " ".join(self.texts)})

    async def finish(self) -> str | None:
        """The whole transcript, or None if a chunk failed (then transcribe the full recording)."""
        if tail := self.chunker.flush():
            self.queue.put_nowait(tail)
        self.queue.put_nowait(None)
        await self.task
        return None if self.failed else " ".join(self.texts)

    def cancel(self) -> None:
        self.task.cancel()


class Session:
    def __init__(self, ws: WebSocket, svc: Services):
        self.ws, self.svc = ws, svc
        self.cfg, self.lm, self.models, self.store = svc.cfg, svc.lm, svc.models, svc.store
        self._out: asyncio.Queue = asyncio.Queue()
        self._closed = False

        self.mode: str | None = None
        self.coach: ModuleType | None = None     # the speaking-coach skill's modes.py (practice modes)
        self.session_id: str | None = None
        self.scenario: str | None = None
        self.details = ""
        self.history: list[dict] = []
        self.history_prefix = 0     # leading messages that are never trimmed
        self.started_at = 0.0
        self.answers: list[Answer] = []
        self.timeline_s = 0.0

        self.recording = False
        self.rec = bytearray()
        self.live: LiveTranscript | None = None
        self.reply_id = 0
        self.turn_task: asyncio.Task | None = None
        self.feedback_tasks: set[asyncio.Task] = set()
        self.pending_confirm: tuple[str, asyncio.Future] | None = None   # (call id, answer)

        # presentation
        self.chunker: PresentationChunker | None = None
        self.chunk_q: asyncio.Queue | None = None
        self.chunk_task: asyncio.Task | None = None
        self.captions: list[tuple[float, str]] = []
        self.talk_pcm = bytearray()

    # ---------- plumbing ----------

    def send(self, msg: dict | bytes) -> None:
        if not self._closed:
            self._out.put_nowait(msg)

    async def _writer(self) -> None:
        while True:
            msg = await self._out.get()
            try:
                if isinstance(msg, bytes):
                    await self.ws.send_bytes(msg)
                else:
                    await self.ws.send_text(json.dumps(msg))
            except Exception:  # socket gone; keep draining so producers never block
                self._closed = True

    def _state(self, state: str, detail: str = "") -> None:
        self.send({"type": "state", "state": state, "detail": detail})

    def _notice(self, text: str) -> None:
        self.send({"type": "notice", "text": text})

    def _error(self, err: Exception | str) -> None:
        log.warning("session error: %s", err)
        self.send({"type": "error", "text": str(err)})
        self._state("idle")

    def _on_models(self, snapshot: dict) -> None:
        self.send({"type": "models", **snapshot})

    @property
    def llm(self) -> str:
        return self.models.names["llm"]

    async def run(self) -> None:
        writer = asyncio.create_task(self._writer())
        self.models.subscribe(self._on_models)
        self._on_models(self.models.snapshot())
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    self._on_audio(msg["bytes"])
                elif msg.get("text") is not None:
                    await self._on_command(json.loads(msg["text"]))
        finally:
            self._closed = True
            self.models.unsubscribe(self._on_models)
            await self._close_session()
            writer.cancel()

    # ---------- commands ----------

    async def _on_command(self, cmd: dict) -> None:
        kind = cmd.get("type")
        if kind == "start":
            await self._start(cmd.get("mode", "chat"), cmd.get("scenario"), cmd.get("details", ""))
        elif kind == "rec_start":
            if self.mode == "presentation" and not self.recording:
                self.details = str(cmd.get("details") or "").strip()[:300]
            self._rec_start()
        elif kind == "rec_stop":
            self._rec_stop()
        elif kind == "end" and self.mode == "interview":
            await self._cancel_turn()
            self._spawn_feedback(self._finish_interview(), self.session_id)
        elif kind == "cancel":
            await self._cancel_turn()
            self._state("idle")
        elif kind == "confirm":
            self._resolve_confirm(str(cmd.get("call_id", "")), bool(cmd.get("allow")))
        elif kind == "speak" and cmd.get("text"):
            await self._cancel_turn()
            self.turn_task = asyncio.create_task(self._guard(self._speak_text(cmd["text"])))
        elif kind == "leave":
            await self._close_session()
            self.mode = None

    async def _start(self, mode: str, scenario: str | None, details: str) -> None:
        if mode not in MODES:
            return self._error(f"Unknown mode {mode!r}")
        coach = None
        if mode != "chat":
            skill = self.svc.skills.mode_skill(mode)
            if skill is None:
                return self._error("Practice comes from the speaking coach skill, which is switched off. "
                                   "Switch it on in Settings (gear icon).")
            coach = skill.modules["modes"]
            if mode == "interview" and scenario not in coach.SCENARIOS:
                return self._error(f"Unknown scenario {scenario!r}")
        await self._close_session()
        self.mode, self.coach, self.scenario, self.details = mode, coach, scenario, (details or "").strip()
        self.answers, self.timeline_s, self.history = [], 0.0, []
        if mode == "presentation":
            # A presentation session is created per talk, when recording starts.
            self.send({"type": "session", "mode": mode, "id": None})
            return
        self.session_id = self.store.create_session(mode, scenario, self.details)
        self.started_at = time.monotonic()
        self.send({"type": "session", "mode": mode, "id": self.session_id})
        if mode == "chat":
            self.history = [{"role": "system", "content": self._chat_prompt()}]
            self.history_prefix = 1
            if self.cfg.chat.greet:
                self.turn_task = asyncio.create_task(self._guard(self._greet()))
            else:
                self.models.ensure_in_background("stt", "tts", "llm")
        else:
            self.turn_task = asyncio.create_task(self._guard(self._opening_line()))

    # ---------- the chat prompt ----------

    def _chat_prompt(self) -> str:
        skills = self.svc.skills
        return chat_system_prompt(self.svc.persona, self.svc.tool_ctx.timezone, skills.chat_instructions(),
                                  self.cfg.user.location, skills.extra_instructions())

    def _refresh_chat_prompt(self) -> None:
        """Follow skill switches and voice changes (the assistant's name). Changing the prompt makes FLM re-read
        the conversation once, so it only happens when something actually changed."""
        if self.mode != "chat" or not self.history:
            return
        prompt = self._chat_prompt()
        if self.history[0]["content"] != prompt:
            self.history[0] = {"role": "system", "content": prompt}

    # ---------- recording ----------

    def _rec_start(self) -> None:
        if self.mode is None:
            return self._notice("Pick a mode first.")
        if self.mode == "presentation":
            if self.feedback_tasks:
                return self._notice("Hang on, I'm still writing feedback for your last talk.")
            self._begin_talk()
        elif self.pending_confirm:
            self.live = None        # the user is answering a yes/no question: don't cut the assistant off
        else:
            # Barge-in: pressing talk while the assistant is speaking or thinking cancels that reply.
            if self.turn_task and not self.turn_task.done():
                self.turn_task.cancel()
            if self.live:
                self.live.cancel()
            s = self.cfg.stt
            self.live = LiveTranscript(
                self, PresentationChunker(target_s=s.live_chunk_s, max_s=s.live_chunk_max_s, quiet_s=s.live_pause_s))
        self.rec = bytearray()
        self.recording = True
        self._state("listening")

    def _rec_stop(self) -> None:
        if not self.recording:
            return
        self.recording = False
        if self.mode == "presentation":
            self._spawn_feedback(self._finish_presentation(), self.session_id)
        elif self.pending_confirm:
            asyncio.create_task(self._guard(self._confirm_answer(bytes(self.rec))))
        else:
            pcm, live, self.live = bytes(self.rec), self.live, None
            self.turn_task = asyncio.create_task(self._guard(self._user_turn(pcm, live)))

    def _on_audio(self, data: bytes) -> None:
        if not self.recording:
            return
        self.rec += data
        if self.live:
            self.live.feed(data)
        if self.mode == "presentation" and self.chunker:
            self.talk_pcm += data
            for chunk in self.chunker.feed(data):
                self.chunk_q.put_nowait(chunk)

    async def _cancel_turn(self) -> None:
        if self.pending_confirm:
            self._resolve_confirm(self.pending_confirm[0], False)
        task, self.turn_task = self.turn_task, None
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _guard(self, coro) -> None:
        """Run a turn coroutine, turning errors into UI messages."""
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except LemonadeError as e:
            self._error(e)
        except TimeoutError:
            self._error(f"{self.svc.persona.name} took too long to answer, so I stopped waiting. Please try again.")
        except Exception as e:
            log.exception("turn failed")
            self._error(f"Something went wrong: {e}")

    def _spawn_feedback(self, coro, sid: str | None) -> None:
        """Run end-of-practice analysis. Any failure is reported against its session, so the UI can
        unblock that session's view even if you've switched modes since."""
        async def run():
            try:
                await coro
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.exception("feedback failed")
                self._cancel_feedback(sid, f"Feedback failed: {e}", error=True)

        task = asyncio.create_task(run())
        self.feedback_tasks.add(task)
        task.add_done_callback(self.feedback_tasks.discard)

    def _cancel_feedback(self, sid: str | None, text: str, error: bool = False) -> None:
        self.send({"type": "feedback_cancelled", "session_id": sid, "text": text, "error": error})
        if self.mode in ("presentation", "interview"):
            self._state("idle")

    async def stop(self) -> None:
        """The app is exiting: stop everything, feedback still being written included, and save the session."""
        for task in list(self.feedback_tasks):
            task.cancel()
        await self._close_session()
        self.mode = None

    async def _close_session(self) -> None:
        """Stop whatever is running and save the current session (feedback in progress keeps going)."""
        await self._cancel_turn()
        self.recording = False
        if self.live:
            self.live.cancel()
            self.live = None
        if self.chunk_task and not self.chunk_task.done():
            # A talk that was never stopped is discarded.
            self.chunk_task.cancel()
            if self.session_id:
                self.store.delete_session(self.session_id)
                self.session_id = None
        self.chunk_task = self.chunker = None
        if self.session_id and self.mode in ("chat", "interview"):
            self.store.finish_session(self.session_id, duration_s=round(time.monotonic() - self.started_at, 1))
        self.session_id = None

    # ---------- notes ----------

    async def _notes(self, query: str, k: int, max_chars: int) -> str:
        """Background or talking points for practice, from the notes: only passages that really match."""
        kb = self.svc.knowledge
        if not (kb and kb.folder and query.strip() and self.svc.skills.is_enabled("notes")):
            return ""
        try:
            return notes_block(await kb.search(query, k, strict=True), max_chars)
        except Exception as e:
            log.warning("note search failed: %s", e)
            return ""

    # ---------- chat / interview turns ----------

    async def _greet(self) -> None:
        """The assistant's opening line in a new chat. It is shown, not spoken, and it primes FLM's prompt cache
        with the system prompt, so the user's first turn is as fast as the later ones."""
        await self.models.ensure("stt", "tts", "llm")
        self._refresh_chat_prompt()
        self.history.append({"role": "user", "content": CHAT_KICKOFF})
        self.history_prefix = 2
        self.reply_id += 1
        rid = self.reply_id
        self.send({"type": "reply_start", "id": rid, "greeting": True})
        parts: list[str] = []
        try:
            async with asyncio.timeout(self.cfg.chat.timeout_s):
                async with contextlib.aclosing(self.models.llm_events(
                        self.history, tools=self._tool_schemas(), think=False, temperature=self.cfg.chat.temperature,
                        max_tokens=60)) as events:
                  async for ev in events:
                    if ev["type"] == "content":
                        parts.append(ev["text"])
                        self.send({"type": "reply_delta", "id": rid, "text": ev["text"]})
        finally:
            text = strip_emoji("".join(parts)) or "Hi! What's on your mind?"
            self.history.append({"role": "assistant", "content": text})
            self.send({"type": "reply_end", "id": rid, "text": text, "interrupted": False, "timing": {},
                       "greeting": True})

    async def _opening_line(self) -> None:
        if any(self.models.state[r] != "ready" for r in ("stt", "tts", "llm")):
            self._state("loading", "Getting the conversation model ready…")
        s = self.coach.SCENARIOS[self.scenario]
        background = ""
        if s["notes_query"]:
            background = await self._notes(f"{self.details} {s['notes_query']}", 4, 2000)
        self.history = [{"role": "system", "content": self.coach.roleplay_system_prompt(
                            self.scenario, self.details, background, self.svc.persona.name)},
                        {"role": "user", "content": self.coach.KICKOFF}]
        self.history_prefix = 2
        await self.models.ensure("stt", "tts", "llm")
        await self._reply(time.perf_counter(), tools=False)

    async def _user_turn(self, pcm: bytes, live: LiveTranscript | None) -> None:
        analysis = analyze_speech(pcm)
        if analysis.speech_s < 0.25:
            if live:
                live.cancel()
            self._notice("I didn't hear anything. Hold the button (or Space) while you talk.")
            return self._state("idle")
        self._state("transcribing")
        if any(self.models.state[r] != "ready" for r in ("stt", "tts", "llm")):
            self._state("loading", "Loading models…")
        await self.models.ensure("stt", "tts", "llm")
        self._refresh_chat_prompt()
        t_end = time.perf_counter()
        audio_s = round(pcm_seconds(pcm), 2)
        self._state("transcribing")
        text = await self._transcript(pcm, live)
        stt_ms = round((time.perf_counter() - t_end) * 1000)
        if not words(text):
            self._notice("Sorry, I couldn't make that out. Try again?")
            return self._state("idle")
        self.send({"type": "user_text", "text": text, "stt_ms": stt_ms})
        self.store.add_turn(self.session_id, "user", text, audio_s)
        if self.mode == "interview":
            self.answers.append(Answer(text, analysis, self.timeline_s))
            self._save_audio(pcm, f"{self.session_id}-{len(self.answers):02d}")
        self.timeline_s += audio_s
        self.history.append({"role": "user", "content": text})
        self._trim_history()
        routed, claim = [], None
        if self.mode == "chat" and self._tool_schemas():
            routed = route(text, self.svc.tools, self.svc.tool_ctx)
            if not routed and (routed := recheck(text, self.history[:-1], self.svc.tools)):
                claim = text      # the user questioned an answer from tools: check again, compare honestly
        await self._reply(t_end, tools=self.mode == "chat", routed=routed, claim=claim, request=text)

    async def _transcript(self, pcm: bytes, live: LiveTranscript | None) -> str:
        text = await live.finish() if live else None
        return text if text is not None else await self.models.transcribe(pcm)

    def _trim_history(self) -> None:
        """Drop the oldest exchanges once over budget. Trimming changes the prompt prefix, so FLM has to
        re-read the whole conversation once; cutting a third at a time keeps that rare."""
        budget = self.cfg.chat.history_word_budget

        def size() -> int:
            return sum(len(str(m.get("content") or "").split()) for m in self.history[self.history_prefix:])
        if size() <= budget:
            return
        while size() > budget * 2 // 3 and len(self.history) > self.history_prefix + 2:
            # Remove one whole exchange: something the user said and everything up to the next thing they said.
            end = self.history_prefix + 1
            while end < len(self.history) and not said_by_user(self.history[end]):
                end += 1
            del self.history[self.history_prefix:end]

    def _tool_schemas(self) -> list[dict] | None:
        return self.svc.tools.schemas() if self.mode == "chat" and self.svc.tools else None

    # ---------- the assistant's reply ----------

    async def _reply(self, t_ref: float, tools: bool = True, routed: list[ToolCall] = (),
                     claim: str | None = None, request: str = "") -> None:
        """Stream the assistant's reply and speak it clause by clause while it is still being written. If the model
        calls tools, say a short filler line, run them, and let the model continue with the results."""
        self.reply_id += 1
        rid = self.reply_id
        self._state("thinking")
        self.send({"type": "reply_start", "id": rid})
        timing: dict = {}
        shown: list[str] = []     # everything the assistant wrote this turn (all rounds)
        spoken: list[str] = []
        meta: dict = {"tools": [], "sources": []}
        pieces: asyncio.Queue = asyncio.Queue()

        async def speaker():
            while (piece := await pieces.get()) is not None:
                if isinstance(piece, Filler):
                    await self._speak_filler(piece.text, rid, timing, t_ref)
                else:
                    spoken.append(piece)
                    await self._tts(piece, rid, timing, t_ref)

        speaker_task = asyncio.create_task(speaker())
        interrupted, final = True, ""
        try:
            final = await self._agent_loop(rid, pieces, timing, t_ref, shown, meta, tools, list(routed), claim,
                                           request)
            pieces.put_nowait(None)
            await speaker_task
            interrupted = False
        finally:
            speaker_task.cancel()
            if interrupted:
                text = strip_emoji(" ".join(spoken))
                if final:   # the answer was complete but cut off while speaking: keep what was heard
                    self.history[-1]["content"] = text or "…"
                elif text:
                    self.history.append({"role": "assistant", "content": text})
            else:
                text = strip_emoji(" ".join(s.strip() for s in shown if s.strip()))
            if text and self.session_id:
                self.store.add_turn(self.session_id, "assistant", text, interrupted=interrupted,
                                    meta=meta if meta["tools"] else None)
            timing["total_ms"] = round((time.perf_counter() - t_ref) * 1000)
            self.send({"type": "reply_end", "id": rid, "text": text, "interrupted": interrupted, "timing": timing,
                       "sources": meta["sources"]})
            if not interrupted:
                self._state("idle")

    async def _agent_loop(self, rid: int, pieces: asyncio.Queue, timing: dict, t_ref: float, shown: list[str],
                          meta: dict, tools_on: bool, routed: list[ToolCall], claim: str | None = None,
                          request: str = "") -> str:
        """LLM rounds until the model answers without calling a tool. Returns the final answer text
        (already appended to the history). `claim`: what the user said when questioning an answer, if the routed
        calls re-check it. `request`: what the user said; when it asks for a change and a tool that changes things
        is on, each round is held back until it's complete, so "it's done" without a tool call is never heard."""
        cap = self.cfg.tools.max_calls_per_turn
        registry = self.svc.tools
        schemas = self._tool_schemas() if tools_on else None
        used, nudged, over_cap_rounds = 0, False, 0
        hold = bool(schemas and _ACTION_REQUEST.search(request) and any(t.confirm for t in registry.tools.values()))
        if routed and schemas:
            # The words were known first and clearly need these tools: make the calls before the model runs,
            # so it answers from the results in one round.
            used += len(routed)
            await self._tool_round(rid, routed, "", "", pieces, meta, timing, t_ref, shown, claim)
        for _ in range(cap + 3):
            rnd = await self._llm_round(rid, pieces, timing, t_ref, shown, schemas, hold and not used)
            if rnd.false_claim:
                if nudged:
                    fallback = "Sorry, I couldn't do that just now. Could you ask me again?"
                    break
                nudged = True
                self.history.append({"role": "assistant", "content": rnd.raw})
                self.history.append({"role": "user", "content": CLAIM_NUDGE})
                continue
            calls = [registry.parse(c) for c in rnd.calls] if schemas else []
            if calls:
                if used >= cap:
                    # Over the limit: answer every call with an error so the history stays valid.
                    over_cap_rounds += 1
                    results = [ToolResult({"error": "Tool limit reached for this turn. Answer the user now with "
                                                    "what you already have."}, ok=False) for _ in calls]
                    self.history.append({"role": "assistant", "content": rnd.raw,
                                         "tool_calls": [c.for_history() for c in calls]})
                    self.history.append(registry.results_message(calls, results))
                    if over_cap_rounds > 1:
                        fallback = "Sorry, I couldn't pin that down. Could you ask me in a different way?"
                        break
                    continue
                calls = calls[: cap - used]
                used += len(calls)
                await self._tool_round(rid, calls, rnd.raw, rnd.text, pieces, meta, timing, t_ref, shown)
                continue
            if schemas and not nudged and used < cap and _ANNOUNCE.search(rnd.text) and len(rnd.text.split()) <= 40:
                # It said it would look something up but didn't call a tool: what it said serves as the
                # filler line; ask it to make the call.
                nudged = True
                self.history.append({"role": "assistant", "content": rnd.raw})
                self.history.append({"role": "user", "content": NUDGE})
                continue
            if rnd.text.strip():
                self.history.append({"role": "assistant", "content": rnd.raw})
                return rnd.text
            fallback = "Sorry, I lost my train of thought there. Could you say that again?"
            break
        else:
            # Still asking for tools after the limit: give up gracefully.
            fallback = "Sorry, I couldn't pin that down. Could you ask me in a different way?"
        shown.append(fallback)
        self.send({"type": "reply_delta", "id": rid, "text": " " + fallback})
        pieces.put_nowait(fallback)
        self.history.append({"role": "assistant", "content": fallback})
        return fallback

    async def _tool_round(self, rid: int, calls: list[ToolCall], raw: str, text: str, pieces: asyncio.Queue,
                          meta: dict, timing: dict, t_ref: float, shown: list[str], claim: str | None = None) -> None:
        """Run a round of tool calls and add them and their results to the history."""
        registry = self.svc.tools
        if not text.strip():
            tool = registry.get(calls[0].name)
            pieces.put_nowait(Filler(tool.filler_line() if tool else "One moment."))
        self._state("tool", self._tool_label(calls))
        results = await self._run_tools(rid, calls, meta, timing, t_ref, pieces, shown)
        self._state("thinking")
        self.history.append({"role": "assistant", "content": raw, "tool_calls": [c.for_history() for c in calls]})
        if claim is None:
            self.history.append(registry.results_message(calls, results))
        else:
            self.history.append(registry.results_message(calls, results, recheck_note(claim, results)))

    async def _llm_round(self, rid: int, pieces: asyncio.Queue, timing: dict, t_ref: float, shown: list[str],
                         schemas: list[dict] | None, hold: bool = False) -> Round:
        """One streamed LLM request, shown and spoken as it's written. With `hold`, nothing is shown or spoken until
        the round is complete; a round that claims something was done without calling a tool is dropped
        (`false_claim`)."""
        chunker = SpeechChunker()
        rnd = Round()
        raw: list[str] = []
        content: list[str] = []
        held: list[str] = []
        gap = bool("".join(shown).strip())   # separate this round's text from the previous round's

        def release(delta: str) -> None:
            content.append(delta)
            self.send({"type": "reply_delta", "id": rid, "text": delta})
            for piece in chunker.feed(delta):
                pieces.put_nowait(piece)

        stream = self.models.llm_events(self.history, tools=schemas, think=False, temperature=self.cfg.chat.temperature,
                                        max_tokens=self.cfg.chat.max_tokens)
        try:
            async with asyncio.timeout(self.cfg.chat.timeout_s):
                async for ev in stream:
                    if ev["type"] == "tool_calls":
                        rnd.calls = ev["calls"]
                        timing.setdefault("tool_call_ms", round((time.perf_counter() - t_ref) * 1000))
                        continue
                    delta = ev["text"]
                    raw.append(delta)
                    if not content and not delta.strip():
                        continue
                    if "llm_first_token_ms" not in timing:
                        timing["llm_first_token_ms"] = round((time.perf_counter() - t_ref) * 1000)
                    if gap and not content and not held:
                        delta = " " + delta.lstrip()
                    if hold:
                        held.append(delta)
                    else:
                        release(delta)
        finally:
            await stream.aclose()       # releases the NPU (FLM stops when the client disconnects)
        if held:
            if not rnd.calls and _CLAIMED.search("".join(held)):
                rnd.false_claim = True
                rnd.raw = "".join(raw)
                return rnd
            release("".join(held))
        for piece in chunker.flush():
            pieces.put_nowait(piece)
        rnd.raw = "".join(raw)
        rnd.text = "".join(content)
        if rnd.text.strip():
            shown.append(rnd.text)
        return rnd

    def _tool_label(self, calls) -> str:
        tool = self.svc.tools.get(calls[0].name)
        label = (tool.label if tool else "Working") + "…"
        detail = tool.describe(calls[0].args) if tool else ""
        return f"{label} {detail}".strip() if detail and len(detail) < 60 else label

    async def _run_tools(self, rid: int, calls: list[ToolCall], meta: dict, timing: dict, t_ref: float,
                         pieces: asyncio.Queue, shown: list[str]) -> list[ToolResult]:
        registry = self.svc.tools
        # Tools that change something ask first, one at a time, before anything runs.
        allowed: dict[str, bool] = {}
        for call in calls:
            tool = registry.get(call.name)
            if tool and tool.confirm and not call.error:
                allowed[call.id] = await self._ask_permission(rid, call, tool, pieces, shown)

        async def one(call: ToolCall) -> ToolResult:
            tool = registry.get(call.name)
            self.send({"type": "tool_start", "id": rid, "call_id": call.id, "name": call.name,
                       "label": tool.label if tool else "Tool", "detail": tool.describe(call.args) if tool else "",
                       "routed": call.routed})
            t0 = time.perf_counter()
            if allowed.get(call.id, True):
                result = await registry.execute(call, self.svc.tool_ctx)
            else:
                result = ToolResult({"error": "The user said no, so this was not done. Don't try it again unless "
                                              "they ask."}, summary="You said no", ok=False)
            ms = round((time.perf_counter() - t0) * 1000)
            self.send({"type": "tool_end", "id": rid, "call_id": call.id, "name": call.name, "ok": result.ok,
                       "summary": result.summary, "sources": result.sources, "ms": ms})
            meta["tools"].append({"name": call.name, "args": call.args, "ok": result.ok, "summary": result.summary,
                                  "ms": ms, **({"repaired": True} if call.repaired else {}),
                                  **({"routed": True} if call.routed else {}),
                                  **({"declined": True} if not allowed.get(call.id, True) else {})})
            for s in result.sources:
                if s not in meta["sources"]:
                    meta["sources"].append(s)
            return result

        results = await asyncio.gather(*(one(c) for c in calls))
        timing["tools_ms"] = timing.get("tools_ms", 0) + max((t["ms"] for t in meta["tools"][-len(calls):]), default=0)
        return results

    # ---------- asking permission ----------

    async def _ask_permission(self, rid: int, call: ToolCall, tool, pieces: asyncio.Queue, shown: list[str]) -> bool:
        """The assistant asks before a tool that changes something; the user answers by voice or with the buttons.
        No answer within skills.confirm_timeout_s counts as no."""
        question = tool.confirm_text(call.args)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self.pending_confirm = (call.id, fut)
        self.send({"type": "confirm_request", "id": rid, "call_id": call.id, "skill": tool.skill, "tool": call.name,
                   "text": question, "args": call.args})
        self.send({"type": "reply_delta", "id": rid, "text": " " + question})
        shown.append(question)      # part of the reply: what follows is spaced after it, and it's saved with it
        pieces.put_nowait(question)
        self._state("confirm", question)
        try:
            allow = await asyncio.wait_for(fut, self.cfg.skills.confirm_timeout_s)
        except TimeoutError:
            allow = False
        finally:
            self.pending_confirm = None
        self.send({"type": "confirm_end", "id": rid, "call_id": call.id, "allowed": allow})
        self._state("tool" if allow else "thinking")
        return allow

    def _resolve_confirm(self, call_id: str, allow: bool) -> None:
        if self.pending_confirm and self.pending_confirm[0] == call_id and not self.pending_confirm[1].done():
            self.pending_confirm[1].set_result(allow)

    async def _confirm_answer(self, pcm: bytes) -> None:
        """A spoken answer to the assistant's yes/no question (Whisper + a pattern, no LLM call)."""
        if not has_speech(pcm):
            return self._notice("I didn't hear anything. Say yes or no, or use the buttons.")
        text = await self.models.transcribe(pcm)
        self.send({"type": "user_text", "text": text})
        answer = yes_or_no(text)
        if answer is None:
            return self._notice("Was that a yes or a no? You can also use the buttons.")
        if self.pending_confirm:
            self._resolve_confirm(self.pending_confirm[0], answer)

    # ---------- speech ----------

    def _send_pcm(self, pcm: bytes, rid: int, timing: dict, t_ref: float) -> None:
        if "first_audio_ms" not in timing:
            timing["first_audio_ms"] = round((time.perf_counter() - t_ref) * 1000)
        self.send(struct.pack("<I", rid) + pcm)

    async def _tts(self, text: str, rid: int, timing: dict, t_ref: float) -> None:
        clean = clean_for_speech(text)
        if not clean:
            return
        async for pcm in self.lm.tts_stream(clean, self.models.names["tts"], self.models.voice,
                                            self.cfg.voice.tts_speed):
            self._send_pcm(pcm, rid, timing, t_ref)

    async def _speak_filler(self, text: str, rid: int, timing: dict, t_ref: float) -> None:
        key = (self.models.voice, text)
        if key not in self.svc.fillers:
            buf = bytearray()
            async for pcm in self.lm.tts_stream(text, self.models.names["tts"], self.models.voice,
                                                self.cfg.voice.tts_speed):
                if not self.svc.fillers.get(key):   # stream the first synthesis straight through
                    self._send_pcm(pcm, rid, timing, t_ref)
                buf += pcm
            self.svc.fillers[key] = bytes(buf)
            return
        pcm = self.svc.fillers[key]
        for i in range(0, len(pcm), 4800):
            self._send_pcm(pcm[i:i + 4800], rid, timing, t_ref)

    async def _speak_text(self, text: str) -> None:
        self.reply_id += 1
        rid = self.reply_id
        self.send({"type": "reply_start", "id": rid, "readout": True})
        chunker = SpeechChunker(first_min_words=6)
        timing: dict = {}
        for piece in chunker.feed(text + " ") + chunker.flush():
            await self._tts(piece, rid, timing, time.perf_counter())
        self.send({"type": "reply_end", "id": rid, "text": "", "interrupted": False, "timing": timing,
                   "readout": True})

    # ---------- presentation ----------

    def _begin_talk(self) -> None:
        self.session_id = self.store.create_session("presentation", None, self.details)
        self.started_at = time.monotonic()
        self.send({"type": "session", "mode": "presentation", "id": self.session_id})
        self.chunker = PresentationChunker(target_s=self.cfg.practice.presentation_chunk_s)
        self.chunk_q = asyncio.Queue()
        self.captions = []
        self.talk_pcm = bytearray()
        self.chunk_task = asyncio.create_task(self._transcribe_chunks(self.chunk_q))
        self.models.ensure_in_background("llm")

    async def _transcribe_chunks(self, q: asyncio.Queue) -> None:
        while (item := await q.get()) is not None:
            start_s, pcm = item
            if not has_speech(pcm):
                continue
            try:
                text = await self.models.transcribe(pcm)
            except LemonadeError as e:
                self._notice(f"Transcription of one part failed: {e}")
                continue
            if words(text):
                self.captions.append((start_s, text))
                self.send({"type": "caption", "text": text, "start_s": round(start_s, 1)})

    async def _finish_presentation(self) -> None:
        sid, chunker, q, task = self.session_id, self.chunker, self.chunk_q, self.chunk_task
        topic = self.details
        self.session_id = None
        self.chunker = self.chunk_task = None
        if not (sid and chunker and q and task):
            return
        self._state("analyzing", "Finishing the transcript…")
        if tail := chunker.flush():
            q.put_nowait(tail)
        q.put_nowait(None)
        await task
        pcm = bytes(self.talk_pcm)
        transcript = " ".join(text for _, text in sorted(self.captions))
        if not words(transcript):
            self.store.delete_session(sid)
            return self._cancel_feedback(sid, "I didn't catch any speech in that recording.")
        self.store.add_turn(sid, "user", transcript, round(pcm_seconds(pcm), 1))
        self._save_audio(pcm, sid)
        metrics = self._metrics([Answer(transcript, analyze_speech(pcm))])
        self.send({"type": "metrics", "metrics": metrics, "session_id": sid})
        points = await self._notes(f"{topic} talking points {keywords(transcript)}", 5, 3000)
        await self._feedback(sid, "presentation", transcript, metrics, details=topic, talking_points=points,
                             duration_s=pcm_seconds(pcm))

    # ---------- interview end ----------

    async def _finish_interview(self) -> None:
        sid, answers, history = self.session_id, self.answers, self.history
        scenario, details = self.scenario, self.details
        self.session_id, self.answers = None, []
        if not sid:
            return
        if not answers:
            self.store.finish_session(sid)
            return self._cancel_feedback(sid, "You haven't answered anything yet, so there's nothing to give feedback on.")
        self._state("analyzing")
        metrics = self._metrics(answers)
        self.send({"type": "metrics", "metrics": metrics, "session_id": sid})
        counterpart = self.coach.SCENARIOS[scenario]["counterpart"]
        dialogue = "\n".join(f"{'You' if m['role'] == 'user' else counterpart}: {m['content']}"
                             for m in history[2:] if m["role"] in ("user", "assistant") and m.get("content"))
        points = ""
        if self.coach.SCENARIOS[scenario]["notes_query"]:
            said = " ".join(a.text for a in answers)
            points = await self._notes(f"{details} talking points {keywords(said)}", 4, 2500)
        await self._feedback(sid, "interview", dialogue, metrics, scenario=scenario, details=details,
                             talking_points=points, duration_s=time.monotonic() - self.started_at)

    # ---------- shared feedback ----------

    def _metrics(self, answers: list[Answer]) -> dict:
        p = self.cfg.practice
        return compute_metrics(answers, long_pause_s=p.long_pause_s, wpm_low=p.target_wpm_low,
                               wpm_high=p.target_wpm_high)

    async def _feedback(self, sid: str, kind: str, transcript: str, metrics: dict, *, scenario: str | None = None,
                        details: str = "", talking_points: str = "", duration_s: float | None = None) -> None:
        model = self.llm
        think = self.cfg.feedback.think
        coach = self.coach

        def state(name: str, detail: str = "") -> None:
            # Feedback keeps running if you switch modes; only drive the status line while you stay.
            if self.mode == kind:
                self._state(name, detail)

        if self.models.state["llm"] != "ready":
            state("analyzing", "Loading the model…")
        await self.models.ensure("llm")
        state("analyzing", "Thinking it over…" if think else "Writing your feedback…")
        self.send({"type": "feedback_status", "text": "started", "model": model})
        parts: list[str] = []
        messages = coach.feedback_messages(kind, transcript, metrics, scenario, details, talking_points,
                                           persona=self.svc.persona.name)
        async with contextlib.aclosing(self.models.llm_text(messages, think=think,
                                                            temperature=self.cfg.feedback.temperature,
                                                            max_tokens=self.cfg.feedback.max_tokens)) as text:
            async for delta in text:
                parts.append(delta)
                self.send({"type": "feedback_delta", "text": delta, "session_id": sid})
        md = "".join(parts).strip()
        path = self.store.finish_session(sid, duration_s=round(duration_s or 0, 1), metrics=metrics,
                                         feedback_md=md, feedback_model=model)
        self.send({"type": "feedback_done", "session_id": sid, "markdown": md, "path": str(path or "")})
        state("idle")

    def _save_audio(self, pcm: bytes, name: str) -> None:
        if not self.cfg.app.save_audio:
            return
        folder = self.cfg.app.data_path / "audio"
        folder.mkdir(parents=True, exist_ok=True)
        with wave.open(str(folder / f"{name}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm)
