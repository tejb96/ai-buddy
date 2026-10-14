"""One browser connection = one Session. Drives chat, presentation, and conversation-practice modes.

Wire protocol (WebSocket /ws):
  browser -> server  binary: 16 kHz mono PCM16 microphone audio (only while recording)
                     JSON:   start{mode, scenario?, details?} | rec_start | rec_stop | end | cancel | speak{text} | leave
  server -> browser  binary: 4-byte little-endian reply id + 24 kHz mono PCM16 speech
                     JSON:   models, state, session, user_partial, user_text, reply_start/delta/end,
                             tool_start/tool_end, caption, metrics, feedback_status/delta/done/cancelled, notice, error
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

from fastapi import WebSocket

from .audio import PresentationChunker, analyze_speech, has_speech, pcm_seconds
from .config import Config
from .knowledge import KnowledgeBase, keywords
from .lemonade import Lemonade, LemonadeError
from .metrics import Answer, compute_metrics, words
from .models import ModelManager
from .prompts import (CHAT_KICKOFF, KICKOFF, NUDGE, SCENARIOS, chat_system_prompt, feedback_messages, notes_block,
                      roleplay_system_prompt)
from .storage import Store
from .text import SpeechChunker, clean_for_speech, strip_emoji
from .tools import RESULTS_PREFIX, ToolContext, ToolRegistry, ToolResult

log = logging.getLogger(__name__)

MODES = ("chat", "presentation", "interview")
# "Let me look that up." with no tool call: the model announced a lookup instead of doing it.
_ANNOUNCE = re.compile(r"\b(let me|i'?ll|i will|one (sec|second|moment)|give me a (sec|second|moment)|hang on|"
                       r"hold on)\b[^.!?]{0,60}\b(look|check|search|find|pull|see|get)", re.I)


@dataclass
class Services:
    """Everything a session needs, shared across connections."""
    cfg: Config
    lm: Lemonade
    models: ModelManager
    store: Store
    tools: ToolRegistry
    tool_ctx: ToolContext
    knowledge: KnowledgeBase | None
    fillers: dict[tuple[str, str], bytes] = field(default_factory=dict)   # (voice, text) -> PCM


@dataclass
class Filler:
    """A canned line spoken while a tool runs (synthesized once, then replayed from memory)."""
    text: str


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
        elif kind == "speak" and cmd.get("text"):
            await self._cancel_turn()
            self.turn_task = asyncio.create_task(self._guard(self._speak_text(cmd["text"])))
        elif kind == "leave":
            await self._close_session()
            self.mode = None

    async def _start(self, mode: str, scenario: str | None, details: str) -> None:
        if mode not in MODES:
            return self._error(f"Unknown mode {mode!r}")
        if mode == "interview" and scenario not in SCENARIOS:
            return self._error(f"Unknown scenario {scenario!r}")
        await self._close_session()
        self.mode, self.scenario, self.details = mode, scenario, (details or "").strip()
        self.answers, self.timeline_s, self.history = [], 0.0, []
        if mode == "presentation":
            # A presentation session is created per talk, when recording starts.
            self.send({"type": "session", "mode": mode, "id": None})
            return
        self.session_id = self.store.create_session(mode, scenario, self.details)
        self.started_at = time.monotonic()
        self.send({"type": "session", "mode": mode, "id": self.session_id})
        if mode == "chat":
            self.history = [{"role": "system", "content": chat_system_prompt(
                self.cfg.persona, self.svc.tool_ctx.timezone, tools=bool(self.svc.tools))}]
            self.history_prefix = 1
            if self.cfg.chat.greet:
                self.turn_task = asyncio.create_task(self._guard(self._greet()))
            else:
                self.models.ensure_in_background("stt", "tts", "llm")
        else:
            self.turn_task = asyncio.create_task(self._guard(self._opening_line()))

    def _rec_start(self) -> None:
        if self.mode is None:
            return self._notice("Pick a mode first.")
        if self.mode == "presentation":
            if self.feedback_tasks:
                return self._notice("Hang on, I'm still writing feedback for your last talk.")
            self._begin_talk()
        else:
            # Barge-in: pressing talk while Kira is speaking or thinking cancels that reply.
            if self.turn_task and not self.turn_task.done():
                self.turn_task.cancel()
            if self.live:
                self.live.cancel()
            s = self.cfg.stt
            self.live = LiveTranscript(self, PresentationChunker(target_s=s.live_chunk_s, max_s=s.live_chunk_max_s,
                                                                 quiet_s=s.live_pause_s))
        self.rec = bytearray()
        self.recording = True
        self._state("listening")

    def _rec_stop(self) -> None:
        if not self.recording:
            return
        self.recording = False
        if self.mode == "presentation":
            self._spawn_feedback(self._finish_presentation(), self.session_id)
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
            self._error("Kira took too long to answer, so I stopped waiting. Please try again.")
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
        kb = self.svc.knowledge
        if not (kb and kb.folder and query.strip()):
            return ""
        try:
            return notes_block(await kb.search(query, k), max_chars)
        except Exception as e:
            log.warning("note search failed: %s", e)
            return ""

    # ---------- chat / interview turns ----------

    async def _greet(self) -> None:
        """Kira's opening line in a new chat. It is shown, not spoken, and it primes FLM's prompt cache
        with the system prompt, so the user's first turn is as fast as the later ones."""
        await self.models.ensure("stt", "tts", "llm")
        self.history.append({"role": "user", "content": CHAT_KICKOFF})
        self.history_prefix = 2
        self.reply_id += 1
        rid = self.reply_id
        self.send({"type": "reply_start", "id": rid, "greeting": True})
        parts: list[str] = []
        try:
            async with asyncio.timeout(self.cfg.chat.timeout_s):
                async for ev in self.lm.chat_events(self.llm, list(self.history), tools=self._tool_schemas(),
                                                    think=False, temperature=self.cfg.chat.temperature,
                                                    max_tokens=60):
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
        s = SCENARIOS[self.scenario]
        background = ""
        if s["notes_query"]:
            background = await self._notes(f"{self.details} {s['notes_query']}", 4, 2000)
        self.history = [{"role": "system", "content": roleplay_system_prompt(self.scenario, self.details, background)},
                        {"role": "user", "content": KICKOFF}]
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
        self._state("transcribing")
        t_end = time.perf_counter()
        text = await live.finish() if live else None
        if text is None:
            text = await self.models.transcribe(pcm)
        stt_ms = round((time.perf_counter() - t_end) * 1000)
        if not words(text):
            self._notice("Sorry, I couldn't make that out. Try again?")
            return self._state("idle")
        audio_s = round(pcm_seconds(pcm), 2)
        self.send({"type": "user_text", "text": text, "stt_ms": stt_ms})
        self.store.add_turn(self.session_id, "user", text, audio_s)
        if self.mode == "interview":
            self.answers.append(Answer(text, analysis, self.timeline_s))
            self._save_audio(pcm, f"{self.session_id}-{len(self.answers):02d}")
        self.timeline_s += audio_s
        self.history.append({"role": "user", "content": text})
        self._trim_history()
        await self._reply(t_end, tools=self.mode == "chat")

    def _trim_history(self) -> None:
        """Drop the oldest exchanges once over budget. Trimming changes the prompt prefix, so FLM has to
        re-read the whole conversation once; cutting a third at a time keeps that rare."""
        budget = self.cfg.chat.history_word_budget
        size = lambda: sum(len(str(m.get("content") or "").split()) for m in self.history[self.history_prefix:])
        if size() <= budget:
            return
        while size() > budget * 2 // 3 and len(self.history) > self.history_prefix + 2:
            # Remove one whole exchange: something the user said and everything up to the next thing they said.
            end = self.history_prefix + 1
            while end < len(self.history) and not self._said_by_user(self.history[end]):
                end += 1
            del self.history[self.history_prefix:end]

    @staticmethod
    def _said_by_user(msg: dict) -> bool:
        content = str(msg.get("content") or "")
        return msg["role"] == "user" and not content.startswith(RESULTS_PREFIX) and content != NUDGE

    def _tool_schemas(self) -> list[dict] | None:
        return self.svc.tools.schemas() if self.svc.tools and self.mode == "chat" else None

    async def _reply(self, t_ref: float, tools: bool = True) -> None:
        """Stream Kira's reply and speak it clause by clause while it is still being written. If the model
        calls tools, say a short filler line, run them, and let the model continue with the results."""
        self.reply_id += 1
        rid = self.reply_id
        self._state("thinking")
        self.send({"type": "reply_start", "id": rid})
        timing: dict = {}
        shown: list[str] = []     # everything Kira wrote this turn (all rounds)
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
            final = await self._agent_loop(rid, pieces, timing, t_ref, shown, meta, tools)
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
                          meta: dict, tools_on: bool) -> str:
        """LLM rounds until the model answers without calling a tool. Returns the final answer text
        (already appended to the history)."""
        cap = self.cfg.tools.max_calls_per_turn
        schemas = self._tool_schemas() if tools_on else None
        used, nudged, over_cap_rounds = 0, False, 0
        for _ in range(cap + 3):
            content, raw_calls = await self._llm_round(rid, pieces, timing, t_ref, shown, schemas)
            if raw_calls and schemas:
                calls = [self.svc.tools.parse(c) for c in raw_calls]
                if used >= cap:
                    # Over the limit: answer every call with an error so the history stays valid.
                    over_cap_rounds += 1
                    results = [ToolResult({"error": "Tool limit reached for this turn. Answer the user now with "
                                                    "what you already have."}, ok=False) for _ in calls]
                    if over_cap_rounds > 1:
                        fallback = "Sorry, I couldn't pin that down. Could you ask me in a different way?"
                        break
                else:
                    calls = calls[: cap - used]
                    used += len(calls)
                    if not content.strip():
                        tool = self.svc.tools.get(calls[0].name)
                        pieces.put_nowait(Filler(tool.filler_line() if tool else "One moment."))
                    self._state("tool", self._tool_label(calls))
                    results = await self._run_tools(rid, calls, meta, timing, t_ref)
                    self._state("thinking")
                self.history.append({"role": "assistant", "content": content,
                                     "tool_calls": [c.for_history() for c in calls]})
                self.history.append(self.svc.tools.results_message(calls, results))
                continue
            if schemas and not nudged and used < cap and _ANNOUNCE.search(content) and len(content.split()) <= 40:
                # It said it would look something up but didn't call a tool: what it said serves as the
                # filler line; ask it to make the call.
                nudged = True
                self.history.append({"role": "assistant", "content": content})
                self.history.append({"role": "user", "content": NUDGE})
                continue
            if content.strip():
                self.history.append({"role": "assistant", "content": content})
                return content
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

    async def _llm_round(self, rid: int, pieces: asyncio.Queue, timing: dict, t_ref: float, shown: list[str],
                         schemas: list[dict] | None) -> tuple[str, list[dict]]:
        chunker = SpeechChunker()
        content: list[str] = []
        calls: list[dict] = []
        gap = bool("".join(shown).strip())   # separate this round's text from the previous round's
        async with asyncio.timeout(self.cfg.chat.timeout_s):
            async for ev in self.lm.chat_events(self.llm, list(self.history), tools=schemas, think=False,
                                                temperature=self.cfg.chat.temperature,
                                                max_tokens=self.cfg.chat.max_tokens):
                if ev["type"] == "tool_calls":
                    calls = ev["calls"]
                    timing.setdefault("tool_call_ms", round((time.perf_counter() - t_ref) * 1000))
                    continue
                delta = ev["text"]
                if not content and not delta.strip():
                    continue
                if "llm_first_token_ms" not in timing:
                    timing["llm_first_token_ms"] = round((time.perf_counter() - t_ref) * 1000)
                if gap and not content:
                    delta = " " + delta.lstrip()
                content.append(delta)
                self.send({"type": "reply_delta", "id": rid, "text": delta})
                for piece in chunker.feed(delta):
                    pieces.put_nowait(piece)
        for piece in chunker.flush():
            pieces.put_nowait(piece)
        text = "".join(content)
        if text.strip():
            shown.append(text)
        return text, calls

    def _tool_label(self, calls) -> str:
        tool = self.svc.tools.get(calls[0].name)
        label = (tool.label if tool else "Working") + "…"
        detail = tool.describe(calls[0].args) if tool else ""
        return f"{label} {detail}".strip() if detail and len(detail) < 60 else label

    async def _run_tools(self, rid: int, calls, meta: dict, timing: dict, t_ref: float) -> list[ToolResult]:
        async def one(call):
            tool = self.svc.tools.get(call.name)
            self.send({"type": "tool_start", "id": rid, "call_id": call.id, "name": call.name,
                       "label": tool.label if tool else "Tool", "detail": tool.describe(call.args) if tool else ""})
            t0 = time.perf_counter()
            result = await self.svc.tools.execute(call, self.svc.tool_ctx)
            ms = round((time.perf_counter() - t0) * 1000)
            self.send({"type": "tool_end", "id": rid, "call_id": call.id, "name": call.name, "ok": result.ok,
                       "summary": result.summary, "sources": result.sources, "ms": ms})
            meta["tools"].append({"name": call.name, "args": call.args, "ok": result.ok, "summary": result.summary,
                                  "ms": ms, **({"repaired": True} if call.repaired else {})})
            for s in result.sources:
                if s not in meta["sources"]:
                    meta["sources"].append(s)
            return result

        results = await asyncio.gather(*(one(c) for c in calls))
        timing["tools_ms"] = timing.get("tools_ms", 0) + max((t["ms"] for t in meta["tools"][-len(calls):]), default=0)
        return results

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
        counterpart = SCENARIOS[scenario]["counterpart"]
        dialogue = "\n".join(f"{'You' if m['role'] == 'user' else counterpart}: {m['content']}"
                             for m in history[2:] if m["role"] in ("user", "assistant") and m.get("content"))
        points = ""
        if SCENARIOS[scenario]["notes_query"]:
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
        messages = feedback_messages(kind, transcript, metrics, scenario, details, talking_points)
        async for delta in self.lm.chat_stream(model, messages, think=think, temperature=self.cfg.feedback.temperature,
                                               max_tokens=self.cfg.feedback.max_tokens):
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
