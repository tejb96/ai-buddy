"""One browser connection = one Session. Drives chat, presentation, and conversation-practice modes.

Wire protocol (WebSocket /ws):
  browser -> server  binary: 16 kHz mono PCM16 microphone audio (only while recording)
                     JSON:   start{mode, scenario?, details?} | rec_start | rec_stop | end | cancel | speak{text} | leave
  server -> browser  binary: 4-byte little-endian reply id + 24 kHz mono PCM16 speech
                     JSON:   models, state, session, user_text, reply_start/delta/end, caption,
                             metrics, feedback_status/delta/done/cancelled, notice, error
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import struct
import time
import wave

from fastapi import WebSocket

from .audio import PresentationChunker, analyze_speech, has_speech, pcm_seconds
from .config import Config
from .lemonade import Lemonade, LemonadeError
from .metrics import Answer, compute_metrics, words
from .models import ModelManager
from .prompts import KICKOFF, SCENARIOS, chat_system_prompt, feedback_messages, roleplay_system_prompt
from .storage import Store
from .text import SpeechChunker, clean_for_speech, strip_emoji

log = logging.getLogger(__name__)

MODES = ("chat", "presentation", "interview")


class Session:
    def __init__(self, ws: WebSocket, cfg: Config, lm: Lemonade, models: ModelManager, store: Store):
        self.ws, self.cfg, self.lm, self.models, self.store = ws, cfg, lm, models, store
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
        self.mode, self.scenario, self.details = mode, scenario, details or ""
        self.answers, self.timeline_s, self.history = [], 0.0, []
        if mode == "presentation":
            # A presentation session is created per talk, when recording starts.
            self.send({"type": "session", "mode": mode, "id": None})
            return
        self.session_id = self.store.create_session(mode, scenario, self.details)
        self.started_at = time.monotonic()
        self.send({"type": "session", "mode": mode, "id": self.session_id})
        if mode == "chat":
            self.history = [{"role": "system", "content": chat_system_prompt(self.cfg.persona)}]
            self.history_prefix = 1
            self.models.ensure_in_background("stt", "tts", "chat")
        else:
            self.history = [{"role": "system", "content": roleplay_system_prompt(scenario, self.details)},
                            {"role": "user", "content": KICKOFF}]
            self.history_prefix = 2
            self.turn_task = asyncio.create_task(self._guard(self._opening_line()))

    def _rec_start(self) -> None:
        if self.mode is None:
            return self._notice("Pick a mode first.")
        if self.mode == "presentation":
            if self.feedback_tasks:
                return self._notice("Hang on, I'm still writing feedback for your last talk.")
            self._begin_talk()
        else:
            # Barge-in: pressing talk while the buddy is speaking or thinking cancels that reply.
            if self.turn_task and not self.turn_task.done():
                self.turn_task.cancel()
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
            pcm = bytes(self.rec)
            self.turn_task = asyncio.create_task(self._guard(self._user_turn(pcm)))

    def _on_audio(self, data: bytes) -> None:
        if not self.recording:
            return
        self.rec += data
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

    # ---------- chat / interview turns ----------

    async def _opening_line(self) -> None:
        if any(self.models.state[r] != "ready" for r in ("stt", "tts", "chat")):
            self._state("loading", "Getting the conversation model ready…")
        await self.models.ensure("stt", "tts", "chat")
        await self._reply(time.perf_counter())

    async def _user_turn(self, pcm: bytes) -> None:
        analysis = analyze_speech(pcm)
        if analysis.speech_s < 0.25:
            self._notice("I didn't hear anything. Hold the button (or Space) while you talk.")
            return self._state("idle")
        self._state("transcribing")
        if any(self.models.state[r] != "ready" for r in ("stt", "tts", "chat")):
            self._state("loading", "Loading models…")
        await self.models.ensure("stt", "tts", "chat")
        self._state("transcribing")
        t_end = time.perf_counter()
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
        await self._reply(t_end)

    def _trim_history(self) -> None:
        budget = self.cfg.chat.history_word_budget
        while (sum(len(m["content"].split()) for m in self.history[self.history_prefix:]) > budget
               and len(self.history) > self.history_prefix + 2):
            del self.history[self.history_prefix:self.history_prefix + 2]

    async def _reply(self, t_ref: float) -> None:
        """Stream an LLM reply and speak it clause by clause while it is still being written."""
        self.reply_id += 1
        rid = self.reply_id
        self._state("thinking")
        self.send({"type": "reply_start", "id": rid})
        timing: dict = {}
        full: list[str] = []
        spoken: list[str] = []
        pieces: asyncio.Queue = asyncio.Queue()
        chunker = SpeechChunker()

        async def speaker():
            while (piece := await pieces.get()) is not None:
                spoken.append(piece)
                await self._tts(piece, rid, timing, t_ref)

        speaker_task = asyncio.create_task(speaker())
        interrupted = True
        try:
            async for delta in self.lm.chat_stream(self.models.names["chat"], list(self.history), think=False,
                                                   temperature=self.cfg.chat.temperature,
                                                   max_tokens=self.cfg.chat.max_tokens):
                if not full:
                    timing["llm_first_token_ms"] = round((time.perf_counter() - t_ref) * 1000)
                full.append(delta)
                self.send({"type": "reply_delta", "id": rid, "text": delta})
                for piece in chunker.feed(delta):
                    pieces.put_nowait(piece)
            for piece in chunker.flush():
                pieces.put_nowait(piece)
            pieces.put_nowait(None)
            await speaker_task
            interrupted = False
        finally:
            speaker_task.cancel()
            text = strip_emoji("".join(full) if not interrupted else " ".join(spoken))
            if text:
                self.history.append({"role": "assistant", "content": text})
                if self.session_id:
                    self.store.add_turn(self.session_id, "assistant", text, interrupted=interrupted)
            timing["total_ms"] = round((time.perf_counter() - t_ref) * 1000)
            self.send({"type": "reply_end", "id": rid, "text": text, "interrupted": interrupted, "timing": timing})
            if not interrupted:
                self._state("idle")

    async def _tts(self, text: str, rid: int, timing: dict, t_ref: float) -> None:
        clean = clean_for_speech(text)
        if not clean:
            return
        async for pcm in self.lm.tts_stream(clean, self.models.names["tts"], self.cfg.voice.tts_voice,
                                            self.cfg.voice.tts_speed):
            if "first_audio_ms" not in timing:
                timing["first_audio_ms"] = round((time.perf_counter() - t_ref) * 1000)
            self.send(struct.pack("<I", rid) + pcm)

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
        self.session_id = self.store.create_session("presentation")
        self.started_at = time.monotonic()
        self.send({"type": "session", "mode": "presentation", "id": self.session_id})
        self.chunker = PresentationChunker(target_s=self.cfg.practice.presentation_chunk_s)
        self.chunk_q = asyncio.Queue()
        self.captions = []
        self.talk_pcm = bytearray()
        self.chunk_task = asyncio.create_task(self._transcribe_chunks(self.chunk_q))
        # The feedback LLM loads while you talk (it evicts the chat LLM; Whisper stays loaded).
        self.models.ensure_in_background("feedback")

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
        await self._feedback(sid, "presentation", transcript, metrics, duration_s=pcm_seconds(pcm))

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
                             for m in history[2:])
        await self._feedback(sid, "interview", dialogue, metrics, scenario=scenario, details=details,
                             duration_s=time.monotonic() - self.started_at)
        # The next thing is usually another conversation, so bring the chat model back now.
        self.models.ensure_in_background("stt", "tts", "chat")

    # ---------- shared feedback ----------

    def _metrics(self, answers: list[Answer]) -> dict:
        p = self.cfg.practice
        return compute_metrics(answers, long_pause_s=p.long_pause_s, wpm_low=p.target_wpm_low,
                               wpm_high=p.target_wpm_high)

    async def _feedback(self, sid: str, kind: str, transcript: str, metrics: dict, *, scenario: str | None = None,
                        details: str = "", duration_s: float | None = None) -> None:
        model = self.models.names["feedback"]
        think = self.cfg.feedback.think

        def state(name: str, detail: str = "") -> None:
            # Feedback keeps running if you switch modes; only drive the status line while you stay.
            if self.mode == kind:
                self._state(name, detail)

        try:
            if self.models.state["feedback"] != "ready":
                state("analyzing", "Loading the feedback model (about 20 seconds)…")
            await self.models.ensure("feedback")
        except LemonadeError as e:
            model, think = self.models.names["chat"], False
            self._notice(f"Couldn't load the feedback model ({e}). Using {model} instead.")
            await self.models.ensure("chat")
        state("analyzing", "Thinking it over…" if think else "Writing your feedback…")
        self.send({"type": "feedback_status", "text": "started", "model": model})
        parts: list[str] = []
        async for delta in self.lm.chat_stream(model, feedback_messages(kind, transcript, metrics, scenario, details),
                                               think=think, temperature=self.cfg.feedback.temperature,
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
