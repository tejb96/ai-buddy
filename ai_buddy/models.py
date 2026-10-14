"""Keep the right models loaded in Lemonade and report their state to the UI.

FastFlowLM in Lemonade has a single LLM slot: loading the feedback LLM evicts the chat LLM
(and vice versa). Whisper (transcription) and Kokoro (TTS) live in their own slots and stay put.
Models are loaded explicitly so our context sizes apply (an implicit load would use the 64k default).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import httpx

from .audio import pcm_to_wav
from .config import Config
from .lemonade import Lemonade, LemonadeError

log = logging.getLogger(__name__)

ROLES = ("stt", "chat", "feedback", "tts")


class ModelManager:
    def __init__(self, lm: Lemonade, cfg: Config):
        self.lm = lm
        self.cfg = cfg
        self.names = {r: getattr(cfg.models, r) for r in ROLES}
        self.ctx = {"chat": cfg.models.chat_ctx_size, "feedback": cfg.models.feedback_ctx_size}
        self.state = {r: "unknown" for r in ROLES}
        self.error: str | None = None
        self.warm = False
        self._lock = asyncio.Lock()
        self._stt_lock = asyncio.Lock()
        self._listeners: set[Callable[[dict], None]] = set()

    # ---------- status ----------

    def snapshot(self) -> dict:
        return {"models": {r: {"name": self.names[r], "state": self.state[r]} for r in ROLES},
                "error": self.error, "warm": self.warm}

    def subscribe(self, fn: Callable[[dict], None]) -> None:
        self._listeners.add(fn)

    def unsubscribe(self, fn: Callable[[dict], None]) -> None:
        self._listeners.discard(fn)

    def _set(self, role: str, state: str) -> None:
        if self.state[role] != state:
            self.state[role] = state
            self._notify()

    def _notify(self) -> None:
        snap = self.snapshot()
        for fn in list(self._listeners):
            fn(snap)

    async def refresh(self) -> dict[str, dict]:
        """Sync our state with what Lemonade actually has loaded."""
        try:
            loaded = await self.lm.loaded_models()
        except LemonadeError as e:
            self.error = str(e)
            for r in ROLES:
                self.state[r] = "error"
            self._notify()
            raise
        self.error = None
        for r in ROLES:
            if self.state[r] != "loading":
                self.state[r] = "ready" if self.names[r] in loaded else "standby"
        self._notify()
        return loaded

    # ---------- loading ----------

    async def ensure(self, *roles: str) -> None:
        """Make sure the models for `roles` are loaded (cheap when they already are)."""
        loaded = await self.refresh()
        if all(self.names[r] in loaded for r in roles):
            return
        async with self._lock:
            loaded = await self.refresh()
            for role in roles:
                if self.names[role] in loaded:
                    continue
                self._set(role, "loading")
                try:
                    await self.lm.load(self.names[role], self.ctx.get(role))
                except LemonadeError as e:
                    self.error = str(e)
                    self._set(role, "error")
                    raise
                self.error = None
                self.state[role] = "ready"
                loaded = await self.refresh()

    async def prepare_live(self) -> None:
        """Load and warm up everything a live conversation needs."""
        await self.ensure("stt", "tts", "chat")
        await self.warm_up()
        self.warm = True
        self._notify()

    async def warm_up(self) -> None:
        try:
            await self.lm.chat(self.names["chat"], [{"role": "user", "content": "Hi"}], think=False, max_tokens=1)
            async for _ in self.lm.tts_stream("Hi.", self.names["tts"], self.cfg.voice.tts_voice):
                pass
        except LemonadeError as e:
            log.warning("warm-up failed: %s", e)

    async def transcribe(self, pcm: bytes) -> str:
        """Speech-to-text with one retry. Calls are serialized: FLM runs one ASR job at a time anyway,
        and we have seen rare NPU errors ("qds_device::wait() unexpected command state") that a retry,
        or reloading Whisper, clears."""
        wav = pcm_to_wav(pcm)
        async with self._stt_lock:
            for attempt in (1, 2):
                try:
                    return await self.lm.transcribe(wav, self.names["stt"], self.cfg.voice.language)
                except (LemonadeError, httpx.HTTPError) as e:
                    log.warning("transcription attempt %d failed: %s", attempt, e)
                    if attempt == 2:
                        await self._reload_stt()
                        raise LemonadeError(f"Speech recognition failed twice ({e}); reloaded Whisper, "
                                            "please try again.") from e
                    await asyncio.sleep(0.5)
        return ""

    async def _reload_stt(self) -> None:
        try:
            self._set("stt", "loading")
            await self.lm.unload(self.names["stt"])
            await self.lm.load(self.names["stt"])
            self.state["stt"] = "ready"
            self._notify()
        except (LemonadeError, httpx.HTTPError) as e:
            log.warning("reloading Whisper failed: %s", e)
            self._set("stt", "error")

    def ensure_in_background(self, *roles: str) -> asyncio.Task:
        async def run():
            try:
                await self.ensure(*roles)
            except LemonadeError as e:
                log.warning("background load of %s failed: %s", roles, e)
        return asyncio.create_task(run())
