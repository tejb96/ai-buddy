"""Keep the right models loaded in Lemonade and report their state to the UI.

One LLM does everything: conversation, tool calls and coaching, so nothing is swapped. Whisper
(speech-to-text) runs in its own FLM process on the NPU (or, as an option, Whisper Large v3 Turbo on
whisper.cpp, on the CPU by default) and Kokoro (TTS) on the CPU. The embedding model for note search runs on
llama.cpp's CPU backend and is loaded the first time the notes need it. Models are loaded explicitly so our
options apply (an implicit load would use Lemonade's defaults: a 64k context, and whatever llama.cpp or
whisper.cpp backend is the default).

One NPU job at a time: when Whisper and the LLM (two FLM processes) run at the same moment, both results come
back corrupted (an empty reply after a minute, "What? What? What?" from Whisper; FLM 1.0.6). Loaded side by
side they're fine. So every FLM Whisper call and every LLM stream goes through one lock, and LLM streams are
always closed when abandoned (FLM stops generating as soon as the client disconnects).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable

import httpx

from .audio import pcm_to_wav
from .config import Config
from .lemonade import Lemonade, LemonadeError

log = logging.getLogger(__name__)

ROLES = ("stt", "llm", "tts")      # what a conversation needs; "embed" (notes) is loaded on demand
BACKEND_LABELS = {"flm": "FastFlowLM", "llamacpp": "llama.cpp", "whispercpp": "whisper.cpp", "kokoro": "Kokoro"}


def on_npu(model: str) -> bool:
    """FLM models (Lemonade names them "...-FLM") run on the NPU; the others on the CPU (or a GPU backend)."""
    return model.endswith("-FLM")


class ModelManager:
    def __init__(self, lm: Lemonade, cfg: Config, llm: str | None = None, voice: str | None = None,
                 stt: str | None = None):
        self.lm = lm
        self.cfg = cfg
        self.names = {"stt": stt or cfg.models.stt, "llm": llm or cfg.models.llm, "tts": cfg.models.tts}
        if cfg.models.embed:
            self.names["embed"] = cfg.models.embed
        self.voice = voice or cfg.voice.tts_voice
        self.state = {r: "unknown" for r in self.names}
        self.error: str | None = None
        self.warm = False
        self.closed = False             # the app is exiting: load nothing more
        self._lock = asyncio.Lock()
        self._npu = asyncio.Lock()      # one FLM job (Whisper or LLM) at a time
        self._listeners: set[Callable[[dict], None]] = set()
        # Lemonade backends our models need that must be updated (or installed) before they load, e.g. FLM after
        # a Lemonade upgrade: its FLM models are then "not found" until `lemonade backends install flm:npu`.
        self.backend_updates: list[dict] = []
        self.updating: dict | None = None     # the backend being installed: {"spec", "label", "percent"}
        self.backend_error: str | None = None

    # ---------- status ----------

    def snapshot(self) -> dict:
        return {"models": {r: {"name": self.names[r], "state": self.state[r], "device": self.device(r)}
                           for r in self.names},
                "error": self.error, "warm": self.warm,
                "backends": {"needed": self.backend_updates, "updating": self.updating, "error": self.backend_error}}

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
            for r in self.names:
                self.state[r] = "error"
            self._notify()
            raise
        self.error = None
        for r in self.names:
            if self.state[r] != "loading":
                self.state[r] = "ready" if self.names[r] in loaded else "standby"
        self._notify()
        return loaded

    # ---------- loading ----------

    async def ensure(self, *roles: str) -> None:
        """Make sure the models for `roles` are loaded (cheap when they already are)."""
        if self.closed:
            raise LemonadeError("The app is shutting down.")
        roles = roles or ROLES
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
                    await self.lm.load(self.names[role], self._ctx_size(role), **self._options(role))
                except LemonadeError as e:
                    self.error = str(e)
                    self._set(role, "error")
                    await self.check_backends()     # often the reason: a backend Lemonade wants updated
                    raise
                self.error = None
                self.state[role] = "ready"
                loaded = await self.refresh()

    def _ctx_size(self, role: str) -> int | None:
        """Context to load with: the LLM's from config; the embedding model's small (Lemonade's 64k default made
        the 0.6B embedding model take 7.7 GB of RAM; a batch of 16 note chunks needs ~5k tokens)."""
        return {"llm": self.cfg.models.llm_ctx_size, "embed": self.cfg.models.embed_ctx_size}.get(role)

    def _options(self, role: str) -> dict:
        m = self.cfg.models
        if role == "embed":
            return {"llamacpp_backend": m.embed_backend, "llamacpp_args": f"--threads {m.embed_threads}"}
        if role == "stt" and not on_npu(self.names["stt"]):
            return {"whispercpp_backend": m.stt_backend}
        return {}

    def device(self, role: str) -> str:
        """Where a role's model runs, for the UI: "NPU", "CPU" or "GPU"."""
        name = self.names.get(role, "")
        if on_npu(name):
            return "NPU"
        backend = {"stt": self.cfg.models.stt_backend, "embed": self.cfg.models.embed_backend}.get(role, "cpu")
        return "CPU" if backend == "cpu" else "GPU"

    # ---------- Lemonade backends ----------

    def backend_of(self, role: str) -> tuple[str, str]:
        """The Lemonade recipe and backend a role's model runs on."""
        if on_npu(self.names[role]):
            return "flm", "npu"
        m = self.cfg.models
        return {"stt": ("whispercpp", m.stt_backend), "embed": ("llamacpp", m.embed_backend)}.get(role, ("kokoro", "cpu"))

    async def check_backends(self) -> list[dict]:
        """Find the backends our models need that Lemonade says must be updated (or, for a conversation's models,
        installed) first. Keeps the last answer when Lemonade can't be reached."""
        try:
            info = await self.lm.backends()
        except (LemonadeError, httpx.HTTPError) as e:
            log.debug("could not read Lemonade's backends: %s", e)
            return self.backend_updates
        needed: dict[str, dict] = {}
        for role in self.names:
            recipe, backend = self.backend_of(role)
            b = info.get(recipe, {}).get(backend) or {}
            state = b.get("state")
            if state != "update_required" and not (state == "installable" and role in ROLES):
                continue
            spec = f"{recipe}:{backend}"
            if spec not in needed:
                url = b.get("release_url") or ""
                needed[spec] = {"spec": spec, "recipe": recipe, "backend": backend,
                                "label": f"{BACKEND_LABELS.get(recipe, recipe)} ({backend.upper()})",
                                "action": "update" if state == "update_required" else "install",
                                "version": b.get("version", "") if state == "update_required" else "",
                                "latest": url.rsplit("/tag/", 1)[1] if "/tag/" in url else "", "roles": []}
            needed[spec]["roles"].append(role)
        updates = list(needed.values())
        if updates != self.backend_updates:
            if updates:
                log.warning("Lemonade backends to update before the models load: %s",
                            ", ".join(u["spec"] for u in updates))
            self.backend_updates = updates
            self._notify()
        return updates

    async def update_backends(self) -> None:
        """Install the backends check_backends() found (what `lemonade backends install` does), one at a time,
        reporting progress to the UI. Model loads wait until it's done."""
        if self.updating or not self.backend_updates:
            return
        self.backend_error = None
        async with self._lock:
            try:
                for u in list(self.backend_updates):
                    self.updating = {"spec": u["spec"], "label": u["label"], "percent": 0}
                    self._notify()
                    log.info("installing Lemonade backend %s", u["spec"])
                    async for ev in self.lm.install_backend(u["recipe"], u["backend"]):
                        pct = ev.get("percent")
                        if isinstance(pct, (int, float)) and int(pct) != self.updating["percent"]:
                            self.updating = {**self.updating, "percent": int(pct)}   # snapshots may be queued
                            self._notify()
                    log.info("installed Lemonade backend %s", u["spec"])
            except (LemonadeError, httpx.HTTPError) as e:
                self.backend_error = str(e) or e.__class__.__name__
                log.warning("backend update failed: %s", self.backend_error)
            finally:
                self.updating = None
                self.error = None
        await self.check_backends()
        self._notify()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeddings for note search (loads the embedding model on the CPU first if needed)."""
        await self.ensure("embed")
        return await self.lm.embed(texts, self.names["embed"])

    async def prepare_live(self) -> None:
        """Load and warm up everything a conversation needs."""
        await self.ensure("stt", "tts", "llm")
        await self.warm_up()
        self.warm = True
        self._notify()

    async def warm_up(self) -> None:
        try:
            async with self._npu:
                await self.lm.chat(self.names["llm"], [{"role": "user", "content": "Hi"}], think=False, max_tokens=1)
            async for _ in self.lm.tts_stream("Hi.", self.names["tts"], self.voice):
                pass
        except LemonadeError as e:
            log.warning("warm-up failed: %s", e)

    def switch_llm(self, name: str) -> asyncio.Task:
        """Use a different LLM from now on (loading it evicts the old one)."""
        return self._switch("llm", name)

    def switch_stt(self, name: str) -> asyncio.Task:
        """Use a different speech-to-text model from now on (Lemonade keeps one loaded, so the old one goes)."""
        return self._switch("stt", name)

    def _switch(self, role: str, name: str) -> asyncio.Task:
        self.names[role] = name
        self.state[role] = "standby"
        self._notify()
        return self.ensure_in_background(role)

    async def unload_all(self) -> None:
        """Exit: unload every model from Lemonade (not just ours), and load nothing more."""
        self.closed = True
        await self.lm.unload_all()
        for role in self.names:
            self.state[role] = "standby"
        self.warm = False
        self._notify()

    async def llm_events(self, messages: list[dict], **kw) -> AsyncIterator[dict]:
        """The chat LLM's streamed events (see Lemonade.chat_events), holding the NPU while it runs.
        Use it with contextlib.aclosing() so an abandoned stream releases the NPU at once."""
        async with self._npu:
            async with contextlib.aclosing(self.lm.chat_events(self.names["llm"], messages, **kw)) as events:
                async for ev in events:
                    yield ev

    async def llm_text(self, messages: list[dict], **kw) -> AsyncIterator[str]:
        async with contextlib.aclosing(self.llm_events(messages, **kw)) as events:
            async for ev in events:
                if ev["type"] == "content":
                    yield ev["text"]

    async def transcribe(self, pcm: bytes) -> str:
        """Speech-to-text with one retry, holding the NPU when Whisper runs there. We have seen rare NPU errors
        ("qds_device::wait() unexpected command state") that a retry, or reloading Whisper, clears."""
        wav = pcm_to_wav(pcm)
        async with self._npu if on_npu(self.names["stt"]) else contextlib.nullcontext():
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
            await self.lm.load(self.names["stt"], **self._options("stt"))
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
