"""Async client for Lemonade's OpenAI-compatible API (plus its /load and /health extensions)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

TTS_SAMPLE_RATE = 24000  # kokoro-v1 streams 16-bit mono PCM at 24 kHz


class LemonadeError(RuntimeError):
    pass


class Lemonade:
    def __init__(self, base_url: str, load_timeout_s: float = 180):
        self.base_url = base_url.rstrip("/")
        self.load_timeout_s = load_timeout_s
        # Generous read timeout: the first request to an unloaded model waits for the load.
        self._http = httpx.AsyncClient(base_url=self.base_url,
                                       timeout=httpx.Timeout(10.0, read=load_timeout_s))

    async def close(self) -> None:
        await self._http.aclose()

    # ---------- server / model management ----------

    async def health(self) -> dict:
        try:
            r = await self._http.get("/health", timeout=5)
        except httpx.HTTPError as e:
            raise LemonadeError(f"Lemonade server not reachable at {self.base_url} ({e.__class__.__name__})") from e
        r.raise_for_status()
        return r.json()

    async def loaded_models(self) -> dict[str, dict]:
        return {m["model_name"]: m for m in (await self.health()).get("all_models_loaded", [])}

    async def models(self) -> dict[str, dict]:
        """The models Lemonade knows (downloaded or not), with their labels ("audio", "embeddings"...)."""
        r = await self._http.get("/models", params={"show_all": "true"}, timeout=10)
        r.raise_for_status()
        return {m["id"]: m for m in r.json().get("data", [])}

    async def load(self, model: str, ctx_size: int | None = None, **options) -> None:
        """Load a model; `options` are Lemonade recipe options for this load only (e.g. llamacpp_backend)."""
        body: dict = {"model_name": model, **options}
        if ctx_size:
            body["ctx_size"] = ctx_size
        r = await self._http.post("/load", json=body, timeout=self.load_timeout_s)
        if r.status_code != 200 or r.json().get("status") != "success":
            raise LemonadeError(f"Loading {model} failed: {r.text[:300]}")

    # ---------- speech-to-text ----------

    async def transcribe(self, wav: bytes, model: str, language: str = "en", timeout: float = 30) -> str:
        r = await self._http.post(
            "/audio/transcriptions",
            data={"model": model, "language": language, "response_format": "json"},
            files={"file": ("speech.wav", wav, "audio/wav")},
            timeout=httpx.Timeout(10.0, read=timeout),
        )
        if r.status_code != 200:
            raise LemonadeError(f"Transcription failed ({r.status_code}): {r.text[:300]}")
        return " ".join(r.json().get("text", "").split())

    async def unload(self, model: str) -> None:
        await self._http.post("/unload", json={"model_name": model}, timeout=60)

    async def unload_all(self) -> None:
        """Unload every loaded model (an empty request body means all of them)."""
        r = await self._http.post("/unload", json={}, timeout=60)
        if r.status_code != 200:
            raise LemonadeError(f"Unloading the models failed ({r.status_code}): {r.text[:300]}")

    # ---------- LLM ----------

    async def chat_events(self, model: str, messages: list[dict], *, tools: list[dict] | None = None,
                          think: bool = False, temperature: float = 0.7,
                          max_tokens: int = 300) -> AsyncIterator[dict]:
        """Stream a completion as events: {"type": "content", "text"} while it is written, then
        {"type": "tool_calls", "calls": [{"id", "name", "arguments"}]} if the model called tools.

        `think` is always sent: FLM keeps the last value per process.
        """
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
        calls: dict[int, dict] = {}
        async with self._http.stream("POST", "/chat/completions", json=body) as r:
            if r.status_code != 200:
                raise LemonadeError(f"Chat request failed ({r.status_code}): {(await r.aread())[:300]!r}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue  # blank lines and ": ping" keep-alives
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise LemonadeError(f"Chat stream error: {chunk['error']}")
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    for tc in delta.get("tool_calls") or []:
                        # FLM sends each call whole; OpenAI-style servers stream the arguments in pieces.
                        call = calls.setdefault(tc.get("index", len(calls)), {"id": "", "name": "", "arguments": ""})
                        fn = tc.get("function") or {}
                        call["id"] = tc.get("id") or call["id"]
                        call["name"] += fn.get("name") or ""
                        call["arguments"] += fn.get("arguments") or ""
                    piece = delta.get("content")
                    if piece:
                        yield {"type": "content", "text": piece}
        if calls:
            yield {"type": "tool_calls", "calls": [calls[i] for i in sorted(calls)]}

    async def chat_stream(self, model: str, messages: list[dict], **kw) -> AsyncIterator[str]:
        """Yield content deltas only (no tools)."""
        async for ev in self.chat_events(model, messages, **kw):
            if ev["type"] == "content":
                yield ev["text"]

    async def chat(self, model: str, messages: list[dict], **kw) -> str:
        return "".join([piece async for piece in self.chat_stream(model, messages, **kw)])

    # ---------- embeddings ----------

    async def embed(self, texts: list[str], model: str, timeout: float = 30) -> list[list[float]]:
        r = await self._http.post("/embeddings", json={"model": model, "input": texts},
                                  timeout=httpx.Timeout(10.0, read=timeout))
        if r.status_code != 200:
            raise LemonadeError(f"Embedding request failed ({r.status_code}): {r.text[:300]}")
        data = sorted(r.json().get("data", []), key=lambda d: d.get("index", 0))
        if len(data) != len(texts):
            raise LemonadeError(f"Embedding request returned {len(data)} vectors for {len(texts)} inputs")
        return [d["embedding"] for d in data]

    # ---------- text-to-speech ----------

    async def tts_stream(self, text: str, model: str, voice: str, speed: float = 1.0,
                         chunk_bytes: int = 4800) -> AsyncIterator[bytes]:
        """Yield 16-bit mono PCM at 24 kHz, in even-length chunks (100 ms by default)."""
        body = {"model": model, "input": text, "voice": voice, "speed": speed,
                "response_format": "pcm", "stream_format": "audio"}
        pending = b""
        async with self._http.stream("POST", "/audio/speech", json=body) as r:
            if r.status_code != 200:
                raise LemonadeError(f"TTS request failed ({r.status_code}): {(await r.aread())[:300]!r}")
            async for data in r.aiter_bytes():
                pending += data
                while len(pending) >= chunk_bytes:
                    yield pending[:chunk_bytes]
                    pending = pending[chunk_bytes:]
        if len(pending) >= 2:
            yield pending[: len(pending) // 2 * 2]
