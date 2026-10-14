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

    async def load(self, model: str, ctx_size: int | None = None) -> None:
        body: dict = {"model_name": model}
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
        return r.json().get("text", "").strip()

    async def unload(self, model: str) -> None:
        await self._http.post("/unload", json={"model_name": model}, timeout=60)

    # ---------- LLM ----------

    async def chat_stream(self, model: str, messages: list[dict], *, think: bool = False,
                          temperature: float = 0.7, max_tokens: int = 300) -> AsyncIterator[str]:
        """Yield content deltas. `think` is always sent: FLM keeps the last value per process."""
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
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
                    piece = choice.get("delta", {}).get("content")
                    if piece:
                        yield piece

    async def chat(self, model: str, messages: list[dict], **kw) -> str:
        return "".join([piece async for piece in self.chat_stream(model, messages, **kw)])

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
