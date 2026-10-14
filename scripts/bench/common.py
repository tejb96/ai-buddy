"""Shared helpers for the Phase 0 benchmarks (speech synthesis, transcription, WER, timing)."""
from __future__ import annotations

import io
import json
import re
import subprocess
import time
import wave
from pathlib import Path

import httpx
import numpy as np

BASE = "http://localhost:13305/v1"
HERE = Path(__file__).parent
CACHE = HERE / "audio_cache"
CACHE.mkdir(exist_ok=True)
http = httpx.Client(base_url=BASE, timeout=httpx.Timeout(10.0, read=600))


def synth(text: str, voice: str = "am_michael", speed: float = 1.0) -> bytes:
    """Kokoro speech for `text` as 16 kHz mono PCM16 (cached on disk)."""
    key = re.sub(r"[^a-z0-9]+", "_", f"{voice}_{speed}_{text.lower()}")[:120] + f"_{abs(hash(text)) % 10**8}"
    f = CACHE / f"{key}.pcm"
    if f.exists():
        return f.read_bytes()
    r = http.post("/audio/speech", json={"model": "kokoro-v1", "input": text, "voice": voice,
                                          "speed": speed, "response_format": "wav"})
    r.raise_for_status()
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ar", "16000", "-ac", "1",
                          "-f", "s16le", "pipe:1"], input=r.content, capture_output=True, check=True).stdout
    f.write_bytes(pcm)
    return pcm


def silence(seconds: float, amp: float = 20.0, seed: int = 1) -> bytes:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(seconds * 16000))).astype("<i2").tobytes()


def add_noise(pcm: bytes, snr_db: float, seed: int = 0) -> bytes:
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    p = np.mean(x ** 2) + 1e-9
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(len(x)).astype(np.float32) * np.sqrt(p / 10 ** (snr_db / 10))
    return np.clip(x + n, -32768, 32767).astype("<i2").tobytes()


def wav(pcm: bytes, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def seconds(pcm: bytes) -> float:
    return len(pcm) / 32000


def transcribe(pcm: bytes, model: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    r = http.post("/audio/transcriptions", data={"model": model, "language": "en", "response_format": "json"},
                  files={"file": ("speech.wav", wav(pcm), "audio/wav")})
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        raise RuntimeError(f"{model}: {r.status_code} {r.text[:300]}")
    return r.json().get("text", "").strip(), dt


def load(model: str, **opts) -> float:
    t0 = time.perf_counter()
    r = http.post("/load", json={"model_name": model, **opts}, timeout=600)
    if r.status_code != 200:
        raise RuntimeError(f"load {model}: {r.status_code} {r.text[:300]}")
    return time.perf_counter() - t0


def unload(model: str | None = None) -> None:
    http.post("/unload", json={"model_name": model} if model else {}, timeout=120)


def loaded() -> dict:
    return {m["model_name"]: m for m in http.get("/health").json().get("all_models_loaded", [])}


# ---------- WER ----------

_NUM = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15", "twenty": "20",
        "thirty": "30", "forty": "40", "fifty": "50", "hundred": "100"}


def norm(text: str) -> list[str]:
    t = text.lower().replace("’", "'")
    t = re.sub(r"(\d),(\d)", r"\1\2", t)
    t = re.sub(r"[^a-z0-9' ]+", " ", t.replace("-", " "))
    out = []
    for w in t.split():
        w = w.strip("'")
        if not w or w in {"um", "uh", "er", "ah", "hmm", "mm", "umm"}:
            continue  # hesitations are scored separately; Whisper often drops them
        out.append(_NUM.get(w, w))
    return out


def edit_distance(a: list[str], b: list[str]) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
            prev, d[j] = d[j], cur
    return d[len(b)]


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = norm(ref), norm(hyp)
    return edit_distance(r, h), len(r)


def dump(name: str, obj) -> None:
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{name}.json").write_text(json.dumps(obj, indent=2))
