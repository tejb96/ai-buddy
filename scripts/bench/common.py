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

_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}


def numbers_to_digits(tokens: list[str]) -> list[str]:
    """'two hundred and forty' -> '240', 'thirty seven' -> '37', so '18%' and 'eighteen percent' compare equal."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        j, total, cur, seen = i, 0, 0, False
        while j < len(tokens):
            w = tokens[j]
            if w in _UNITS and not (seen and cur % 10 and cur >= 20) and not (seen and 0 < cur % 100 < 20 and cur < 100):
                cur += _UNITS[w]
            elif w in _TENS and cur % 100 == 0:
                cur += _TENS[w]
            elif w == "hundred" and cur % 100 == cur:
                cur = max(cur, 1) * 100          # "a hundred": the "a" stays a word of its own
            elif w == "thousand" and seen:
                total, cur = total + max(cur, 1) * 1000, 0
            elif (w == "and" and seen and tokens[j - 1] in ("hundred", "thousand") and j + 1 < len(tokens)
                  and (tokens[j + 1] in _UNITS or tokens[j + 1] in _TENS)):
                pass
            else:
                break
            seen = seen or w != "and"
            j += 1
        if seen:
            out.append(str(total + cur))
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return out


def norm(text: str) -> list[str]:
    t = text.lower().replace("’", "'").replace("%", " percent")
    t = re.sub(r"(\d),(\d)", r"\1\2", t)
    t = re.sub(r"[^a-z0-9' ]+", " ", t.replace("-", " "))
    out = []
    for w in t.split():
        w = w.strip("'")
        if not w or w in {"um", "uh", "er", "ah", "hmm", "mm", "umm"}:
            continue  # hesitations are scored separately; Whisper often drops them
        out.append(_TITLES.get(w, w))
    return numbers_to_digits(out)


_TITLES = {"mr": "mister", "mrs": "missus", "dr": "doctor", "st": "saint"}   # LibriSpeech spells these out


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


# ---------- temperatures ----------

def _hwmon(name: str) -> Path | None:
    for h in Path("/sys/class/hwmon").glob("hwmon*"):
        if (h / "name").exists() and (h / "name").read_text().strip() == name:
            return h
    return None


def sensors() -> dict:
    """CPU (Tctl) and iGPU edge temperature, APU package power (amdgpu PPT), fan RPM, NPU utilization."""
    out = {}
    for key, chip, f, scale in (("tctl_c", "k10temp", "temp1_input", 1000), ("edge_c", "amdgpu", "temp1_input", 1000),
                                ("ppt_w", "amdgpu", "power1_input", 1e6), ("fan_rpm", "thinkpad", "fan1_input", 1)):
        h = _hwmon(chip)
        try:
            out[key] = round(int((h / f).read_text()) / scale, 1) if h else None
        except (OSError, ValueError):
            out[key] = None
    try:
        out["npu_pct"] = http.get("/system-stats", timeout=2).json().get("npu_percent")
    except (httpx.HTTPError, ValueError):
        out["npu_pct"] = None
    return out


class ThermalLog:
    """Samples `sensors()` every `every_s` seconds in a background thread while a run is in progress."""

    def __init__(self, every_s: float = 10):
        import threading
        self.every_s, self.samples = every_s, []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        t0 = time.monotonic()
        while not self._stop.is_set():
            self.samples.append({"t_s": round(time.monotonic() - t0), **sensors()})
            self._stop.wait(self.every_s)

    def __enter__(self) -> "ThermalLog":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def summary(self) -> dict:
        out = {"samples": len(self.samples)}
        for key in ("tctl_c", "edge_c", "ppt_w", "fan_rpm", "npu_pct"):
            vals = [s[key] for s in self.samples if s.get(key) is not None]
            if vals:
                out[key] = {"min": min(vals), "avg": round(sum(vals) / len(vals), 1), "max": max(vals)}
        return out
