"""Whisper-Small vs Large-v3-Turbo (iGPU, filler prompt) under clean, babble-noise and reverberant conditions.

Usage: python stt_robust.py <model> [<model> ...]
"""
import json
import statistics
import sys
from pathlib import Path
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo root
from ai_buddy.metrics import count_fillers  # noqa: E402

from common import dump, http, load, silence, synth, wav, wer  # noqa: E402
from stt_accuracy import LS, PROMPTS, VOICES  # noqa: E402
from stt_fillers import TEXTS as DISFLUENT  # noqa: E402

PROMPT = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."
BABBLE_TEXTS = [
    "The quarterly numbers came in higher than we expected, so the team is pretty happy about it.",
    "Did you see the game last night? It went into overtime and the crowd was going wild.",
    "I need to pick up groceries after work, maybe some pasta and a bag of salad.",
    "We should book the meeting room for Thursday, the big one with the projector.",
]


def f32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32)


def i16(x: np.ndarray) -> bytes:
    return np.clip(x, -32768, 32767).astype("<i2").tobytes()


_babble = None


def babble(n: int, seed: int) -> np.ndarray:
    """Background chatter: several other voices talking at once, looped to length n."""
    global _babble
    if _babble is None:
        tracks = [f32(synth(t, v)) for t, v in zip(BABBLE_TEXTS, ["af_sarah", "am_eric", "bf_isabella", "am_liam"])]
        L = max(len(t) for t in tracks)
        _babble = sum(np.pad(t, (0, L - len(t))) for t in tracks)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, len(_babble)))
    return np.resize(np.roll(_babble, -start), n)


def add_babble(pcm: bytes, snr_db: float, seed: int) -> bytes:
    x = f32(pcm)
    b = babble(len(x), seed)
    ps, pb = np.mean(x ** 2) + 1e-9, np.mean(b ** 2) + 1e-9
    return i16(x + b * np.sqrt(ps / pb / 10 ** (snr_db / 10)))


def add_reverb(pcm: bytes, rt60: float, seed: int) -> bytes:
    """Convolve with a synthetic room impulse response (direct path + exponentially decaying tail)."""
    rng = np.random.default_rng(seed)
    n = int(rt60 * 16000)
    t = np.arange(n) / 16000
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60) * 0.35
    rir[:80] = 0
    rir[0] = 1.0
    x = f32(pcm)
    y = np.convolve(x, rir)[: len(x)]
    return i16(y * (np.abs(x).max() / (np.abs(y).max() + 1e-9)))


def clips():
    out = []
    for i, text in enumerate(DISFLUENT):
        for k in range(2):
            out.append(("disfluent", text, silence(0.25) + synth(text, VOICES[(i + k) % 4]) + silence(0.35)))
    for i, text in enumerate(PROMPTS):
        out.append(("prompt", text, silence(0.25) + synth(text, VOICES[i % len(VOICES)]) + silence(0.35)))
    for m in json.loads((LS / "manifest.json").read_text()):
        if m["seconds"] <= 14:
            out.append(("librispeech", m["text"], (LS / m["file"]).read_bytes()))
    return out


def tx(pcm: bytes, model: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    r = http.post("/audio/transcriptions", data={"model": model, "language": "en", "response_format": "json",
                                                 "prompt": PROMPT}, files={"file": ("a.wav", wav(pcm), "audio/wav")})
    r.raise_for_status()
    return " ".join(r.json()["text"].split()), time.perf_counter() - t0


if __name__ == "__main__":
    data = clips()
    conditions = {"clean": lambda p, i: p, "babble10dB": lambda p, i: add_babble(p, 10, i),
                  "reverb0.5s": lambda p, i: add_reverb(p, 0.5, i)}
    table = {}
    for model in sys.argv[1:]:
        load(model)
        tx(silence(1.0), model)
        for cname, fn in conditions.items():
            agg = {}
            kept = total = 0
            times = []
            rows = []
            for i, (kind, ref, pcm) in enumerate(data):
                hyp, dt = tx(fn(pcm, i), model)
                times.append(dt)
                e, n = wer(ref, hyp)
                a = agg.setdefault(kind, [0, 0])
                a[0] += e
                a[1] += n
                if kind == "disfluent":
                    fr, fh = count_fillers(ref), count_fillers(hyp)
                    total += sum(fr.values())
                    kept += sum(min(v, fr.get(k, 0)) for k, v in fh.items())
                rows.append({"kind": kind, "ref": ref, "hyp": hyp, "err": e})
            res = {k: round(100 * e / n, 1) for k, (e, n) in agg.items()}
            res["fillers_kept"] = f"{kept}/{total}"
            res["median_ms"] = round(statistics.median(times) * 1000)
            table[(model, cname)] = res
            dump(f"stt_robust_{model}_{cname}", rows)
            print(f"{model:<24} {cname:<11} WER% {json.dumps(res)}", flush=True)
