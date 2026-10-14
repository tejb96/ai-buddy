"""Does cutting speech at pauses (as a live chunker would) hurt accuracy, and how long is the leftover tail?

Usage: python stt_chunking.py <model> [min_chunk_s] [cut_pause_s]
"""
import json
import sys
from pathlib import Path
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo root
from ai_buddy.audio import FRAME_MS, FRAME_SAMPLES, frame_db, speech_threshold  # noqa: E402

from common import http, load, silence, synth, wav, wer  # noqa: E402
from stt_accuracy import LS, PROMPTS, VOICES  # noqa: E402

MODEL = sys.argv[1]
MIN_CHUNK_S = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
CUT_PAUSE_S = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
FILLER_PROMPT = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."
LONG = [
    ("So I have a presentation on Friday about our quarterly results, and I want to practice the opening. "
     "It covers revenue growth, the new hires, and the product roadmap for next year."),
    ("Okay, here is the thing. I have been thinking about switching careers for a while now. I studied biology, "
     "but I have spent the last three years teaching myself to code, and I built a couple of small apps. "
     "I am not sure whether I should apply for junior developer jobs, or go back to school for a master's degree."),
    ("Um, so in my last role I led a small team of four analysts. We were, uh, responsible for the weekly sales "
     "dashboard, and I rebuilt the whole pipeline in Python, which cut the refresh time from two hours to ten minutes."),
]


def tx(pcm: bytes, prompt: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    r = http.post("/audio/transcriptions", data={"model": MODEL, "language": "en", "response_format": "json",
                                                 "prompt": prompt}, files={"file": ("a.wav", wav(pcm), "audio/wav")})
    r.raise_for_status()
    return " ".join(r.json()["text"].split()), time.perf_counter() - t0


def cut_points(pcm: bytes) -> list[int]:
    """Byte offsets where a live chunker would cut: the middle of the first pause >= CUT_PAUSE_S
    once the pending chunk is >= MIN_CHUNK_S long (threshold estimated on the audio so far)."""
    cuts, start = [], 0
    frame_b = FRAME_SAMPLES * 2
    need = int(CUT_PAUSE_S * 1000 / FRAME_MS)
    pos = start
    while True:
        seg = pcm[start:]
        db = frame_db(seg)
        thr = speech_threshold(frame_db(pcm[:max(start + len(seg), 1)]))
        if thr is None:
            break
        first = int(MIN_CHUNK_S * 1000 / FRAME_MS)
        run, found = 0, None
        for i in range(first, len(db)):
            run = run + 1 if db[i] < thr else 0
            if run >= need:
                found = i - run // 2
                break
        if found is None:
            break
        pos = start + found * frame_b
        cuts.append(pos)
        start = pos
    return cuts


def main():
    load(MODEL)
    tx(silence(1.0), FILLER_PROMPT)
    items = [(t, silence(0.25) + synth(t, VOICES[i % 4]) + silence(0.4)) for i, t in enumerate(LONG + PROMPTS[8:12])]
    for m in json.loads((LS / "manifest.json").read_text()):
        if 6 <= m["seconds"] <= 30:
            items.append((m["text"], (LS / m["file"]).read_bytes()))
    whole_e = chunk_e = words = 0
    tails, whole_ms, tail_ms, nchunks = [], [], [], []
    for ref, pcm in items:
        hyp_whole, dt_whole = tx(pcm, FILLER_PROMPT)
        cuts = cut_points(pcm)
        bounds = [0] + cuts + [len(pcm)]
        texts, dt_last = [], 0.0
        for a, b in zip(bounds, bounds[1:]):
            chunk = pcm[a:b]
            prev = " ".join(texts)[-200:]
            if np.abs(np.frombuffer(chunk, dtype="<i2")).max(initial=0) < 300:
                continue
            t, dt_last = tx(chunk, (FILLER_PROMPT + " " + prev).strip())
            texts.append(t)
        hyp_chunks = " ".join(texts)
        e1, n = wer(ref, hyp_whole)
        e2, _ = wer(ref, hyp_chunks)
        whole_e += e1
        chunk_e += e2
        words += n
        tails.append((len(pcm) - bounds[-2]) / 32000)
        whole_ms.append(dt_whole * 1000)
        tail_ms.append(dt_last * 1000)
        nchunks.append(len(bounds) - 1)
        if e2 > e1:
            print(f"  chunking cost {e2 - e1} words ({len(bounds) - 1} chunks):\n    whole: {hyp_whole[:150]}\n    chunk: {hyp_chunks[:150]}")
    print(f"{MODEL} min_chunk={MIN_CHUNK_S}s cut_pause={CUT_PAUSE_S}s on {len(items)} utterances of 6-30 s: "
          f"WER whole {100 * whole_e / words:.1f}% vs chunked {100 * chunk_e / words:.1f}%; "
          f"chunks/utt median {sorted(nchunks)[len(nchunks) // 2]}; tail audio median {sorted(tails)[len(tails) // 2]:.1f}s "
          f"max {max(tails):.1f}s; whole-utterance STT median {sorted(whole_ms)[len(whole_ms) // 2]:.0f} ms vs "
          f"tail-chunk STT median {sorted(tail_ms)[len(tail_ms) // 2]:.0f} ms")


main()
