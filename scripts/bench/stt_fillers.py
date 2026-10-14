"""Filler-word retention with and without a disfluent Whisper prompt.

Usage: python stt_fillers.py <model> [<model> ...]
"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo root
from ai_buddy.metrics import count_fillers  # noqa: E402

from common import http, load, silence, synth, wav, wer  # noqa: E402

TEXTS = [
    "Um, so I studied statistics at university, and, uh, I did an internship where I, like, built dashboards.",
    "So, basically, the main thing I learned is that, you know, communication matters more than the code.",
    "I mean, I'm not sure, um, whether I should take the job or, uh, wait for a better offer.",
    "Uh, the project was, like, kind of a mess at first, but, um, we turned it around in about two months.",
    "Hmm, that's a good question. Um, I guess my biggest weakness is, uh, that I take on too much.",
    "And, uh, we launched it in, um, March, and, like, the users really loved it.",
    "So yeah, um, I think, uh, the team was, you know, pretty happy with the result.",
    "Well, um, I, uh, I think I would, like, start by talking to the customer first.",
]
VOICES = ["am_michael", "af_heart", "bf_emma", "bm_george"]
PROMPT = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. Uh, so, you know, I mean..."


def tx(pcm, model, prompt=None):
    data = {"model": model, "language": "en", "response_format": "json"}
    if prompt:
        data["prompt"] = prompt
    t0 = time.perf_counter()
    r = http.post("/audio/transcriptions", data=data, files={"file": ("a.wav", wav(pcm), "audio/wav")})
    r.raise_for_status()
    return " ".join(r.json()["text"].split()), time.perf_counter() - t0


for model in sys.argv[1:]:
    load(model)
    tx(silence(1), model)
    for label, prompt in (("no prompt", None), ("disfluent prompt", PROMPT)):
        kept = ref_total = um_kept = um_total = errs = words = 0
        times = []
        for i, text in enumerate(TEXTS):
            for v in VOICES[:2]:
                pcm = silence(0.25) + synth(text, VOICES[(i + VOICES.index(v)) % 4]) + silence(0.35)
                hyp, dt = tx(pcm, model, prompt)
                times.append(dt)
                fr, fh = count_fillers(text), count_fillers(hyp)
                ref_total += sum(fr.values())
                kept += sum(min(n, fr.get(k, 0)) for k, n in fh.items())
                um_total += fr.get("um", 0) + fr.get("uh", 0)
                um_kept += min(fh.get("um", 0), fr.get("um", 0)) + min(fh.get("uh", 0), fr.get("uh", 0))
                e, n = wer(text, hyp)
                errs += e
                words += n
                if label != "no prompt" and fh != fr:
                    print(f"     {dict(fr)} vs {dict(fh)}: {hyp}")
        print(f"{model:<24} {label:<17} fillers kept {kept}/{ref_total}, um/uh kept {um_kept}/{um_total}, "
              f"WER(excl. um/uh) {100 * errs / words:.1f}%, median {sorted(times)[len(times) // 2] * 1000:.0f} ms")
