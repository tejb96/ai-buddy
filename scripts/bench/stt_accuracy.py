"""STT accuracy + latency: synthetic (clean and noisy) prompts, disfluent answers, and real LibriSpeech speech.

Usage: python stt_accuracy.py <model> [<model> ...]
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo root
from ai_buddy.metrics import count_fillers  # noqa: E402

import os, time
from common import add_noise, dump, http, load, silence, synth, wav, wer  # noqa: E402
LOAD_ARGS = json.loads(os.environ.get("LOAD_ARGS", "{}"))
MAX_S = float(os.environ.get("MAX_S", "999"))
PROMPT = os.environ.get("PROMPT", "")
TAG = os.environ.get("TAG", "")

def transcribe(pcm, model):
    data = {"model": model, "language": "en", "response_format": "json"}
    if PROMPT: data["prompt"] = PROMPT
    t0 = time.perf_counter()
    r = http.post("/audio/transcriptions", data=data, files={"file": ("a.wav", wav(pcm), "audio/wav")})
    r.raise_for_status()
    return r.json()["text"], time.perf_counter() - t0


PROMPTS = [
    "What's the weather going to be like in Calgary tomorrow?",
    "What time is it in Tokyo right now?",
    "Can you search for the latest news about the Mars rover?",
    "How much does a Tesla Model 3 cost in Canada these days?",
    "What's eighteen percent of two hundred and forty?",
    "Look up who won the Stanley Cup last year.",
    "Check my notes for the talking points about the Henderson project.",
    "Did I mention the budget cuts and the hiring plan in that answer?",
    "I just got back from a long walk by the river. What should I cook for dinner tonight?",
    "Honestly, I'm a bit nervous about my interview on Thursday with the data team at Shopify.",
    "Can you explain the difference between Kubernetes and Docker in simple terms?",
    "My sister is visiting from Vancouver next weekend, and I need ideas for things to do downtown.",
    "What's the exchange rate between the Canadian dollar and the euro?",
    "Remind me how many kilometers are in a marathon.",
    "I think the second slide is too busy, so I'm going to cut the chart and keep just the three numbers.",
    "Tell me something interesting about octopuses.",
]
DISFLUENT = [
    "Um, so I studied statistics at university, and, uh, I did an internship where I, like, built dashboards.",
    "So, basically, the main thing I learned is that, you know, communication matters more than the code.",
    "I mean, I'm not sure, um, whether I should take the job or, uh, wait for a better offer.",
    "Uh, the project was, like, kind of a mess at first, but, um, we turned it around in about two months.",
]
VOICES = ["am_michael", "af_heart", "bf_emma", "bm_george", "af_bella", "am_adam"]
LS = Path(__file__).parent / "librispeech"


def clips():
    out = []
    for i, text in enumerate(PROMPTS):
        pcm = silence(0.25) + synth(text, VOICES[i % len(VOICES)]) + silence(0.35)
        out.append(("prompt", text, pcm))
        out.append(("prompt_noisy", text, add_noise(pcm, snr_db=12, seed=i)))
    for i, text in enumerate(DISFLUENT):
        pcm = silence(0.25) + synth(text, VOICES[(i + 2) % len(VOICES)]) + silence(0.35)
        out.append(("disfluent", text, pcm))
    for m in json.loads((LS / "manifest.json").read_text()):
        if m["seconds"] <= MAX_S:
            out.append(("librispeech", m["text"], (LS / m["file"]).read_bytes()))
    return out


def run(model: str, data) -> dict:
    load(model, **LOAD_ARGS)
    transcribe(silence(1.0), model)
    per = {}
    rows = []
    hes_ref = hes_hyp = fill_ref = fill_hyp = 0
    for kind, ref, pcm in data:
        hyp, dt = transcribe(pcm, model)
        hyp = " ".join(hyp.split())
        e, n = wer(ref, hyp)
        p = per.setdefault(kind, {"err": 0, "words": 0, "ms": []})
        p["err"] += e
        p["words"] += n
        p["ms"].append(dt * 1000)
        rows.append({"kind": kind, "ref": ref, "hyp": hyp, "err": e, "n": n, "ms": round(dt * 1000)})
        if kind == "disfluent":
            fr, fh = count_fillers(ref), count_fillers(hyp)
            hes_ref += sum(v for k, v in fr.items() if k in ("um", "uh"))
            hes_hyp += sum(min(v, fr.get(k, 0)) for k, v in fh.items() if k in ("um", "uh"))
            fill_ref += sum(fr.values())
            fill_hyp += sum(min(v, fr.get(k, 0)) for k, v in fh.items())
    summary = {kind: {"wer_pct": round(100 * p["err"] / p["words"], 1), "n": len(p["ms"]),
                      "median_ms": round(statistics.median(p["ms"])),
                      "p90_ms": round(sorted(p["ms"])[int(0.9 * (len(p["ms"]) - 1))])}
               for kind, p in per.items()}
    summary["fillers_kept"] = f"{fill_hyp}/{fill_ref}"
    summary["um_uh_kept"] = f"{hes_hyp}/{hes_ref}"
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    data = clips()
    results = {}
    for model in sys.argv[1:]:
        res = run(model, data)
        results[model] = res["summary"]
        dump(f"stt_accuracy_{model}{TAG}", res)
        print(model, json.dumps(res["summary"]))
        for r in res["rows"]:
            if r["err"] and r["kind"] != "librispeech":
                print(f"   [{r['kind']}] {r['err']}/{r['n']}  REF: {r['ref']}\n{'':18}HYP: {r['hyp']}")
    print(json.dumps(results, indent=1))
