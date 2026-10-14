"""One model that hears (Gemma 4 audio input) vs. Whisper + text LLM, on the same spoken test set.

Usage: python audio_llm.py <pipeline> [sections]
  pipeline: "whisper:<llm>"  transcribe with whisper-v3-turbo-FLM, then send the text to <llm>
            "audio:<llm>"    send the recording itself to <llm> (input_audio content part)
  sections: comma list of tools,multi,understand,transcribe (default: all)

Uses Kira's real system prompt and tool schemas (ai_buddy.prompts / ai_buddy.tools). Quality only:
no latency numbers are reported. Calls are spaced out (PAUSE_S, default 1 s) to keep the laptop cool.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo root
from ai_buddy.config import load_config  # noqa: E402
from ai_buddy.metrics import count_fillers  # noqa: E402
from ai_buddy.prompts import chat_system_prompt  # noqa: E402
from ai_buddy.tools import ToolRegistry  # noqa: E402

from common import add_noise, dump, http, load, silence, synth, transcribe, wav, wer  # noqa: E402
_argv, sys.argv = sys.argv, sys.argv[:1]   # llm_bench reads its own arguments at import
from llm_bench import TOOL_CASES  # noqa: E402
sys.argv = _argv

PIPE = sys.argv[1]
KIND, LLM = PIPE.split(":", 1)
SECTIONS = (sys.argv[2] if len(sys.argv) > 2 else "tools,multi,understand,transcribe").split(",")
STT = "whisper-v3-turbo-FLM"
PAUSE_S = float(os.environ.get("PAUSE_S", "1"))
# REDUCED=1: ~20 cases picked from the full sections (same prompts, voices and noise), for a quick comparison.
REDUCED = os.environ.get("REDUCED") == "1"
R_TOOLS = {0, 3, 8, 9, 11, 12, 14, 18, 19, 23}         # indexes into TOOL_CASES, clean audio only
R_MULTI = {0, 2, 4}
R_UNDERSTAND = {0, 1, 2, 5}
R_LIBRISPEECH = 3
CHAT_TEMPERATURE = 0.6       # Kira's chat.temperature; transcription requests use 0
VOICES = ["am_michael", "af_heart", "bf_emma", "bm_george", "af_bella", "am_adam"]
LS = Path(__file__).parent / "librispeech"

CFG = load_config()
REG = ToolRegistry.from_config(CFG)
SYSTEM = chat_system_prompt(CFG.persona, "America/Edmonton")
if KIND == "audio":
    SYSTEM = SYSTEM.replace("The user's words come from speech recognition and may contain small errors; "
                            "infer what they meant.", "You hear the user's voice directly.")


def spoken(text: str, i: int, noisy: bool = False) -> bytes:
    pcm = silence(0.25) + synth(text, VOICES[i % len(VOICES)]) + silence(0.35)
    return add_noise(pcm, snr_db=12, seed=i) if noisy else pcm


def audio_part(pcm: bytes) -> dict:
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav(pcm)).decode(), "format": "wav"}}


def user_msg(pcm: bytes, text_hint: str | None = None) -> tuple[dict, str]:
    """The user message for this pipeline, and what the text pipeline heard (for the record)."""
    if KIND == "whisper":
        heard, _ = transcribe(pcm, STT)
        return {"role": "user", "content": heard}, heard
    return {"role": "user", "content": [audio_part(pcm)]}, ""


def complete(messages: list[dict], tools: bool = True, max_tokens: int = 300) -> dict:
    body = {"model": LLM, "messages": messages, "stream": False, "think": False, "temperature": CHAT_TEMPERATURE,
            "max_completion_tokens": max_tokens}
    if tools:
        body["tools"] = REG.schemas()
    r = http.post("/chat/completions", json=body)
    time.sleep(PAUSE_S)
    if r.status_code != 200:
        return {"error": f"{r.status_code} {r.text[:300]}", "text": "", "calls": []}
    msg = r.json()["choices"][0]["message"]
    calls = [{"id": c.get("id", ""), "name": c["function"].get("name", ""),
              "arguments": c["function"].get("arguments", "")} for c in msg.get("tool_calls") or []]
    return {"text": msg.get("content") or "", "calls": calls}


def spoken_style(text: str) -> list[str]:
    """Rule breaks for a spoken reply."""
    bad = []
    if len(text.split()) > 60:
        bad.append(f"{len(text.split())} words")
    if re.search(r"^\s*([-*•]|\d+\.)\s", text, re.M) or "**" in text or "#" in text:
        bad.append("markdown/list")
    if re.search(r"https?://|www\.", text):
        bad.append("url")
    if re.search(r"<\s*/?\s*(think|tool|heard)", text):
        bad.append("tag leak")
    return bad


# ---------- tools: 24 prompts x clean + noisy ----------

def run_tools() -> dict:
    rows, ok_raw, ok_fixed = [], 0, 0
    for i, (prompt, allowed, argcheck) in enumerate(TOOL_CASES):
        if REDUCED and i not in R_TOOLS:
            continue
        for noisy in ((False,) if REDUCED else (False, True)):
            pcm = spoken(prompt, i + noisy, noisy)
            msg, heard = user_msg(pcm)
            out = complete([{"role": "system", "content": SYSTEM}, msg])
            parsed = [REG.parse(c) for c in out["calls"]]
            raw_names = {c["name"] for c in out["calls"]}
            names = {p.name for p in parsed if not p.error}
            args = json.dumps([p.args for p in parsed]).lower()
            arg_ok = argcheck is None or not argcheck or bool(re.search(argcheck, args))
            good_raw = raw_names in allowed and arg_ok
            good = names in allowed and arg_ok
            ok_raw += good_raw
            ok_fixed += good
            rows.append({"prompt": prompt, "noisy": noisy, "heard": heard, "calls": out["calls"],
                         "names": sorted(names), "ok": good, "text": out["text"], "style": spoken_style(out["text"]),
                         "error": out.get("error")})
            flag = "ok " if good else "BAD"
            print(f"  {flag} {'noisy' if noisy else 'clean'} {prompt[:55]:<55} -> {sorted(names) or out['text'][:60]!r}"
                  + (f"  (heard: {heard})" if heard and wer(prompt, heard)[0] else ""))
    n = len(rows)
    return {"tool_choice_ok": f"{ok_fixed}/{n}", "tool_choice_ok_before_name_repair": f"{ok_raw}/{n}", "rows": rows}


# ---------- multi-turn: no guessing after a tool turn ----------

WEATHER_RESULT = ("web_search: " + json.dumps({"query": "Calgary weather tomorrow", "results": [
    {"title": "Calgary forecast - Environment Canada", "url": "https://weather.gc.ca/city/pages/ab-52_metric_e.html",
     "content": "Tomorrow: A mix of sun and cloud. 40 percent chance of showers in the afternoon. High 14. Low plus 2."}]}))

WEATHER_ANSWER = ("Tomorrow in Calgary looks like a mix of sun and cloud with a high of fourteen, and there's a "
                  "forty percent chance of showers in the afternoon.")

MULTI = [
    # (turns, check on the last reply). Each turn: ("say", text) | ("tool", name, args, result text)
    ("time after a tool turn",
     [("say", "What's the weather going to be like in Calgary tomorrow?"),
      ("tool", "web_search", {"query": "Calgary weather tomorrow"}, WEATHER_RESULT),
      ("reply", WEATHER_ANSWER),
      ("say", "Nice. And what time is it in Tokyo right now?")], {"get_datetime"}),
    ("date after small talk",
     [("say", "Hey Kira, I just finished a long run and I'm exhausted."),
      ("reply", "Nice work! How far did you go?"),
      ("say", "About ten kilometers. What's today's date, by the way?")], {"get_datetime"}),
    ("unknown fact after chat: look it up",
     [("say", "I'm thinking about visiting the Royal Tyrrell Museum this weekend."),
      ("reply", "Oh, that's a great trip. Dinosaurs?"),
      ("say", "Yes! Is it open on Sunday, and until when?")], {"web_search"}),
    ("arithmetic after a tool turn",
     [("say", "What's the weather going to be like in Calgary tomorrow?"),
      ("tool", "web_search", {"query": "Calgary weather tomorrow"}, WEATHER_RESULT),
      ("reply", WEATHER_ANSWER),
      ("say", "Okay. Quick one: what's eighteen percent of two hundred and forty?")], {"calculate"}),
    ("no tool needed after a tool turn",
     [("say", "What's the weather going to be like in Calgary tomorrow?"),
      ("tool", "web_search", {"query": "Calgary weather tomorrow"}, WEATHER_RESULT),
      ("reply", WEATHER_ANSWER),
      ("say", "Thanks. Should I bring a jacket then?")], set()),
]


def run_multi() -> dict:
    from ai_buddy.tools import ToolCall, ToolResult
    rows, ok = [], 0
    for rep in range(1 if REDUCED else 2):
        for k, (label, turns, want) in enumerate(MULTI):
            if REDUCED and k not in R_MULTI:
                continue
            msgs = [{"role": "system", "content": SYSTEM}]
            last = None
            for j, t in enumerate(turns):
                if t[0] == "say":
                    msg, heard = user_msg(spoken(t[1], k + j + rep))
                    if j < len(turns) - 1 and KIND == "audio":
                        msg = {"role": "user", "content": t[1]}  # earlier turns as text (what the app would keep)
                    msgs.append(msg)
                elif t[0] == "reply":
                    msgs.append({"role": "assistant", "content": t[1]})
                else:
                    call = ToolCall(id="call_1", name=t[1], args=t[2], raw_arguments=json.dumps(t[2]))
                    msgs.append({"role": "assistant", "content": "", "tool_calls": [call.for_history()]})
                    msgs.append(REG.results_message([call], [ToolResult(json.loads(t[3].split(": ", 1)[1]))]))
            last = complete(msgs)
            names = {p.name for p in (REG.parse(c) for c in last["calls"]) if not p.error}
            good = names == want
            ok += good
            rows.append({"case": label, "rep": rep, "names": sorted(names), "text": last["text"], "ok": good})
            print(f"  {'ok ' if good else 'BAD'} {label:<38} -> {sorted(names) or last['text'][:80]!r}")
    return {"multi_ok": f"{ok}/{len(rows)}", "rows": rows}


# ---------- understanding: spoken details the reply must use ----------

UNDERSTAND = [
    ("My sister Priya is visiting from Vancouver on the twenty third. What are two things we could do downtown "
     "that she'd like if she loves art?", [r"art|gallery|museum"]),
    ("I have three meetings tomorrow, at nine, eleven thirty and four. Which one should I prepare for tonight if "
     "the eleven thirty one is with the CEO?", [r"eleven|11|ceo"]),
    ("Um, so, I was, like, thinking about switching from Python to Rust for my side project, uh, but I'm worried "
     "about the learning curve. What do you think?", [r"rust", r"python|learning|curve"]),
    ("Can you give me a word that rhymes with orange, or is that impossible?", [r"orange|rhyme|door ?hinge|sporange"]),
    ("I'm practicing my pitch for a vegan bakery called Crumb and Kindness. Does that name sound friendly?",
     [r"crumb"]),
    ("I said to my manager I'd finish the report by Friday, but my daughter Ava is sick. How do I ask for an "
     "extension without sounding flaky?", [r"manager|extension|friday|ava"]),
]


def run_understand() -> dict:
    rows, ok, style_bad = [], 0, 0
    for i, (text, checks) in enumerate(UNDERSTAND):
        if REDUCED and i not in R_UNDERSTAND:
            continue
        for noisy in ((False,) if REDUCED else (False, True)):
            msg, heard = user_msg(spoken(text, i + 3, noisy))
            out = complete([{"role": "system", "content": SYSTEM}, msg])
            reply = out["text"]
            good = all(re.search(c, reply, re.I) for c in checks) and not out["calls"]
            bad = spoken_style(reply)
            ok += good
            style_bad += bool(bad)
            rows.append({"said": text, "noisy": noisy, "heard": heard, "reply": reply, "calls": out["calls"],
                         "uses_details": good, "style": bad})
            print(f"  {'ok ' if good else 'BAD'}{' style:' + ','.join(bad) if bad else ''} "
                  f"{'noisy' if noisy else 'clean'}: {reply[:140]!r}")
    return {"uses_details": f"{ok}/{len(rows)}", "style_breaks": style_bad, "rows": rows}


# ---------- transcription (what coaching needs) ----------

FILLERS = [
    "Um, so I studied statistics at university, and, uh, I did an internship where I, like, built dashboards.",
    "So, basically, the main thing I learned is that, you know, communication matters more than the code.",
    "I mean, I'm not sure, um, whether I should take the job or, uh, wait for a better offer.",
    "Uh, the project was, like, kind of a mess at first, but, um, we turned it around in about two months.",
    "Hmm, that's a good question. Um, I guess my biggest weakness is, uh, that I take on too much.",
    "And, uh, we launched it in, um, March, and, like, the users really loved it.",
    "So yeah, um, I think, uh, the team was, you know, pretty happy with the result.",
    "Well, um, I, uh, I think I would, like, start by talking to the customer first.",
]
VERBATIM = ("Transcribe this audio verbatim in English. Keep every filler word and hesitation exactly as spoken "
            "(um, uh, like, you know, so) and any repeated words. Output only the transcript.")


def tx(pcm: bytes) -> str:
    if KIND == "whisper":
        return transcribe(pcm, STT)[0]
    r = http.post("/chat/completions", json={
        "model": LLM, "stream": False, "think": False, "temperature": 0, "max_completion_tokens": 400,
        "messages": [{"role": "user", "content": [{"type": "text", "text": VERBATIM}, audio_part(pcm)]}]})
    time.sleep(PAUSE_S / 2)
    r.raise_for_status()
    return " ".join((r.json()["choices"][0]["message"].get("content") or "").split())


def run_transcribe() -> dict:
    kept = total = 0
    rows = []
    for i, text in enumerate([] if REDUCED else FILLERS):
        for v in range(2):
            hyp = tx(spoken(text, i + v))
            fr, fh = count_fillers(text), count_fillers(hyp)
            total += sum(fr.values())
            kept += sum(min(n, fr.get(k, 0)) for k, n in fh.items())
            rows.append({"kind": "fillers", "ref": text, "hyp": hyp, "ref_fillers": dict(fr), "hyp_fillers": dict(fh)})
    err = n = truncated = 0
    for m in json.loads((LS / "manifest.json").read_text())[:R_LIBRISPEECH if REDUCED else None]:
        hyp = tx((LS / m["file"]).read_bytes())
        e, w = wer(m["text"], hyp)
        err, n = err + e, n + w
        truncated += len(hyp.split()) < 0.7 * len(m["text"].split())
        rows.append({"kind": "librispeech", "ref": m["text"], "hyp": hyp, "err": e, "n": w})
    for r in rows:
        if r["kind"] == "librispeech" and REDUCED:
            print(f"  REF: {r['ref']}\n  HYP: {r['hyp']}")
    print(f"  fillers kept {kept}/{total}, LibriSpeech WER {100 * err / n:.1f}% ({truncated} cut short)")
    return {"fillers_kept": f"{kept}/{total}", "librispeech_wer_pct": round(100 * err / n, 1),
            "librispeech_cut_short": truncated, "rows": rows}


if __name__ == "__main__":
    load(LLM, ctx_size=CFG.models.llm_ctx_size)
    if KIND == "whisper":
        load(STT)
    results = {}
    for section, fn in (("tools", run_tools), ("multi", run_multi), ("understand", run_understand),
                        ("transcribe", run_transcribe)):
        if section in SECTIONS:
            print(f"== {section} ({PIPE})")
            results[section] = fn()
            time.sleep(5 * PAUSE_S)
    dump(f"audio_llm_{PIPE.replace(':', '_')}{'_reduced' if REDUCED else ''}", results)
    print(json.dumps({s: {k: v for k, v in r.items() if k != "rows"} for s, r in results.items()}, indent=1))
