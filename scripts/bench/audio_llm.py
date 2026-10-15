"""One model that hears (Gemma 4 audio input) vs. Whisper + text LLM, on the same spoken test set.

Usage: python audio_llm.py <pipeline> [sections]
  pipeline: "whisper:<llm>"  transcribe with whisper-v3-turbo-FLM, then send the text to <llm>
            "audio:<llm>"    send the recording itself to <llm> (input_audio content part)
            "heard:<llm>"    like audio, but the model first writes what it heard in <heard>...</heard>
                             (one pass: transcript and reply in the same turn)
  sections: comma list of tools,multi,understand,transcribe (default: all); for heard: heard
            (tool prompts, filler answers and LS_N LibriSpeech clips, scored against Whisper's
            transcripts of the same clips from the whisper:qwen3.6-moe-35b-a3b-FLM results)

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
from ai_buddy.prompts import Persona, chat_system_prompt  # noqa: E402
from ai_buddy.skills import SkillSet  # noqa: E402

from common import (AUDIO_HEARING, ThermalLog, add_noise, dump, hearing, http, load, loaded, norm, silence,  # noqa: E402
                    synth, transcribe, wav, wer)
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
SKILLS = SkillSet.from_config(CFG)
REG = SKILLS.registry()
# Kira's real chat prompt (as of this commit; runs before 2026-10-14 used the pre-skills prompt).
# The test prompts call the assistant Kira (its name until 2026-10-14; now it is named after its voice).
SYSTEM = hearing(chat_system_prompt(Persona("Kira", CFG.persona.style), "America/Edmonton",
                                    SKILLS.chat_instructions()), KIND)
HEARD_RULE = AUDIO_HEARING["heard"]
LS_N = int(os.environ.get("LS_N", "30"))     # LibriSpeech clips in the heard section


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
    return {"text": msg.get("content") or "", "calls": calls, "usage": r.json().get("usage", {})}


HEARD = re.compile(r"\s*<heard>(.*?)</heard>\s*", re.S)


def complete_heard(messages: list[dict], max_tokens: int = 400) -> dict:
    """Streamed completion (FLM streams one token per chunk), split into the <heard> transcript and the reply.
    heard_tokens = output tokens up to and including </heard>: the extra cost of the transcript."""
    body = {"model": LLM, "messages": messages, "stream": True, "think": False, "temperature": CHAT_TEMPERATURE,
            "max_completion_tokens": max_tokens, "tools": REG.schemas(), "stream_options": {"include_usage": True}}
    chunks, calls, usage = [], {}, {}
    with http.stream("POST", "/chat/completions", json=body) as r:
        if r.status_code != 200:
            time.sleep(PAUSE_S)
            return {"error": f"{r.status_code} {r.read()[:300]!r}", "text": "", "heard": "", "heard_ok": False,
                    "heard_tokens": None, "calls": [], "usage": {}}
        for line in r.iter_lines():
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for ch in d.get("choices", []):
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    chunks.append(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", len(calls)), {"id": "", "name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    c["id"] = tc.get("id") or c["id"]
                    c["name"] += fn.get("name") or ""
                    c["arguments"] += fn.get("arguments") or ""
    time.sleep(PAUSE_S)
    text = "".join(chunks)
    heard_tokens, acc = None, ""
    for k, piece in enumerate(chunks, 1):
        acc += piece
        if "</heard>" in acc:
            heard_tokens = k
            break
    m = HEARD.match(text)
    return {"text": text[m.end():] if m else text, "heard": " ".join(m.group(1).split()) if m else "",
            "heard_ok": bool(m), "heard_tokens": heard_tokens, "calls": [calls[i] for i in sorted(calls)],
            "usage": usage}


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


# ---------- heard: transcript + reply in one pass, against Whisper on the same clips ----------

BASELINE = Path(__file__).parent / "results" / "audio_llm_whisper_qwen3.6-moe-35b-a3b-FLM.json"


def _errors(ref: str, hyp: str) -> tuple[int, int]:
    return wer(ref, hyp) if hyp else (len(norm(ref)), len(norm(ref)))


def run_heard() -> dict:
    base = json.loads(BASELINE.read_text())
    w_tools = {(r["prompt"], r["noisy"]): r["heard"] for r in base["tools"]["rows"]}
    w_fill = [r for r in base["transcribe"]["rows"] if r["kind"] == "fillers"]
    w_ls = [r for r in base["transcribe"]["rows"] if r["kind"] == "librispeech"]
    rows, diffs = [], []

    def turn(kind: str, ref: str, pcm: bytes, whisper: str, **extra) -> dict:
        out = complete_heard([{"role": "system", "content": SYSTEM}, user_msg(pcm)[0]])
        e_g, n = _errors(ref, out["heard"])
        e_w, _ = _errors(ref, whisper)
        u = out["usage"]
        row = {"kind": kind, "ref": ref, "heard": out["heard"], "heard_ok": out["heard_ok"], "whisper": whisper,
               "err_gemma": e_g, "err_whisper": e_w, "n": n, "heard_tokens": out["heard_tokens"],
               "completion_tokens": u.get("completion_tokens"), "decode_tps": u.get("decoding_speed_tps"),
               "reply": out["text"], "calls": out["calls"], "style": spoken_style(out["text"]),
               "error": out.get("error"), **extra}
        rows.append(row)
        if e_g or e_w or not out["heard_ok"]:
            diffs.append(row)
        flag = "ok " if out["heard_ok"] else "NO-HEARD"
        print(f"  {flag} {kind:<6} gemma {e_g}/{n} whisper {e_w}/{n} {out['heard_tokens']} tok | {out['heard'][:70]!r}")
        return row

    # 1. The 24 tool prompts, clean and noisy: transcript accuracy and tool choice in heard mode.
    tool_ok = 0
    for i, (prompt, allowed, argcheck) in enumerate(TOOL_CASES):
        for noisy in (False, True):
            row = turn("tools", prompt, spoken(prompt, i + noisy, noisy), w_tools.get((prompt, noisy), ""),
                       noisy=noisy)
            names = {p.name for p in (REG.parse(c) for c in row["calls"]) if not p.error}
            arg_ok = argcheck is None or not argcheck or bool(re.search(
                argcheck, json.dumps([REG.parse(c).args for c in row["calls"]]).lower()))
            row["tool_ok"] = names in allowed and arg_ok
            tool_ok += row["tool_ok"]
    # 2. Answers full of fillers (same clips as the Whisper transcription rows).
    kept_g = kept_w = total = 0
    k = 0
    for i, text in enumerate(FILLERS):
        for v in range(2):
            row = turn("filler", text, spoken(text, i + v), w_fill[k]["hyp"])
            k += 1
            fr = count_fillers(text)
            total += sum(fr.values())
            kept_g += sum(min(n, fr.get(w, 0)) for w, n in count_fillers(row["heard"]).items())
            kept_w += sum(min(n, fr.get(w, 0)) for w, n in count_fillers(row["whisper"]).items())
            row["fillers_ref"] = dict(fr)
            row["fillers_gemma"] = dict(count_fillers(row["heard"]))
    # 3. Real speech: the first LS_N LibriSpeech clips.
    for j, m in enumerate(json.loads((LS / "manifest.json").read_text())[:LS_N]):
        turn("libri", m["text"], (LS / m["file"]).read_bytes(), w_ls[j]["hyp"])

    def wer_pct(kind: str, who: str, noisy: bool | None = None) -> float:
        sel = [r for r in rows if r["kind"] == kind and (noisy is None or r.get("noisy") == noisy)]
        return round(100 * sum(r[f"err_{who}"] for r in sel) / max(1, sum(r["n"] for r in sel)), 1)

    toks = sorted(r["heard_tokens"] for r in rows if r["heard_tokens"])
    words_heard = sum(len(r["heard"].split()) for r in rows if r["heard_tokens"])
    tps = sorted(r["decode_tps"] for r in rows if r["decode_tps"])
    summary = {
        "turns": len(rows), "heard_present": f"{sum(r['heard_ok'] for r in rows)}/{len(rows)}",
        "heard_present_on_tool_turns": f"{sum(r['heard_ok'] for r in rows if r['calls'])}/{sum(1 for r in rows if r['calls'])}",
        "tool_choice_ok": f"{tool_ok}/{sum(1 for r in rows if r['kind'] == 'tools')}",
        "wer_tools_clean": {"gemma": wer_pct("tools", "gemma", False), "whisper": wer_pct("tools", "whisper", False)},
        "wer_tools_noisy": {"gemma": wer_pct("tools", "gemma", True), "whisper": wer_pct("tools", "whisper", True)},
        "wer_fillers": {"gemma": wer_pct("filler", "gemma"), "whisper": wer_pct("filler", "whisper")},
        "wer_librispeech": {"gemma": wer_pct("libri", "gemma"), "whisper": wer_pct("libri", "whisper"),
                            "clips": min(LS_N, len(w_ls))},
        "fillers_kept": {"gemma": f"{kept_g}/{total}", "whisper": f"{kept_w}/{total}"},
        "heard_tokens_per_turn": {"median": toks[len(toks) // 2] if toks else None,
                                  "mean": round(sum(toks) / len(toks), 1) if toks else None,
                                  "max": max(toks, default=None),
                                  "per_word": round(sum(toks) / max(1, words_heard), 2)},
        "decode_tps_median": round(tps[len(tps) // 2], 2) if tps else None,
        "style_breaks": sum(bool(r["style"]) for r in rows),
    }
    print("  differences (REF / GEMMA / WHISPER):")
    for r in diffs:
        print(f"   [{r['kind']}{' noisy' if r.get('noisy') else ''}] {r['ref']}\n      G: {r['heard']}\n      W: {r['whisper']}")
    return {**summary, "rows": rows}


if __name__ == "__main__":
    load(LLM, ctx_size=CFG.models.llm_ctx_size)
    if KIND == "whisper":
        load(STT)
    pids = {k: v.get("pid") for k, v in loaded().items()}
    results = {}
    with ThermalLog(10) as thermal:
        for section, fn in (("tools", run_tools), ("multi", run_multi), ("understand", run_understand),
                            ("transcribe", run_transcribe), ("heard", run_heard)):
            if section in SECTIONS:
                print(f"== {section} ({PIPE})")
                results[section] = fn()
                time.sleep(5 * PAUSE_S)
    results["thermal"] = thermal.summary()
    after = {k: v.get("pid") for k, v in loaded().items()}
    results["processes"] = {"before": pids, "after": after}
    if pids != after:
        print(f"!! loaded models changed during the run: {pids} -> {after}")
    dump(f"audio_llm_{PIPE.replace(':', '_')}{'_reduced' if REDUCED else ''}{'_' + os.environ['TAG'] if os.environ.get('TAG') else ''}",
         results)
    print(json.dumps({s: {k: v for k, v in r.items() if k != "rows"} for s, r in results.items()}, indent=1))
