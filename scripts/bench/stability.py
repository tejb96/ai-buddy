"""A long spoken conversation with an audio-input LLM, plus a short practice session: crashes, evictions,
prompt-cache misses, drift, tool use and temperatures. Usage: python stability.py [llm] [heard|whisper]

  heard    the model writes what it heard in <heard>...</heard> before replying (the transcript for the UI)
  whisper  the transcript comes from whisper-v3-turbo-FLM, run before the model's turn (one NPU job at a time)

Uses the app's real tools (web search, clock, calculator, notes over a sample folder), its system prompt,
and its history format (tool results as a user message). Earlier user turns stay in the history as
audio, which keeps FLM's prompt cache (see gemma_audio_cache.py). Turns are spaced like real use
(GAP_S, default 8 s between turns). Whisper stays loaded throughout; the practice part at the end
transcribes a talk with it while the LLM stays loaded, then has the LLM write feedback.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo root
from ai_buddy.audio import PresentationChunker, analyze_speech, has_speech  # noqa: E402
from ai_buddy.config import load_config  # noqa: E402
from ai_buddy.knowledge import KnowledgeBase  # noqa: E402
from ai_buddy.metrics import Answer, compute_metrics, count_fillers  # noqa: E402
from ai_buddy.prompts import chat_system_prompt  # noqa: E402
from ai_buddy.skills import SkillSet  # noqa: E402
from ai_buddy.tools import ToolContext, ToolRegistry  # noqa: E402

from common import BASE, ThermalLog, dump, load, loaded, silence, synth, transcribe, wav, wer  # noqa: E402

LLM = sys.argv[1] if len(sys.argv) > 1 else "gemma4-it-12b-FLM"
MODE = sys.argv[2] if len(sys.argv) > 2 else "heard"
STT = "whisper-v3-turbo-FLM"
GAP_S = float(os.environ.get("GAP_S", "8"))
CFG = load_config()
TZ = "America/Edmonton"
SKILLS = SkillSet.from_config(CFG)
HEARD = re.compile(r"\s*<heard>(.*?)</heard>\s*", re.S)

NOTES = {
    "work/henderson-pitch.md": """# Henderson project: talking points
- Open with the Henderson ops team story: they saved ten hours a week after switching to automated reports.
- Numbers: revenue up 12% quarter over quarter, churn down to 3%, NPS 61.
- Roadmap: SSO in Q1, audit log in Q2, data residency for EU customers in Q3.
- Ask: a two-year renewal with the analytics add-on.
- Likely objection: budget freeze. Answer: phased rollout, pay per seat.
""",
    "resume.md": """# Resume: Sam Rivera
## Experience
- Data analyst, Prairie Health (2021-2026): built Python ETL pipelines with pandas and SQLAlchemy that cut
  reporting time from two days to three hours; maintained a FastAPI service for dashboard data.
- Intern, City of Calgary open data team (2020): cleaned transit datasets in Python and wrote Jupyter tutorials.
## Skills
Python (five years: pandas, FastAPI, pytest), SQL, Tableau, a little Rust.
""",
    "personal/budget-2026.md": """# Budget notes, October 2026
Rent is 1,650 a month. Groceries target 500, eating out capped at 150 after going over in September.
Saving 400 a month for a new bike; 1,200 saved so far. Cancel the streaming service I don't use.
""",
    "work/quarterly-review-q3.md": """# Q3 quarterly review
Three main numbers: revenue 4.2 million (up 12%), 38 new enterprise customers, support response time down
to 3.5 hours. Risks: two large renewals in Q4, hiring freeze on the platform team.
""",
}

# (what the user says, tools that should be used ("any" = don't score), what to check by hand)
TURNS = [
    ("Hey Kira, how's it going? I just got back from a long walk by the river.", set(), "chat"),
    ("What's the weather going to be like in Calgary tomorrow?", {"web_search"}, "forecast from the search"),
    ("Thanks. Should I bring an umbrella then?", set(), "uses the earlier forecast, no new facts"),
    ("What time is it in Tokyo right now?", {"get_datetime"}, "time after a tool turn: must call the clock"),
    ("What are my talking points for the Henderson project?", {"search_notes"}, "from notes only"),
    ("Which of those numbers do you think will impress them most?", set(), "refers back to the notes"),
    ("Okay, quick one. What's eighteen percent of two hundred and forty?", {"calculate"}, "43.2"),
    ("If I split a hundred and thirty seven fifty bill four ways with a fifteen percent tip, how much does each "
     "person pay?", {"calculate"}, "39.53"),
    ("Um, so, I was, like, thinking about switching from Python to Rust for my side project. What do you think?",
     set(), "opinion, no invented personal story"),
    ("What does my resume say about my Python experience?", {"search_notes"}, "five years, pandas, FastAPI"),
    ("Who won the Stanley Cup this year?", {"web_search"}, "current fact: must search"),
    ("How many days until Christmas?", {"get_datetime"}, "72 days from Oct 14"),
    ("Tell me a joke about cats.", set(), "chat"),
    ("Is the Royal Tyrrell Museum open on Sundays, and until when?", {"web_search"}, "must search, no guessing"),
    ("Remind me what I wrote about the budget in my notes.", {"search_notes"}, "rent 1650, groceries 500"),
    ("What's today's date?", {"get_datetime"}, "Oct 14 2026 (the prompt has it too)"),
    ("I'm really nervous about my job interview on Thursday.", set(), "support, no invented details"),
    ("Where am I going on my trip next month?", "any", "unknown: say so or check notes, never invent a place"),
    ("What did you say the weather would be like tomorrow, again?", set(), "same facts as turn 2"),
    ("Thanks Kira, that's all for now. Good night!", set(), "chat"),
]
PRACTICE_TALK = [
    "Um, so today I want to talk about, uh, why our team switched to automated reports.",
    "Basically, the old process took like two days every month, and, you know, people hated it.",
    "So we, um, we built a small Python pipeline, and now it takes about three hours.",
    "The main thing I learned is that, uh, you have to show people the time they save, not the code.",
]


def audio_msg(pcm: bytes) -> dict:
    return {"role": "user", "content": [{"type": "input_audio", "input_audio": {
        "data": base64.b64encode(wav(pcm)).decode(), "format": "wav"}}]}


def system_prompt() -> str:
    return chat_system_prompt(CFG.persona, TZ, SKILLS.chat_instructions(), "heard" if MODE == "heard" else "audio")


def spoken_breaks(text: str) -> list[str]:
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


async def stream(client: httpx.AsyncClient, messages: list[dict], tools: list[dict] | None,
                 max_tokens: int = 400) -> dict:
    body = {"model": LLM, "messages": messages, "stream": True, "think": False, "temperature": CFG.chat.temperature,
            "max_completion_tokens": max_tokens, "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
    chunks, calls, usage = [], {}, {}
    t0 = time.perf_counter()
    async with client.stream("POST", "/chat/completions", json=body) as r:
        if r.status_code != 200:
            return {"error": f"{r.status_code} {(await r.aread())[:300]!r}", "text": "", "calls": [], "usage": {},
                    "wall_s": time.perf_counter() - t0}
        async for line in r.aiter_lines():
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            d = json.loads(line[5:])
            if "error" in d:
                return {"error": str(d["error"]), "text": "".join(chunks), "calls": [], "usage": usage,
                        "wall_s": time.perf_counter() - t0}
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
    return {"text": "".join(chunks), "calls": [calls[i] for i in sorted(calls)], "usage": usage,
            "wall_s": round(time.perf_counter() - t0, 2)}


def processes() -> dict:
    return {k: v.get("pid") for k, v in loaded().items()}


async def conversation(client, reg: ToolRegistry, ctx: ToolContext, log: list[dict]) -> list[dict]:
    history = [{"role": "system", "content": system_prompt()}]
    voices = ["af_heart", "am_michael", "bf_emma", "am_adam"]
    pids0 = processes()
    for n, (said, want, check) in enumerate(TURNS, 1):
        pcm = silence(0.25) + synth(said, voices[n % len(voices)]) + silence(0.35)
        # One NPU job at a time (two FLM processes running at once corrupt each other's results): Whisper
        # first, as the app does; in heard mode, Whisper checks every 5th turn after the model's turn.
        whisper = None
        if MODE == "whisper":
            try:
                whisper = (await asyncio.to_thread(transcribe, pcm, STT))[0]
            except Exception as e:      # a failed Whisper call is a finding, not a crash of the run
                whisper = f"ERROR {e}"
        history.append(audio_msg(pcm))
        rounds, used, heard, tools_used, reply, heard_again = [], 0, None, [], "", False
        for rnd in range(CFG.tools.max_calls_per_turn + 2):
            out = await stream(client, history, reg.schemas())
            rounds.append({k: out.get(k) for k in ("wall_s", "error")} | {"usage": {
                k: out["usage"].get(k) for k in ("prompt_tokens", "completion_tokens", "active_kv_tokens",
                                                 "decoding_speed_tps", "prefill_speed_tps")}})
            if out.get("error"):
                print(f"  !! turn {n} round {rnd}: {out['error']}")
                break
            text = out["text"]
            m = HEARD.match(text)
            if m and rnd == 0:
                heard = " ".join(m.group(1).split())
            elif m:
                heard_again = True   # wrote a transcript again after tool results
            if m:
                text = text[m.end():]
            calls = [reg.parse(c) for c in out["calls"]]
            if calls and used < CFG.tools.max_calls_per_turn:
                calls = calls[: CFG.tools.max_calls_per_turn - used]
                used += len(calls)
                results = [await reg.execute(c, ctx) for c in calls]
                tools_used += [{"name": c.name, "args": c.args, "ok": r.ok, "summary": r.summary}
                               for c, r in zip(calls, results)]
                history.append({"role": "assistant", "content": out["text"], "tool_calls": [c.for_history() for c in calls]})
                history.append(reg.results_message(calls, results))
                continue
            reply = text.strip()
            history.append({"role": "assistant", "content": out["text"].strip() or "…"})
            break
        if MODE == "heard" and n % 5 == 0:
            try:
                whisper = (await asyncio.to_thread(transcribe, pcm, STT))[0]
            except Exception as e:
                whisper = f"ERROR {e}"
        names = {t["name"] for t in tools_used}
        pids = processes()
        u0 = rounds[0]["usage"]
        row = {"turn": n, "said": said, "check": check, "want": sorted(want) if want != "any" else "any",
               "tools": tools_used, "tool_ok": want == "any" or names == want, "heard": heard,
               "heard_err": wer(said, heard)[0] if heard else None, "whisper": whisper,
               "whisper_err": wer(said, whisper)[0] if whisper and not whisper.startswith("ERROR") else None,
               "reply": reply, "breaks": spoken_breaks(reply), "rounds": rounds, "heard_again": heard_again,
               "cache_miss": bool(u0.get("prompt_tokens") and u0.get("active_kv_tokens")
                                  and u0["prompt_tokens"] > 0.6 * u0["active_kv_tokens"] and n > 1),
               "pids_changed": pids != pids0, "pids": pids}
        log.append(row)
        print(f"  turn {n:>2} {'ok ' if row['tool_ok'] else 'BAD'} tools={sorted(names) or '-'} "
              f"prefill={[r['usage']['prompt_tokens'] for r in rounds]} kv={u0.get('active_kv_tokens')} "
              f"heard={heard!r} {'PIDS CHANGED ' if row['pids_changed'] else ''}\n         -> {reply[:150]!r}")
        await asyncio.sleep(GAP_S)
    return history


async def practice(client) -> dict:
    """A short talk transcribed by Whisper in chunks (as the presentation mode does), then feedback from the LLM."""
    pids0 = processes()
    gap = silence(0.9)
    talk = b"".join(silence(0.3) + synth(t, "am_michael") + gap for t in PRACTICE_TALK)
    chunker = PresentationChunker(target_s=8)
    captions = []
    for i in range(0, len(talk), 3200):   # fed in 100 ms frames, like the browser
        for start_s, pcm in chunker.feed(talk[i:i + 3200]):
            if has_speech(pcm):
                captions.append(await asyncio.to_thread(lambda p=pcm: transcribe(p, STT)[0]))
    if tail := chunker.flush():
        captions.append(await asyncio.to_thread(lambda: transcribe(tail[1], STT)[0]))
    transcript = " ".join(captions)
    metrics = compute_metrics([Answer(transcript, analyze_speech(talk))])
    coach = SKILLS.mode_skill("presentation").modules["modes"]
    msgs = coach.feedback_messages("presentation", transcript, metrics, details="automated reports")
    out = await stream(client, msgs, None, max_tokens=CFG.feedback.max_tokens)
    pids = processes()
    ref = " ".join(PRACTICE_TALK)
    return {"transcript": transcript, "wer": wer(ref, transcript), "fillers_ref": sum(count_fillers(ref).values()),
            "fillers_whisper": sum(count_fillers(transcript).values()), "metrics_wpm": metrics["wpm"],
            "feedback": out["text"], "feedback_error": out.get("error"), "usage": out["usage"],
            "pids_before": pids0, "pids_after": pids, "pids_changed": pids != pids0}


async def main() -> None:
    load(LLM, ctx_size=CFG.models.llm_ctx_size)
    load(STT)
    print("loaded:", processes())
    notes = Path(__file__).parent / "results" / "sample-notes"   # git-ignored
    for rel, text in NOTES.items():
        (notes / rel).parent.mkdir(parents=True, exist_ok=True)
        (notes / rel).write_text(text)
    kb = KnowledgeBase(notes / ".index.db", notes, watch=False)
    await kb.reindex()
    reg = SKILLS.registry()
    log: list[dict] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=600), follow_redirects=True) as web, \
            httpx.AsyncClient(base_url=BASE, timeout=httpx.Timeout(10.0, read=600)) as client:
        ctx = ToolContext(CFG, web, kb, TZ)
        with ThermalLog(10) as thermal:
            history = await conversation(client, reg, ctx, log)
            await asyncio.sleep(GAP_S)
            print("== practice: Whisper transcribes a talk, then the LLM writes feedback")
            prac = await practice(client)
    await kb.close()
    print(f"  practice transcript: {prac['transcript']!r}\n  fillers {prac['fillers_whisper']}/{prac['fillers_ref']}, "
          f"pids changed: {prac['pids_changed']}\n  feedback: {prac['feedback'][:300]!r}")
    size = len(json.dumps(history))
    summary = {
        "llm": LLM, "mode": MODE, "turns": len(log), "tool_ok": f"{sum(r['tool_ok'] for r in log)}/{len(log)}",
        "errors": sum(1 for r in log for x in r["rounds"] if x.get("error")),
        "cache_misses": [r["turn"] for r in log if r["cache_miss"]],
        "pids_changed": any(r["pids_changed"] for r in log) or prac["pids_changed"],
        "heard_missing": [r["turn"] for r in log if MODE == "heard" and not r["heard"]],
        "heard_word_errors": sum(r["heard_err"] or 0 for r in log),
        "whisper_word_errors": sum(r["whisper_err"] or 0 for r in log if r["whisper_err"] is not None),
        "spoken_breaks": {r["turn"]: r["breaks"] for r in log if r["breaks"]},
        "final_kv_tokens": log[-1]["rounds"][0]["usage"].get("active_kv_tokens") if log else None,
        "request_json_kb_at_end": round(size / 1024),
        "thermal": thermal.summary(),
    }
    print(json.dumps(summary, indent=1))
    dump(f"stability_{LLM}_{MODE}", {"summary": summary, "turns": log, "practice": prac,
                                     "thermal_samples": thermal.samples})


if __name__ == "__main__":
    asyncio.run(main())
