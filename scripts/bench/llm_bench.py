"""LLM benchmark for the agent: time to first spoken word, prefill speed, tool-calling reliability,
and tool-result -> spoken answer. Usage: python llm_bench.py <model> [ctx_size] [sections]

sections: comma list of ttfw,prefill,tools,answer (default: all)
"""
from __future__ import annotations

import json
import random
import re
import statistics
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo root
from ai_buddy.text import SpeechChunker, clean_for_speech  # noqa: E402

from common import dump, http, load  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.6-moe-35b-a3b-FLM"
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
SECTIONS = (sys.argv[3] if len(sys.argv) > 3 else "ttfw,prefill,tools,answer").split(",")
TEMP = 0.6

SYSTEM = """You are Kira, the user's voice companion and speaking coach. You're warm and encouraging, but honest: you give real, specific opinions and never empty praise.

Everything you say is read aloud by a text-to-speech voice, so:
- Keep each turn short: one to three sentences and under 50 words, like real conversation.
- Plain spoken English only: no emojis, lists, markdown, URLs or stage directions.
- Use contractions and natural phrasing. Say numbers the way a person would.
The user's words come from speech recognition and may contain small errors; infer what they meant.

You have tools. Use them when they help, and answer directly when they don't:
- web_search for anything current or that you aren't sure about: weather, news, prices, sports, people, products, opening hours, events.
- get_datetime for the current date or time anywhere.
- calculate for any arithmetic beyond the trivial.
- search_notes for anything about the user's own notes, talking points, resume or documents.
Never invent facts a tool could give you. After using a tool, answer in one to three spoken sentences."""

TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web and read the top results. Use for current or factual information: weather, news, prices, sports, people, products, opening hours, events.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "A concise search query, like you would type into a search engine."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_datetime",
        "description": "Get the current date and time, locally or in another timezone.",
        "parameters": {"type": "object", "properties": {
            "timezone": {"type": "string", "description": "IANA timezone such as 'Asia/Tokyo'. Omit for the user's local time."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression exactly, e.g. '137.50 * 1.15 / 4'. Supports + - * / ** %, parentheses, sqrt, round.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "The expression to evaluate."}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "search_notes",
        "description": "Search the user's own notes, talking points, resume and documents.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for."}},
            "required": ["query"]}}},
]

# (prompt, acceptable tool sets, argument check)
TOOL_CASES = [
    ("What's the weather going to be like in Calgary tomorrow?", [{"web_search"}], r"calgary"),
    ("Any big news today about the Mars rover?", [{"web_search"}], r"mars|rover"),
    ("How much is a Tesla Model 3 in Canada right now?", [{"web_search"}], r"tesla|model 3"),
    ("Who won the Stanley Cup this year?", [{"web_search"}], r"stanley"),
    ("What's the exchange rate from Canadian dollars to euros?", [{"web_search"}], r"cad|canadian|eur"),
    ("Is the Calgary Public Library open on Sunday?", [{"web_search"}], r"library"),
    ("Can you look up the opening hours of the Royal Tyrrell Museum?", [{"web_search"}], r"tyrrell"),
    ("Search the web for the latest version of Python and tell me what's new.", [{"web_search"}], r"python"),
    ("What time is it in Tokyo right now?", [{"get_datetime"}], r"tokyo"),
    ("What's today's date?", [{"get_datetime"}], r""),
    ("How many days until Christmas?", [{"get_datetime"}, {"get_datetime", "calculate"}], r""),
    ("What's eighteen percent of two hundred and forty?", [{"calculate"}], r"18|0\.18|240"),
    ("If I split a hundred and thirty seven fifty bill four ways with a fifteen percent tip, how much does each person pay?",
     [{"calculate"}], r"137\.5|4"),
    ("What's the square root of seven hundred and twenty nine times three?", [{"calculate"}], r"729|sqrt"),
    ("What are my talking points for the Henderson project?", [{"search_notes"}], r"henderson"),
    ("What does my resume say about my Python experience?", [{"search_notes"}], r"python|resume"),
    ("Remind me what I wrote about the budget in my notes.", [{"search_notes"}], r"budget"),
    ("Find my notes on the quarterly review and tell me the three main numbers.", [{"search_notes"}], r"quarter"),
    ("What's the weather in Paris, and what time is it there?", [{"web_search", "get_datetime"}], r"paris"),
    ("Hey Kira, how's it going?", [set()], None),
    ("Tell me a joke about cats.", [set()], None),
    ("I'm really nervous about my job interview on Thursday.", [set()], None),
    ("Why is the sky blue?", [set()], None),
    ("What's the capital of Australia?", [set()], None),
]

CHAT_TURNS = [
    "Hey Kira! I just got back from a long walk by the river. What should I cook for dinner tonight?",
    ("Hmm, I like that idea, but I only have some leftover rice, a couple of eggs, half an onion, frozen peas and "
     "some soy sauce. I don't really want to go to the store because it's raining and I'm pretty tired after the "
     "walk. Is there something quick I could make with just that, maybe in twenty minutes or less?"),
    "Nice. How do I keep the rice from getting mushy?",
    ("Okay, while that cooks, can I tell you about my week? Work has been a lot. My manager asked me to lead the "
     "client presentation on Friday, which is exciting, but I've never presented to that client before and they're "
     "known for asking really tough questions about budgets and timelines. I've got the slides mostly done but I "
     "keep second guessing the opening."),
    "What would make a strong opening, do you think?",
    ("That's helpful. I was planning to open with a slide of our quarterly numbers, revenue up twelve percent and "
     "churn down to three percent, then go straight into the roadmap. But maybe that's too dry? My coworker said "
     "I should open with a customer story instead, about how one of their teams saved ten hours a week."),
    "Yeah, let's go with the story. Can you help me practice it later?",
    "Thanks Kira. Honestly, you've made me feel a lot better about Friday.",
]

SEARCH_RESULT = {
    "query": "Calgary weather tomorrow",
    "results": [
        {"title": "Calgary, AB 7-Day Weather Forecast - Environment Canada", "url": "https://weather.gc.ca/city/pages/ab-52_metric_e.html",
         "content": ("Tomorrow: A mix of sun and cloud. 40 percent chance of showers in the afternoon. Wind west 20 km/h "
                     "gusting to 40. High 14. UV index 4 or moderate. Tomorrow night: Cloudy periods. Low plus 2. "
                     "Thursday: Sunny. High 17. Thursday night: Clear. Low 4. Friday: A mix of sun and cloud. High 19. "
                     "Saturday: Chance of showers. High 11. Sunday: Periods of rain. High 8. ") * 3},
        {"title": "Calgary weather forecast | The Weather Network", "url": "https://www.theweathernetwork.com/ca/weather/alberta/calgary",
         "content": ("Calgary forecast for Wednesday: High 13°C, low 1°C. Feels like 10. Showers possible after 2 PM "
                     "with about 2 mm of rain. Winds W 25 km/h. Sunrise 7:52 AM, sunset 6:38 PM. Air quality: low risk. "
                     "Pollen: low. Hourly: 9 AM 6°C sunny, noon 11°C partly cloudy, 3 PM 13°C showers, 6 PM 9°C cloudy. ") * 3},
        {"title": "Calgary, Alberta, Canada Weather Forecast | AccuWeather", "url": "https://www.accuweather.com/en/ca/calgary/t2p/weather-forecast/52455",
         "content": ("Tomorrow: Times of clouds and sun with a shower in the afternoon. Hi: 13°. Tomorrow night: Partly "
                     "cloudy. Lo: 2°. RealFeel 11°. Max UV Index 3 Moderate. Wind W 22 km/h, gusts 41 km/h. "
                     "Probability of precipitation 45%. Thunderstorms 0%. Cloud cover 60%. ") * 3},
    ],
}
NOTES_RESULT = {
    "query": "Henderson project talking points",
    "results": [
        {"file": "work/henderson-pitch.md", "text": (
            "# Henderson project: talking points\n- Open with the Henderson ops team story: they saved ten hours a week "
            "after switching to automated reports.\n- Numbers: revenue up 12% quarter over quarter, churn down to 3%, "
            "NPS 61.\n- Roadmap: SSO in Q1, audit log in Q2, data residency for EU customers in Q3.\n- Ask: a two-year "
            "renewal with the analytics add-on.\n- Likely objection: budget freeze. Answer: phased rollout, pay per seat.")},
        {"file": "work/meeting-notes-2026-09.md", "text": (
            "Sept 30 sync with Dana: Henderson wants a clearer timeline for SSO. Their CFO is skeptical about the "
            "analytics add-on price. Prepare a one-slide ROI summary: 10 hours/week saved x 40 staff x $55/hour.")},
    ],
}


def stream(messages, tools=TOOLS, max_tokens=220, think=False):
    """Run one streamed completion; return timing and outputs."""
    body = {"model": MODEL, "messages": messages, "stream": True, "think": think, "temperature": TEMP,
            "max_completion_tokens": max_tokens}
    if tools:
        body["tools"] = tools
    t0 = time.perf_counter()
    out = {"text": "", "tool_calls": [], "first_token": None, "first_piece": None, "first_piece_text": None}
    chunker = SpeechChunker()
    with http.stream("POST", "/chat/completions", json=body) as r:
        if r.status_code != 200:
            raise RuntimeError(r.read()[:300])
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            if "usage" in d and d["usage"]:
                out["usage"] = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                for tc in delta.get("tool_calls") or []:
                    out["tool_calls"].append(tc)
                    out.setdefault("tool_at", time.perf_counter() - t0)
                piece = delta.get("content")
                rc = delta.get("reasoning_content")
                if rc:
                    out["reasoning"] = out.get("reasoning", "") + rc
                if piece:
                    if out["first_token"] is None:
                        out["first_token"] = time.perf_counter() - t0
                    out["text"] += piece
                    if out["first_piece"] is None:
                        pieces = chunker.feed(piece)
                        if pieces:
                            out["first_piece"] = time.perf_counter() - t0
                            out["first_piece_text"] = pieces[0]
    out["total"] = time.perf_counter() - t0
    if out["first_piece"] is None and out["text"].strip():
        out["first_piece"] = out["total"]
        out["first_piece_text"] = out["text"].strip()
    return out


def tts_first_audio(text: str) -> float:
    t0 = time.perf_counter()
    with http.stream("POST", "/audio/speech", json={"model": "kokoro-v1", "input": clean_for_speech(text),
                                                     "voice": "af_heart", "response_format": "pcm",
                                                     "stream_format": "audio"}) as r:
        dt = None
        for _ in r.iter_bytes():
            if dt is None:
                dt = time.perf_counter() - t0
    return dt


def bench_ttfw():
    """Growing conversation with tools attached: first token, first speakable piece, + TTS first audio."""
    rows = []
    for run in range(2):
        msgs = [{"role": "system", "content": SYSTEM}]
        for i, user in enumerate(CHAT_TURNS):
            msgs.append({"role": "user", "content": user})
            o = stream(msgs)
            tts = tts_first_audio(o["first_piece_text"] or "Okay.")
            ttfw = (o["first_piece"] or o["total"]) + tts
            words = len(user.split())
            rows.append({"run": run, "turn": i + 1, "user_words": words, "first_token": o["first_token"],
                         "first_piece": o["first_piece"], "tts": tts, "ttfw": ttfw, "total": o["total"],
                         "prompt_tokens": o.get("usage", {}).get("prompt_tokens"),
                         "tool_calls": [tc["function"]["name"] for tc in o["tool_calls"]],
                         "reply": o["text"], "reply_words": len(o["text"].split())})
            print(f"  run {run} turn {i + 1} ({words:>3} words in, {o.get('usage', {}).get('prompt_tokens')} tok ctx): "
                  f"first token {o['first_token'] or 0:.2f}s, first piece {o['first_piece'] or 0:.2f}s, "
                  f"+TTS {tts:.2f}s => first word {ttfw:.2f}s | {len(o['text'].split())} words: {o['text'][:90]!r}")
            msgs.append({"role": "assistant", "content": o["text"] or "Okay."})
    short = [r["ttfw"] for r in rows if r["user_words"] < 25 and r["turn"] > 1]
    long_ = [r["ttfw"] for r in rows if r["user_words"] >= 25]
    first = [r["ttfw"] for r in rows if r["turn"] == 1]
    summary = {"first_turn_s": round(statistics.median(first), 2), "short_turn_median_s": round(statistics.median(short), 2),
               "long_turn_median_s": round(statistics.median(long_), 2),
               "reply_words_median": statistics.median(r["reply_words"] for r in rows)}
    print("  TTFW summary:", summary)
    return {"summary": summary, "rows": rows}


def bench_prefill():
    rows = []
    filler = ("The committee reviewed the annual report, discussed the budget for the coming year, and agreed to "
              "revisit the hiring plan after the second quarter results are published. ")
    for n_rep in (12, 25, 50, 100):
        nonce = random.randint(0, 10**9)
        text = f"[{nonce}] " + filler * n_rep
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text + "\nSummarize that in one sentence."}]
        o = stream(msgs, max_tokens=30)
        u = o.get("usage", {})
        rows.append({"prompt_tokens": u.get("prompt_tokens"), "ttft": o["first_token"],
                     "prefill_tps": u.get("prefill_speed_tps"), "decode_tps": u.get("decoding_speed_tps")})
        print(f"  prefill {u.get('prompt_tokens')} tokens: first token {o['first_token']:.2f}s, "
              f"prefill {u.get('prefill_speed_tps', 0):.0f} tok/s, decode {u.get('decoding_speed_tps', 0):.1f} tok/s")
    return rows


def bench_tools(reps=2):
    rows = []
    ok = 0
    for prompt, expected, arg_re in TOOL_CASES:
        for rep in range(reps):
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            o = stream(msgs, max_tokens=160)
            names = {tc["function"]["name"] for tc in o["tool_calls"]}
            args_ok = True
            args = []
            for tc in o["tool_calls"]:
                try:
                    a = json.loads(tc["function"].get("arguments") or "{}")
                    args.append(a)
                except json.JSONDecodeError:
                    args_ok = False
                    args.append(tc["function"].get("arguments"))
            choice_ok = names in expected
            if arg_re and choice_ok:
                args_ok = args_ok and bool(re.search(arg_re, json.dumps(args).lower()))
            leaked = bool(re.search(r"<tool_call>|\{\"name\"|<think>|</think>", o["text"]))
            good = choice_ok and args_ok and not leaked
            ok += good
            rows.append({"prompt": prompt, "rep": rep, "tools": sorted(names), "args": args, "text": o["text"][:200],
                         "ok": good, "choice_ok": choice_ok, "args_ok": args_ok, "leaked": leaked,
                         "decision_s": round(o.get("tool_at") or o["first_token"] or o["total"], 2)})
            flag = "OK " if good else "BAD"
            print(f"  {flag} {prompt[:60]:<60} -> {sorted(names) or '-'} {json.dumps(args)[:70]} "
                  f"{'' if names else repr(o['text'][:50])} ({rows[-1]['decision_s']}s)")
    n = len(rows)
    tool_rows = [r for r in rows if r["tools"]]
    summary = {"correct": f"{ok}/{n}", "pct": round(100 * ok / n, 1),
               "tool_decision_median_s": round(statistics.median(r["decision_s"] for r in tool_rows), 2) if tool_rows else None}
    print("  TOOLS summary:", summary)
    return {"summary": summary, "rows": rows}


def bench_answer():
    """Tool result -> spoken answer: latency from tool result to first spoken word, and answer style."""
    rows = []
    cases = [("What's the weather going to be like in Calgary tomorrow?", "web_search", {"query": "Calgary weather tomorrow"}, SEARCH_RESULT),
             ("What are my talking points for the Henderson project?", "search_notes", {"query": "Henderson project talking points"}, NOTES_RESULT)]
    for rep in range(2):
        for prompt, tool, args, result in cases:
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": tool, "arguments": json.dumps(args)}}]},
                    {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(result)}]
            o = stream(msgs, max_tokens=220)
            tts = tts_first_audio(o["first_piece_text"] or "Okay.")
            u = o.get("usage", {})
            bad_style = bool(re.search(r"https?://|\*\*|^\s*[-*]\s|#", o["text"], re.M))
            rows.append({"case": tool, "prompt_tokens": u.get("prompt_tokens"), "first_token": o["first_token"],
                         "ttfw": (o["first_piece"] or o["total"]) + tts, "words": len(o["text"].split()),
                         "bad_style": bad_style, "tool_calls": [t["function"]["name"] for t in o["tool_calls"]],
                         "text": o["text"]})
            print(f"  {tool:<13} {u.get('prompt_tokens')} tok: first token {o['first_token'] or 0:.2f}s, first word "
                  f"{rows[-1]['ttfw']:.2f}s, {rows[-1]['words']} words, style {'BAD' if bad_style else 'ok'} | {o['text'][:160]!r}")
    return rows


if __name__ == "__main__":
    print(f"load {MODEL} ctx={CTX}: {load(MODEL, ctx_size=CTX):.1f}s")
    stream([{"role": "user", "content": "Hi"}], tools=None, max_tokens=2)  # warm-up
    res = {"model": MODEL, "ctx": CTX}
    if "ttfw" in SECTIONS:
        print("== time to first spoken word (growing conversation, tools attached)")
        res["ttfw"] = bench_ttfw()
    if "prefill" in SECTIONS:
        print("== prefill speed")
        res["prefill"] = bench_prefill()
    if "tools" in SECTIONS:
        print("== tool calling")
        res["tools"] = bench_tools()
    if "answer" in SECTIONS:
        print("== tool result -> spoken answer")
        res["answer"] = bench_answer()
    dump(f"llm_{MODEL}_{'-'.join(SECTIONS)}", res)
