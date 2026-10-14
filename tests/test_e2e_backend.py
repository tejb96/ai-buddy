"""End-to-end tests of every mode through the WebSocket, against the real models.

Run: uv run pytest -m lemonade -s   (takes a few minutes: includes two swaps to the 35B model)
"""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.request

import pytest
import websockets

from conftest import Client, say, silence

pytestmark = [pytest.mark.lemonade, pytest.mark.asyncio(loop_scope="session")]

QUESTION = "Hey Kira, I just got back from a long walk by the river. What should I cook for dinner tonight?"


async def test_rejects_foreign_origin(server):
    with pytest.raises(websockets.exceptions.InvalidStatus):
        ws = await websockets.connect(server["ws"], origin="https://evil.example")
        await ws.recv()


async def test_chat_turn_latency_and_speech(server):
    c = await Client.connect(server)
    await c.send(type="start", mode="chat")
    await c.until(lambda e: e["type"] == "session")
    speech = say(QUESTION)
    await c.send(type="rec_start")
    await c.stream(silence(0.3) + speech + silence(0.4))
    await c.send(type="rec_stop")
    t_stop = time.monotonic()
    user = await c.until(lambda e: e["type"] == "user_text")
    end = await c.until(lambda e: e["type"] == "reply_end")
    rid = end["id"]
    first_audio = c.first_audio_at[rid] - t_stop
    print(f"\n  transcript: {user['text']!r}\n  reply: {end['text']!r}\n  STT {user['stt_ms']} ms, "
          f"server timing {end['timing']}, end-of-speech -> first audio frame {first_audio:.2f}s, "
          f"speech {c.audio[rid] / 48000:.1f}s")
    assert "river" in user["text"].lower() and "dinner" in user["text"].lower()
    assert end["text"] and not end["interrupted"]
    assert c.audio[rid] / 48000 > 1.0          # at least a second of 24 kHz 16-bit speech
    assert first_audio < 6.0
    assert not any(ch in end["text"] for ch in "😊🍽️🌿")

    # Second turn reuses the conversation (history + FLM prefix cache).
    await c.send(type="rec_start")
    await c.stream(say("Ooh, that sounds good. Is it hard to make?") + silence(0.3))
    await c.send(type="rec_stop")
    t_stop = time.monotonic()
    end2 = await c.until(lambda e: e["type"] == "reply_end" and e["id"] != rid)
    print(f"  turn 2 reply: {end2['text']!r} first audio {c.first_audio_at[end2['id']] - t_stop:.2f}s")
    await c.close()


async def test_barge_in_interrupts_reply(server):
    c = await Client.connect(server)
    await c.send(type="start", mode="chat")
    await c.send(type="rec_start")
    await c.stream(say("Tell me a long story about a dragon who learns to bake bread.") + silence(0.3))
    await c.send(type="rec_stop")
    start = await c.until(lambda e: e["type"] == "reply_start")
    await c.until(lambda e: e["type"] == "reply_delta")
    await c.send(type="rec_start")       # user starts talking over the buddy
    end = await c.until(lambda e: e["type"] == "reply_end" and e["id"] == start["id"])
    assert end["interrupted"]
    await c.send(type="rec_stop")        # nothing recorded -> "didn't hear anything"
    await c.until(lambda e: e["type"] == "notice")
    await c.close()


async def test_silence_is_not_transcribed(server):
    c = await Client.connect(server)
    await c.send(type="start", mode="chat")
    await c.send(type="rec_start")
    await c.stream(silence(1.5))
    await c.send(type="rec_stop")
    notice = await c.until(lambda e: e["type"] == "notice", timeout=10)
    assert "didn't hear" in notice["text"]
    assert not c.of("user_text")
    await c.close()


async def test_interview_with_feedback(server):
    c = await Client.connect(server)
    await c.send(type="start", mode="interview", scenario="job_interview",
                 details="junior data analyst at a healthcare startup")
    session = await c.until(lambda e: e["type"] == "session")
    opening = await c.until(lambda e: e["type"] == "reply_end")
    print(f"\n  interviewer: {opening['text']!r}")
    assert opening["text"] and c.audio.get(opening["id"], 0) > 0
    answers = [
        "Um, sure. So I studied statistics at university, and, uh, I did an internship where I, like, "
        "built dashboards for a hospital. I really enjoy turning messy data into something useful.",
        "One time our data had, um, a lot of duplicate patient records. I wrote a script to find them, "
        "and, you know, I explained the fix to the nurses so they could avoid it in the future.",
    ]
    last = opening["id"]
    for answer in answers:
        await c.send(type="rec_start")
        await c.stream(say(answer) + silence(0.3))
        await c.send(type="rec_stop")
        reply = await c.until(lambda e: e["type"] == "reply_end" and e["id"] > last)
        last = reply["id"]
        print(f"  interviewer: {reply['text']!r}")
    t0 = time.monotonic()
    await c.send(type="end")
    metrics = (await c.until(lambda e: e["type"] == "metrics"))["metrics"]
    done = await c.until(lambda e: e["type"] == "feedback_done", timeout=300)
    print(f"  metrics: {metrics['words']} words, {metrics['wpm']} wpm, fillers {metrics['fillers']}")
    print(f"  feedback in {time.monotonic() - t0:.0f}s ({len(done['markdown'])} chars):\n{done['markdown'][:600]}")
    assert metrics["fillers"]["total"] >= 3 and len(metrics["answers"]) == 2
    assert "## Top 3 things to improve" in done["markdown"]
    detail = json.load(urllib.request.urlopen(f"{server['base']}/api/sessions/{session['id']}"))
    assert detail["feedback_md"] and detail["metrics"]["words"] == metrics["words"]
    assert detail["user_turns"] == 2
    await c.close()


async def test_presentation_with_feedback(server):
    c = await Client.connect(server)
    await c.send(type="start", mode="presentation")
    await c.until(lambda e: e["type"] == "session")
    paragraphs = [
        "Good morning everyone. Today I want to talk about, um, why sleep matters so much for learning. "
        "So, basically, when we sleep, our brain consolidates the memories we formed during the day.",
        "There are, like, two main stages that matter here. Deep sleep helps us keep facts, and, uh, "
        "dream sleep helps us connect ideas in new and creative ways.",
        "So what can you actually do about it? Um, keep a regular schedule, avoid screens before bed, "
        "and, you know, don't pull all nighters before an exam. Thank you.",
    ]
    talk = silence(0.5) + say(paragraphs[0]) + silence(3.2) + say(paragraphs[1]) + silence(0.8) + say(paragraphs[2])
    talk_s = len(talk) / 32000
    await c.send(type="rec_start")
    session = await c.until(lambda e: e["type"] == "session" and e["id"])
    await c.stream(talk, realtime=4.0)    # 4x faster than live, enough to exercise chunking
    t0 = time.monotonic()
    await c.send(type="rec_stop")
    metrics = (await c.until(lambda e: e["type"] == "metrics", timeout=120))["metrics"]
    t_metrics = time.monotonic() - t0
    done = await c.until(lambda e: e["type"] == "feedback_done", timeout=300)
    print(f"\n  talk {talk_s:.0f}s, captions: {len(c.of('caption'))}, metrics after {t_metrics:.1f}s, "
          f"feedback after {time.monotonic() - t0:.0f}s")
    print(f"  metrics: {json.dumps({k: metrics[k] for k in ('words', 'wpm', 'fillers', 'pauses')})}")
    print(f"  feedback:\n{done['markdown'][:700]}")
    # 3.2 s inserted + Kokoro's own ~0.6 s tail and ~0.4 s lead silence around the clips = ~4.2 s
    assert metrics["pauses"]["count"] == 1 and 3.9 <= metrics["pauses"]["longest_s"] <= 4.5
    assert metrics["fillers"]["total"] >= 5
    assert 100 <= metrics["wpm"] <= 220
    assert "## Overall" in done["markdown"]
    con = sqlite3.connect(server["data"] / "buddy.db")
    row = con.execute("SELECT metrics_json, feedback_md FROM sessions WHERE id=?", (session["id"],)).fetchone()
    assert row and row[0] and row[1]
    assert (server["data"] / "sessions" / f"{session['id']}.md").exists()
    await c.close()


async def test_silent_presentation_is_cancelled_not_stuck(server):
    c = await Client.connect(server)
    await c.send(type="start", mode="presentation")
    await c.send(type="rec_start")
    session = await c.until(lambda e: e["type"] == "session" and e["id"])
    await c.stream(silence(3.0))
    await c.send(type="rec_stop")
    ev = await c.until(lambda e: e["type"] == "feedback_cancelled", timeout=60)
    assert ev["session_id"] == session["id"] and not ev["error"]
    assert "didn't catch any speech" in ev["text"]
    sessions = json.load(urllib.request.urlopen(f"{server['base']}/api/sessions"))
    assert session["id"] not in [s["id"] for s in sessions]   # empty takes aren't saved
    await c.close()


async def test_interview_end_without_answers(server):
    c = await Client.connect(server)
    await c.send(type="start", mode="interview", scenario="small_talk", details="")
    session = await c.until(lambda e: e["type"] == "session")
    await c.send(type="end")
    ev = await c.until(lambda e: e["type"] == "feedback_cancelled", timeout=60)
    assert ev["session_id"] == session["id"] and not ev["error"]
    assert not c.of("metrics")
    await c.close()
