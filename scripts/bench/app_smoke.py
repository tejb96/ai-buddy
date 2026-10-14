"""Scripted conversation against the running app: greeting, plain turn, and tool turns.
Usage: python smoke.py <port> [turn ...]"""
import asyncio
import json
import sys
import time

import websockets

from common import silence, synth

PORT = sys.argv[1]
TURNS = {
    "plain": "Hey Kira, how's your day going?",
    "calc": "What's eighteen percent of two hundred and forty?",
    "notes": "What are my talking points for the Henderson project?",
    "web": "What's the weather going to be like in Calgary tomorrow?",
    "time": "What time is it in Tokyo right now?",
    "news": "Any big news today about the Mars rover?",
}
ORDER = sys.argv[2:] or ["plain", "calc", "notes", "web", "time"]
HOLD = 0.35


async def recv_until(ws, pred, log, t0, timeout=120):
    while True:
        msg = await asyncio.wait_for(ws.recv(), timeout)
        if isinstance(msg, bytes):
            log.setdefault("first_audio", time.perf_counter() - t0)
            continue
        ev = json.loads(msg)
        t = time.perf_counter() - t0
        if ev["type"] in ("tool_start", "tool_end", "user_text", "user_partial", "notice", "error", "reply_end"):
            brief = {k: ev.get(k) for k in ("name", "detail", "summary", "ok", "ms", "text", "stt_ms") if ev.get(k) is not None}
            if ev["type"] == "reply_end":
                brief = {"text": ev["text"], "timing": ev.get("timing"), "sources": len(ev.get("sources") or [])}
            print(f"   {t:6.2f}s {ev['type']:<12} {json.dumps(brief)[:300]}")
        if ev["type"] == "user_text":
            log["transcript"] = t
        if pred(ev):
            return ev


async def main():
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws", origin=f"http://127.0.0.1:{PORT}", max_size=None) as ws:
        t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "start", "mode": "chat"}))
        print("greeting:")
        await recv_until(ws, lambda e: e["type"] == "reply_end" and e.get("greeting"), {}, t0)
        for name in ORDER:
            text = TURNS[name]
            pcm = silence(0.3) + synth(text) + silence(HOLD)
            await ws.send(json.dumps({"type": "rec_start"}))
            for i in range(0, len(pcm), 3200):
                await ws.send(pcm[i:i + 3200])
                await asyncio.sleep(0.1)
            await ws.send(json.dumps({"type": "rec_stop"}))
            t0 = time.perf_counter()
            log = {}
            print(f"[{name}] {text}")
            await recv_until(ws, lambda e: e["type"] == "reply_end", log, t0)
            print(f"   => stop speaking -> transcript {log.get('transcript', 0) + HOLD:.2f}s, "
                  f"-> Kira's first audio {log.get('first_audio', 0) + HOLD:.2f}s")
            await asyncio.sleep(3)


asyncio.run(main())
