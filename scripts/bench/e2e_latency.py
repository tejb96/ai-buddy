"""End-to-end turn latency through the app's WebSocket, like the browser: stream mic audio at live pace,
release, then time "release -> transcript shown" and "release -> first audio frame".

Usage: python e2e_latency.py <ws_url> <origin> [label]
"""
import asyncio
import json
import statistics
import sys
import time

import websockets

from common import silence, synth

WS, ORIGIN = sys.argv[1], sys.argv[2]
LABEL = sys.argv[3] if len(sys.argv) > 3 else ""
TURNS = [
    ("short", "Hey Kira, how's your day going?"),
    ("short", "What should I cook for dinner tonight?"),
    ("short", "Tell me something interesting about octopuses."),
    ("long", "I just got back from a long walk by the river, and honestly I'm pretty tired. I've got some rice, "
             "a couple of eggs and frozen peas. What could I make in twenty minutes?"),
    ("long", "So I have a presentation on Friday about our quarterly results, and I want to practice the opening. "
             "It covers revenue growth, the new hires, and the product roadmap for next year."),
    ("long", "Okay, here is the thing. I have been thinking about switching careers for a while now. I studied "
             "biology, but I have spent the last three years teaching myself to code, and I built a couple of small apps."),
]
HOLD_AFTER_S = 0.35   # people let go of the key a beat after their last word


async def main():
    async with websockets.connect(WS, origin=ORIGIN, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "mode": "chat"}))
        rows = []
        for kind, text in TURNS:
            pcm = silence(0.3) + synth(text) + silence(HOLD_AFTER_S)
            await ws.send(json.dumps({"type": "rec_start"}))
            for i in range(0, len(pcm), 3200):            # 100 ms frames at live pace
                await ws.send(pcm[i:i + 3200])
                await asyncio.sleep(0.1)
            await ws.send(json.dumps({"type": "rec_stop"}))
            t0 = time.perf_counter()
            t_text = t_audio = None
            heard = ""
            while True:
                msg = await asyncio.wait_for(ws.recv(), 60)
                if isinstance(msg, bytes):
                    t_audio = t_audio or time.perf_counter() - t0
                    continue
                ev = json.loads(msg)
                if ev["type"] == "user_text":
                    t_text = time.perf_counter() - t0
                    heard = ev["text"]
                    print("     stt_ms", ev.get("stt_ms"))
                elif ev["type"] == "reply_end":
                    print("     server timing (ms from STT start):", ev.get("timing"), "| reply:", ev.get("text", "")[:70])
                    break
                elif ev["type"] in ("error", "notice"):
                    print("  !!", ev)
                    break
            # speech ends HOLD_AFTER_S before the release; report from the end of speech, as the user feels it
            rows.append((kind, t_text + HOLD_AFTER_S, t_audio + HOLD_AFTER_S if t_audio else None))
            print(f"  {kind:<5} {len(pcm) / 32000:4.1f}s audio: stop speaking -> transcript {t_text + HOLD_AFTER_S:.2f}s, "
                  f"-> first audio {t_audio + HOLD_AFTER_S if t_audio else float('nan'):.2f}s | heard {heard[:60]!r}")
            await asyncio.sleep(1.0)
        for kind in ("short", "long"):
            tt = [r[1] for r in rows if r[0] == kind]
            ta = [r[2] for r in rows if r[0] == kind and r[2]]
            print(f"{LABEL} {kind}: transcript median {statistics.median(tt):.2f}s, first audio median {statistics.median(ta):.2f}s")


asyncio.run(main())
