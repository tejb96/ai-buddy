"""Probe Lemonade's /realtime WebSocket: stream speech at real-time pace, then commit, and time each event.

Usage: python stt_realtime.py <model> [vad|manual]
"""
import asyncio
import base64
import json
import sys
import time

import websockets

from common import load, seconds, silence, synth

MODEL = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "vad"
TEXT = ("So I have a presentation on Friday about our quarterly results, and I want to practice the opening. "
        "It covers revenue growth, the new hires, and the product roadmap for next year.")


async def main():
    load(MODEL)
    pcm = silence(0.3) + synth(TEXT) + silence(0.4)
    print(f"audio {seconds(pcm):.1f}s, mode={MODE}")
    async with websockets.connect(f"ws://localhost:13305/v1/realtime?model={MODEL}", max_size=None) as ws:
        print("<-", (await ws.recv())[:200])
        if MODE == "manual":
            await ws.send(json.dumps({"type": "session.update", "session": {"model": MODEL, "turn_detection": None}}))
        t0 = time.perf_counter()
        events = []

        async def reader():
            async for msg in ws:
                ev = json.loads(msg)
                t = time.perf_counter() - t0
                events.append((t, ev))
                txt = ev.get("delta") or ev.get("transcript") or ev.get("error") or ""
                print(f"  {t:6.2f}s {ev['type']:<55} {str(txt)[:90]}")

        rtask = asyncio.create_task(reader())
        step = 2720  # 85 ms of 16 kHz PCM16
        for i in range(0, len(pcm), step):
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(pcm[i:i + step]).decode()}))
            await asyncio.sleep(0.085)
        t_end = time.perf_counter() - t0
        print(f"  {t_end:6.2f}s -- audio finished, committing")
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        try:
            await asyncio.wait_for(rtask, timeout=15)
        except asyncio.TimeoutError:
            pass
        done = [t for t, e in events if e["type"].endswith("transcription.completed")]
        if done:
            print(f"last completed event {done[-1] - t_end:.2f}s after end of audio")


asyncio.run(main())
