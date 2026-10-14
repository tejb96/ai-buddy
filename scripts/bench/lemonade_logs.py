"""Dump recent Lemonade server log lines (via the /logs/stream WebSocket). Usage: python lemonade_logs.py [grep] [n]"""
import asyncio
import json
import re
import sys

import websockets

PAT = re.compile(sys.argv[1], re.I) if len(sys.argv) > 1 and sys.argv[1] else None
N = int(sys.argv[2]) if len(sys.argv) > 2 else 120


async def main():
    async with websockets.connect("ws://localhost:13305/v1/logs/stream", max_size=None) as ws:
        await ws.send(json.dumps({"type": "logs.subscribe"}))
        msg = json.loads(await asyncio.wait_for(ws.recv(), 10))
        entries = msg.get("entries") or msg.get("logs") or []
        lines = []
        for e in entries:
            text = e.get("line") or e.get("message") or e.get("text") or json.dumps(e)
            if PAT is None or PAT.search(text):
                lines.append(f"{e.get('timestamp', '')} {text}")
        print("\n".join(line[:400] for line in lines[-N:]))


asyncio.run(main())
