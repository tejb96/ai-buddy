"""Fixtures for end-to-end tests against a real Lemonade server.

Speech input is synthesized with Kokoro through Lemonade and resampled to 16 kHz with ffmpeg,
so these tests exercise the same STT -> LLM -> TTS path as a real microphone (minus the browser).
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import sys

import numpy as np
import pytest
import uvicorn
import websockets

from ai_buddy.config import McpServerCfg, load_config
from ai_buddy.server import create_app

FIXTURES = Path(__file__).parent / "fixtures"
NOTES = {
    "work/henderson-pitch.md": "# Henderson project: talking points\n- Open with the Henderson ops team story: they "
                               "saved ten hours a week after switching to automated reports.\n- Numbers: revenue up "
                               "12% quarter over quarter, churn down to 3%, NPS 61.\n- Ask: a two-year renewal.\n",
    "resume.md": "# Resume\n## Experience\nData analyst at Prairie Health (2021-2026): Python ETL pipelines with "
                 "pandas, and a FastAPI service for dashboards.\n",
    "personal/budget.md": "# Budget notes\nRent is 1,650 a month. Groceries target 500, eating out capped at 150.\n",
}

LEMONADE = "http://localhost:13305/v1"
_speech_cache: dict[tuple[str, str], bytes] = {}


def lemonade_up() -> bool:
    try:
        urllib.request.urlopen(f"{LEMONADE}/health", timeout=2)
        return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    if not lemonade_up():
        skip = pytest.mark.skip(reason="Lemonade server not running")
        for item in items:
            if "lemonade" in item.keywords:
                item.add_marker(skip)


def say(text: str, voice: str = "am_michael") -> bytes:
    """Speech for `text` as 16 kHz mono PCM16."""
    key = (text, voice)
    if key not in _speech_cache:
        req = urllib.request.Request(f"{LEMONADE}/audio/speech", data=json.dumps(
            {"model": "kokoro-v1", "input": text, "voice": voice, "response_format": "wav"}).encode(),
            headers={"Content-Type": "application/json"})
        wav = urllib.request.urlopen(req, timeout=120).read()
        _speech_cache[key] = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-f", "s16le", "pipe:1"],
            input=wav, capture_output=True, check=True).stdout
    return _speech_cache[key]


def silence(seconds: float) -> bytes:
    rng = np.random.default_rng(1)
    return (20 * rng.standard_normal(int(seconds * 16000))).astype("<i2").tobytes()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    """The real app on the configured models, with a sample knowledge folder and a test MCP server (a to-do list
    whose add_item tool asks before it acts)."""
    data = tmp_path_factory.mktemp("data")
    notes = tmp_path_factory.mktemp("notes")
    for rel, text in NOTES.items():
        (notes / rel).parent.mkdir(parents=True, exist_ok=True)
        (notes / rel).write_text(text)
    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    cfg.app.data_dir = str(data)
    cfg.app.port = free_port()
    cfg.app.open_window = False
    cfg.rag.folder = str(notes)
    cfg.mcp.servers = {"todo": McpServerCfg(command=sys.executable, description="The user's to-do list",
                                            args=[str(FIXTURES / "mcp_notes_server.py"), str(data / "todo.txt")])}
    cfg.skills.enabled = cfg.skills.enabled + ["todo"]
    srv = uvicorn.Server(uvicorn.Config(create_app(cfg), host="127.0.0.1", port=cfg.app.port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{cfg.app.port}"
    deadline = time.time() + 240
    while time.time() < deadline:
        try:
            status = json.load(urllib.request.urlopen(f"{base}/api/status", timeout=5))
            if status["warm"] and all(status["models"][r]["state"] == "ready" for r in ("stt", "llm", "tts")):
                break
        except OSError:
            pass
        time.sleep(1)
    else:
        pytest.fail("models never became ready")
    yield {"base": base, "ws": f"ws://127.0.0.1:{cfg.app.port}/ws", "cfg": cfg, "data": data, "notes": notes}
    srv.should_exit = True
    thread.join(timeout=10)


class Client:
    """Minimal stand-in for the browser: sends mic audio, collects JSON events and speech."""

    def __init__(self, ws):
        self.ws = ws
        self.events: list[dict] = []
        self.audio: dict[int, int] = {}       # reply id -> bytes of speech received
        self.first_audio_at: dict[int, float] = {}

    @classmethod
    async def connect(cls, server):
        origin = server["base"]
        ws = await websockets.connect(server["ws"], origin=origin, max_size=None)
        return cls(ws)

    async def send(self, **msg):
        await self.ws.send(json.dumps(msg))

    async def stream(self, pcm: bytes, realtime: float = 0.0):
        """Send PCM in 100 ms frames; `realtime` = 1.0 paces it like a live microphone."""
        step = 3200
        for i in range(0, len(pcm), step):
            await self.ws.send(pcm[i:i + step])
            if realtime:
                await asyncio.sleep(0.1 / realtime)

    async def until(self, pred, timeout: float = 120) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise TimeoutError
                msg = await asyncio.wait_for(self.ws.recv(), remaining)
            except TimeoutError:
                said = [e["text"] for e in self.events if e["type"] in ("user_text", "reply_end")][-4:]
                raise TimeoutError(f"timed out; last events: {[e.get('type') for e in self.events[-8:]]}; "
                                   f"last words: {said}") from None
            if isinstance(msg, bytes):
                rid = int.from_bytes(msg[:4], "little")
                self.audio[rid] = self.audio.get(rid, 0) + len(msg) - 4
                self.first_audio_at.setdefault(rid, time.monotonic())
                continue
            ev = json.loads(msg)
            self.events.append(ev)
            if ev["type"] == "error":
                raise AssertionError(f"server error: {ev['text']}")
            if pred(ev):
                return ev

    def of(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["type"] == kind]

    async def close(self):
        await self.ws.close()
