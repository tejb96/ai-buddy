"""Smoke test for Lemonade + FLM after an upgrade: load Kira's default models, one spoken turn, one tool call.

Usage: .venv/bin/python scripts/flm_smoke.py      (exit code 0 = passed)
Reads the models from config.toml. Talks to the running Lemonade server only; changes nothing.
"""
from __future__ import annotations

import asyncio
import io
import subprocess
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_buddy.config import load_config  # noqa: E402
from ai_buddy.lemonade import Lemonade, LemonadeError  # noqa: E402
from ai_buddy.prompts import chat_system_prompt  # noqa: E402
from ai_buddy.tools import ToolRegistry  # noqa: E402

SPOKEN = "Hey Kira, what's a good name for a grey cat?"
TOOL_PROMPT = "What time is it in Tokyo right now?"


def to_wav16k(wav24: bytes) -> bytes:
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-f", "s16le",
                          "pipe:1"], input=wav24, capture_output=True, check=True).stdout
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return buf.getvalue()


async def main() -> int:
    cfg = load_config()
    lm = Lemonade(cfg.lemonade.base_url, cfg.lemonade.load_timeout_s)
    m = cfg.models
    failures: list[str] = []

    def check(ok: bool, what: str, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {what}{': ' + detail if detail else ''}")
        if not ok:
            failures.append(what)

    try:
        for name, ctx in ((m.stt, None), (m.tts, None), (m.llm, m.llm_ctx_size)):
            try:
                await lm.load(name, ctx)
                check(True, f"load {name}")
            except LemonadeError as e:
                check(False, f"load {name}", str(e)[:200])
        if failures:
            return 1
        loaded = await lm.loaded_models()
        check(all(n in loaded for n in (m.stt, m.llm, m.tts)), "all three models stay loaded together",
              ", ".join(loaded))

        # One spoken turn: Kokoro speech -> Whisper -> LLM reply.
        r = await lm._http.post("/audio/speech", json={"model": m.tts, "input": SPOKEN, "voice": "am_michael",
                                                       "response_format": "wav"})
        check(r.status_code == 200 and len(r.content) > 10000, "text-to-speech")
        heard = await lm.transcribe(to_wav16k(r.content), m.stt)
        check("cat" in heard.lower(), "speech-to-text", repr(heard))
        system = {"role": "system", "content": chat_system_prompt(cfg.persona, "UTC")}
        reply = await lm.chat(m.llm, [system, {"role": "user", "content": heard or SPOKEN}], think=False,
                              temperature=0.3, max_tokens=120)
        check(len(reply.split()) >= 3 and "<think" not in reply, "spoken reply", repr(reply[:120]))

        # One tool call (blank names are repaired from the arguments, as in the app).
        reg = ToolRegistry.from_config(cfg)
        got = []
        for _ in range(2):
            async for ev in lm.chat_events(m.llm, [system, {"role": "user", "content": TOOL_PROMPT}],
                                           tools=reg.schemas(), think=False, temperature=0.3, max_tokens=200):
                if ev["type"] == "tool_calls":
                    got = [reg.parse(c) for c in ev["calls"]]
            if got:
                break
        names = [c.name for c in got if not c.error]
        check(names == ["get_datetime"], "tool call", f"{names} (raw: {[c.raw_arguments for c in got]})")
    except Exception as e:  # noqa: BLE001 - any error fails the smoke test
        check(False, "unexpected error", f"{e.__class__.__name__}: {e}")
    finally:
        await lm.close()
    print("smoke test", "FAILED: " + ", ".join(failures) if failures else "passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
