"""Audio input and FLM's prompt cache: what an audio turn costs in context, and which history format
keeps the cache. Usage: python gemma_audio_cache.py [llm]

FLM reuses its KV cache only when the previous request's prompt is a prefix of the new one. With
audio input, the open question is what goes into the history for earlier user turns: the audio
itself (append-only, but every clip stays in the context) or its transcript (shorter, but it
replaces what the model saw, so the prefix changes).
"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo root
from ai_buddy.config import load_config  # noqa: E402
from ai_buddy.prompts import chat_system_prompt  # noqa: E402

from common import http, load, loaded, seconds, silence, synth, wav  # noqa: E402

LLM = sys.argv[1] if len(sys.argv) > 1 else "gemma4-it-12b-FLM"
CFG = load_config()
SYSTEM = chat_system_prompt(CFG.persona, "America/Edmonton", hearing="audio")
SAYS = ["Hey Kira, I just finished a long run and I'm exhausted.",
        "About ten kilometers, along the river. It was beautiful.",
        "What should I eat to recover?",
        "Thanks. I think I'll make an omelette."]


def clip(text: str) -> bytes:
    return silence(0.25) + synth(text, "af_heart") + silence(0.35)


def audio(pcm: bytes) -> dict:
    return {"role": "user", "content": [{"type": "input_audio", "input_audio": {
        "data": base64.b64encode(wav(pcm)).decode(), "format": "wav"}}]}


def ask(messages: list[dict], max_tokens: int = 40) -> tuple[str, dict]:
    r = http.post("/chat/completions", json={"model": LLM, "messages": messages, "stream": False, "think": False,
                                              "temperature": 0.6, "max_completion_tokens": max_tokens})
    r.raise_for_status()
    d = r.json()
    time.sleep(2)
    return d["choices"][0]["message"].get("content") or "", d.get("usage", {})


def tokens(u: dict) -> str:
    keys = ("prompt_tokens", "completion_tokens", "prompt_tokens_details")
    return json.dumps({k: u[k] for k in keys if k in u})


if __name__ == "__main__":
    load(LLM, ctx_size=CFG.models.llm_ctx_size)
    print("loaded:", {k: v.get("pid") for k, v in loaded().items()})
    pcms = [clip(t) for t in SAYS]

    # 1. Context cost of one audio turn (fresh prompts: a nonce in the system prompt defeats the cache).
    print("== context cost")
    for i, pcm in enumerate(pcms[:3]):
        sys_n = f"[{time.time_ns()}] " + SYSTEM
        _, u_text = ask([{"role": "system", "content": sys_n}, {"role": "user", "content": SAYS[i]}], 1)
        sys_n = f"[{time.time_ns()}] " + SYSTEM
        _, u_audio = ask([{"role": "system", "content": sys_n}, audio(pcm)], 1)
        d = u_audio["prompt_tokens"] - u_text["prompt_tokens"]
        print(f"  {seconds(pcm):.1f} s clip: text {u_text['prompt_tokens']} vs audio {u_audio['prompt_tokens']} "
              f"prompt tokens -> audio costs ~{d + len(SAYS[i].split()) * 1.3:.0f} tokens "
              f"(~{(d + len(SAYS[i].split()) * 1.3) / seconds(pcm):.1f}/s)")

    # 2. History keeps the audio: is each new turn read incrementally?
    print("== history keeps the audio")
    msgs = [{"role": "system", "content": f"[{time.time_ns()}] " + SYSTEM}]
    for i, pcm in enumerate(pcms):
        msgs.append(audio(pcm))
        reply, u = ask(msgs)
        print(f"  turn {i + 1}: {tokens(u)}  {reply[:60]!r}")
        msgs.append({"role": "assistant", "content": reply})

    # 3. History replaces the audio with its transcript after each turn.
    print("== history replaces audio with text")
    msgs = [{"role": "system", "content": f"[{time.time_ns()}] " + SYSTEM}]
    for i, pcm in enumerate(pcms):
        msgs.append(audio(pcm))
        reply, u = ask(msgs)
        print(f"  turn {i + 1}: {tokens(u)}  {reply[:60]!r}")
        msgs[-1] = {"role": "user", "content": SAYS[i]}
        msgs.append({"role": "assistant", "content": reply})
    print("loaded:", {k: v.get("pid") for k, v in loaded().items()})
