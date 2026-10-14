"""Stream a chat completion and report time-to-first-token and decode speed.

Usage: python3 test_llm.py [model] [runs]
"""
import json
import sys
import time

from _common import post_json

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3-it-4b-FLM"
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

MESSAGES = [
    {"role": "system", "content": "You are a friendly voice companion. Keep replies to 1-3 short spoken sentences."},
    {"role": "user", "content": "Hey! I just got back from a long walk by the river. What should I cook for dinner tonight?"},
]


def run_once():
    t0 = time.perf_counter()
    ttft = first_sentence = None
    text, chunks = "", 0
    with post_json("/chat/completions", {"model": MODEL, "messages": MESSAGES, "stream": True,
                                         "max_completion_tokens": 200, "temperature": 0.7}) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            delta = json.loads(data)["choices"][0].get("delta", {})
            piece = delta.get("content") or ""
            if piece:
                chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text += piece
                if first_sentence is None and any(p in piece for p in ".!?"):
                    first_sentence = time.perf_counter() - t0
    total = time.perf_counter() - t0
    tps = (chunks - 1) / (total - ttft) if chunks > 1 else 0
    return ttft, first_sentence, total, chunks, tps, text


if __name__ == "__main__":
    print(f"model={MODEL}")
    for i in range(RUNS):
        ttft, fs, total, n, tps, text = run_once()
        print(f"run {i+1}: TTFT={ttft:.3f}s  first-sentence={fs if fs is None else round(fs,3)}s  "
              f"total={total:.2f}s  chunks={n}  ~{tps:.1f} tok/s")
    print("reply:", text.strip())
