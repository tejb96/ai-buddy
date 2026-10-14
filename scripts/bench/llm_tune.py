"""Quick TTFT probe under different FLM launch args. Usage: python llm_tune.py <model> <ctx> '<flm_args>'"""
import json
import sys
import time

from common import http, unload
from llm_bench import CHAT_TURNS, SYSTEM, TOOLS

MODEL, CTX, ARGS = sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else ""


def first_token(messages):
    t0 = time.perf_counter()
    text, usage = "", {}
    with http.stream("POST", "/chat/completions", json={"model": MODEL, "messages": messages, "tools": TOOLS,
                                                         "stream": True, "think": False, "temperature": 0.6,
                                                         "max_completion_tokens": 60}) as r:
        ft = None
        for line in r.iter_lines():
            if line.startswith("data:") and "[DONE]" not in line:
                d = json.loads(line[5:])
                usage = d.get("usage") or usage
                for ch in d.get("choices", []):
                    c = ch.get("delta", {}).get("content")
                    if c:
                        text += c
                        ft = ft or time.perf_counter() - t0
    return ft, text, usage


unload(MODEL)
t0 = time.perf_counter()
body = {"model_name": MODEL, "ctx_size": CTX}
if ARGS:
    body["flm_args"] = ARGS
r = http.post("/load", json=body, timeout=600)
print(f"{MODEL} args={ARGS!r}: load {time.perf_counter() - t0:.1f}s {r.status_code}")
first_token([{"role": "user", "content": "Hi"}])
msgs = [{"role": "system", "content": SYSTEM}]
for i, user in enumerate(CHAT_TURNS[:4]):
    msgs.append({"role": "user", "content": user})
    ft, text, u = first_token(msgs)
    print(f"  turn {i + 1}: first token {ft:.2f}s (new prompt tokens {u.get('prompt_tokens')}, "
          f"prefill {u.get('prefill_speed_tps', 0):.0f} tok/s, decode {u.get('decoding_speed_tps', 0):.1f} tok/s)")
    msgs.append({"role": "assistant", "content": text or "Okay."})
