"""Can the LLM, an embedding model and Whisper stay loaded together? Alternate requests and watch PIDs.

Usage: python coresidency.py <llm> <stt> [ctx]
"""
import subprocess
import sys
import time

from common import http, load, loaded, silence, synth, transcribe

LLM, STT = sys.argv[1], sys.argv[2]
CTX = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
EMB = sys.argv[4] if len(sys.argv) > 4 else "embed-gemma-300m-FLM"


def embed(texts):
    t0 = time.perf_counter()
    r = http.post("/embeddings", json={"model": EMB, "input": texts})
    r.raise_for_status()
    d = r.json()["data"]
    return len(d), len(d[0]["embedding"]), time.perf_counter() - t0


def chat(prompt):
    t0 = time.perf_counter()
    r = http.post("/chat/completions", json={"model": LLM, "messages": [{"role": "user", "content": prompt}],
                                             "think": False, "max_completion_tokens": 20})
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content", "")[:40], time.perf_counter() - t0


def mem():
    out = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.split("\n")[1].split()
    return f"used {int(out[2]) / 1024:.1f} GB, available {int(out[6]) / 1024:.1f} GB"


for m, opts in ((EMB, {"llamacpp_backend": "vulkan"} if "GGUF" in EMB else {}), (LLM, {"ctx_size": CTX}), (STT, {})):
    print(f"load {m}: {load(m, **opts):.1f}s")
before = {k: v.get("pid") for k, v in loaded().items()}
print("loaded:", before, "|", mem())
pcm = silence(0.2) + synth("Can you look up the weather in Calgary for tomorrow?") + silence(0.3)
for i in range(4):
    t, dt_stt = transcribe(pcm, STT)
    n, dim, dt_emb = embed([f"chunk {j}: the quarterly review covered revenue, churn and hiring." for j in range(16)])
    txt, dt_llm = chat(f"Reply in five words: {t}")
    print(f"round {i + 1}: STT {dt_stt:.2f}s | embed 16x{dim} {dt_emb:.2f}s | LLM {dt_llm:.2f}s {txt!r}")
after = {k: v.get("pid") for k, v in loaded().items()}
print("loaded after:", after, "|", mem())
same = all(after.get(k) == v for k, v in before.items())
print("ALL CO-RESIDENT, NO RELOADS" if same and len(after) >= 4 else "SOMETHING WAS EVICTED/RELOADED")
