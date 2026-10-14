"""Check that STT (whisper-v3-turbo-FLM) and the chat LLM stay co-resident on the NPU.

Alternates STT -> LLM -> TTS turns and verifies the backend PIDs never change
(a change would mean Lemonade evicted and reloaded a model between turns).

Usage: python3 test_coresidency.py <wav16k> [stt_model] [llm_model] [turns]
"""
import json
import sys
import time

from _common import Timer, loaded_models, post_json, post_multipart

WAV = sys.argv[1]
STT = sys.argv[2] if len(sys.argv) > 2 else "whisper-v3-turbo-FLM"
LLM = sys.argv[3] if len(sys.argv) > 3 else "qwen3-it-4b-FLM"
TURNS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
TTS = "kokoro-v1"


def pids():
    return {k: v["pid"] for k, v in loaded_models().items()}


def llm_ttft(prompt):
    t0 = time.perf_counter()
    with post_json("/chat/completions", {"model": LLM, "stream": True, "max_completion_tokens": 40,
                                         "messages": [{"role": "user", "content": prompt}]}) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                if json.loads(line[5:])["choices"][0].get("delta", {}).get("content"):
                    ttft = time.perf_counter() - t0
                    for _ in r:  # drain the rest of the stream
                        pass
                    return ttft


def tts_first_audio(text):
    t0 = time.perf_counter()
    with post_json("/audio/speech", {"model": TTS, "input": text, "voice": "af_heart",
                                     "response_format": "pcm", "stream_format": "audio"}) as r:
        first = None
        while chunk := r.read(4096):
            if first is None:
                first = time.perf_counter() - t0
        return first, time.perf_counter() - t0


before = pids()
print("loaded before:", before)
for i in range(TURNS):
    with Timer() as t_stt:
        text = post_multipart("/audio/transcriptions", {"model": STT, "language": "en"}, "file", WAV)["text"]
    ttft = llm_ttft(f"Reply in one short sentence: {text}")
    first_audio, tts_total = tts_first_audio("Sure, that sounds like a great plan. Let's do it!")
    print(f"turn {i+1}: STT={t_stt.dt:.2f}s  LLM-TTFT={ttft:.2f}s  TTS-first-audio={first_audio:.2f}s "
          f"(TTS total {tts_total:.2f}s)")
after = pids()
print("loaded after: ", after)
stable = all(after.get(k) == v for k, v in before.items() if k in (STT, LLM))
print("CO-RESIDENT, NO RELOADS" if stable and STT in after and LLM in after else "MODEL WAS RELOADED/EVICTED")
