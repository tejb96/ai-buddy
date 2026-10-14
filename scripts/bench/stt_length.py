"""How does STT latency scale with clip length? Usage: python stt_length.py <model> [runs]"""
import sys

from common import dump, load, seconds, silence, synth, transcribe

MODEL = sys.argv[1]
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
TEXTS = [
    "Hi there.",
    "What time is it in Tokyo?",
    "Can you look up the weather in Calgary for tomorrow?",
    "I just got back from a long walk by the river and I'm wondering what I should cook for dinner tonight.",
    ("So I have a presentation on Friday about our quarterly results, and I want to practice the opening. "
     "It covers revenue growth, the new hires, and the product roadmap for next year."),
    ("Okay, here is the thing. I have been thinking about switching careers for a while now. I studied biology, "
     "but I have spent the last three years teaching myself to code, and I built a couple of small apps. "
     "I am not sure whether I should apply for junior developer jobs, or go back to school for a master's degree."),
]

print(f"load {MODEL}: {load(MODEL):.1f}s")
transcribe(silence(1.0), MODEL)  # warm-up
rows = []
for text in TEXTS:
    pcm = silence(0.2) + synth(text) + silence(0.3)
    times = []
    for _ in range(RUNS):
        hyp, dt = transcribe(pcm, MODEL)
        times.append(dt)
    rows.append({"audio_s": round(seconds(pcm), 2), "ms": [round(t * 1000) for t in times], "hyp": hyp})
    print(f"{seconds(pcm):5.1f}s audio -> {min(times):.2f}-{max(times):.2f}s  | {hyp[:80]}")
dump(f"stt_length_{MODEL}", rows)
