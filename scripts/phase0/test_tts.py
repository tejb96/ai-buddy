"""Synthesize speech through Lemonade and report latency (buffered and streaming).

Writes the buffered result to a WAV file so you can listen to it.
Usage: python3 test_tts.py [out.wav] [voice]
"""
import sys
import time

from _common import Timer, post_json

OUT = sys.argv[1] if len(sys.argv) > 1 else "tts_test.wav"
VOICE = sys.argv[2] if len(sys.argv) > 2 else "af_heart"
SHORT = "Sure, that sounds great!"
LONG = ("Hi there. Today I want to talk about why sleep matters so much for learning, "
        "and how a few small habits can make a big difference.")

if __name__ == "__main__":
    for label, text in (("short", SHORT), ("long", LONG)):
        with Timer() as t:
            with post_json("/audio/speech", {"model": "kokoro-v1", "input": text, "voice": VOICE,
                                             "response_format": "wav"}) as r:
                audio = r.read()
        secs = (len(audio) - 44) / 4 / 24000  # float32 mono @ 24 kHz
        print(f"buffered {label:<5}: {t.dt:.2f}s for {secs:.1f}s of audio (RTF {t.dt/secs:.2f})")
        if label == "long":
            with open(OUT, "wb") as f:
                f.write(audio)

    t0 = time.perf_counter()
    first = None
    total = 0
    with post_json("/audio/speech", {"model": "kokoro-v1", "input": LONG, "voice": VOICE,
                                     "response_format": "pcm", "stream_format": "audio"}) as r:
        while chunk := r.read(4800):
            first = first or time.perf_counter() - t0
            total += len(chunk)
    print(f"streaming pcm  : first audio after {first:.2f}s, done {time.perf_counter()-t0:.2f}s "
          f"({total/2/24000:.1f}s of 16-bit 24 kHz audio)")
    print(f"wrote {OUT}")
