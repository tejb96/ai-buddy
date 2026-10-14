"""Transcribe a WAV file through Lemonade and report latency.

Lemonade's /audio/transcriptions only accepts WAV; 16 kHz mono PCM16 is safest.
Usage: python3 test_stt.py <file.wav> [model] [runs]
"""
import sys

from _common import Timer, post_multipart

WAV = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "whisper-v3-turbo-FLM"
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

if __name__ == "__main__":
    print(f"model={MODEL} file={WAV}")
    for i in range(RUNS):
        with Timer() as t:
            resp = post_multipart("/audio/transcriptions", {"model": MODEL, "language": "en"}, "file", WAV)
        print(f"run {i+1}: {t.dt:.2f}s")
    print("text:", resp["text"].strip())
