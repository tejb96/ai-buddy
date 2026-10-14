import io
import wave

import numpy as np

from ai_buddy.audio import (SAMPLE_RATE, PresentationChunker, analyze_speech, has_speech,
                            pcm_seconds, pcm_to_wav)

rng = np.random.default_rng(0)


def tone(seconds, amp=8000.0):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    # Amplitude-modulated tone: speech-like loudness with small natural dips.
    env = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 3 * t))
    return (amp * env * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds, amp=30.0):
    return (amp * rng.standard_normal(int(seconds * SAMPLE_RATE))).astype(np.float32)


def pcm(*parts):
    return np.concatenate(parts).clip(-32768, 32767).astype("<i2").tobytes()


def test_wav_roundtrip():
    data = pcm(tone(0.5))
    with wave.open(io.BytesIO(pcm_to_wav(data))) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert w.readframes(w.getnframes()) == data


def test_analyze_speech_finds_segments_and_long_pause():
    audio = pcm(silence(1.0), tone(3.0), silence(0.2), tone(2.0), silence(2.5), tone(1.5), silence(1.0))
    a = analyze_speech(audio)
    assert abs(a.duration_s - 11.2) < 0.05
    assert len(a.segments) == 2  # the 0.2 s gap is merged into one segment
    assert abs(a.segments[0][0] - 1.0) < 0.1 and abs(a.span_s - 9.2) < 0.15
    pauses = a.pauses(2.0)
    assert len(pauses) == 1 and abs(pauses[0][1] - 2.5) < 0.1


def test_silence_and_short_clicks_are_not_speech():
    assert not has_speech(pcm(silence(2.0)))
    assert not has_speech(pcm(silence(1.0), tone(0.06), silence(1.0)))
    assert has_speech(pcm(silence(0.5), tone(1.0), silence(0.5)))


def test_chunker_cuts_in_a_pause_after_target():
    c = PresentationChunker(target_s=10, max_s=14)
    audio = pcm(tone(6.5), silence(0.6), tone(5.0), silence(0.8), tone(4.0))
    chunks = []
    step = int(0.1 * SAMPLE_RATE) * 2
    for i in range(0, len(audio), step):
        chunks += c.feed(audio[i:i + step])
    chunks.append(c.flush())
    starts = [s for s, _ in chunks]
    lengths = [pcm_seconds(p) for _, p in chunks]
    assert abs(lengths[0] - 6.8) < 0.2  # cut in the middle of the first pause past 5 s
    assert starts[1] == lengths[0]
    assert abs(sum(lengths) - 16.9) < 0.01
    assert all(n <= 14 for n in lengths)


def test_chunker_forces_cut_without_pauses():
    c = PresentationChunker(target_s=10, max_s=12)
    chunks = c.feed(pcm(tone(30)))
    assert len(chunks) == 2  # ~6-12 s remain buffered until more audio or flush()
    assert all(9.0 <= pcm_seconds(p) <= 12.0 for _, p in chunks)
