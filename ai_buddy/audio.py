"""PCM helpers: WAV encoding, energy-based speech/pause detection, and presentation chunking.

All microphone audio in the app is 16 kHz mono signed 16-bit little-endian PCM.
FLM's Whisper returns no timestamps, so pauses are measured from the audio itself.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480


def pcm_seconds(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> float:
    return len(pcm) / BYTES_PER_SAMPLE / sample_rate


def pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(BYTES_PER_SAMPLE)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def frame_db(pcm: bytes) -> np.ndarray:
    """Loudness (dBFS) of each 30 ms frame."""
    samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32)
    n = len(samples) // FRAME_SAMPLES
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = samples[: n * FRAME_SAMPLES].reshape(n, FRAME_SAMPLES)
    rms = np.sqrt(np.mean(frames * frames, axis=1)) / 32768.0
    return 20 * np.log10(rms + 1e-9)


def speech_threshold(db: np.ndarray) -> float | None:
    """dB level separating speech from background, or None if the clip has no clear speech."""
    if len(db) == 0:
        return None
    floor, peak = np.percentile(db, 10), np.percentile(db, 97)
    if peak < -50 or peak - floor < 12:
        return None  # silence, or no contrast between speech and background
    return float(max(floor + 10, floor + 0.3 * (peak - floor)))


@dataclass
class SpeechAnalysis:
    segments: list[tuple[float, float]]  # (start_s, end_s) of speech
    duration_s: float                    # whole recording

    @property
    def speech_s(self) -> float:
        return sum(e - s for s, e in self.segments)

    @property
    def span_s(self) -> float:
        """First speech onset to last speech offset (leading/trailing silence excluded)."""
        return self.segments[-1][1] - self.segments[0][0] if self.segments else 0.0

    def pauses(self, min_s: float) -> list[tuple[float, float]]:
        """(start_s, length_s) of silences between speech segments lasting at least `min_s`."""
        return [(a_end, b_start - a_end)
                for (_, a_end), (b_start, _) in zip(self.segments, self.segments[1:])
                if b_start - a_end >= min_s]


def analyze_speech(pcm: bytes, merge_gap_s: float = 0.3, min_speech_s: float = 0.12) -> SpeechAnalysis:
    db = frame_db(pcm)
    duration = pcm_seconds(pcm)
    thr = speech_threshold(db)
    if thr is None:
        return SpeechAnalysis([], duration)
    frame_s = FRAME_MS / 1000
    segments: list[list[float]] = []
    for i, is_speech in enumerate(db > thr):
        if not is_speech:
            continue
        t = i * frame_s
        if segments and t - segments[-1][1] <= merge_gap_s:
            segments[-1][1] = t + frame_s
        else:
            segments.append([t, t + frame_s])
    kept = [(round(s, 3), round(e, 3)) for s, e in segments if e - s >= min_speech_s]
    return SpeechAnalysis(kept, duration)


def has_speech(pcm: bytes, min_speech_s: float = 0.25) -> bool:
    return analyze_speech(pcm).speech_s >= min_speech_s


class PresentationChunker:
    """Split a live recording into ~`target_s` chunks, cutting in quiet moments.

    Whisper encodes 30 s windows, so chunks are capped at `max_s` and cut at the first
    silence of at least `quiet_s` once a chunk is `target_s - 5` seconds long.
    """

    def __init__(self, target_s: float = 25, max_s: float = 28.5, quiet_s: float = 0.3):
        self.min_bytes = int(max(target_s - 5, 5) * SAMPLE_RATE) * BYTES_PER_SAMPLE
        self.max_bytes = int(max_s * SAMPLE_RATE) * BYTES_PER_SAMPLE
        self.quiet_frames = max(1, int(quiet_s * 1000 / FRAME_MS))
        self.buf = bytearray()
        self.offset_s = 0.0  # recording time at which `buf` starts

    def feed(self, pcm: bytes) -> list[tuple[float, bytes]]:
        """Add audio; return any finished chunks as (start_s, pcm)."""
        self.buf += pcm
        out = []
        while (cut := self._find_cut()) is not None:
            out.append(self._take(cut))
        return out

    def flush(self) -> tuple[float, bytes] | None:
        if not self.buf:
            return None
        return self._take(len(self.buf))

    def _take(self, nbytes: int) -> tuple[float, bytes]:
        chunk = bytes(self.buf[:nbytes])
        del self.buf[:nbytes]
        start = self.offset_s
        self.offset_s += pcm_seconds(chunk)
        return start, chunk

    def _find_cut(self) -> int | None:
        if len(self.buf) < self.min_bytes:
            return None
        db = frame_db(bytes(self.buf[: self.max_bytes]))
        thr = speech_threshold(db)
        first = self.min_bytes // (FRAME_SAMPLES * BYTES_PER_SAMPLE)
        if thr is not None:
            run = 0
            for i in range(first, len(db)):
                run = run + 1 if db[i] < thr else 0
                if run >= self.quiet_frames:
                    middle = i - run // 2
                    return middle * FRAME_SAMPLES * BYTES_PER_SAMPLE
        if len(self.buf) >= self.max_bytes:
            # No pause found: cut at the quietest frame of the last 3 s before the cap,
            # keeping chunks long (every Whisper call costs the same ~2 s).
            lo = max(first, len(db) - int(3000 / FRAME_MS))
            quietest = lo + int(np.argmin(db[lo:]))
            return max(quietest, 1) * FRAME_SAMPLES * BYTES_PER_SAMPLE
        return None
