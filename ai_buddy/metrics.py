"""Objective speaking metrics: filler words, pace, pauses, and rambling indicators."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .audio import SpeechAnalysis

HESITATIONS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "mm"}
_NORMALIZE = {"umm": "um", "uhm": "um", "uhh": "uh", "erm": "er", "mm": "hmm"}
CRUTCH_WORDS = {"basically", "literally"}
_DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her",
                "our", "their", "its", "to"}
_LIKE_FILLER_PREV = {"so", "and", "but", "was", "were", "i'm", "it's", "that's", "like", "the",
                     "with", "of", "just", "is", "are"} | HESITATIONS
_SO_FILLER_NEXT = {"yeah", "like", "basically", "anyway", "anyways", "um", "uh", "okay", "ok", "right"}
_YOU_KNOW_NOT_PREV = {"do", "did", "don't", "didn't", "if", "would", "as", "you", "they", "we", "i"}
_YOU_KNOW_NOT_NEXT = {"what", "how", "why", "where", "when", "who", "that", "if", "about", "the", "a",
                      "an", "him", "her", "them", "it", "me", "this", "those", "these", "anyone", "everything"}
_KIND_OF_NOT_PREV = {"what", "this", "that", "a", "the", "some", "any", "every", "which", "one", "same",
                     "different", "all", "those", "these", "other", "another", "certain", "new", "right"}
_WORD = re.compile(r"[A-Za-z']+|[.!?,;:]")
_STOPWORDS = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "is", "it",
              "that", "this", "i", "you", "we", "they", "be", "was", "are", "so", "like", "with", "my"}


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text.replace("’", "'"))]


def words(text: str) -> list[str]:
    return [t for t in _tokens(text) if t[0].isalpha()]


def count_fillers(text: str) -> Counter:
    """Count filler words/phrases, using context so "I like pizza" or "so much" are not fillers."""
    toks = _tokens(text)
    found: Counter = Counter()

    def at(i: int) -> str:
        return toks[i] if 0 <= i < len(toks) else ""

    def sentence_start(i: int) -> bool:
        return i == 0 or at(i - 1) in {".", "!", "?"}

    for i, t in enumerate(toks):
        prev, nxt = at(i - 1), at(i + 1)
        if t in HESITATIONS:
            found[_NORMALIZE.get(t, t)] += 1
        elif t in CRUTCH_WORDS:
            found[t] += 1
        elif t == "like":
            comma = prev == "," or nxt == ","
            if comma or sentence_start(i) or (prev in _LIKE_FILLER_PREV and nxt not in _DETERMINERS):
                found["like"] += 1
        elif t == "so":
            if nxt in _SO_FILLER_NEXT or nxt == "," or prev in HESITATIONS or (
                    sentence_start(i) and nxt not in {"much", "many", "far", "that", "long"}):
                found["so"] += 1
        elif t == "you" and nxt == "know":
            after = at(i + 2)
            if after in {",", ".", "?", "!", ""} or prev == "," or (
                    prev not in _YOU_KNOW_NOT_PREV and after not in _YOU_KNOW_NOT_NEXT):
                found["you know"] += 1
        elif t == "i" and nxt == "mean":
            if at(i + 2) not in {"it", "that", "to", "what"}:
                found["I mean"] += 1
        elif t in {"kind", "sort"} and nxt == "of":
            if prev not in _KIND_OF_NOT_PREV:
                found[f"{t} of"] += 1
    return found


def sentence_stats(text: str) -> dict | None:
    """Sentence-length stats, or None when the transcript has too little punctuation to trust."""
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    lengths = [len(words(s)) for s in sentences]
    total = sum(lengths)
    if not lengths or total < 20 or total / len(lengths) > 45:
        return None
    return {
        "count": len(lengths),
        "avg_words": round(total / len(lengths), 1),
        "longest_words": max(lengths),
        "long_count": sum(1 for n in lengths if n > 35),
    }


def repeated_phrases(text: str, n: int = 3, min_count: int = 3, top: int = 5) -> list[list]:
    """Phrases of n+ words said at least `min_count` times; overlapping n-grams merge into one phrase."""
    grams: Counter = Counter()
    for sentence in re.split(r"[.!?]+", text):  # phrases never span sentence breaks
        toks = [w for w in words(sentence) if w not in HESITATIONS]
        grams.update(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    kept: list[list] = []  # [words, count]
    for g, c in grams.most_common():
        if c < min_count:
            break
        if all(w in _STOPWORDS for w in g):
            continue
        for k in kept:
            if k[1] == c and list(g[:-1]) == k[0][-(n - 1):]:
                k[0].append(g[-1])
                break
            if k[1] == c and list(g[1:]) == k[0][:n - 1]:
                k[0].insert(0, g[0])
                break
        else:
            kept.append([list(g), c])
    return [[" ".join(w), c] for w, c in kept][:top]


@dataclass
class Answer:
    """One stretch of the user's speech (a whole presentation, or one interview answer)."""
    text: str
    speech: SpeechAnalysis
    start_s: float = 0.0  # offset of this audio within the session timeline


def pace_label(wpm: float, low: int, high: int) -> str:
    if wpm <= 0:
        return "n/a"
    return "slow" if wpm < low else "fast" if wpm > high else "good"


def compute_metrics(answers: list[Answer], *, long_pause_s: float = 2.0,
                    wpm_low: int = 120, wpm_high: int = 160) -> dict:
    text = " ".join(a.text for a in answers if a.text)
    n_words = len(words(text))
    span = sum(a.speech.span_s for a in answers)
    speech = sum(a.speech.speech_s for a in answers)
    fillers = count_fillers(text)
    n_fillers = sum(fillers.values())
    # (time in the session, length, which answer it happened in)
    pauses = [(round(a.start_s + t, 1), round(d, 1), i + 1)
              for i, a in enumerate(answers) for t, d in a.speech.pauses(long_pause_s)]
    wpm = n_words / (span / 60) if span > 0 else 0.0

    result = {
        "words": n_words,
        "speaking_time_s": round(span, 1),
        "wpm": round(wpm),
        "articulation_wpm": round(n_words / (speech / 60)) if speech > 0 else 0,
        "pace": pace_label(wpm, wpm_low, wpm_high),
        "target_wpm": [wpm_low, wpm_high],
        "fillers": {
            "total": n_fillers,
            "per_min": round(n_fillers / (span / 60), 1) if span > 0 else 0.0,
            "per_100_words": round(100 * n_fillers / n_words, 1) if n_words else 0.0,
            "by_word": dict(fillers.most_common()),
        },
        "pauses": {
            "threshold_s": long_pause_s,
            "count": len(pauses),
            "longest_s": max((d for _, d, _ in pauses), default=0.0),
            "total_s": round(sum(d for _, d, _ in pauses), 1),
            "at": pauses[:20],
        },
        "sentences": sentence_stats(text),
        "repeated_phrases": repeated_phrases(text),
    }
    if len(answers) > 1:
        result["answers"] = [{
            "words": len(words(a.text)),
            "seconds": round(a.speech.span_s, 1),
            "wpm": round(len(words(a.text)) / (a.speech.span_s / 60)) if a.speech.span_s > 0 else 0,
            "fillers": sum(count_fillers(a.text).values()),
        } for a in answers]
        lengths = [x["seconds"] for x in result["answers"]]
        result["longest_answer_s"] = max(lengths)
        result["avg_answer_s"] = round(sum(lengths) / len(lengths), 1)
    return result
