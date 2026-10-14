"""Turn a streaming LLM reply into speakable chunks, as early as possible."""

from __future__ import annotations

import re

_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.",
                  "e.g.", "i.e.", "a.m.", "p.m.", "approx.", "no."}
_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*\s")
_CLAUSE_END = re.compile(r"[,;:]\s|\s?[—–]\s?")  # models often write "amazing—coastal" with no spaces
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF"
    "\U0000FE0F\U0000200D\U000020E3]+"
)
_STAGE_WORDS = r"(?:smiles?|laughs?|chuckles?|grins?|nods?|sighs?|winks?|pauses?|giggles?|shrugs?|leans?|clears)"
_STAGE_DIRECTION = re.compile(rf"\*{_STAGE_WORDS}[^*\n]{{0,30}}\*|\({_STAGE_WORDS}[^)\n]{{0,30}}\)", re.I)
_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(\S(?:.*?\S)?)\1")


_UNITS = [(re.compile(r"\s?°\s?C\b"), " degrees Celsius"), (re.compile(r"\s?°\s?F\b"), " degrees Fahrenheit"),
          (re.compile(r"\s?°"), " degrees"), (re.compile(r"\bkm/h\b"), "kilometers per hour"),
          (re.compile(r"\bmph\b"), "miles per hour")]


def clean_for_speech(text: str) -> str:
    """Remove things a TTS voice should not read out: emojis, markdown, links, stage directions."""
    text = _STAGE_DIRECTION.sub(" ", text)
    text = re.sub(r"\[([^\]]+)\]\((?:https?://|/)[^)]*\)", r"\1", text)   # [label](url) -> label
    text = _EMPHASIS.sub(r"\2", text)
    text = _EMOJI.sub(" ", text)
    text = re.sub(r"\(?\s*(?:https?://|www\.)\S+?\s*\)?(?=[\s,.;!?]|$)", " ", text)
    for pattern, spoken in _UNITS:
        text = pattern.sub(spoken, text)
    text = re.sub(r"[*_#`>]+", " ", text)
    text = re.sub(r"^\s*(?:[-•]|\d+[.)])\s+", "", text, flags=re.M)
    text = text.replace("—", ", ").replace("–", ", ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text if re.search(r"[A-Za-z0-9]", text) else ""


def strip_emoji(text: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", _EMOJI.sub("", text)).strip()


class SpeechChunker:
    """Accumulates streamed tokens and releases text at sentence (or early clause) boundaries.

    The first chunk may end at a clause break once it has `first_min_words`, so the voice
    starts sooner; later chunks prefer whole sentences unless they grow past `max_words`.
    """

    def __init__(self, first_min_words: int = 3, max_words: int = 30):
        self.first_min_words = first_min_words
        self.max_words = max_words
        self.buf = ""
        self.emitted = 0

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out = []
        while (cut := self._find_cut()) is not None:
            piece, self.buf = self.buf[:cut], self.buf[cut:]
            if piece.strip():
                out.append(piece.strip())
                self.emitted += 1
        return out

    def flush(self) -> list[str]:
        piece, self.buf = self.buf.strip(), ""
        return [piece] if piece else []

    def _find_cut(self) -> int | None:
        for m in _SENTENCE_END.finditer(self.buf):
            words_before = self.buf[: m.start() + 1].split()
            if words_before and words_before[-1].lower() in _ABBREVIATIONS:
                continue
            if len(words_before) >= 2 or self.emitted == 0:
                return m.end()
        words = len(self.buf.split())
        if self.emitted == 0 and words >= self.first_min_words or words >= self.max_words:
            for m in _CLAUSE_END.finditer(self.buf):
                if len(self.buf[: m.start()].split()) >= self.first_min_words:
                    return m.end()
        return None
