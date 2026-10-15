"""Make the right tool call in code when the user's intent is unambiguous.

The model is good at choosing tools, but not perfect: after a tool turn the 35B sometimes answered "what
time is it in Tokyo" from memory (and got it wrong), and Gemma searched Paris' weather but skipped the clock
for "and what time is it there". Forcing the call through the API isn't possible: FLM ignores
`tool_choice` (tested on FLM 1.0.6). So the app makes such calls itself, without extra LLM calls: Whisper's
transcript is routed before the model runs, and the routed calls and their results go into the history before
the first request, so the model answers from them in one round instead of two.

Each tool decides what it claims (`Tool.route`); patterns are strict, so anything less clear is left to the
model and its instructions.

The same goes for re-checking (`recheck`): when the user questions an answer that came from tools ("isn't it 16
degrees and feels like 12?", "are you sure?"), the model must not simply agree or invent a reason for the
difference. The app makes that turn's read-only calls again (a web search reads one more source), and the
results go back with an instruction to compare them honestly with what the user said.
"""

from __future__ import annotations

import json
import logging
import re

from .prompts import NUDGE
from .tools import RECHECK_SUFFIX, RESULTS_PREFIX, ToolCall, ToolContext, ToolRegistry, ToolResult

log = logging.getLogger(__name__)


def route(text: str, registry: ToolRegistry, ctx: ToolContext) -> list[ToolCall]:
    """The tool calls this request unambiguously needs (usually none)."""
    calls = []
    for tool in registry.tools.values():
        try:
            args = tool.route(text, ctx)
        except Exception:
            log.exception("routing with %s failed", tool.name)
            continue
        if args is not None:
            calls.append(ToolCall(id=f"route_{tool.name}_{len(calls)}", name=tool.name, args=args,
                                  raw_arguments=json.dumps(args), routed=True))
    return calls


def said_by_user(msg: dict) -> bool:
    """Something the user said (not tool results or a nudge the app added)."""
    content = str(msg.get("content") or "")
    return msg["role"] == "user" and not content.startswith(RESULTS_PREFIX) and content != NUDGE


# Explicit doubt: "are you sure?", "that's wrong", "can you double-check?"
_DOUBT = re.compile(r"\b(?:are you (?:sure|certain)|you sure\b|(?:that|this)(?:'s| is) (?:not right|wrong|incorrect|"
                    r"not correct|off|not true)|that doesn'?t (?:sound|seem|look) right|double[- ]?check|check (?:that |"
                    r"it |this )?again|look (?:that |it )?up again)", re.I)
# A counter-claim: "isn't it 16 degrees?", "I thought it was 3:30", "my app says 12", "actually it's 16".
_CLAIM = re.compile(r"\b(?:is|are|was|were|does|did|do|has|have|should|would|could|wo|ca)n'?t (?:it|that|they|there|this)\b"
                    r"|\bis it not\b|\bi (?:thought|heard|read|saw)\b|\b(?:my|the) (?:\w+ )?(?:app|phone|watch|forecast|"
                    r"news|website|site|google)\b[^.?!]{0,40}\b(?:says|said|shows|showed|has)\b|\bactually,? (?:it'?s|"
                    r"it is|they|there)\b|^\W*no,? it'?s\b|\bit'?s actually\b", re.I)
_FIGURE = re.compile(r"\d|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
                     r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|"
                     r"ninety|hundred|thousand|half|minus)\b", re.I)


def questions_a_fact(text: str) -> bool:
    """Is the user disputing or correcting something? Explicit doubt always counts; a counter-claim only with a
    figure in it ("isn't it 16 degrees?"), so "isn't it lovely" doesn't."""
    return bool(_DOUBT.search(text) or (_CLAIM.search(text) and _FIGURE.search(text)))


def recheck(text: str, history: list[dict], registry: ToolRegistry) -> list[ToolCall]:
    """The calls to make again when the user questions an answer that came from tools: the read-only calls of
    the previous exchange (`Tool.recheck` decides; a web search reads one more source). `history` ends with
    the assistant's last answer, before the user's new words."""
    if not questions_a_fact(text):
        return []
    calls, seen = [], set()
    for msg in reversed(history):
        if said_by_user(msg):
            break
        for c in reversed(msg.get("tool_calls") or []):
            fn = c.get("function") or {}
            tool = registry.get(fn.get("name", ""))
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            again = tool.recheck(args) if tool and isinstance(args, dict) else None
            key = (fn.get("name"), json.dumps(again, sort_keys=True))
            if again is None or key in seen:
                continue
            seen.add(key)
            calls.append(ToolCall(id=f"recheck_{tool.name}_{len(calls)}", name=tool.name, args=again,
                                  raw_arguments=json.dumps(again), routed=True))
    return calls[::-1]


_UNITS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
         "seventeen eighteen nineteen".split()
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_NUMBER = re.compile(r"\d+(?::\d{2}|[.,]\d+)?|\b(?:" + "|".join(_TENS) + r")(?:[- ](?:" + "|".join(_UNITS[1:10]) +
                     r"))?\b|\b(?:" + "|".join(_UNITS) + r")\b", re.I)


def figures(text: str) -> list[str]:
    """The numbers in what the user said, as digits ("sixteen" -> "16", "3:30" stays)."""
    out = []
    for m in _NUMBER.finditer(text):
        word = m.group(0).lower()
        if word[0].isdigit():
            out.append(word.replace(",", "") if re.fullmatch(r"\d{1,3}(?:,\d{3})+", word) else word)
        elif word in _UNITS:
            out.append(str(_UNITS.index(word)))
        else:
            tens, _, unit = word.replace("-", " ").partition(" ")
            out.append(str(_TENS[tens] + (_UNITS.index(unit) if unit else 0)))
    return list(dict.fromkeys(out))


_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"


def _shows(figure: str, data: str) -> bool:
    """Whether `figure` appears in `data` as a figure, not as part of a date ("Fri Oct 16") or a time ("12am")."""
    for m in re.finditer(rf"(?<![\d.:]){re.escape(figure)}(?![\d]|[.,:]\d)", data):
        before, after = data[max(0, m.start() - 12):m.start()], data[m.end():m.end() + 8]
        if ":" not in figure and (re.search(rf"\b{_MONTH}\s*$", before, re.I) or
                                  re.match(rf"\s*(?:[ap]\.?m\b|{_MONTH}\b)", after, re.I)):
            continue
        return True
    return False


def recheck_note(claim: str, results: list[ToolResult]) -> str:
    """What the re-checked results go back with: the comparison instruction, plus the user's figures that appear
    nowhere in them, so the model can't credit a source with the user's number (seen with qwen3.5-9b: "You're
    right, The Weather Network shows 16" when no result said 16)."""
    data = " ".join(json.dumps(r.content, ensure_ascii=False) for r in results)
    absent = [f for f in figures(claim) if not _shows(f, data)]
    if not absent:
        return RECHECK_SUFFIX
    which = f"figure {absent[0]}" if len(absent) == 1 else "figures " + ", ".join(absent)
    return (f"{RECHECK_SUFFIX} The user's {which} {'is' if len(absent) == 1 else 'are'} not in any of these results: "
            f"don't say a source shows {'it' if len(absent) == 1 else 'them'}.")
