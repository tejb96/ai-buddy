"""The speaking coach's practice modes: role-play prompts for conversation practice and the feedback prompt.

The app's practice runtime (ai_buddy.session) records, transcribes with Whisper and measures delivery;
this module turns that into prompts. Instructions come from SKILL.md, scenarios from assets/scenarios.toml.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from ai_buddy.prompts import SPOKEN_RULES
from ai_buddy.skills import parse_skill_md

HERE = Path(__file__).parent
SCENARIOS: dict[str, dict] = tomllib.loads((HERE / "assets" / "scenarios.toml").read_text())
KICKOFF = "[The practice conversation starts now. Say your opening line.]"


def section(title: str, persona: str = "Bella") -> str:
    """One '## title' section of SKILL.md, with its wrapped lines joined into paragraphs."""
    _, body = parse_skill_md((HERE / "SKILL.md").read_text())
    m = re.search(rf"^## {re.escape(title)}\s*$(.*?)(?=^## |\Z)", body, re.M | re.S)
    text = m.group(1).strip() if m else ""
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    return "\n\n".join(paras).replace("{persona}", persona)


def roleplay_system_prompt(scenario: str, details: str, background: str = "", persona: str = "Bella") -> str:
    s = SCENARIOS[scenario]
    parts = [s["prompt"].format(details=details.strip() or s["default"]), section("Role-play", persona)]
    if background:
        parts.append("Background from the user's own documents (use it to make the conversation realistic and specific, "
                     "for example to ask about their real experience; never read it out or mention these notes):\n"
                     + background)
    parts.append(SPOKEN_RULES)
    return "\n\n".join(parts)


_FEEDBACK_FORMAT = """Write the feedback in Markdown with exactly these sections:

## Overall
One or two honest sentences with your overall impression.

## What worked
Two bullet points, each with specific evidence.

## Clarity & structure
Two to four sentences on clarity, structure, and any rambling, pointing to specific moments.{extra}
{talking_points}
## Delivery
Two or three sentences interpreting the measured pace, filler words, and pauses. Say which filler habit to work on first.

## Top 3 things to improve
1. **Short title**: concrete advice, with an example of a better way to say it.
2. **Short title**: ...
3. **Short title**: ...

Address the user as "you". Keep it under {words} words."""

_TALKING_POINTS_SECTION = """
## Talking points
Compare the talk with the user's talking points below. List each point that applies to this {kind} as a bullet marked **Covered**, **Partly** or **Missed**, with a short quote as evidence for covered ones. Ignore notes that clearly belong to something else; if none apply, say so in one sentence.
"""


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def _pause_places(m: dict) -> str:
    at = m["pauses"]["at"][:6]
    if not at:
        return ""
    if m.get("answers"):
        return " (" + ", ".join(f"{d} s in answer {a}" for _, d, a in at) + ")"
    return " (" + ", ".join(f"{d} s at {_clock(t)}" for t, d, _ in at) + " into the talk)"


def format_metrics(m: dict) -> str:
    mins, secs = divmod(int(m["speaking_time_s"]), 60)
    f = m["fillers"]
    lines = [
        f"- Speaking time: {mins} min {secs} s, {m['words']} words",
        f"- Pace: {m['wpm']} words per minute ({m['pace']}; comfortable range {m['target_wpm'][0]}-{m['target_wpm'][1]})",
        f"- Filler words: {f['total']} total, {f['per_min']} per minute"
        + (": " + ", ".join(f"\"{w}\" x{n}" for w, n in f["by_word"].items()) if f["by_word"] else ""),
        f"- Long pauses (at least {m['pauses']['threshold_s']:g} s): {m['pauses']['count']}"
        + (f", longest {m['pauses']['longest_s']} s" if m["pauses"]["count"] else "") + _pause_places(m),
    ]
    if m.get("sentences"):
        s = m["sentences"]
        lines.append(f"- Sentences: average {s['avg_words']} words, longest {s['longest_words']} words, "
                     f"{s['long_count']} longer than 35 words")
    if m.get("repeated_phrases"):
        lines.append("- Repeated phrases: " + ", ".join(f"\"{p}\" x{c}" for p, c in m["repeated_phrases"]))
    if m.get("answers"):
        lines.append(f"- Answers: {len(m['answers'])}, average {m['avg_answer_s']} s, longest {m['longest_answer_s']} s")
    return "\n".join(lines)


def feedback_messages(kind: str, transcript: str, metrics: dict, scenario: str | None = None, details: str = "",
                      talking_points: str = "", persona: str = "Bella") -> list[dict]:
    if kind == "presentation":
        intro = "The user just practiced a presentation, speaking without interruption."
        if details.strip():
            intro += f" The talk is about: {details.strip()}."
        extra = ""
    else:
        s = SCENARIOS[scenario or "custom"]
        intro = (f"The user just finished a practice conversation. Scenario: {s['title']}: "
                 f"{details.strip() or s['default']}. Lines marked \"You\" are the user; coach only the user.")
        extra = (" Also assess the quality of the user's answers: did they answer the actual question, give "
                 "concrete examples, and stay concise?") if scenario == "job_interview" else (
                 " Also assess how well the user handled the other person: listening, tone, and getting to the point.")
    tp_section = _TALKING_POINTS_SECTION.format(kind="talk" if kind == "presentation" else "conversation") \
        if talking_points else ""
    user = (f"{intro}\n\n## Measured delivery (computed from the audio; trust these numbers)\n{format_metrics(metrics)}"
            f"\n\n## Transcript\n{transcript}\n\n(The transcript comes from speech recognition: it can miss some "
            f"\"uh\" sounds and mishear words, and its punctuation and sentence breaks were added by the recognizer, "
            f"so don't judge sentence structure from punctuation alone. It has no timestamps, so only say where a "
            f"pause happened if the measurements above say so.)\n\n")
    if talking_points:
        user += f"## The user's talking points (from their notes)\n{talking_points}\n\n"
    user += _FEEDBACK_FORMAT.format(extra=extra, talking_points=tp_section, words=500 if talking_points else 400)
    return [{"role": "system", "content": section("Feedback", persona)}, {"role": "user", "content": user}]
