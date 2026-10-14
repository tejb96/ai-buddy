"""System prompts for Kira, the role-play scenarios, and the feedback coach."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from .config import PersonaCfg

SPOKEN_RULES = """Everything you say is read aloud by a text-to-speech voice, so:
- Keep each turn short: one to three sentences and under 50 words, like real conversation.
- Plain spoken English only: no emojis, lists, markdown, URLs, stage directions or reading out source names.
- Use contractions and natural phrasing, and say numbers the way a person would.
The user's words come from speech recognition and may contain small errors; infer what they meant."""

TOOL_RULES = """You have tools. Use them when they help, and answer directly when they don't:
- web_search for anything current or that you aren't sure of: weather, news, prices, exchange rates, sports results, people, products, opening hours, events, or anything "latest". Your own knowledge stops in 2024, so don't answer those from memory.
- fetch_page to read a specific page, such as a link from your search results.
- get_datetime whenever the user asks what time or date it is anywhere, or how long until a date. Never work out a time yourself.
- calculate for arithmetic beyond the trivial, including percentages, tips and splitting bills.
- search_notes for anything about the user's own notes, talking points, resume or documents.
When you need a tool, call it right away without announcing it; the app tells the user you're checking.
After a tool, answer in one to three spoken sentences with the key facts. The sources are shown on screen, so don't read out links or site names. If the results don't answer the question, say so honestly instead of guessing.
Never invent facts, numbers, names or quotes."""


def today_line(tz_name: str, now: datetime | None = None) -> str:
    # The zone itself is left out: models then tend to tell the user they're "here in Edmonton".
    now = now or datetime.now(ZoneInfo(tz_name))
    return f"Today is {now.strftime('%A, %B %-d, %Y')}."


def persona_intro(persona: PersonaCfg) -> str:
    return (f"You are {persona.name}, the user's voice companion and speaking coach. Your personality: {persona.style}. "
            "You're on the user's side, which is exactly why you're honest with them: you give real, specific "
            "opinions and feedback instead of empty praise or flattery, and you say so kindly when something isn't "
            "working. Be genuinely curious about their life, share small opinions of your own, and ask a follow-up "
            "question now and then (not every turn).")


def chat_system_prompt(persona: PersonaCfg, tz_name: str, tools: bool = True) -> str:
    parts = [persona_intro(persona), today_line(tz_name), SPOKEN_RULES]
    if tools:
        parts.append(TOOL_RULES)
    return "\n\n".join(parts)


CHAT_KICKOFF = "[The user just opened the app. Greet them in one short, natural sentence.]"
NUDGE = "[You said you'd look it up but didn't call a tool. Call the tool now; don't announce it again.]"


SCENARIOS: dict[str, dict] = {
    "job_interview": {
        "title": "Job interview",
        "blurb": "A hiring manager interviews you for a role you choose.",
        "placeholder": "Role and company, e.g. junior data analyst at a healthcare startup",
        "default": "a professional role that fits the candidate's background",
        "counterpart": "Interviewer",
        "notes_query": "resume experience skills projects",
        "prompt": ("You are Alex, a hiring manager interviewing the user for: {details}. Start by greeting them and "
                   "asking them to introduce themselves. Over the conversation ask a realistic mix of questions: "
                   "background, a behavioral question (\"tell me about a time when...\"), a role-specific question, "
                   "and how they handle challenges or failure. Ask one question at a time and probe vague answers "
                   "with a follow-up. After about six questions, ask whether they have questions for you."),
    },
    "tough_conversation": {
        "title": "Tough conversation",
        "blurb": "Practice a difficult talk: asking for a raise, pushing back, giving hard feedback.",
        "placeholder": "Who it's with and what it's about, e.g. telling my manager I'm overloaded",
        "default": "telling their manager that they are overloaded and need to drop a project",
        "counterpart": "Other person",
        "notes_query": "talking points",
        "prompt": ("The user wants to practice a difficult conversation: {details}. You play the other person. "
                   "Be realistic: a little resistant, with genuine concerns and objections, but not hostile. React "
                   "emotionally the way a real person would and let the user steer. Open with one short, natural "
                   "line that sets the scene, such as \"Hey, you wanted to talk?\""),
    },
    "small_talk": {
        "title": "Small talk",
        "blurb": "Chat with a friendly stranger at an event and keep the conversation going.",
        "placeholder": "Setting, e.g. a tech meetup, a wedding, a conference coffee break",
        "default": "a professional networking event",
        "counterpart": "Stranger",
        "notes_query": "",
        "prompt": ("You are Jordan, a friendly stranger the user meets at {details}. Invent a plausible life for "
                   "yourself (job, hobbies, why you are here) and share small details. Make light small talk, ask "
                   "questions, and keep it flowing naturally, but leave room for the user to lead. Open by "
                   "introducing yourself."),
    },
    "custom": {
        "title": "Custom scenario",
        "blurb": "Describe any situation and who Kira should play.",
        "placeholder": "e.g. You're a customer returning a broken laptop and I'm the store clerk",
        "default": "a casual conversation between colleagues",
        "counterpart": "Partner",
        "notes_query": "talking points",
        "prompt": "Role-play this scenario with the user: {details}. You play the other person. Open the conversation naturally.",
    },
}

ROLEPLAY_RULES = """Stay fully in character the whole time. Never mention being an AI, and never coach or give feedback during the conversation."""

KICKOFF = "[The practice conversation starts now. Say your opening line.]"


def notes_block(hits, max_chars: int = 2500) -> str:
    """Excerpts from the user's knowledge folder, formatted for a prompt."""
    out, used = [], 0
    for h in hits:
        where = h.file + (f" › {h.heading}" if h.heading else "") + (f" (page {h.page})" if h.page else "")
        piece = f"[{where}]\n{h.text.strip()}"
        if used + len(piece) > max_chars:
            piece = piece[: max(0, max_chars - used)]
        if piece:
            out.append(piece)
            used += len(piece)
        if used >= max_chars:
            break
    return "\n\n".join(out)


def roleplay_system_prompt(scenario: str, details: str, background: str = "") -> str:
    s = SCENARIOS[scenario]
    parts = [s["prompt"].format(details=details.strip() or s["default"]), ROLEPLAY_RULES]
    if background:
        parts.append("Background from the user's own documents (use it to make the conversation realistic and specific, "
                     "for example to ask about their real experience; never read it out or mention these notes):\n"
                     + background)
    parts.append(SPOKEN_RULES)
    return "\n\n".join(parts)


FEEDBACK_SYSTEM = (
    "You are Kira, an experienced speaking coach: warm and encouraging, but honest and specific. You never give "
    "generic praise: every compliment and every criticism points to something the user actually said, quoting short "
    "phrases as evidence. When something didn't work, say so plainly and show a better way to say it. Ground your "
    "comments in the transcript and the measured numbers you are given. Never invent numbers; use only the "
    "measurements provided."
)

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


def _format_metrics(m: dict) -> str:
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


def feedback_messages(kind: str, transcript: str, metrics: dict, scenario: str | None = None,
                      details: str = "", talking_points: str = "") -> list[dict]:
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
    user = (f"{intro}\n\n## Measured delivery (computed from the audio; trust these numbers)\n{_format_metrics(metrics)}"
            f"\n\n## Transcript\n{transcript}\n\n(The transcript comes from speech recognition: it can miss some "
            f"\"uh\" sounds and mishear words, and its punctuation and sentence breaks were added by the recognizer, "
            f"so don't judge sentence structure from punctuation alone. It has no timestamps, so only say where a "
            f"pause happened if the measurements above say so.)\n\n")
    if talking_points:
        user += f"## The user's talking points (from their notes)\n{talking_points}\n\n"
    user += _FEEDBACK_FORMAT.format(extra=extra, talking_points=tp_section, words=500 if talking_points else 400)
    return [{"role": "system", "content": FEEDBACK_SYSTEM}, {"role": "user", "content": user}]
