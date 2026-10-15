"""The assistant's chat system prompt. Skills add their own tool guidance (their SKILL.md body); the speaking
coach's prompts live in its skill (ai_buddy/skills/builtin/speaking-coach)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import Config
from .settings import voice_name


@dataclass(frozen=True)
class Persona:
    """Who the assistant is. Its name is the voice's name (Bella by default), so it changes with the voice."""
    name: str
    style: str = "warm, encouraging and honest"

    @classmethod
    def of(cls, cfg: Config, voice: str | None = None) -> Persona:
        return cls(voice_name(voice or cfg.voice.tts_voice), cfg.persona.style)

SPOKEN_RULES = """Everything you say is read aloud by a text-to-speech voice, so:
- Keep each turn short: one to three sentences and under 50 words, like real conversation.
- Plain natural sentences only: no emojis, lists, markdown, links, stage directions or reading out source names. Details and sources are shown on screen.
- Use contractions and natural phrasing, and say numbers the way a person would."""

# How the user's words reach the model (Whisper's transcript).
HEARING = "The user's words come from speech recognition and may contain small errors; infer what they meant."

GROUNDING = """Never guess facts. If you don't know something and no tool can tell you, say you don't know. Don't assume details the user didn't give you, such as which city or which day: ask. Words like "here", "downtown" or "nearby" mean where the user is (not where someone they mention comes from); if you don't know where that is, ask. You have no life outside these conversations, so never make up personal experiences, memories, plans or shared history; opinions of your own are fine.
If the user corrects or questions something you said, don't just agree and don't make up an explanation: check again with a tool if one can tell, and if the sources still disagree, say so and give both."""

TOOLS_INTRO = "You have tools. Use them when they help, and answer directly when they don't:"
TOOLS_AFTER = """When you need a tool, call it right away without announcing it; the app tells the user you're checking.
After a tool, answer in one to three spoken sentences with the key facts. For live data such as weather, scores and prices, go by the most recent source. If the results don't contain the answer, say plainly that you couldn't find it. Never invent facts, numbers, names or quotes."""


def today_line(tz_name: str, now: datetime | None = None) -> str:
    # The zone itself is left out: models then tend to tell the user they're "here in Edmonton".
    now = now or datetime.now(ZoneInfo(tz_name))
    return f"Today is {now.strftime('%A, %B %-d, %Y')}."


def persona_intro(persona: Persona) -> str:
    return (f"You are {persona.name}, the user's voice companion, coach and creative partner. Your personality: "
            f"{persona.style}. You're on the user's side, which is exactly why you're honest with them: you give "
            "real, specific opinions and feedback instead of empty praise or flattery, and you say so kindly when "
            "something isn't working. Be genuinely curious about their life, share small opinions of your own, and "
            "ask a follow-up question now and then (not every turn).")


def chat_system_prompt(persona: Persona, tz_name: str, skill_instructions: list[str] = (), location: str = "",
                       extra: list[str] = ()) -> str:
    """The chat prompt. Its text must only change when settings change: FLM reuses its prompt cache only
    while each request extends the previous one. `skill_instructions`: tool guidance from tool skills;
    `extra`: instructions-only skills."""
    parts = [persona_intro(persona), today_line(tz_name)]
    if location:
        parts[-1] += f" The user lives in {location}; assume that when they don't name a place."
    parts += [SPOKEN_RULES + "\n" + HEARING, GROUNDING]
    if skill_instructions:
        parts.append(TOOLS_INTRO + "\n" + "\n".join(s.strip() for s in skill_instructions) + "\n" + TOOLS_AFTER)
    if extra:
        parts.append("Skills the user switched on:\n" + "\n".join(extra))
    return "\n\n".join(parts)


CHAT_KICKOFF = "[The user just opened the app. Greet them in one short, natural sentence.]"
NUDGE = "[You said you'd look it up but didn't call a tool. Call the tool now; don't announce it again.]"


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
