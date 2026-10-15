"""Skills: SKILL.md parsing and validation (Agent Skills spec), discovery, switching on and off."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_buddy.config import load_config
from ai_buddy.prompts import Persona, chat_system_prompt
from ai_buddy.skills import BUILTIN_DIR, MAX_PROMPT_CHARS, SkillError, SkillSet, discover, load_skill, parse_skill_md, \
    validate

TOOL_SKILL = '''---
name: coin
description: Flips a coin. Use when the user wants a random yes or no.
metadata:
  kira-label: Coin flip
---
- flip_coin when the user wants a random choice.
'''
TOOL_CODE = '''
from ai_buddy.tools import Tool, ToolResult

class Coin(Tool):
    name = "flip_coin"
    description = "Flip a coin."
    label = "Flipping a coin"

    async def run(self, args, ctx):
        return ToolResult({"result": "heads"}, summary="heads")
'''


def make_skill(base: Path, name: str, md: str, tools: str | None = None) -> Path:
    folder = base / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(md)
    if tools:
        (folder / "tools.py").write_text(tools)
    return folder


# ---------- SKILL.md ----------

def test_parse_front_matter_and_body():
    meta, body = parse_skill_md(TOOL_SKILL)
    assert meta["name"] == "coin" and meta["metadata"] == {"kira-label": "Coin flip"}
    assert body == "- flip_coin when the user wants a random choice."


@pytest.mark.parametrize("text", ["no front matter", "---\nname: x\n", "---\n: [\n---\nbody", "---\n- a list\n---\n"])
def test_parse_rejects_broken_front_matter(text):
    with pytest.raises(SkillError):
        parse_skill_md(text)


@pytest.mark.parametrize("name,ok", [("pdf-processing", True), ("a", True), ("x" * 64, True), ("x" * 65, False),
                                     ("PDF-Processing", False), ("-pdf", False), ("pdf-", False), ("pdf--x", False),
                                     ("pdf_x", False), ("", False)])
def test_name_rules(name, ok):
    assert (not validate({"name": name, "description": "Does a thing."})) == ok


def test_other_field_rules():
    assert validate({"name": "a", "description": "d"}, "b") == ["'name' 'a' must match the folder name 'b'"]
    assert validate({"name": "a"}) == ["'description' is required"]
    assert validate({"name": "a", "description": "d" * 1025})
    assert validate({"name": "a", "description": "d", "compatibility": "c" * 501})
    assert validate({"name": "a", "description": "d", "metadata": {"version": 1}})
    assert not validate({"name": "a", "description": "d", "metadata": {"version": "1"}, "license": "MIT",
                         "allowed-tools": "Read"})


# ---------- built-in skills ----------

def test_builtin_skills_follow_the_spec():
    for folder in sorted(p for p in BUILTIN_DIR.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
        meta, body = parse_skill_md((folder / "SKILL.md").read_text())
        assert validate(meta, folder.name) == [], folder.name
        assert body, f"{folder.name} has no instructions"


def test_builtin_skills_and_their_tools():
    skills = discover()
    assert set(skills) >= {"web-search", "notes", "clock", "calculator", "speaking-coach"}
    tools = {s.name: sorted(t.name for t in s.tools) for s in skills.values()}
    assert tools["web-search"] == ["fetch_page", "web_search"]
    assert tools["notes"] == ["search_notes"] and tools["clock"] == ["get_datetime"]
    assert tools["calculator"] == ["calculate"] and tools["speaking-coach"] == []
    assert skills["speaking-coach"].modes == ["presentation", "interview"]
    assert all(t.skill == s.name for s in skills.values() for t in s.tools)
    assert not any(t.confirm for s in skills.values() for t in s.tools)    # built-ins only read


def test_speaking_coach_prompts_come_from_the_skill():
    coach = discover()["speaking-coach"].modules["modes"]
    assert set(coach.SCENARIOS) == {"job_interview", "tough_conversation", "small_talk", "custom"}
    feedback = coach.section("Feedback", "Nova")
    assert feedback.startswith("You are Nova, an experienced speaking coach") and "\n" not in feedback
    prompt = coach.roleplay_system_prompt("job_interview", "data analyst", "Resume: five years of Python")
    assert "interviewing the user for: data analyst" in prompt and "Never mention being an AI" in prompt
    assert "five years of Python" in prompt
    metrics = {"speaking_time_s": 70, "words": 150, "wpm": 128, "pace": "good", "target_wpm": [120, 160],
               "fillers": {"total": 4, "per_min": 3.4, "by_word": {"um": 3, "like": 1}},
               "pauses": {"threshold_s": 2.0, "count": 1, "longest_s": 2.5, "at": [[30.0, 2.5, 1]]}}
    system, user = coach.feedback_messages("presentation", "Um, hello.", metrics, details="sleep", persona="Nova")
    assert system["content"].startswith("You are Nova") and "## Top 3 things to improve" in user["content"]
    assert '"um" x3' in user["content"] and "The talk is about: sleep." in user["content"]


# ---------- your own skills ----------

def test_user_skill_with_a_tool(tmp_path):
    folder = make_skill(tmp_path, "coin", TOOL_SKILL, TOOL_CODE)
    skill = load_skill(folder, "user")
    assert skill.label == "Coin flip" and skill.source == "user"
    assert [t.name for t in skill.tools] == ["flip_coin"] and skill.tools[0].skill == "coin"


def test_discover_skips_bad_and_clashing_skills(tmp_path):
    make_skill(tmp_path, "coin", TOOL_SKILL, TOOL_CODE)
    make_skill(tmp_path, "Bad Name", TOOL_SKILL.replace("name: coin", "name: Bad Name"))
    make_skill(tmp_path, "clock", "---\nname: clock\ndescription: A second clock.\n---\nbody")   # name taken
    make_skill(tmp_path, "my-calc", "---\nname: my-calc\ndescription: Clashes.\n---\n- calculate",
               TOOL_CODE.replace("flip_coin", "calculate"))                                         # tool name taken
    skills = discover([tmp_path])
    assert skills["coin"].source == "user" and skills["clock"].source == "built-in"
    assert "Bad Name" not in skills and "my-calc" not in skills


def test_instructions_only_skill_goes_into_the_chat_prompt(tmp_path):
    make_skill(tmp_path, "pirate", "---\nname: pirate\ndescription: Talk like a pirate.\n---\nSay arr.\nOften.")
    make_skill(tmp_path, "essay", "---\nname: essay\ndescription: Long.\n---\n" + "word " * 1000)
    skills = SkillSet(discover([tmp_path]), ["pirate", "essay", "clock"])
    extra = skills.extra_instructions()
    assert extra[1] == "- Pirate: Say arr. Often."
    assert len(extra[0]) <= MAX_PROMPT_CHARS + len("- Essay: ")
    cfg = load_config()
    prompt = chat_system_prompt(Persona.of(cfg), "UTC", skills.chat_instructions(), extra=extra)
    assert "Skills the user switched on:\n- Essay:" in prompt and "- Pirate: Say arr. Often." in prompt


# ---------- switching ----------

def test_skillset_enabled_overrides_and_listeners():
    skills = SkillSet(discover(), ["web-search", "clock", "speaking-coach"], {"clock": False, "calculator": True})
    assert [s.name for s in skills.enabled] == ["calculator", "speaking-coach", "web-search"]   # sorted: stable prompt
    assert set(skills.registry().tools) == {"calculate", "fetch_page", "web_search"}
    assert skills.modes() == ["presentation", "interview"] and skills.mode_skill("interview").name == "speaking-coach"
    calls = []
    skills.on_change(lambda: calls.append(1))
    skills.set_enabled("calculator", True)          # no change, no event
    skills.set_enabled("speaking-coach", False)
    assert calls == [1] and skills.mode_skill("presentation") is None and skills.modes() == []
    view = {v["name"]: v for v in skills.view()}
    assert view["clock"]["enabled"] is False and view["calculator"]["enabled"] is True


def test_prompt_is_stable_and_lists_enabled_tools_only():
    cfg = load_config()
    a = SkillSet(discover(), ["clock", "web-search"])
    b = SkillSet(discover(), ["web-search", "clock"])
    pa = chat_system_prompt(Persona.of(cfg), "UTC", a.chat_instructions())
    assert pa == chat_system_prompt(Persona.of(cfg), "UTC", b.chat_instructions())
    assert "get_datetime" in pa and "web_search" in pa and "search_notes" not in pa and "calculate" not in pa
