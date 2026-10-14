"""Kira's skills: every capability is a skill that can be switched on or off.

A skill is a folder in the Agent Skills format (https://agentskills.io/specification):

    my-skill/
        SKILL.md      YAML front matter (name, description, optional metadata) + instructions for Kira
        tools.py      optional: Tool subclasses the model can call (see ai_buddy.tools)
        modes.py      optional: practice modes the app can run (see the speaking-coach skill)
        assets/, references/   optional data and documents the skill's code reads

How Kira uses the instructions (progressive disclosure, kept cheap for a small local model):
- skills with tools: the SKILL.md body is a few lines of tool guidance that go into the chat system prompt
  while the skill is on (the prompt is cached, so they cost nothing per turn after the first);
- skills with only instructions (no tools, no modes): the body goes into the chat prompt while the skill is
  on, up to MAX_PROMPT_CHARS; keep such skills short, since every chat turn carries them;
- skills with modes: the body is loaded only when that mode runs (e.g. writing practice feedback);
- MCP servers (ai_buddy.skills.mcp) become skills too: their tools come from the server.

Built-in skills live in ai_buddy/skills/builtin/; `skills.dirs` in config.toml adds folders with your own.
Which skills are on comes from `skills.enabled`, overridden by the switches in the app's settings.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import yaml

from ..tools import Tool, ToolRegistry

log = logging.getLogger(__name__)

BUILTIN_DIR = Path(__file__).parent / "builtin"
MAX_PROMPT_CHARS = 1500     # an instructions-only skill's share of the chat prompt
_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SkillError(ValueError):
    pass


def parse_skill_md(text: str) -> tuple[dict, str]:
    """Split a SKILL.md file into its YAML front matter and its Markdown body."""
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillError("SKILL.md must start with YAML front matter between '---' lines")
    try:
        end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration:
        raise SkillError("SKILL.md front matter is not closed with '---'") from None
    try:
        meta = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError as e:
        raise SkillError(f"SKILL.md front matter is not valid YAML: {e}") from None
    if not isinstance(meta, dict):
        raise SkillError("SKILL.md front matter must be a mapping")
    return meta, "\n".join(lines[end + 1:]).strip()


def validate(meta: dict, folder_name: str | None = None) -> list[str]:
    """Problems with a skill's front matter, following the Agent Skills specification."""
    problems = []
    name = meta.get("name")
    if not isinstance(name, str) or not name:
        problems.append("'name' is required")
    elif len(name) > 64 or not _NAME.match(name):
        problems.append(f"'name' {name!r} must be 1-64 lowercase letters, digits and single hyphens")
    elif folder_name is not None and name != folder_name:
        problems.append(f"'name' {name!r} must match the folder name {folder_name!r}")
    desc = meta.get("description")
    if not isinstance(desc, str) or not desc.strip():
        problems.append("'description' is required")
    elif len(desc) > 1024:
        problems.append("'description' must be at most 1024 characters")
    compat = meta.get("compatibility")
    if compat is not None and (not isinstance(compat, str) or not 1 <= len(compat) <= 500):
        problems.append("'compatibility' must be 1-500 characters")
    md = meta.get("metadata")
    if md is not None and not (isinstance(md, dict) and all(isinstance(k, str) and isinstance(v, str)
                                                           for k, v in md.items())):
        problems.append("'metadata' must map strings to strings")
    return problems


@dataclass
class Skill:
    name: str
    description: str
    instructions: str = ""                              # the SKILL.md body
    metadata: dict[str, str] = field(default_factory=dict)
    path: Path | None = None
    source: str = "built-in"                            # built-in | user | mcp
    tools: list[Tool] = field(default_factory=list)
    modules: dict[str, ModuleType] = field(default_factory=dict)
    status: str = "ready"                               # ready | starting | error (MCP servers)
    error: str | None = None

    @property
    def modes(self) -> list[str]:
        """Practice modes this skill provides (metadata `kira-modes`, space-separated)."""
        return self.metadata.get("kira-modes", "").split()

    @property
    def label(self) -> str:
        return self.metadata.get("kira-label") or self.name.replace("-", " ").capitalize()

    def view(self, enabled: bool) -> dict:
        return {"name": self.name, "label": self.label, "description": self.description, "enabled": enabled,
                "source": self.source, "status": self.status, "error": self.error, "modes": self.modes,
                "tools": [{"name": t.name, "confirm": t.confirm} for t in self.tools]}


def _import(path: Path, skill_name: str) -> ModuleType:
    mod_name = f"kira_skills.{skill_name.replace('-', '_')}.{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _tools_in(module: ModuleType) -> list[type[Tool]]:
    return [obj for _, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, Tool) and obj.__module__ == module.__name__ and obj.name]


def load_skill(folder: Path, source: str = "built-in") -> Skill:
    """Read one skill folder: SKILL.md, plus tools.py and modes.py if present."""
    md = folder / "SKILL.md"
    if not md.is_file():
        raise SkillError(f"{folder} has no SKILL.md")
    meta, body = parse_skill_md(md.read_text())
    if problems := validate(meta, folder.name):
        raise SkillError(f"{md}: " + "; ".join(problems))
    skill = Skill(meta["name"], " ".join(meta["description"].split()), body, dict(meta.get("metadata") or {}),
                  folder, source)
    for stem in ("tools", "modes"):
        if (folder / f"{stem}.py").is_file():
            skill.modules[stem] = _import(folder / f"{stem}.py", skill.name)
    if "tools" in skill.modules:
        for cls in _tools_in(skill.modules["tools"]):
            tool = cls()
            tool.skill = skill.name
            skill.tools.append(tool)
    return skill


def discover(dirs: list[str | Path] = ()) -> dict[str, Skill]:
    """All skills: the built-in ones, then those in `dirs` (each a folder of skill folders)."""
    found: dict[str, Skill] = {}
    for base, source in [(BUILTIN_DIR, "built-in")] + [(Path(d).expanduser(), "user") for d in dirs]:
        if not base.is_dir():
            log.warning("skills folder not found: %s", base)
            continue
        for folder in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
            try:
                skill = load_skill(folder, source)
            except Exception as e:
                log.warning("skipping skill %s: %s", folder, e)
                continue
            if skill.name in found:
                log.warning("skipping skill %s: a skill named %r is already loaded", folder, skill.name)
                continue
            names = {t.name for s in found.values() for t in s.tools}
            if clash := names & {t.name for t in skill.tools}:
                log.warning("skipping skill %s: tool name(s) %s already taken", folder, sorted(clash))
                continue
            found[skill.name] = skill
    return found


class SkillSet:
    """The skills Kira has and which of them are on. Changes notify listeners (the server rebuilds the
    tool registry and the chat prompt, and tells the UI)."""

    def __init__(self, skills: dict[str, Skill], enabled: list[str], overrides: dict[str, bool] | None = None,
                 default_timeout_s: float = 15):
        self.skills = dict(skills)
        self.default_timeout_s = default_timeout_s
        self._enabled = set(enabled)
        for name, on in (overrides or {}).items():
            (self._enabled.add if on else self._enabled.discard)(name)
        self._listeners: list[Callable[[], None]] = []
        unknown = [n for n in enabled if n not in self.skills]
        if unknown:
            log.info("enabled skills not loaded (yet): %s", unknown)

    @classmethod
    def from_config(cls, cfg, overrides: dict[str, bool] | None = None) -> SkillSet:
        return cls(discover(cfg.skills.dirs), cfg.skills.enabled, overrides, cfg.tools.timeout_s)

    def on_change(self, fn: Callable[[], None]) -> None:
        self._listeners.append(fn)

    def changed(self) -> None:
        """Tell listeners that skills changed (one was switched, or an MCP server connected)."""
        for fn in self._listeners:
            fn()

    # ---------- membership ----------

    def add(self, skill: Skill) -> None:
        """Add or replace a skill (MCP servers are added once they connect)."""
        self.skills[skill.name] = skill
        self.changed()

    def remove(self, name: str) -> None:
        if self.skills.pop(name, None):
            self.changed()

    def is_enabled(self, name: str) -> bool:
        return name in self._enabled and name in self.skills

    def set_enabled(self, name: str, on: bool) -> None:
        if (name in self._enabled) == on:
            return
        (self._enabled.add if on else self._enabled.discard)(name)
        self.changed()

    def wanted(self, name: str) -> bool:
        """Switched on, whether or not the skill is loaded (an MCP server may still be starting)."""
        return name in self._enabled

    @property
    def enabled(self) -> list[Skill]:
        """The skills that are on and usable, in a stable order (so the chat prompt stays cacheable)."""
        return [self.skills[n] for n in sorted(self.skills) if n in self._enabled and self.skills[n].status == "ready"]

    # ---------- what the rest of the app uses ----------

    def registry(self) -> ToolRegistry:
        return ToolRegistry([t for s in self.enabled for t in s.tools], self.default_timeout_s)

    def chat_instructions(self) -> list[str]:
        """The enabled tool skills' guidance for the chat prompt."""
        return [s.instructions for s in self.enabled if s.tools and s.instructions]

    def extra_instructions(self) -> list[str]:
        """The enabled instructions-only skills (no tools, no modes), each cut to MAX_PROMPT_CHARS."""
        out = []
        for s in self.enabled:
            if s.tools or s.modes or not s.instructions:
                continue
            if len(s.instructions) > MAX_PROMPT_CHARS:
                log.warning("skill %s: instructions cut to %d characters for the chat prompt", s.name, MAX_PROMPT_CHARS)
            out.append(f"- {s.label}: " + " ".join(s.instructions[:MAX_PROMPT_CHARS].split()))
        return out

    def mode_skill(self, mode: str) -> Skill | None:
        return next((s for s in self.enabled if mode in s.modes), None)

    def modes(self) -> list[str]:
        return [m for s in self.enabled for m in s.modes]

    def view(self) -> list[dict]:
        order = sorted(self.skills.values(), key=lambda s: ({"built-in": 0, "user": 1, "mcp": 2}[s.source], s.name))
        return [s.view(s.name in self._enabled) for s in order]
