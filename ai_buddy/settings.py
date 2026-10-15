"""Choices made in the UI (voice, models, knowledge folder, which skills are on), saved in
<data_dir>/settings.json.

They start from config.toml and override it once changed in the settings panel.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)

# Kokoro voices worth offering (the best-rated English ones), in display order. The assistant is named after
# the voice: the first word of its label (Bella by default).
VOICES: dict[str, str] = {
    "af_bella": "Bella (American, female)",
    "af_heart": "Heart (American, female)",
    "af_nicole": "Nicole (American, female, soft)",
    "af_sarah": "Sarah (American, female)",
    "bf_emma": "Emma (British, female)",
    "bf_isabella": "Isabella (British, female)",
    "am_michael": "Michael (American, male)",
    "am_fenrir": "Fenrir (American, male)",
    "bm_george": "George (British, male)",
}



def voice_name(voice: str) -> str:
    """The assistant's name for a Kokoro voice: "af_bella" -> "Bella" (also for voices not listed above)."""
    label = VOICES.get(voice)
    return label.split(" (")[0] if label else voice.split("_", 1)[-1].replace("_", " ").title()


class Settings:
    KEYS = ("voice", "llm", "stt", "knowledge_folder")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.path = cfg.app.data_path / "settings.json"
        self.defaults = {"voice": cfg.voice.tts_voice, "llm": cfg.models.llm, "stt": cfg.models.stt,
                         "knowledge_folder": cfg.rag.folder}
        self.values = dict(self.defaults)
        self.skills: dict[str, bool] = {}    # skill switches flipped in the app (override skills.enabled)
        self.mcp_folders: dict[str, str] = {}   # the folder picked for each MCP server that uses {folder}
        self._listeners: list[Callable[[str, str], None]] = []
        try:
            saved = json.loads(self.path.read_text())
            self.values.update({k: v for k, v in saved.items() if k in self.KEYS and isinstance(v, str)})
            if isinstance(saved.get("skills"), dict):
                self.skills = {k: v for k, v in saved["skills"].items() if isinstance(k, str) and isinstance(v, bool)}
            if isinstance(saved.get("mcp_folders"), dict):
                self.mcp_folders = {k: v for k, v in saved["mcp_folders"].items() if isinstance(v, str)}
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as e:
            log.warning("ignoring unreadable %s: %s", self.path, e)

    @property
    def voice(self) -> str:
        return self.values["voice"]

    @property
    def llm(self) -> str:
        return self.values["llm"]

    @property
    def stt(self) -> str:
        return self.values["stt"]

    @property
    def knowledge_folder(self) -> Path | None:
        f = self.values["knowledge_folder"]
        return Path(os.path.expanduser(f)) if f else None

    def on_change(self, fn: Callable[[str, str], None]) -> None:
        self._listeners.append(fn)

    def update(self, **changes: str) -> None:
        changed = {k: v for k, v in changes.items() if k in self.KEYS and v != self.values.get(k)}
        if not changed:
            return
        self.values.update(changed)
        self._save()
        for key, value in changed.items():
            for fn in self._listeners:
                fn(key, value)

    def set_skill(self, name: str, on: bool) -> None:
        self.skills[name] = on
        self._save()

    def set_mcp_folder(self, name: str, folder: str) -> None:
        self.mcp_folders[name] = folder
        self._save()

    def _save(self) -> None:
        """Only choices that differ from config.toml are saved, so a new default there still applies."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        chosen = {k: v for k, v in self.values.items() if v != self.defaults[k]}
        extra = {"mcp_folders": self.mcp_folders} if self.mcp_folders else {}
        tmp.write_text(json.dumps({**chosen, "skills": self.skills, **extra}, indent=2))
        tmp.replace(self.path)
