"""Choices made in the UI (voice, model, knowledge folder), saved in <data_dir>/settings.json.

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

# Kokoro voices worth offering (the best-rated English ones), in display order.
VOICES: dict[str, str] = {
    "af_heart": "Heart (American, female)",
    "af_bella": "Bella (American, female)",
    "af_nicole": "Nicole (American, female, soft)",
    "af_sarah": "Sarah (American, female)",
    "bf_emma": "Emma (British, female)",
    "bf_isabella": "Isabella (British, female)",
    "am_michael": "Michael (American, male)",
    "am_fenrir": "Fenrir (American, male)",
    "bm_george": "George (British, male)",
}


class Settings:
    KEYS = ("voice", "llm", "knowledge_folder")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.path = cfg.app.data_path / "settings.json"
        self.values = {"voice": cfg.voice.tts_voice, "llm": cfg.models.llm, "knowledge_folder": cfg.rag.folder}
        self._listeners: list[Callable[[str, str], None]] = []
        try:
            saved = json.loads(self.path.read_text())
            self.values.update({k: v for k, v in saved.items() if k in self.KEYS and isinstance(v, str)})
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.values, indent=2))
        tmp.replace(self.path)
        for key, value in changed.items():
            for fn in self._listeners:
                fn(key, value)
