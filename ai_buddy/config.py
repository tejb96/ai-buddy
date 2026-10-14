"""Load config.toml into typed settings."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"


@dataclass
class LemonadeCfg:
    base_url: str = "http://localhost:13305/v1"
    load_timeout_s: float = 180


@dataclass
class ModelsCfg:
    stt: str = "whisper-v3-turbo-FLM"
    chat: str = "qwen3-it-4b-FLM"
    feedback: str = "qwen3.6-moe-35b-a3b-FLM"
    tts: str = "kokoro-v1"
    chat_ctx_size: int = 8192
    feedback_ctx_size: int = 16384


@dataclass
class ChatCfg:
    temperature: float = 0.7
    max_tokens: int = 200
    history_word_budget: int = 3500


@dataclass
class FeedbackCfg:
    think: bool = False
    max_tokens: int = 1200
    temperature: float = 0.4


@dataclass
class VoiceCfg:
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0
    language: str = "en"


@dataclass
class PersonaCfg:
    name: str = "Sunny"
    style: str = "warm, curious, upbeat, and a little playful"


@dataclass
class PracticeCfg:
    long_pause_s: float = 2.0
    target_wpm_low: int = 120
    target_wpm_high: int = 160
    presentation_chunk_s: float = 25


@dataclass
class AppCfg:
    host: str = "127.0.0.1"
    port: int = 8765
    data_dir: str = "~/.local/share/ai-buddy"
    save_audio: bool = False
    open_window: bool = True

    @property
    def data_path(self) -> Path:
        return Path(os.path.expanduser(self.data_dir))


@dataclass
class Config:
    lemonade: LemonadeCfg = field(default_factory=LemonadeCfg)
    models: ModelsCfg = field(default_factory=ModelsCfg)
    chat: ChatCfg = field(default_factory=ChatCfg)
    feedback: FeedbackCfg = field(default_factory=FeedbackCfg)
    voice: VoiceCfg = field(default_factory=VoiceCfg)
    persona: PersonaCfg = field(default_factory=PersonaCfg)
    practice: PracticeCfg = field(default_factory=PracticeCfg)
    app: AppCfg = field(default_factory=AppCfg)


def _build(cls, values: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown setting(s) in [{cls.__name__}]: {', '.join(sorted(unknown))}")
    return cls(**values)


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Read the TOML file (default: ./config.toml); missing sections fall back to defaults."""
    path = Path(path or os.environ.get("AI_BUDDY_CONFIG", DEFAULT_CONFIG))
    raw = tomllib.loads(path.read_text()) if path.exists() else {}
    cfg = Config()
    for f in fields(Config):
        if f.name in raw:
            setattr(cfg, f.name, _build(type(getattr(cfg, f.name)), raw[f.name]))
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown section(s) in {path}: {', '.join(sorted(unknown))}")
    return cfg
