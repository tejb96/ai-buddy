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
    llm: str = "qwen3.6-moe-35b-a3b-FLM"
    tts: str = "kokoro-v1"
    llm_ctx_size: int = 32768
    # Optional embedding model for semantic note search; empty = keyword search only (see README).
    embed: str = ""
    # LLMs offered in the settings panel.
    llm_choices: list[str] = field(default_factory=lambda: ["qwen3.6-moe-35b-a3b-FLM", "qwen3-it-4b-FLM"])


@dataclass
class ChatCfg:
    temperature: float = 0.6
    max_tokens: int = 300
    # Drop the oldest turns when the conversation history gets longer than this many words.
    history_word_budget: int = 8000
    greet: bool = True
    timeout_s: float = 90


@dataclass
class FeedbackCfg:
    think: bool = False
    max_tokens: int = 1500
    temperature: float = 0.4


@dataclass
class SttCfg:
    # Long push-to-talk turns are transcribed in chunks while you talk, cut in pauses.
    live_chunk_s: float = 15
    live_chunk_max_s: float = 25
    live_pause_s: float = 0.5


@dataclass
class VoiceCfg:
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0
    language: str = "en"


@dataclass
class PersonaCfg:
    name: str = "Kira"
    style: str = "warm, encouraging and honest"


@dataclass
class ToolsCfg:
    enabled: list[str] = field(default_factory=lambda: ["web_search", "fetch_page", "get_datetime", "calculate",
                                                        "search_notes"])
    max_calls_per_turn: int = 3
    timeout_s: float = 15
    plugin_dir: str = ""
    search_backend: str = "duckduckgo"
    searxng_url: str = ""
    search_results: int = 5
    pages_to_read: int = 2
    page_chars: int = 1500
    fetch_timeout_s: float = 6


@dataclass
class RagCfg:
    folder: str = "~/Documents/ai-buddy-notes"
    watch: bool = True
    chunk_chars: int = 1200
    chunk_overlap: int = 200
    top_k: int = 4
    max_file_mb: float = 20

    @property
    def folder_path(self) -> Path | None:
        return Path(os.path.expanduser(self.folder)) if self.folder else None


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
    stt: SttCfg = field(default_factory=SttCfg)
    voice: VoiceCfg = field(default_factory=VoiceCfg)
    persona: PersonaCfg = field(default_factory=PersonaCfg)
    tools: ToolsCfg = field(default_factory=ToolsCfg)
    rag: RagCfg = field(default_factory=RagCfg)
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
