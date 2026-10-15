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
    llm: str = "qwen3.5-9b-FLM"
    tts: str = "kokoro-v1"
    # Speech-to-text models offered in the settings panel, and the whisper.cpp backend for the non-FLM ones.
    stt_choices: list[str] = field(default_factory=lambda: ["whisper-v3-turbo-FLM", "Whisper-Large-v3-Turbo"])
    stt_backend: str = "cpu"
    llm_ctx_size: int = 32768
    # Embedding model for note search (llama.cpp on the CPU), e.g. "Qwen3-Embedding-0.6B-GGUF"; "" = keywords only.
    embed: str = ""
    embed_backend: str = "cpu"          # llama.cpp backend for the embedding model
    embed_threads: int = 4              # CPU threads for it (leaves cores for Kokoro)
    embed_ctx_size: int = 8192          # its context (Lemonade's 64k default costs ~7 GB of RAM)
    # LLMs offered in the settings panel.
    llm_choices: list[str] = field(default_factory=lambda: ["qwen3.5-9b-FLM", "gemma4-it-12b-FLM",
                                                            "qwen3.6-moe-35b-a3b-FLM"])


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
    tts_voice: str = "af_bella"         # the assistant is named after the voice: Bella
    tts_speed: float = 1.0
    language: str = "en"


@dataclass
class PersonaCfg:
    # The assistant's name is the voice's name (Bella, Emma, George...), so it changes with the voice.
    style: str = "warm, encouraging and honest"


@dataclass
class UserCfg:
    location: str = ""      # e.g. "Calgary, Alberta": assumed when you don't name a place


@dataclass
class SkillsCfg:
    enabled: list[str] = field(default_factory=lambda: ["web-search", "notes", "clock", "calculator",
                                                        "speaking-coach"])
    dirs: list[str] = field(default_factory=list)     # folders with your own skills
    confirm_timeout_s: float = 60                      # how long to wait for a yes/no before an action


@dataclass
class McpServerCfg:
    command: str = ""               # a local server started over stdio ...
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""                   # ... or a remote one (Streamable HTTP)
    headers: dict[str, str] = field(default_factory=dict)  # for a URL, e.g. Authorization = "Bearer ${TOKEN}"
    description: str = ""           # what it's for (shown in the app and to the model)
    tools: list[str] = field(default_factory=list)          # only these of its tools (default: all)
    auto_approve: list[str] = field(default_factory=list)   # tools that may run without asking ("*" = all)
    timeout_s: float = 30


@dataclass
class McpCfg:
    servers: dict[str, McpServerCfg] = field(default_factory=dict)

    def __post_init__(self):
        self.servers = {name: v if isinstance(v, McpServerCfg) else _build(McpServerCfg, v)
                        for name, v in self.servers.items()}


@dataclass
class ToolsCfg:
    max_calls_per_turn: int = 3
    timeout_s: float = 15
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
    path: Path | None = None        # the file this was read from (not a section)
    lemonade: LemonadeCfg = field(default_factory=LemonadeCfg)
    models: ModelsCfg = field(default_factory=ModelsCfg)
    chat: ChatCfg = field(default_factory=ChatCfg)
    feedback: FeedbackCfg = field(default_factory=FeedbackCfg)
    stt: SttCfg = field(default_factory=SttCfg)
    voice: VoiceCfg = field(default_factory=VoiceCfg)
    persona: PersonaCfg = field(default_factory=PersonaCfg)
    user: UserCfg = field(default_factory=UserCfg)
    skills: SkillsCfg = field(default_factory=SkillsCfg)
    mcp: McpCfg = field(default_factory=McpCfg)
    tools: ToolsCfg = field(default_factory=ToolsCfg)
    rag: RagCfg = field(default_factory=RagCfg)
    practice: PracticeCfg = field(default_factory=PracticeCfg)
    app: AppCfg = field(default_factory=AppCfg)


_MOVED = {("ToolsCfg", "enabled"): "[tools] enabled has moved to skills.enabled (skills are named web-search, notes, "
                                   "clock, calculator...)",
          ("ToolsCfg", "plugin_dir"): "[tools] plugin_dir has moved to skills.dirs (a tool plugin is now a skill folder "
                                      "with SKILL.md and tools.py)",
          ("PersonaCfg", "name"): "[persona] name is gone: the assistant is named after its voice ([voice] tts_voice, "
                                  "or pick one in the app)"}


def _build(cls, values: dict):
    for (owner, key), message in _MOVED.items():
        if cls.__name__ == owner and key in values:
            raise ValueError(message)
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown setting(s) in [{cls.__name__}]: {', '.join(sorted(unknown))}")
    return cls(**values)


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Read the TOML file (default: ./config.toml); missing sections fall back to defaults."""
    path = Path(path or os.environ.get("AI_BUDDY_CONFIG", DEFAULT_CONFIG))
    raw = tomllib.loads(path.read_text()) if path.exists() else {}
    cfg = Config(path=path)
    sections = [f.name for f in fields(Config) if f.name != "path"]
    for name in sections:
        if name in raw:
            setattr(cfg, name, _build(type(getattr(cfg, name)), raw[name]))
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"Unknown section(s) in {path}: {', '.join(sorted(unknown))}")
    return cfg
