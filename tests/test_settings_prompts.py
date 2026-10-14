"""Settings saved from the app, the config file's layout, and the chat prompt's standing rules."""

from __future__ import annotations

import json

import pytest

from ai_buddy.config import Config, _build, load_config
from ai_buddy.prompts import GROUNDING, HEARING, SPOKEN_RULES, chat_system_prompt
from ai_buddy.settings import Settings


def test_settings_persist_voice_model_folder_and_skills(tmp_path):
    cfg = Config()
    cfg.app.data_dir = str(tmp_path)
    s = Settings(cfg)
    seen = []
    s.on_change(lambda k, v: seen.append((k, v)))
    s.update(voice="bf_emma", llm="qwen3-it-4b-FLM")
    s.set_skill("web-search", False)
    s.set_skill("todo", True)
    assert seen == [("voice", "bf_emma"), ("llm", "qwen3-it-4b-FLM")]
    saved = json.loads((tmp_path / "settings.json").read_text())
    assert saved["skills"] == {"web-search": False, "todo": True} and saved["voice"] == "bf_emma"
    again = Settings(cfg)
    assert again.voice == "bf_emma" and again.skills == {"web-search": False, "todo": True}


def test_unreadable_settings_fall_back_to_the_config(tmp_path):
    cfg = Config()
    cfg.app.data_dir = str(tmp_path)
    (tmp_path / "settings.json").write_text("{not json")
    assert Settings(cfg).voice == cfg.voice.tts_voice


def test_repo_config_loads_and_old_tool_keys_point_to_skills():
    cfg = load_config()
    assert "speaking-coach" in cfg.skills.enabled and cfg.models.embed_backend == "cpu"
    with pytest.raises(ValueError, match="moved to skills.enabled"):
        _build(type(cfg.tools), {"enabled": ["web_search"]})
    with pytest.raises(ValueError, match="moved to skills.dirs"):
        _build(type(cfg.tools), {"plugin_dir": "~/x"})


def test_mcp_servers_from_toml():
    from ai_buddy.config import McpCfg
    mcp = _build(McpCfg, {"servers": {"files": {"command": "npx", "args": ["-y", "server"], "auto_approve": ["read"]}}})
    assert mcp.servers["files"].command == "npx" and mcp.servers["files"].auto_approve == ["read"]
    with pytest.raises(ValueError):
        _build(McpCfg, {"servers": {"files": {"comand": "typo"}}})


def test_chat_prompt_carries_the_standing_rules():
    cfg = load_config()
    p = chat_system_prompt(cfg.persona, "UTC", ["- calculate for arithmetic."], "audio", "Calgary, Alberta")
    assert GROUNDING in p and SPOKEN_RULES in p and HEARING["audio"] in p
    assert "The user lives in Calgary, Alberta" in p
    assert "say you don't know" in p and "never make up personal experiences" in p
    assert "no emojis, lists, markdown, links" in p and "Details and sources are shown on screen" in p
    bare = chat_system_prompt(cfg.persona, "UTC")
    assert "You have tools" not in bare and GROUNDING in bare and "lives in" not in bare
