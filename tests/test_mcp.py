"""MCP servers as skills, through the official SDK, against a tiny stdio server (tests/fixtures)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from ai_buddy.config import Config, McpServerCfg, load_config
from ai_buddy.skills import SkillSet, discover
from ai_buddy.skills.mcp import McpManager, McpServer, skill_name, tool_name

SERVER = Path(__file__).parent / "fixtures" / "mcp_notes_server.py"


def server_cfg(tmp_path: Path, **kw) -> McpServerCfg:
    return McpServerCfg(command=sys.executable, args=[str(SERVER), str(tmp_path / "todo.txt")], **kw)


def test_names():
    assert skill_name("My Files!") == "my-files" and skill_name("github") == "github"
    assert tool_name("my-files", "read file") == "my_files_read_file"


async def test_server_tools_confirmation_and_calls(tmp_path):
    srv = McpServer("todo", server_cfg(tmp_path, description="A shared to-do list"))
    await srv.start()
    try:
        assert srv.skill.status == "ready" and srv.skill.source == "mcp"
        tools = {t.remote: t for t in srv.skill.tools}
        assert set(tools) == {"list_items", "add_item", "fail"}
        assert not tools["list_items"].confirm               # the server marks it read-only
        assert tools["add_item"].confirm and tools["fail"].confirm
        assert tools["add_item"].schema()["function"]["parameters"]["required"] == ["text"]
        assert "todo_add_item" in srv.skill.instructions and "A shared to-do list" in srv.skill.instructions
        added = await tools["add_item"].run({"text": "buy milk"}, None)
        assert added.ok and added.content == {"result": "Added: buy milk"}
        listed = await tools["list_items"].run({}, None)
        assert listed.content["result"] == "1. buy milk"
        failed = await tools["fail"].run({"reason": "nope"}, None)
        assert not failed.ok and failed.content["error"]
        assert tools["add_item"].confirm_text({"text": "buy milk"}) == "Should I use add item from todo with text: buy milk?"
    finally:
        await srv.stop()
    assert srv.skill.status == "off" and srv.skill.tools == []


async def test_auto_approve(tmp_path):
    srv = McpServer("todo", server_cfg(tmp_path, auto_approve=["add_item"]))
    await srv.start()
    try:
        confirm = {t.remote: t.confirm for t in srv.skill.tools}
        assert confirm == {"list_items": False, "add_item": False, "fail": True}
    finally:
        await srv.stop()


async def test_a_broken_server_reports_an_error(tmp_path):
    srv = McpServer("broken", McpServerCfg(command=sys.executable, args=["-c", "import sys; sys.exit(3)"], timeout_s=5))
    await srv.start(timeout_s=10)
    assert srv.skill.status == "error" and srv.skill.error and srv.skill.tools == []
    await srv.stop()
    nothing = McpServer("empty", McpServerCfg())
    await nothing.start()
    assert nothing.skill.status == "error" and "command or url" in nothing.skill.error


async def test_manager_follows_the_skill_switch(tmp_path):
    cfg: Config = load_config()
    cfg.mcp.servers = {"todo": server_cfg(tmp_path)}
    skills = SkillSet(discover(), ["clock"])
    manager = McpManager(cfg, skills)
    events = []
    skills.on_change(lambda: events.append(1))
    try:
        assert skills.skills["todo"].status == "off" and "todo_add_item" not in skills.registry().tools
        skills.set_enabled("todo", True)
        await manager.sync()
        assert skills.skills["todo"].status == "ready"
        assert {"todo_add_item", "todo_list_items", "get_datetime"} <= set(skills.registry().tools)
        assert any("todo_list_items" in line for line in skills.chat_instructions())
        skills.set_enabled("todo", False)
        await manager.sync()
        assert skills.skills["todo"].status == "off" and "todo_add_item" not in skills.registry().tools
        assert len(events) >= 4          # two switches, plus a change each time the connection changed
    finally:
        await manager.close()
        await asyncio.sleep(0)
