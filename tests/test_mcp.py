"""MCP servers as skills, through the official SDK, against a tiny stdio server (tests/fixtures)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from ai_buddy.config import Config, McpServerCfg, load_config
from ai_buddy.skills import SkillSet, discover
from ai_buddy.skills.mcp import McpManager, McpServer, find_command, read_secrets, short_description, skill_name, tool_name

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
        assert tools["add_item"].confirm_text({"text": "buy milk"}) == "Should I add item “buy milk” using todo?"
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
    cfg.app.data_dir = str(tmp_path)            # server logs go here, not to the real data folder
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


async def test_folder_placeholder_secrets_and_tool_allowlist(tmp_path):
    (tmp_path / "secrets.env").write_text("# tokens\nTODO_FILE='" + str(tmp_path / "todo.txt") + "'\n")
    cfg = McpServerCfg(command=sys.executable, args=[str(SERVER), "${TODO_FILE}"], tools=["list_items", "add_item"],
                       env={"WHERE": "{folder}"})
    srv = McpServer("todo", cfg, data_dir=tmp_path)
    assert srv.needs_folder and srv.details()["folder"] == ""
    await srv.start()
    assert srv.skill.status == "error" and "choose its folder" in srv.skill.error
    await srv.stop()
    srv.folder = str(tmp_path)
    await srv.start()
    try:
        assert srv.skill.status == "ready", srv.skill.error
        assert {t.remote for t in srv.skill.tools} == {"list_items", "add_item"}          # "fail" left out
        assert f"Its folder is {tmp_path}" in srv.skill.instructions
        await srv.skill.tools[0].server.call("add_item", {"text": "from secrets"})
        assert "from secrets" in (tmp_path / "todo.txt").read_text()                     # ${TODO_FILE} was filled
        assert (tmp_path / "logs" / "mcp-todo.log").exists()
    finally:
        await srv.stop()
    missing = McpServer("todo", McpServerCfg(command=sys.executable, args=["${NOT_SET_ANYWHERE}"]), data_dir=tmp_path)
    await missing.start()
    assert missing.skill.status == "error" and "NOT_SET_ANYWHERE" in missing.skill.error
    await missing.stop()


def test_helpers(tmp_path, monkeypatch):
    (tmp_path / "s.env").write_text('export A=1\nB = "two words"\n\n# C=3\n')
    assert read_secrets(tmp_path / "s.env") == {"A": "1", "B": "two words"}
    assert read_secrets(tmp_path / "missing.env") == {}
    long = "Read a file. " + "It handles many encodings and errors. " * 20
    assert short_description(long).endswith(".") and len(short_description(long)) <= 320
    assert find_command(sys.executable) == sys.executable and find_command("no-such-program-xyz") is None
    bin_dir = tmp_path / ".nvm" / "versions" / "node" / "v24.1.0" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "npx-test").write_text("#!/bin/sh\n")
    (bin_dir / "npx-test").chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert find_command("npx-test") == str(bin_dir / "npx-test")       # nvm's node, though not on PATH


async def test_reload_adds_changes_and_removes_servers(tmp_path):
    cfg: Config = load_config()
    cfg.app.data_dir = str(tmp_path)
    cfg.mcp.servers = {"todo": server_cfg(tmp_path)}
    skills = SkillSet(discover(), ["clock", "todo"])
    manager = McpManager(cfg, skills)
    try:
        await manager.sync()
        assert skills.skills["todo"].status == "ready"
        changes = await manager.reload({"todo": server_cfg(tmp_path, description="Changed"),
                                        "notes2": server_cfg(tmp_path)})
        assert changes == {"added": ["notes2"], "removed": [], "changed": ["todo"]}
        assert skills.skills["todo"].status == "ready" and skills.skills["todo"].description == "Changed"
        assert skills.skills["notes2"].status == "off"                    # new, not switched on
        changes = await manager.reload({"notes2": server_cfg(tmp_path)})
        assert changes["removed"] == ["todo"] and "todo" not in skills.skills
        assert "todo_add_item" not in skills.registry().tools
    finally:
        await manager.close()


async def test_choosing_a_folder_reconnects(tmp_path):
    cfg: Config = load_config()
    cfg.app.data_dir = str(tmp_path)
    cfg.mcp.servers = {"todo": McpServerCfg(command=sys.executable, args=[str(SERVER), "{folder}/todo.txt"])}
    skills = SkillSet(discover(), ["todo"])
    manager = McpManager(cfg, skills)
    try:
        await manager.sync()
        assert skills.skills["todo"].status == "error"
        (tmp_path / "work").mkdir()
        await manager.set_folder("todo", str(tmp_path / "work"))
        assert skills.skills["todo"].status == "ready" and skills.view()[-1]["folder"] == str(tmp_path / "work")
        await manager.servers["todo"].call("add_item", {"text": "x"})
        assert (tmp_path / "work" / "todo.txt").exists()
    finally:
        await manager.close()


def test_files_are_named_and_resolved_inside_the_servers_folder(tmp_path):
    from mcp import types as mt
    from ai_buddy.skills.mcp import McpTool, _summary
    srv = McpServer("files", McpServerCfg(command="npx", args=["server", "{folder}"]), folder=str(tmp_path))
    write = McpTool(srv, mt.Tool(name="write_file", description="Write a file.", input_schema={"type": "object"}))
    args = {"path": f"{tmp_path}/Shopping.txt", "content": "milk\neggs\nbread"}
    assert write.confirm_text(args) == "Should I write file “Shopping.txt” using files?"     # no path or content read out
    assert write.describe(args) == "path: Shopping.txt, content: milk eggs bread"
    assert write._in_folder({"path": "notes/a.md", "content": "x"}) == {"path": f"{tmp_path}/notes/a.md", "content": "x"}
    assert write._in_folder({"paths": ["a.md", "/etc/x"]}) == {"paths": [f"{tmp_path}/a.md", "/etc/x"]}
    fetch = McpTool(McpServer("fetch", McpServerCfg(command="uvx")),
                    mt.Tool(name="fetch", description="Fetch a URL.", input_schema={"type": "object"}))
    assert fetch._prepare({"url": "example.org"}) == {"url": "https://example.org"}
    assert fetch._prepare({"url": "www.python.org/about"}) == {"url": "https://www.python.org/about"}
    assert fetch._prepare({"url": "http://localhost:8000/x", "max_length": 5}) == {"url": "http://localhost:8000/x",
                                                                                  "max_length": 5}
    assert fetch._prepare({"url": "not a url"}) == {"url": "not a url"}
    assert _summary('{"timezone": "Asia/Tokyo", "datetime": "2026-10-15T09:36:00+09:00", "is_dst": false}') == \
        "timezone: Asia/Tokyo, datetime: 2026-10-15T09:36:00+09:00"
    assert _summary("[FILE] ideas.md\n[FILE] shopping.txt") == "[FILE] ideas.md"
