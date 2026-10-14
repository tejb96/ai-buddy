"""MCP servers as skills, through the official MCP Python SDK.

Each `[mcp.servers.<name>]` entry in config.toml is a skill named <name>. While the skill is on, the app keeps
a connection to the server (a local process over stdio, or a URL over Streamable HTTP) and offers its tools to
the model as `<name>_<tool>`. A tool asks the user before it runs unless the server marks it read-only
(`readOnlyHint`) or it's listed in the server's `auto_approve`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re

from mcp import Client, StdioServerParameters
from mcp import types as mt

from ..config import Config, McpServerCfg
from ..tools import Tool, ToolContext, ToolResult
from . import Skill, SkillSet

log = logging.getLogger(__name__)

MAX_TEXT = 6000


def skill_name(server: str) -> str:
    """A valid skill name for a server name ('My Files' -> 'my-files')."""
    name = re.sub(r"[^a-z0-9]+", "-", server.lower()).strip("-")
    return re.sub(r"-{2,}", "-", name)[:64] or "mcp-server"


def tool_name(server: str, tool: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", f"{server.replace('-', '_')}_{tool}")[:64]


def result_text(result: mt.CallToolResult) -> str:
    parts = []
    for item in result.content or []:
        if isinstance(item, mt.TextContent):
            parts.append(item.text)
        elif isinstance(item, mt.EmbeddedResource) and isinstance(item.resource, mt.TextResourceContents):
            parts.append(item.resource.text)
        elif isinstance(item, mt.ResourceLink):
            parts.append(f"[{item.name}]({item.uri})")
        else:
            parts.append(f"[{getattr(item, 'type', 'content')}]")
    if not parts and result.structured_content is not None:
        parts.append(json.dumps(result.structured_content, ensure_ascii=False))
    text = "\n".join(parts).strip()
    return text[:MAX_TEXT] + (" …[truncated]" if len(text) > MAX_TEXT else "")


class McpTool(Tool):
    def __init__(self, server: McpServer, spec: mt.Tool):
        self.server, self.remote = server, spec.name
        self.name = tool_name(server.name, spec.name)
        self.description = (spec.description or spec.title or spec.name).strip()[:1000]
        self.input_schema = dict(spec.input_schema or {"type": "object", "properties": {}})
        self.parameters = self.input_schema.get("properties") or {}
        self.required = list(self.input_schema.get("required") or [])
        self.label = f"Using {server.name}"
        self.filler = "One moment."
        self.timeout_s = server.cfg.timeout_s
        self.skill = server.name
        read_only = bool(spec.annotations and spec.annotations.read_only_hint)
        approved = "*" in server.cfg.auto_approve or spec.name in server.cfg.auto_approve
        self.confirm = not (read_only or approved)

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.input_schema}}

    def describe(self, args: dict) -> str:
        text = ", ".join(f"{k}: {v}" for k, v in args.items() if isinstance(v, (str, int, float, bool)))
        return text[:80]

    def confirm_text(self, args: dict) -> str:
        detail = self.describe(args)
        return f"Should I use {self.remote.replace('_', ' ')} from {self.server.name}" + \
            (f" with {detail}?" if detail else "?")

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        result = await self.server.call(self.remote, args)
        text = result_text(result)
        first = text.splitlines()[0][:80] if text else ""
        if result.is_error:
            return ToolResult({"error": text or "the tool reported an error"}, summary=first or "Failed", ok=False)
        return ToolResult({"result": text}, summary=first or "Done")


class McpServer:
    """One configured server: a long-lived connection in its own task (the SDK's context managers must be
    entered and left in the same task)."""

    def __init__(self, name: str, cfg: McpServerCfg):
        self.name, self.cfg = name, cfg
        self.client: Client | None = None
        self.skill = Skill(name, cfg.description or f"Tools from the MCP server '{name}'.", source="mcp",
                           status="off")
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._ready = asyncio.Event()

    def _target(self):
        if self.cfg.url:
            return self.cfg.url
        env = {**os.environ, **self.cfg.env} if self.cfg.env else None
        return StdioServerParameters(command=self.cfg.command, args=[os.path.expanduser(a) for a in self.cfg.args],
                                     env=env)

    async def start(self, timeout_s: float = 30) -> None:
        if self._task and not self._task.done():
            return
        if not (self.cfg.url or self.cfg.command):
            self._fail("set either command or url in config.toml")
            return
        self._stop.clear()
        self._ready.clear()
        self.skill.status, self.skill.error = "starting", None
        self._task = asyncio.create_task(self._run())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout_s)
        except TimeoutError:
            self._fail(f"no answer within {timeout_s:g} s")
            await self.stop()

    async def _run(self) -> None:
        try:
            async with Client(self._target(), read_timeout_seconds=self.cfg.timeout_s) as client:
                self.client = client
                specs = (await client.list_tools()).tools
                self.skill.tools = [McpTool(self, s) for s in specs]
                about = self.cfg.description or (client.instructions or "").strip().split("\n")[0][:300]
                if about:
                    self.skill.description = about
                names = ", ".join(t.name for t in self.skill.tools)
                self.skill.instructions = (f"- {self.name} ({self.skill.description.rstrip('.')}): its tools are "
                                           f"{names}. Use them when the user asks for that; tools that change "
                                           "something ask the user first, so just call them.")
                self.skill.status = "ready"
                self._ready.set()
                await self._stop.wait()
        except asyncio.CancelledError:
            raise
        except BaseException as e:   # the SDK raises exception groups from its task group
            self._fail(_describe(e))
            log.warning("MCP server %s stopped: %s", self.name, self.skill.error)
        finally:
            self.client = None
            if self.skill.status == "ready":
                self.skill.status = "off"
            self._ready.set()

    def _fail(self, error: str) -> None:
        self.skill.status, self.skill.error = "error", error
        self.skill.tools = []

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), 5)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            except Exception:
                pass
        self._task = None
        if self.skill.status != "error":
            self.skill.status = "off"
        self.skill.tools = []

    async def call(self, tool: str, args: dict) -> mt.CallToolResult:
        if self.client is None:
            raise RuntimeError(f"{self.name} is not connected")
        return await self.client.call_tool(tool, args, read_timeout_seconds=self.cfg.timeout_s)


def _describe(e: BaseException) -> str:
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    return f"{type(e).__name__}: {e}" if str(e) else type(e).__name__


class McpManager:
    """Starts and stops the configured servers as their skills are switched on and off."""

    def __init__(self, cfg: Config, skills: SkillSet):
        self.skills = skills
        self.servers = {skill_name(n): McpServer(skill_name(n), c) for n, c in cfg.mcp.servers.items()}
        for server in self.servers.values():
            skills.skills.setdefault(server.name, server.skill)
        self._lock = asyncio.Lock()

    async def sync(self) -> None:
        """Connect the servers that are switched on, disconnect the others."""
        async with self._lock:
            changed = False
            for name, server in self.servers.items():
                running = server.skill.status in ("ready", "starting")
                if self.skills.wanted(name) and not running:
                    await server.start()
                    changed = True
                elif not self.skills.wanted(name) and running:
                    await server.stop()
                    changed = True
            if changed:
                self.skills.changed()

    async def close(self) -> None:
        for server in self.servers.values():
            await server.stop()
