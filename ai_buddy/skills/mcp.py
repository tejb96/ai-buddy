"""MCP servers as skills, through the official MCP Python SDK.

Each `[mcp.servers.<name>]` entry in config.toml is a skill named <name>. While the skill is on, the app keeps
a connection to the server (a local process over stdio, or a URL over Streamable HTTP) and offers its tools to
the model as `<name>_<tool>` (all of them, or those listed in `tools`). A tool asks the user before it runs
unless the server marks it read-only (`readOnlyHint`) or it's listed in the server's `auto_approve`.

In a server's command, args, env and headers:
- `{folder}` is a folder the user picks in Settings (the server doesn't start until one is chosen);
- `{timezone}` is the local time zone;
- `${NAME}` is a secret from <data_dir>/secrets.env (NAME=value lines, kept out of config.toml), or else from
  the environment.
Local servers' own messages go to <data_dir>/logs/mcp-<name>.log. `npx`, `uvx` and the like are also found in
the usual per-user places (nvm, ~/.local/bin), since an app started from the desktop doesn't get the shell's PATH.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
from pathlib import Path

from mcp import Client, StdioServerParameters
from mcp import types as mt
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from ..config import Config, McpServerCfg
from ..tools import Tool, ToolContext, ToolResult
from . import Skill, SkillSet

log = logging.getLogger(__name__)

MAX_TEXT = 6000
MAX_DESCRIPTION = 320           # characters of a tool's description the model reads (every request carries them)
USER_BIN_DIRS = ("~/.local/bin", "~/.volta/bin", "~/.cargo/bin", "/usr/local/bin")
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
# Arguments that name a file or folder: shown and spoken by their name, and resolved inside a server's {folder}.
PATH_KEYS = {"path", "paths", "source", "destination", "file", "filename", "filepath", "file_path", "directory", "dir"}
URL_KEYS = {"url", "uri", "href", "link"}
_BARE_DOMAIN = re.compile(r"^(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}(?::\d+)?(?:/\S*)?$", re.I)


class McpConfigError(ValueError):
    pass


def find_command(command: str) -> str | None:
    """The program to run: on PATH, or in nvm's newest Node or another per-user bin folder."""
    if os.sep in command:
        path = Path(command).expanduser()
        return str(path) if os.access(path, os.X_OK) else None
    if found := shutil.which(command):
        return found

    def version(p: Path) -> tuple:
        return tuple(int(x) if x.isdigit() else 0 for x in p.parent.name.lstrip("v").split("."))
    nvm = sorted(Path("~/.nvm/versions/node").expanduser().glob("*/bin"), key=version, reverse=True)
    for folder in [*nvm, *(Path(d).expanduser() for d in USER_BIN_DIRS)]:
        if os.access(folder / command, os.X_OK) and (folder / command).is_file():
            return str(folder / command)
    return None


def read_secrets(path: Path) -> dict[str, str]:
    """NAME=value lines (blank lines and # comments ignored; values may be quoted)."""
    out = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.removeprefix("export ").partition("=")
            out[key.strip()] = value.strip().strip("'\"")
    return out


def short_description(text: str, limit: int = MAX_DESCRIPTION) -> str:
    """A tool's description cut to whole sentences within `limit` characters."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return cut[:end + 1] if end > limit // 3 else cut.rstrip() + "…"


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
        self.description = short_description(spec.description or spec.title or spec.name)
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
        """Short text for the UI: a file by its name, other values cut short."""
        parts = []
        for k, v in args.items():
            if k in PATH_KEYS and isinstance(v, str):
                parts.append(f"{k}: {Path(v).name or v}")
            elif isinstance(v, (str, int, float, bool)):
                v = " ".join(str(v).split())
                parts.append(f"{k}: {v[:40] + '…' if len(v) > 40 else v}")
        return ", ".join(parts)[:80]

    def confirm_text(self, args: dict) -> str:
        """The question asked out loud: the action and what it's about (a file by its name), not every argument;
        the card shows them all."""
        target = next((Path(v).name for k, v in args.items() if k in PATH_KEYS and isinstance(v, str) and v), "")
        if not target:
            target = next((" ".join(v.split()) for v in args.values() if isinstance(v, str) and 0 < len(v) <= 40), "")
        about = f" “{target}”" if target else ""
        return f"Should I {self.remote.replace('_', ' ')}{about} using {self.server.name}?"

    def _prepare(self, args: dict) -> dict:
        """Fix what a small model gets slightly wrong: "example.org" for a URL means https://example.org (as the
        built-in fetch_page reads it), and relative paths mean paths inside the server's folder."""
        args = {k: "https://" + v if k in URL_KEYS and isinstance(v, str) and _BARE_DOMAIN.match(v) else v
                for k, v in args.items()}
        return self._in_folder(args)

    def _in_folder(self, args: dict) -> dict:
        """Relative paths for a server that works in a folder ({folder}) mean paths inside that folder."""
        folder = self.server.folder if self.server.needs_folder else ""
        if not folder:
            return args

        def inside(v):
            if isinstance(v, list):
                return [inside(x) for x in v]
            if not isinstance(v, str) or os.path.isabs(os.path.expanduser(v)):
                return os.path.expanduser(v) if isinstance(v, str) else v
            return os.path.normpath(os.path.join(folder, v))
        return {k: inside(v) if k in PATH_KEYS else v for k, v in args.items()}

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        result = await self.server.call(self.remote, self._prepare(args))
        text = result_text(result)
        if result.is_error:
            return ToolResult({"error": text or "the tool reported an error"}, summary=_summary(text) or "Failed",
                              ok=False)
        return ToolResult({"result": text}, summary=_summary(text) or "Done")


def _summary(text: str) -> str:
    """One line for the UI from a tool's text result (a JSON result shows its first values)."""
    if text[:1] in "{[":
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, dict):
            pairs = [f"{k}: {v}" for k, v in data.items() if isinstance(v, (str, int, float, bool))]
            return ", ".join(pairs[:2])[:80]
        if isinstance(data, list):
            return f"{len(data)} item{'s' if len(data) != 1 else ''}"
    return text.splitlines()[0][:80] if text else ""


class McpServer:
    """One configured server: a long-lived connection in its own task (the SDK's context managers must be
    entered and left in the same task)."""

    def __init__(self, name: str, cfg: McpServerCfg, *, folder: str = "", timezone: str = "UTC",
                 data_dir: Path | None = None):
        self.name, self.cfg = name, cfg
        self.folder, self.timezone, self.data_dir = folder, timezone, data_dir
        self.client: Client | None = None
        self.skill = Skill(name, cfg.description or f"Tools from the MCP server '{name}'.", source="mcp",
                           status="off")
        self.skill.extra = self.details()
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._ready = asyncio.Event()

    @property
    def needs_folder(self) -> bool:
        c = self.cfg
        return any("{folder}" in v for v in [c.command, c.url, *c.args, *c.env.values(), *c.headers.values()])

    @property
    def secrets_path(self) -> Path | None:
        return self.data_dir / "secrets.env" if self.data_dir else None

    @property
    def log_path(self) -> Path | None:
        return self.data_dir / "logs" / f"mcp-{self.name}.log" if self.data_dir and not self.cfg.url else None

    def details(self) -> dict:
        """What Settings shows besides the switch."""
        return {"needs_folder": self.needs_folder, "folder": self.folder, "kind": "url" if self.cfg.url else "command",
                "target": self.cfg.url or " ".join([self.cfg.command, *self.cfg.args]),
                "log": str(self.log_path) if self.log_path else None}

    def _fill(self, text: str, secrets: dict[str, str]) -> str:
        text = text.replace("{folder}", self.folder).replace("{timezone}", self.timezone)

        def var(m: re.Match) -> str:
            if m[1] in secrets:
                return secrets[m[1]]
            if m[1] in os.environ:
                return os.environ[m[1]]
            raise McpConfigError(f"${{{m[1]}}} isn't set: add {m[1]}=... to {self.secrets_path or 'secrets.env'}")
        return _VAR.sub(var, text)

    async def _transport(self, stack: contextlib.AsyncExitStack):
        """The SDK transport for this server (entered by the Client), with placeholders and secrets filled in."""
        if self.needs_folder and not self.folder:
            raise McpConfigError("choose its folder first")
        secrets = read_secrets(self.secrets_path) if self.secrets_path else {}
        if self.cfg.url:
            url = self._fill(self.cfg.url, secrets)
            if not self.cfg.headers:
                return streamable_http_client(url)
            headers = {k: self._fill(v, secrets) for k, v in self.cfg.headers.items()}
            http = await stack.enter_async_context(create_mcp_http_client(headers=headers))
            return streamable_http_client(url, http_client=http)
        command = find_command(self._fill(self.cfg.command, secrets))
        if not command:
            raise McpConfigError(f"'{self.cfg.command}' isn't installed (not found on PATH, in nvm or ~/.local/bin)")
        env = get_default_environment()
        env.update({k: self._fill(v, secrets) for k, v in self.cfg.env.items()})
        env["PATH"] = os.pathsep.join([str(Path(command).parent), env.get("PATH", "")])    # npx needs its node
        args = [os.path.expanduser(self._fill(a, secrets)) for a in self.cfg.args]
        errlog = None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            errlog = stack.enter_context(self.log_path.open("a", buffering=1))
        params = StdioServerParameters(command=command, args=args, env=env)
        return stdio_client(params, errlog=errlog) if errlog else stdio_client(params)

    async def start(self, timeout_s: float = 60) -> None:
        if self._task and not self._task.done():
            return
        self.skill.extra = self.details()
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
            async with contextlib.AsyncExitStack() as stack:
                target = await self._transport(stack)
                client = await stack.enter_async_context(Client(target, read_timeout_seconds=self.cfg.timeout_s))
                self.client = client
                specs = (await client.list_tools()).tools
                if self.cfg.tools:
                    specs = [s for s in specs if s.name in self.cfg.tools]
                self.skill.tools = [McpTool(self, s) for s in specs]
                about = self.cfg.description or (client.instructions or "").strip().split("\n")[0][:300]
                if about:
                    self.skill.description = about
                names = ", ".join(t.name for t in self.skill.tools)
                where = (f" Its folder is {self.folder}: when the user says \"my folder\" or \"my files\", they "
                         f"mean this folder; use full paths inside it." if self.needs_folder else "")
                self.skill.instructions = (f"- {self.name} ({self.skill.description.rstrip('.')}): its tools are "
                                           f"{names}. Use them when the user asks for that; tools that change "
                                           f"something ask the user first, so just call them.{where}")
                self.skill.status = "ready"
                self._ready.set()
                await self._stop.wait()
        except asyncio.CancelledError:
            raise
        except McpConfigError as e:
            self._fail(str(e))
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
    """Starts and stops the configured servers as their skills are switched on and off, follows folder choices
    and re-reads the server list from config.toml on request."""

    def __init__(self, cfg: Config, skills: SkillSet, *, folders: dict[str, str] | None = None,
                 timezone: str = "UTC"):
        self.cfg, self.skills, self.timezone = cfg, skills, timezone
        self.folders = dict(folders or {})
        self.servers: dict[str, McpServer] = {}
        for name, scfg in cfg.mcp.servers.items():
            self._add(skill_name(name), scfg)
        self._lock = asyncio.Lock()

    def _add(self, name: str, scfg: McpServerCfg) -> McpServer | None:
        if name in self.skills.skills and self.skills.skills[name].source != "mcp":
            log.warning("MCP server %r skipped: a built-in skill has that name", name)
            return None
        server = McpServer(name, scfg, folder=self.folders.get(name, ""), timezone=self.timezone,
                           data_dir=self.cfg.app.data_path)
        self.servers[name] = server
        self.skills.skills[name] = server.skill
        return server

    async def sync(self) -> None:
        """Connect the servers that are switched on, disconnect the others."""
        async with self._lock:
            await self._sync()

    async def _sync(self) -> None:
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

    async def set_folder(self, name: str, folder: str) -> None:
        """A new folder for a server that uses {folder}: reconnect it if it's on."""
        async with self._lock:
            self.folders[name] = folder
            server = self.servers.get(name)
            if not server:
                return
            server.folder = folder
            server.skill.extra = server.details()
            if server.skill.status in ("ready", "starting", "error"):
                await server.stop()
            await self._sync()
            self.skills.changed()

    async def reload(self, servers: dict[str, McpServerCfg]) -> dict[str, list[str]]:
        """Apply a new [mcp.servers] list: add new servers, drop removed ones, restart changed ones."""
        wanted = {skill_name(n): c for n, c in servers.items()}
        out: dict[str, list[str]] = {"added": [], "removed": [], "changed": []}
        async with self._lock:
            for name in [n for n in self.servers if n not in wanted]:
                await self.servers.pop(name).stop()
                self.skills.skills.pop(name, None)
                out["removed"].append(name)
            for name, scfg in wanted.items():
                old = self.servers.get(name)
                if old and old.cfg == scfg:
                    continue
                if old:
                    await old.stop()
                if self._add(name, scfg):
                    out["changed" if old else "added"].append(name)
            await self._sync()
        self.skills.changed()
        return out

    async def close(self) -> None:
        for server in self.servers.values():
            await server.stop()
