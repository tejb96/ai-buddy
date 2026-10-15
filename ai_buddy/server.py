"""FastAPI app: serves the UI, the session WebSocket, and a small REST API for history and settings."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import subprocess
import wave
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import Config, load_config
from .knowledge import KnowledgeBase
from .lemonade import TTS_SAMPLE_RATE, Lemonade, LemonadeError
from .models import ModelManager
from .session import Services, Session
from .settings import VOICES, Settings, voice_name
from .skills import SkillSet
from .skills.mcp import McpManager
from .storage import Store
from .tools import ToolContext

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
PREVIEW_LINE = "Hi, I'm {name}. That opening was clear and confident, but you rushed the numbers. Try pausing after each one."


def local_timezone() -> str:
    tz = os.environ.get("TZ", "")
    if tz and "/" in tz:
        return tz.lstrip(":")
    try:
        link = os.readlink("/etc/localtime")
        return link.split("zoneinfo/", 1)[1]
    except (OSError, IndexError):
        return "UTC"


def pcm_to_wav24(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TTS_SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def create_app(cfg: Config, shutdown: Callable[[], None] | None = None) -> FastAPI:
    """The app. `shutdown` stops the server (the Exit button); without it, Exit only unloads the models."""
    settings = Settings(cfg)
    lm = Lemonade(cfg.lemonade.base_url, cfg.lemonade.load_timeout_s)
    models = ModelManager(lm, cfg, llm=settings.llm, voice=settings.voice, stt=settings.stt)
    skills = SkillSet.from_config(cfg, settings.skills)
    timezone = local_timezone()
    mcp = McpManager(cfg, skills, folders=settings.mcp_folders, timezone=timezone)
    coach = skills.skills.get("speaking-coach")
    scenarios = coach.modules["modes"].SCENARIOS if coach and "modes" in coach.modules else {}
    store = Store(cfg.app.data_path, voice_name(models.voice), scenarios)
    web = httpx.AsyncClient(timeout=httpx.Timeout(10.0), follow_redirects=True)
    folder = settings.knowledge_folder
    if folder and str(folder) == os.path.expanduser(cfg.rag.folder) and not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)   # the default folder is created on first run
    knowledge = KnowledgeBase(cfg.app.data_path / "knowledge.db", folder, chunk_chars=cfg.rag.chunk_chars,
                              chunk_overlap=cfg.rag.chunk_overlap, max_file_mb=cfg.rag.max_file_mb,
                              watch=cfg.rag.watch, embed=models.embed if cfg.models.embed else None,
                              embed_model=cfg.models.embed)
    svc = Services(cfg, lm, models, store, skills, ToolContext(cfg, web, knowledge, timezone), knowledge)
    allowed_origins = {f"http://{h}:{cfg.app.port}" for h in ("127.0.0.1", "localhost", cfg.app.host)}
    notes_running = False
    sessions: set[Session] = set()      # one per open window
    background: list[asyncio.Task] = []
    exiting = False

    async def sync_notes() -> None:
        """Index the knowledge folder (and watch it) only while the notes skill is on."""
        nonlocal notes_running
        if skills.is_enabled("notes") and not notes_running:
            notes_running = True
            await knowledge.start()
        elif not skills.is_enabled("notes") and notes_running:
            notes_running = False
            await knowledge.close()

    def on_skills() -> None:
        svc.skills_changed()
        loop = asyncio.get_running_loop()
        loop.create_task(mcp.sync())
        loop.create_task(sync_notes())
    skills.on_change(on_skills)

    def on_setting(key: str, value: str) -> None:
        if key == "voice":
            models.voice = value
            store.assistant_name = voice_name(value)     # the assistant is named after its voice
        elif key == "llm":
            models.switch_llm(value)
        elif key == "stt":
            models.switch_stt(value)
        elif key == "knowledge_folder" and notes_running:
            asyncio.get_running_loop().create_task(knowledge.set_folder(settings.knowledge_folder))
        elif key == "knowledge_folder":
            knowledge.folder = settings.knowledge_folder
    settings.on_change(on_setting)

    async def warm_start():
        while True:
            try:
                await models.prepare_live()
                log.info("models ready: %s", models.names)
                break
            except LemonadeError as e:
                log.warning("waiting for Lemonade: %s", e)
                await asyncio.sleep(5)
        # Synthesize the tool filler lines once, so they play instantly.
        for tool in svc.tools.tools.values():
            for line in tool.filler if isinstance(tool.filler, list) else [tool.filler]:
                try:
                    pcm = b"".join([c async for c in lm.tts_stream(line, models.names["tts"], models.voice,
                                                                     cfg.voice.tts_speed)])
                    svc.fillers[(models.voice, line)] = pcm
                except LemonadeError:
                    break

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        background[:] = [asyncio.create_task(warm_start()), asyncio.create_task(sync_notes()),
                         asyncio.create_task(mcp.sync())]
        yield
        for t in background:
            t.cancel()
        await mcp.close()
        await knowledge.close()
        await web.aclose()
        await lm.close()

    def same_origin(request: Request) -> None:
        """Refuse state-changing requests from other web pages (there is no CORS, but simple POSTs get through)."""
        origin = request.headers.get("origin")
        if origin is not None and origin not in allowed_origins:
            raise HTTPException(403, "Cross-origin request refused")

    app = FastAPI(title="AI Buddy", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", cfg.app.host])
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    def config_view() -> dict:
        return {
            "persona": svc.persona.name,
            "models": models.names,
            "voice": models.voice,
            "practice": vars(cfg.practice),
            "data_dir": str(cfg.app.data_path),
            "tools": list(svc.tools.tools),
            "skills": skills.view(),
            "modes": ["chat"] + skills.modes(),
            "scenarios": {k: {f: v[f] for f in ("title", "blurb", "placeholder", "counterpart")}
                          for k, v in scenarios.items()},
        }

    @app.get("/api/config")
    async def get_config():
        return config_view()

    # ---------- skills ----------

    @app.get("/api/skills")
    async def get_skills():
        return skills.view()

    @app.post("/api/skills", dependencies=[Depends(same_origin)])
    async def post_skills(body: dict):
        name, on = str(body.get("name", "")), body.get("enabled")
        if name not in skills.skills or not isinstance(on, bool):
            raise HTTPException(400, "Expected {name: <a skill>, enabled: true|false}")
        settings.set_skill(name, on)
        skills.set_enabled(name, on)
        await asyncio.sleep(0)      # let an MCP server start connecting before answering
        return config_view()

    @app.post("/api/exit", dependencies=[Depends(same_origin)])
    async def exit_app():
        """The Exit button: stop every session (feedback still being written too), unload all models from
        Lemonade, then stop the server. The window closes itself, and __main__ closes the app window it opened."""
        nonlocal exiting
        exiting = True
        warm = background[0] if background else None
        if warm and not warm.done():
            warm.cancel()           # still loading models at startup: stop before unloading
        for session in list(sessions):
            await session.stop()
        unloaded = True
        try:
            await models.unload_all()
        except (LemonadeError, httpx.HTTPError) as e:
            log.warning("unloading the models failed: %s", e)
            unloaded = False
        if shutdown:
            asyncio.get_running_loop().call_later(0.3, shutdown)     # after this response has gone out
        log.info("Exit: models unloaded=%s, stopping=%s", unloaded, shutdown is not None)
        return {"unloaded": unloaded, "stopping": shutdown is not None}

    # ---------- MCP servers ----------

    @app.post("/api/mcp/folder", dependencies=[Depends(same_origin)])
    async def mcp_folder(body: dict):
        """The folder an MCP server that uses {folder} works in (the files server); reconnects it."""
        name = str(body.get("name", ""))
        path = Path(os.path.expanduser(str(body.get("folder", "")).strip()))
        if name not in mcp.servers or not mcp.servers[name].needs_folder:
            raise HTTPException(400, f"{name!r} is not an MCP server that uses a folder")
        if not path.is_dir():
            raise HTTPException(400, f"Not a folder: {path}")
        settings.set_mcp_folder(name, str(path))
        await mcp.set_folder(name, str(path))
        return config_view()

    @app.post("/api/mcp/reload", dependencies=[Depends(same_origin)])
    async def mcp_reload():
        """Re-read [mcp.servers] from config.toml: new servers appear, removed ones go, changed ones restart."""
        try:
            fresh = load_config(cfg.path)
        except (OSError, ValueError) as e:
            raise HTTPException(400, f"config.toml has a problem: {e}") from e
        changes = await mcp.reload(fresh.mcp.servers)
        cfg.mcp = fresh.mcp
        return {"changes": changes, **config_view()}

    @app.post("/api/config/open", dependencies=[Depends(same_origin)])
    async def open_config():
        """Open config.toml in the default text editor."""
        if not cfg.path or not cfg.path.is_file():
            raise HTTPException(404, "config.toml not found")
        await asyncio.create_subprocess_exec("xdg-open", str(cfg.path), stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL)
        return {"path": str(cfg.path)}

    @app.post("/api/pick-folder", dependencies=[Depends(same_origin)])
    async def pick_folder(body: dict):
        """GNOME's folder chooser, for an MCP server's folder."""
        return {"path": await choose_folder(str(body.get("title") or "Choose a folder"),
                                            str(body.get("start") or Path.home()))}

    @app.get("/api/status")
    async def get_status():
        try:
            await models.refresh()
        except LemonadeError:
            pass
        return models.snapshot()

    # ---------- settings ----------

    def settings_view() -> dict:
        llms = list(dict.fromkeys(cfg.models.llm_choices + [models.names["llm"]]))
        stts = list(dict.fromkeys(cfg.models.stt_choices + [models.names["stt"]]))
        return {"voice": models.voice, "voices": VOICES, "default_voice": cfg.voice.tts_voice,
                "persona": svc.persona.name, "llm": models.names["llm"], "llms": llms,
                "default_llm": cfg.models.llm, "stt": models.names["stt"], "stts": stts,
                "default_stt": cfg.models.stt, "stt_device": models.device("stt"), "tts": models.names["tts"],
                "embed": models.names.get("embed"),
                "knowledge": knowledge.status(), "skills": skills.view(),
                "can_pick_folder": bool(shutil.which("zenity"))}

    @app.get("/api/settings")
    async def get_settings():
        return settings_view()

    @app.post("/api/settings", dependencies=[Depends(same_origin)])
    async def post_settings(body: dict):
        changes = {}
        if (voice := body.get("voice")) is not None:
            if voice not in VOICES:
                raise HTTPException(400, f"Unknown voice {voice!r}")
            changes["voice"] = voice
        if (llm := body.get("llm")) is not None:
            if llm not in cfg.models.llm_choices and llm != models.names["llm"]:
                raise HTTPException(400, f"{llm!r} is not in models.llm_choices")
            changes["llm"] = llm
        if (stt := body.get("stt")) is not None:
            if stt not in cfg.models.stt_choices and stt != models.names["stt"]:
                raise HTTPException(400, f"{stt!r} is not in models.stt_choices")
            changes["stt"] = stt
        if (folder := body.get("knowledge_folder")) is not None:
            path = Path(os.path.expanduser(str(folder).strip())) if str(folder).strip() else None
            if path and not path.is_dir():
                raise HTTPException(400, f"Not a folder: {path}")
            changes["knowledge_folder"] = str(path) if path else ""
        settings.update(**changes)
        return settings_view()

    @app.get("/api/voice/preview")
    async def voice_preview(voice: str):
        if voice not in VOICES:
            raise HTTPException(400, "Unknown voice")
        try:
            pcm = b"".join([c async for c in lm.tts_stream(PREVIEW_LINE.format(name=voice_name(voice)),
                                                            models.names["tts"], voice, cfg.voice.tts_speed)])
        except LemonadeError as e:
            raise HTTPException(503, str(e)) from e
        return Response(pcm_to_wav24(pcm), media_type="audio/wav")

    # ---------- knowledge folder ----------

    @app.get("/api/knowledge")
    async def knowledge_status():
        return knowledge.status()

    @app.post("/api/knowledge/reindex", dependencies=[Depends(same_origin)])
    async def knowledge_reindex():
        asyncio.get_running_loop().create_task(knowledge.reindex(force=True))
        await asyncio.sleep(0.05)
        return knowledge.status()

    async def choose_folder(title: str, start: str) -> str | None:
        """Open GNOME's folder chooser on the desktop; the chosen path, or None."""
        if not shutil.which("zenity"):
            raise HTTPException(501, "zenity is not installed; type the folder path instead")
        proc = await asyncio.create_subprocess_exec(
            "zenity", "--file-selection", "--directory", f"--title={title}", f"--filename={start.rstrip('/')}/",
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, _ = await proc.communicate()
        return out.decode().strip() or None

    @app.post("/api/knowledge/pick", dependencies=[Depends(same_origin)])
    async def knowledge_pick():
        return {"path": await choose_folder("Choose the knowledge folder", str(settings.knowledge_folder or Path.home()))}

    @app.post("/api/knowledge/open", dependencies=[Depends(same_origin)])
    async def knowledge_open(body: dict):
        """Open a source file from the knowledge folder in the default app."""
        root = settings.knowledge_folder
        path = Path(str(body.get("path", ""))).resolve()
        if not root or not path.is_file() or not path.is_relative_to(root.resolve()):
            raise HTTPException(400, "Not a file in the knowledge folder")
        await asyncio.create_subprocess_exec("xdg-open", str(path), stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL)
        return {"ok": True}

    # ---------- history ----------

    @app.get("/api/sessions")
    async def list_sessions():
        return store.list_sessions()

    @app.get("/api/sessions/{sid}")
    async def get_session(sid: str):
        if not (s := store.get_session(sid)):
            raise HTTPException(404, "No such session")
        return s

    @app.delete("/api/sessions/{sid}", dependencies=[Depends(same_origin)])
    async def delete_session(sid: str):
        store.delete_session(sid)
        return {"ok": True}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        # WebSockets are not covered by CORS: refuse pages from other origins.
        if ws.headers.get("origin") not in allowed_origins:
            await ws.close(code=4403)
            return
        if exiting:
            await ws.close(code=4410)       # the app is shutting down
            return
        await ws.accept()
        session = Session(ws, svc)
        sessions.add(session)
        try:
            await session.run()
        finally:
            sessions.discard(session)

    return app
