"""FastAPI app: serves the UI, the session WebSocket, and a small REST API for history."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import Config
from .lemonade import Lemonade, LemonadeError
from .models import ModelManager
from .prompts import SCENARIOS
from .session import Session
from .storage import Store

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


def create_app(cfg: Config) -> FastAPI:
    lm = Lemonade(cfg.lemonade.base_url, cfg.lemonade.load_timeout_s)
    models = ModelManager(lm, cfg)
    store = Store(cfg.app.data_path)
    allowed_origins = {f"http://{h}:{cfg.app.port}" for h in ("127.0.0.1", "localhost", cfg.app.host)}

    async def warm_start():
        while True:
            try:
                await models.prepare_live()
                log.info("models ready: %s", models.names)
                return
            except LemonadeError as e:
                log.warning("waiting for Lemonade: %s", e)
                await asyncio.sleep(5)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(warm_start())
        yield
        task.cancel()
        await lm.close()

    app = FastAPI(title="AI Buddy", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", cfg.app.host])
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/config")
    async def get_config():
        return {
            "persona": cfg.persona.name,
            "models": models.names,
            "voice": cfg.voice.tts_voice,
            "practice": vars(cfg.practice),
            "data_dir": str(cfg.app.data_path),
            "scenarios": {k: {f: v[f] for f in ("title", "blurb", "placeholder", "counterpart")} for k, v in SCENARIOS.items()},
        }

    @app.get("/api/status")
    async def get_status():
        try:
            await models.refresh()
        except LemonadeError:
            pass
        return models.snapshot()

    @app.get("/api/sessions")
    async def list_sessions():
        return store.list_sessions()

    @app.get("/api/sessions/{sid}")
    async def get_session(sid: str):
        if not (s := store.get_session(sid)):
            raise HTTPException(404, "No such session")
        return s

    @app.delete("/api/sessions/{sid}")
    async def delete_session(sid: str):
        store.delete_session(sid)
        return {"ok": True}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        # WebSockets are not covered by CORS: refuse pages from other origins.
        if ws.headers.get("origin") not in allowed_origins:
            await ws.close(code=4403)
            return
        await ws.accept()
        await Session(ws, cfg, lm, models, store).run()

    return app
