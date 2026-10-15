"""Entry point: `uv run ai-buddy` (or `python -m ai_buddy`)."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
import webbrowser

import uvicorn

from .config import load_config
from .server import create_app

log = logging.getLogger("ai_buddy")


def launch_window(url: str, profile_dir: str) -> subprocess.Popen | None:
    """Open the UI as a Chrome app window (falls back to the default browser)."""
    chrome = next((shutil.which(b) for b in ("google-chrome", "google-chrome-stable", "chromium",
                                             "chromium-browser") if shutil.which(b)), None)
    if not chrome:
        log.info("Chrome not found; opening %s in the default browser", url)
        webbrowser.open(url)
        return None
    cmd = [chrome, f"--app={url}", f"--user-data-dir={profile_dir}", "--no-first-run",
           "--no-default-browser-check", "--ozone-platform-hint=auto", "--window-size=1180,860",
           "--class=ai-buddy"]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def open_window(url: str, profile_dir: str, server: uvicorn.Server, window: dict) -> None:
    """Open the app window once the server is up; stop the server when that window is closed."""
    while not server.started:
        time.sleep(0.1)
    started = time.monotonic()
    proc = launch_window(url, profile_dir)
    if proc is None:
        return
    window["proc"] = proc
    proc.wait()
    # Chrome hands off to an already-open window of this profile and exits at once; only a
    # window that stayed open for a while means the user closed the app.
    if time.monotonic() - started > 5:
        log.info("App window closed; shutting down")
        server.should_exit = True


def already_running(host: str, port: int) -> bool:
    """True if an AI Buddy server already answers on host:port."""
    with socket.socket() as s:
        s.settimeout(0.5)
        if s.connect_ex((host, port)) != 0:
            return False
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/config", timeout=2) as r:
            return "persona" in json.load(r)
    except (OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Buddy: local voice companion on the Ryzen AI NPU")
    parser.add_argument("--config", help="path to config.toml (default: ./config.toml)")
    parser.add_argument("--no-window", action="store_true", help="only run the server")
    parser.add_argument("--port", type=int, help="override the port from config.toml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_config(args.config)
    if args.port:
        cfg.app.port = args.port
    url = f"http://{cfg.app.host}:{cfg.app.port}/"

    if already_running(cfg.app.host, cfg.app.port):
        print(f"AI Buddy is already running at {url}; opening a window onto it.")
        if not args.no_window:
            launch_window(url, str(cfg.app.data_path / "chrome-profile"))
        return

    server: uvicorn.Server | None = None

    def shutdown() -> None:         # the app's Exit button
        server.should_exit = True

    server = uvicorn.Server(uvicorn.Config(create_app(cfg, shutdown), host=cfg.app.host, port=cfg.app.port,
                                           log_level="warning", ws_max_size=4 * 1024 * 1024))
    window: dict = {}
    if cfg.app.open_window and not args.no_window:
        profile = str(cfg.app.data_path / "chrome-profile")
        threading.Thread(target=open_window, args=(url, profile, server, window), daemon=True).start()
    print(f"AI Buddy running at {url}  (Ctrl+C to quit)", flush=True)
    server.run()
    proc = window.get("proc")
    if proc and proc.poll() is None:
        proc.terminate()            # close the app window we opened (Exit, or Ctrl+C in the terminal)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
