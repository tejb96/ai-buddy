"""Local persistence: SQLite for sessions/turns/metrics, plus a Markdown file per session."""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,             -- chat | presentation | interview
    scenario TEXT,
    details TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_s REAL,
    metrics_json TEXT,
    feedback_md TEXT,
    feedback_model TEXT
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,             -- user | assistant
    text TEXT NOT NULL,
    at TEXT NOT NULL,
    audio_s REAL,
    interrupted INTEGER DEFAULT 0,
    meta_json TEXT                  -- tools Kira used and the sources she read
);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, id);
"""

MODE_TITLES = {"chat": "Chat", "presentation": "Presentation", "interview": "Conversation practice"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Store:
    def __init__(self, data_dir: Path, assistant_name: str = "Kira", scenarios: dict[str, dict] | None = None):
        self.dir = data_dir
        self.assistant_name = assistant_name
        self.scenarios = scenarios or {}     # practice scenarios (titles, counterparts) from the coach skill
        (data_dir / "sessions").mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(data_dir / "buddy.db", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)
        if "meta_json" not in {r[1] for r in self._db.execute("PRAGMA table_info(turns)")}:
            self._db.execute("ALTER TABLE turns ADD COLUMN meta_json TEXT")   # databases from before tools
            self._db.commit()
        self._lock = threading.Lock()

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur

    def create_session(self, mode: str, scenario: str | None = None, details: str = "") -> str:
        sid = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
        self._exec("INSERT INTO sessions (id, mode, scenario, details, started_at) VALUES (?,?,?,?,?)",
                   (sid, mode, scenario, details, _now()))
        return sid

    def add_turn(self, session_id: str, role: str, text: str, audio_s: float | None = None,
                 interrupted: bool = False, meta: dict | None = None) -> None:
        self._exec("INSERT INTO turns (session_id, role, text, at, audio_s, interrupted, meta_json) "
                   "VALUES (?,?,?,?,?,?,?)", (session_id, role, text, _now(), audio_s, int(interrupted),
                                              json.dumps(meta) if meta else None))

    def finish_session(self, session_id: str, *, duration_s: float | None = None, metrics: dict | None = None,
                       feedback_md: str | None = None, feedback_model: str | None = None) -> Path | None:
        """Close a session and write its Markdown file. Sessions without any turns are deleted."""
        if not self._exec("SELECT 1 FROM turns WHERE session_id=? LIMIT 1", (session_id,)).fetchone():
            self.delete_session(session_id)
            return None
        self._exec("UPDATE sessions SET ended_at=?, duration_s=COALESCE(?, duration_s), "
                   "metrics_json=COALESCE(?, metrics_json), feedback_md=COALESCE(?, feedback_md), "
                   "feedback_model=COALESCE(?, feedback_model) WHERE id=?",
                   (_now(), duration_s, json.dumps(metrics) if metrics else None, feedback_md, feedback_model,
                    session_id))
        return self.export_markdown(session_id)

    def list_sessions(self, limit: int = 500) -> list[dict]:
        rows = self._exec(
            "SELECT s.*, (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.id AND t.role='user') AS user_turns "
            "FROM sessions s WHERE EXISTS (SELECT 1 FROM turns t WHERE t.session_id = s.id) "
            "ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._summary(r) for r in rows]

    def get_session(self, session_id: str) -> dict | None:
        row = self._exec("SELECT *, 0 AS user_turns FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            return None
        out = self._summary(row)
        out["metrics"] = json.loads(row["metrics_json"]) if row["metrics_json"] else None
        out["feedback_md"] = row["feedback_md"]
        out["feedback_model"] = row["feedback_model"]
        out["turns"] = []
        for t in self._exec("SELECT role, text, at, audio_s, interrupted, meta_json FROM turns WHERE session_id=? "
                            "ORDER BY id", (session_id,)).fetchall():
            turn = dict(t)
            turn["meta"] = json.loads(turn.pop("meta_json")) if t["meta_json"] else None
            out["turns"].append(turn)
        out["user_turns"] = sum(1 for t in out["turns"] if t["role"] == "user")
        return out

    def delete_session(self, session_id: str) -> None:
        self._exec("DELETE FROM turns WHERE session_id=?", (session_id,))
        self._exec("DELETE FROM sessions WHERE id=?", (session_id,))
        (self.dir / "sessions" / f"{session_id}.md").unlink(missing_ok=True)

    def _summary(self, r: sqlite3.Row) -> dict:
        m = json.loads(r["metrics_json"]) if r["metrics_json"] else None
        title = MODE_TITLES.get(r["mode"], r["mode"])
        if r["scenario"]:
            title = self.scenarios.get(r["scenario"], {}).get("title", r["scenario"])
        return {
            "id": r["id"], "mode": r["mode"], "scenario": r["scenario"], "details": r["details"],
            "title": title, "started_at": r["started_at"], "ended_at": r["ended_at"],
            "duration_s": r["duration_s"], "user_turns": r["user_turns"],
            "has_feedback": bool(r["feedback_md"]),
            "wpm": m and m.get("wpm"), "fillers_per_min": m and m["fillers"]["per_min"],
            "long_pauses": m and m["pauses"]["count"], "words": m and m.get("words"),
        }

    def export_markdown(self, session_id: str) -> Path:
        s = self.get_session(session_id)
        assert s is not None
        counterpart = self.scenarios.get(s["scenario"] or "", {}).get("counterpart", self.assistant_name)
        lines = [f"# {s['title']}: {s['started_at'].replace('T', ' ')}", ""]
        if s["details"]:
            lines += [f"*{'Topic' if s['mode'] == 'presentation' else 'Scenario'}:* {s['details']}", ""]
        m = s["metrics"]
        if m:
            f, p = m["fillers"], m["pauses"]
            lines += ["## Metrics", "",
                      f"- Words: {m['words']} in {m['speaking_time_s']} s",
                      f"- Pace: {m['wpm']} wpm ({m['pace']})",
                      f"- Filler words: {f['total']} ({f['per_min']}/min) "
                      + ", ".join(f"{w} x{n}" for w, n in f["by_word"].items()),
                      f"- Long pauses (>= {p['threshold_s']} s): {p['count']}, longest {p['longest_s']} s", ""]
        if s["feedback_md"]:
            lines += ["## Feedback", "", s["feedback_md"].strip(), ""]
        lines += ["## Transcript", ""]
        for t in s["turns"]:
            who = "You" if t["role"] == "user" else counterpart
            lines.append(f"**{who}:** {t['text']}{' *(interrupted)*' if t['interrupted'] else ''}")
            for src in (t["meta"] or {}).get("sources", []):
                where = src.get("url") or src.get("path") or ""
                lines.append(f"  - Source: [{src.get('title') or where}]({where})" if src.get("url")
                             else f"  - Source: {src.get('title') or where}")
            lines.append("")
        path = self.dir / "sessions" / f"{session_id}.md"
        path.write_text("\n".join(lines))
        return path
