"""Local knowledge (RAG): index a folder of notes, talking points, resumes and docs for Kira.

Pipeline: walk the folder -> extract text (Markdown/text, PDF via pypdf, DOCX, HTML) -> split into
~1200-character chunks along headings and paragraphs, with overlap -> store in SQLite with an FTS5
(BM25) index and an embedding per chunk -> re-index files whose size or modification time changed, when
the folder changes (watchfiles) or on request.

Search is hybrid: the embedding model (Qwen3-Embedding-0.6B on llama.cpp's CPU backend, through Lemonade)
finds passages by meaning, BM25 finds exact names and numbers ("Henderson", "Q3"), and the two rankings are
fused (reciprocal rank fusion). Without an embedding model it falls back to keywords alone.
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import sqlite3
import threading
import zipfile
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text", ".rst", ".org"}
SUFFIXES = TEXT_SUFFIXES | {".pdf", ".docx", ".html", ".htm"}
BARE_NAMES = {"readme", "notes", "todo", "changelog"}
SKIP_DIRS = {"node_modules", "__pycache__", "venv", "site-packages", "dist", "build", "target"}
QUERY_STOPWORDS = set("""a an and are as at be but by can could did do does for from had has have how i if in into is
it its me my of on or our please should so that the their them then there these they this to us was we were what when
where which who why will with would you your about tell find look search show give get notes note wrote write written
say says said file files document documents folder remind""".split())

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, mtime REAL NOT NULL, size INTEGER NOT NULL,
    chunks INTEGER NOT NULL DEFAULT 0, error TEXT, indexed_at TEXT);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL, heading TEXT, page INTEGER, text TEXT NOT NULL, vec BLOB);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, heading, name, tokenize='porter unicode61');
"""


@dataclass
class Chunk:
    text: str
    heading: str = ""
    page: int | None = None


@dataclass
class Hit:
    file: str          # path relative to the knowledge folder
    path: str          # absolute path
    heading: str
    page: int | None
    text: str
    score: float


# ---------- files ----------

def is_indexable(path: Path) -> bool:
    name = path.name
    if name.startswith((".", "~$")) or name.endswith(("~", ".tmp", ".part", ".swp")):
        return False
    suffix = path.suffix.lower()
    return suffix in SUFFIXES or (suffix == "" and name.lower() in BARE_NAMES)


def iter_files(folder: Path) -> Iterator[Path]:
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS)
        for f in sorted(files):
            p = Path(root, f)
            if is_indexable(p) and p.is_file():
                yield p


def _docx_text(path: Path) -> str:
    """Paragraph text of a .docx file; Word headings become Markdown headings."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    paras = []
    for p in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
        text = "".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", p.replace("<w:tab/>", "<w:t>\t</w:t>")))
        if not text.strip():
            continue
        level = re.search(r'<w:pStyle w:val="Heading(\d)"', p)
        paras.append(("#" * int(level.group(1)) + " " if level else "") + text)
    return html.unescape("\n\n".join(paras))


def extract(path: Path) -> tuple[list[tuple[str, int | None]], bool]:
    """([(text, page_number or None), ...], is_markdown) for a supported file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return [((page.extract_text() or ""), i + 1) for i, page in enumerate(reader.pages)], False
    if suffix == ".docx":
        return [(_docx_text(path), None)], True
    if suffix in (".html", ".htm"):
        from .web import html_to_text
        _, text = html_to_text(path.read_text(errors="replace"))
        return [(text, None)], False
    return [(path.read_text(errors="replace"), None)], suffix in (".md", ".markdown") or suffix == ""


# ---------- chunking ----------

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def sections(text: str, markdown: bool) -> list[tuple[str, str]]:
    """Split Markdown by headings into (heading path, body) pairs."""
    if not markdown:
        return [("", text)]
    out: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else _HEADING.match(line)
        if m:
            if "".join(buf).strip():
                out.append((" › ".join(t for _, t in stack), "\n".join(buf)))
            level = len(m.group(1))
            stack = [s for s in stack if s[0] < level] + [(level, m.group(2).strip())]
            buf = []
        else:
            buf.append(line)
    if "".join(buf).strip() or not out:
        out.append((" › ".join(t for _, t in stack), "\n".join(buf)))
    return out


def _split_long(paragraph: str, size: int) -> list[str]:
    pieces, cur = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+|\n", paragraph):
        while len(sentence) > size:
            cut = sentence.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            pieces.append((cur + " " + sentence[:cut]).strip() if cur else sentence[:cut])
            cur, sentence = "", sentence[cut:].strip()
        if len(cur) + len(sentence) + 1 > size and cur:
            pieces.append(cur)
            cur = sentence
        else:
            cur = (cur + " " + sentence).strip()
    if cur:
        pieces.append(cur)
    return pieces


def pack(paragraphs: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack paragraphs into chunks of about `size` characters; each chunk after the first
    starts with the last ~`overlap` characters of the previous one."""
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    fresh = False   # does `cur` hold anything beyond the carried-over overlap?
    for para in paragraphs:
        for piece in (_split_long(para, size) if len(para) > size else [para]):
            if fresh and cur_len + len(piece) + 2 > size:
                chunks.append("\n\n".join(cur))
                tail = chunks[-1][-overlap:] if overlap else ""
                if tail and len(tail) < len(chunks[-1]):
                    space = tail.find(" ")
                    tail = tail[space + 1:] if 0 <= space < len(tail) // 2 else tail
                cur, cur_len, fresh = ([f"…{tail}"], len(tail) + 1, False) if tail else ([], 0, False)
            cur.append(piece)
            cur_len += len(piece) + 2
            fresh = True
    if fresh:
        chunks.append("\n\n".join(cur))
    return chunks


def chunk_document(segments: list[tuple[str, int | None]], markdown: bool, size: int = 1200,
                   overlap: int = 200) -> list[Chunk]:
    out: list[Chunk] = []
    for text, page in segments:
        for heading, body in sections(text, markdown):
            paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
            out += [Chunk(c, heading, page) for c in pack(paras, size, overlap)]
    return out


# ---------- search helpers ----------

def fts_query(text: str) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in QUERY_STOPWORDS]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))


def keywords(text: str, n: int = 12) -> str:
    """The most frequent content words of a longer text (to search notes for a talk's topic)."""
    counts: dict[str, int] = {}
    for w in re.findall(r"[a-z][a-z'-]{3,}", text.lower()):
        if w not in QUERY_STOPWORDS:
            counts[w] = counts.get(w, 0) + 1
    return " ".join(sorted(counts, key=lambda w: -counts[w])[:n])


Embed = Callable[[list[str]], Awaitable[list[list[float]]]]

# How each embedding model family wants its inputs (query template, document template).
EMBED_FORMATS = {
    # Qwen3-Embedding: an instruction on the query side only; documents are embedded as they are.
    "qwen3-embedding": ("Instruct: Given a question about the user's own notes, retrieve the passages that answer "
                        "it\nQuery:{query}", "{title}\n{text}"),
    # nomic-embed-text v1/v1.5: a task prefix on both sides.
    "nomic-embed-text": ("search_query: {query}", "search_document: {title}\n{text}"),
}


def embed_format(model: str) -> tuple[str, str]:
    key = model.lower()
    return next((fmt for family, fmt in EMBED_FORMATS.items() if family in key), ("{query}", "{title}\n{text}"))


class KnowledgeBase:
    def __init__(self, db_path: Path, folder: Path | None, *, chunk_chars: int = 1200, chunk_overlap: int = 200,
                 max_file_mb: float = 20, watch: bool = True, embed: Embed | None = None, embed_model: str = ""):
        self.folder = folder
        self.chunk_chars, self.chunk_overlap = chunk_chars, chunk_overlap
        self.max_bytes = int(max_file_mb * 1024 * 1024)
        self.watch = watch
        self.embed = embed
        self.embed_model = embed_model
        self.query_fmt, self.doc_fmt = embed_format(embed_model)
        self.embed_state = "off" if embed is None else "pending"
        self._matrix: tuple[int, np.ndarray, np.ndarray] | None = None   # (version, chunk ids, vectors)
        self._version = 0
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._scan_lock = asyncio.Lock()
        self._dirty = False
        self.indexing = False
        self.last_indexed: str | None = None
        self.last_error: str | None = None
        self._watch_task: asyncio.Task | None = None
        self._stop_watch: asyncio.Event | None = None

    # ---------- lifecycle ----------

    async def start(self) -> None:
        await self.reindex()
        self._start_watcher()

    async def close(self) -> None:
        await self._stop_watcher()

    async def set_folder(self, folder: Path | None) -> None:
        await self._stop_watcher()
        self.folder = folder
        await self.reindex()
        self._start_watcher()

    def _start_watcher(self) -> None:
        if self.watch and self.folder and self.folder.is_dir():
            self._stop_watch = asyncio.Event()
            self._watch_task = asyncio.create_task(self._watch(self.folder, self._stop_watch))

    async def _stop_watcher(self) -> None:
        if self._stop_watch:
            self._stop_watch.set()
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except (asyncio.CancelledError, Exception):
                pass
        self._watch_task = self._stop_watch = None

    async def _watch(self, folder: Path, stop: asyncio.Event) -> None:
        from watchfiles import awatch

        def interesting(_change, path: str) -> bool:
            rel = Path(path).relative_to(folder) if path.startswith(str(folder)) else Path(path)
            return not any(part.startswith(".") for part in rel.parts) and not path.endswith(("~", ".swp"))

        try:
            async for _ in awatch(folder, stop_event=stop, debounce=1500, watch_filter=interesting):
                await self.reindex()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("stopped watching %s: %s", folder, e)

    # ---------- indexing ----------

    async def reindex(self, force: bool = False) -> None:
        """Index new and changed files, drop deleted ones. Coalesces requests made during a scan."""
        if self._scan_lock.locked():
            self._dirty = True
            return
        async with self._scan_lock:
            while True:
                self._dirty = False
                self.indexing = True
                try:
                    await asyncio.to_thread(self._scan, force)
                    if self.embed is not None:
                        self._check_embed_model()
                        await self._embed_missing()
                except Exception as e:
                    log.exception("indexing failed")
                    self.last_error = str(e)
                finally:
                    self.indexing = False
                force = False
                if not self._dirty:
                    break

    def _query(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def _scan(self, force: bool) -> None:
        folder = self.folder
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key='folder'").fetchone()
            if not row or row[0] != str(folder):
                self._db.executescript("DELETE FROM chunks_fts; DELETE FROM chunks; DELETE FROM files;")
                self._version += 1
                self._db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('folder', ?)", (str(folder),))
                self._db.commit()
        if not folder or not folder.is_dir():
            self.last_error = f"Folder not found: {folder}" if folder else None
            return
        self.last_error = None
        known = {r[0]: (r[1], r[2], r[3]) for r in self._query("SELECT path, mtime, size, id FROM files")}
        seen: set[str] = set()
        for path in iter_files(folder):
            key = str(path)
            seen.add(key)
            try:
                st = path.stat()
            except OSError:
                continue
            old = known.get(key)
            if not force and old and old[0] == st.st_mtime and old[1] == st.st_size:
                continue
            self._index_file(path, st, old[2] if old else None)
        with self._lock:
            for key, (_, _, fid) in known.items():
                if key not in seen:
                    self._delete_file(fid)
            self._db.commit()
        self.last_indexed = datetime.now().isoformat(timespec="seconds")

    def _delete_file(self, fid: int) -> None:
        self._db.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE file_id=?)", (fid,))
        self._db.execute("DELETE FROM chunks WHERE file_id=?", (fid,))
        self._db.execute("DELETE FROM files WHERE id=?", (fid,))
        self._version += 1

    def _index_file(self, path: Path, st: os.stat_result, old_id: int | None) -> None:
        error, chunks = None, []
        if st.st_size > self.max_bytes:
            error = f"larger than {self.max_bytes // (1024 * 1024)} MB, skipped"
        else:
            try:
                segments, markdown = extract(path)
                chunks = chunk_document(segments, markdown, self.chunk_chars, self.chunk_overlap)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
        rel = str(path.relative_to(self.folder)) if self.folder else path.name
        name = re.sub(r"[_\-.]+", " ", Path(rel).with_suffix("").as_posix().replace("/", " "))
        with self._lock:
            if old_id is not None:
                self._delete_file(old_id)
            cur = self._db.execute("INSERT INTO files (path, mtime, size, chunks, error, indexed_at) VALUES (?,?,?,?,?,?)",
                                   (str(path), st.st_mtime, st.st_size, len(chunks), error,
                                    datetime.now().isoformat(timespec="seconds")))
            fid = cur.lastrowid
            for i, c in enumerate(chunks):
                cid = self._db.execute("INSERT INTO chunks (file_id, idx, heading, page, text) VALUES (?,?,?,?,?)",
                                       (fid, i, c.heading, c.page, c.text)).lastrowid
                self._db.execute("INSERT INTO chunks_fts (rowid, text, heading, name) VALUES (?,?,?,?)",
                                 (cid, c.text, c.heading, name))
            self._db.commit()
        if error:
            log.info("not indexed %s: %s", path, error)

    def _check_embed_model(self) -> None:
        """Vectors from another embedding model don't compare: drop them so they're recomputed."""
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key='embed_model'").fetchone()
            if not row or row[0] != self.embed_model:
                self._db.execute("UPDATE chunks SET vec=NULL")
                self._db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_model', ?)",
                                 (self.embed_model,))
                self._db.commit()
                self._version += 1
        if self.embed_state.startswith("unavailable"):
            self.embed_state = "pending"      # try again on every scan (the model may have been loading)

    def _doc_text(self, text: str, heading: str, path: str) -> str:
        title = Path(path).stem.replace("_", " ").replace("-", " ") + (f" › {heading}" if heading else "")
        return self.doc_fmt.format(title=title, text=text)

    async def _embed_missing(self, batch: int = 16) -> None:
        if self.embed_state.startswith("unavailable"):
            return
        self.embed_state = "indexing"
        while True:
            rows = self._query("SELECT c.id, c.text, c.heading, f.path FROM chunks c JOIN files f ON f.id = c.file_id "
                               "WHERE c.vec IS NULL LIMIT ?", (batch,))
            if not rows:
                self.embed_state = "ready"
                return
            try:
                vecs = await self.embed([self._doc_text(t, h, p) for _, t, h, p in rows])
            except Exception as e:
                self.embed_state = f"unavailable ({e})"
                log.warning("embeddings unavailable, using keyword search only: %s", e)
                return
            with self._lock:
                for (cid, *_), v in zip(rows, vecs):
                    a = np.asarray(v, dtype=np.float32)
                    a /= np.linalg.norm(a) + 1e-9
                    self._db.execute("UPDATE chunks SET vec=? WHERE id=?", (a.tobytes(), cid))
                self._db.commit()
                self._version += 1

    # ---------- status & search ----------

    def status(self) -> dict:
        files, chunks = self._query("SELECT COUNT(*), COALESCE(SUM(chunks), 0) FROM files")[0]
        errors = [{"file": str(Path(p).relative_to(self.folder)) if self.folder and p.startswith(str(self.folder)) else p,
                   "error": e} for p, e in self._query("SELECT path, error FROM files WHERE error IS NOT NULL LIMIT 5")]
        embedded = self._query("SELECT COUNT(*) FROM chunks WHERE vec IS NOT NULL")[0][0]
        return {"folder": str(self.folder) if self.folder else None,
                "exists": bool(self.folder and self.folder.is_dir()), "files": files, "chunks": chunks,
                "indexing": self.indexing, "last_indexed": self.last_indexed, "error": self.last_error,
                "errors": errors, "embeddings": self.embed_state, "embed_model": self.embed_model or None,
                "embedded_chunks": embedded}

    async def search(self, query: str, k: int = 4) -> list[Hit]:
        ranked = await asyncio.to_thread(self._bm25, query, max(k * 4, 20))
        if self.embed is not None and self.embed_state in ("ready", "indexing"):
            try:
                qv = np.asarray((await self.embed([self.query_fmt.format(query=query)]))[0], dtype=np.float32)
                dense = await asyncio.to_thread(self._dense, qv, max(k * 4, 20))
                ranked = rrf([ranked, dense]) if ranked else dense
            except Exception as e:
                log.warning("embedding search failed, using keywords only: %s", e)
        return await asyncio.to_thread(self._hits, ranked[:k])

    def _bm25(self, query: str, limit: int) -> list[tuple[int, float]]:
        q = fts_query(query)
        if not q:
            return []
        rows = self._query("SELECT rowid, bm25(chunks_fts, 1.0, 1.5, 2.0) FROM chunks_fts WHERE chunks_fts MATCH ? "
                           "ORDER BY 2 LIMIT ?", (q, limit))
        return [(cid, -score) for cid, score in rows]

    def _dense(self, qv: np.ndarray, limit: int) -> list[tuple[int, float]]:
        if self._matrix is None or self._matrix[0] != self._version:
            rows = self._query("SELECT id, vec FROM chunks WHERE vec IS NOT NULL")
            ids = np.array([r[0] for r in rows], dtype=np.int64)
            mat = (np.frombuffer(b"".join(v for _, v in rows), dtype=np.float32).reshape(len(rows), -1) if rows
                   else np.zeros((0, len(qv)), dtype=np.float32))
            self._matrix = (self._version, ids, mat)
        _, ids, mat = self._matrix
        if not len(ids) or mat.shape[1] != len(qv):
            return []
        sims = mat @ (qv / (np.linalg.norm(qv) + 1e-9))
        order = np.argsort(-sims)[:limit]
        return [(int(ids[i]), float(sims[i])) for i in order]

    def _hits(self, ranked: list[tuple[int, float]]) -> list[Hit]:
        out = []
        for cid, score in ranked:
            rows = self._query("SELECT c.text, c.heading, c.page, f.path FROM chunks c JOIN files f ON f.id = c.file_id "
                               "WHERE c.id=?", (cid,))
            if rows:
                text, heading, page, path = rows[0]
                rel = str(Path(path).relative_to(self.folder)) if self.folder and path.startswith(str(self.folder)) \
                    else Path(path).name
                out.append(Hit(rel, path, heading or "", page, text, round(score, 4)))
        return out


def rrf(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion of several ranked lists of (id, score)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])
