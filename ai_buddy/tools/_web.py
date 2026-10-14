"""Web helpers shared by the search and page tools: search backends, page fetching, text extraction.

No API keys: DuckDuckGo's HTML endpoint by default, or a self-hosted SearXNG instance.
"""

from __future__ import annotations

import asyncio
import html
import re
from datetime import date, timedelta
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import httpx

USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/140.0.0.0 Safari/537.36")
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9",
           "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
MAX_PAGE_BYTES = 2_000_000

STOPWORDS = set("""a an and are as at be but by can could did do does for from had has have how i if in into is it its
me my of on or our please should so than that the their them then there these they this to up us was we were what
when where which who why will with would you your about tell find look search show give get latest current now
today right""".split())


class SearchError(RuntimeError):
    pass


def terms(text: str) -> list[str]:
    """Lowercase content words with a light plural/suffix strip, for matching passages to a query."""
    out = []
    for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower()):
        if w in STOPWORDS or len(w) < 2:
            continue
        for suffix in ("ing", "es", "s"):
            if len(w) > len(suffix) + 3 and w.endswith(suffix):
                w = w[: -len(suffix)]
                break
        out.append(w)
    return out


# ---------- search ----------

class _DDGParser(HTMLParser):
    """Results on html.duckduckgo.com: <a class="result__a" href=…>title</a> … <a class="result__snippet">…</a>."""

    def __init__(self):
        super().__init__()
        self.results: list[dict] = []
        self._field: str | None = None
        self._ad = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class") or ""
        if tag == "div" and "result" in cls.split():
            self._ad = "result--ad" in cls
        if tag == "a" and "result__a" in cls and not self._ad:
            self.results.append({"title": "", "url": _ddg_url(a.get("href", "")), "snippet": ""})
            self._field = "title"
        elif tag in ("a", "div") and "result__snippet" in cls and self.results and not self._ad:
            self._field = "snippet"

    def handle_endtag(self, tag):
        if tag in ("a", "div"):
            self._field = None

    def handle_data(self, data):
        if self._field and self.results:
            self.results[-1][self._field] += data


def _ddg_url(href: str) -> str:
    if "duckduckgo.com/l/" in href:
        target = parse_qs(urlparse(href).query).get("uddg", [""])[0]
        return unquote(target) if target else ""
    return "https:" + href if href.startswith("//") else href


async def search_duckduckgo(http: httpx.AsyncClient, query: str, n: int, timeout: float) -> list[dict]:
    r = await http.post("https://html.duckduckgo.com/html/", data={"q": query, "b": ""}, headers=HEADERS,
                        timeout=timeout, follow_redirects=True)
    if r.status_code != 200:
        raise SearchError(f"DuckDuckGo answered {r.status_code}")
    p = _DDGParser()
    p.feed(r.text)
    results = [{k: " ".join(v.split()) for k, v in res.items()} for res in p.results if res["url"].startswith("http")]
    if not results and "anomaly" in r.text.lower():
        raise SearchError("DuckDuckGo is rate-limiting searches right now")
    return _dedupe(results)[:n]


async def search_searxng(http: httpx.AsyncClient, base_url: str, query: str, n: int, timeout: float) -> list[dict]:
    r = await http.get(base_url.rstrip("/") + "/search", params={"q": query, "format": "json"},
                       headers=HEADERS, timeout=timeout)
    if r.status_code != 200:
        raise SearchError(f"SearXNG answered {r.status_code} (is the JSON format enabled?)")
    results = [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")}
               for x in r.json().get("results", []) if x.get("url", "").startswith("http")]
    return _dedupe(results)[:n]


def _dedupe(results: list[dict]) -> list[dict]:
    seen, out = set(), []
    for res in results:
        key = res["url"].split("#")[0].rstrip("/")
        if key not in seen:
            seen.add(key)
            out.append(res)
    return out


async def search(http: httpx.AsyncClient, query: str, *, backend: str, searxng_url: str, n: int,
                 timeout: float) -> list[dict]:
    if backend == "searxng":
        if not searxng_url:
            raise SearchError("tools.searxng_url is not set")
        return await search_searxng(http, searxng_url, query, n, timeout)
    return await search_duckduckgo(http, query, n, timeout)


# ---------- pages ----------

_SKIP = {"script", "style", "noscript", "template", "svg", "canvas", "iframe", "nav", "aside", "footer", "form",
         "button", "select", "option", "head"}
_BLOCK = {"p", "div", "section", "article", "main", "li", "ul", "ol", "tr", "table", "h1", "h2", "h3", "h4", "h5",
          "h6", "br", "dd", "dt", "dl", "blockquote", "pre", "header", "figcaption", "summary", "details", "hr"}
_VOID = {"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "embed", "source", "track", "wbr"}


class _TextExtractor(HTMLParser):
    """Visible text with line breaks at block boundaries; navigation, scripts and forms are dropped."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.main_parts: list[str] = []
        self.title = ""
        self.description = ""
        self._skip = 0
        self._main = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "meta":
            a = dict(attrs)
            if (a.get("name") or a.get("property") or "").lower() in ("description", "og:description"):
                self.description = self.description or (a.get("content") or "")
            return
        if tag == "title":
            self._in_title = True
        if tag in _VOID:
            if tag == "br":
                self._emit("\n")
            return
        if tag in _SKIP:
            self._skip += 1
        elif tag in ("main", "article"):
            self._main += 1
        if tag in _BLOCK:
            self._emit("\n")
        elif tag in ("td", "th"):
            self._emit(" | ")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in _VOID:
            return
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in ("main", "article"):
            self._main = max(0, self._main - 1)
        if tag in _BLOCK:
            self._emit("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if not self._skip:
            self._emit(data)

    def _emit(self, text: str) -> None:
        self.parts.append(text)
        if self._main:
            self.main_parts.append(text)


def _clean_lines(raw: str) -> list[str]:
    lines, seen = [], set()
    for line in raw.split("\n"):
        line = re.sub(r"\s+", " ", line.replace(" | ", "\x00")).strip().replace("\x00", " | ").strip(" |")
        if len(line) < 3 or not re.search(r"[A-Za-z0-9]", line) or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines


def html_to_text(page: str) -> tuple[str, str]:
    """(title, readable text) of an HTML page. Prefers <main>/<article> when they hold real content."""
    p = _TextExtractor()
    try:
        p.feed(page)
        p.close()
    except Exception:
        pass
    main = _clean_lines("".join(p.main_parts))
    lines = main if sum(len(x) for x in main) > 400 else _clean_lines("".join(p.parts))
    if p.description and p.description.strip() not in lines:
        lines.insert(0, p.description.strip())
    return " ".join(html.unescape(p.title).split()), "\n".join(lines)


_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]


def _day_terms(d: date) -> set[str]:
    day, month = _DAYS[d.weekday()], _MONTHS[d.month - 1]
    return {day, day[:3], str(d.day), month, month[:3]}


def expand_query(query: str, today: date | None) -> set[str]:
    """Query terms, plus the weekday and date that words like "tomorrow" or "Friday" refer to, so
    "weather tomorrow" matches a forecast line such as "Thu, 15 Oct | Sunny. High 11."."""
    q = set(terms(query))
    words = set(re.findall(r"[a-z]+", query.lower()))
    if today:
        if "tomorrow" in words:
            q |= _day_terms(today + timedelta(days=1))
        if words & {"today", "tonight", "now", "currently", "current"}:
            q |= _day_terms(today) | {"today", "tonight"}
        if "weekend" in words:
            q |= {"saturday", "sat", "sunday", "sun"}
    q |= {d[:3] for d in _DAYS if d in words}
    return q


def _score(block: str, q: set[str]) -> float:
    t = set(terms(block))
    score = 2.0 * len(q & t)
    score += 0.15 * min(len(re.findall(r"\d", block)), 8)
    score += 0.4 * min(len(re.findall(r"°|%|\$|€|£|km/h|mph|\bmm\b|\bcm\b", block)), 3)
    if re.search(r"[A-Za-z][^.!?]{25,}[.!?]", block):
        score += 0.5            # reads like a sentence, not a menu
    if block.count(" · ") >= 4 and len(re.findall(r"\d", block)) < 4:
        score -= 1.5            # a list of links
    return score


def relevant_passages(text: str, query: str, max_chars: int, today: date | None = None) -> str:
    """The lines of `text` that best match the query (kept in page order), up to max_chars."""
    q = expand_query(query, today)
    lines = [x for x in text.split("\n") if x.strip()]
    blocks: list[str] = []
    for line in lines:   # merge short lines (table cells, list items) into ~300-char blocks
        if blocks and len(blocks[-1]) < 120 and len(blocks[-1]) + len(line) < 300:
            blocks[-1] += " · " + line
        else:
            blocks.append(line)
    scored = [(_score(block, q), i, block) for i, block in enumerate(blocks)]
    picked, used = [], 0
    for score, i, block in sorted(scored, key=lambda x: (-x[0], x[1])):
        if used >= max_chars:
            break
        if score <= 0 and picked:
            break
        piece = block[: max_chars - used]
        picked.append((i, piece))
        used += len(piece) + 1
    return "\n".join(b for _, b in sorted(picked))


async def fetch_page(http: httpx.AsyncClient, url: str, timeout: float) -> tuple[str, str]:
    """(title, text) of a web page. Raises on non-HTML/text content, errors and oversized pages."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"not a web address: {url}")
    async with http.stream("GET", url, headers=HEADERS, timeout=timeout, follow_redirects=True) as r:
        if r.status_code != 200:
            raise httpx.HTTPStatusError(f"{r.status_code}", request=r.request, response=r)
        ctype = r.headers.get("content-type", "").lower()
        if ctype and not any(t in ctype for t in ("html", "text/plain", "xml")):
            raise ValueError(f"unsupported content type {ctype.split(';')[0]}")
        body = bytearray()
        async for chunk in r.aiter_bytes():
            body += chunk
            if len(body) > MAX_PAGE_BYTES:
                break
        encoding = r.encoding or "utf-8"
    text = bytes(body).decode(encoding, errors="replace")
    if "html" in ctype or text.lstrip()[:200].lower().startswith(("<!doctype", "<html")):
        return html_to_text(text)
    return "", "\n".join(_clean_lines(text))


async def read_pages(http: httpx.AsyncClient, results: list[dict], query: str, *, pages: int, chars: int,
                     timeout: float, today: date | None = None) -> list[dict]:
    """Fetch a few of the top results concurrently and attach the passages relevant to the query."""
    candidates = results[: pages + 1]

    async def one(res: dict) -> dict | None:
        try:
            title, text = await fetch_page(http, res["url"], timeout)
        except (httpx.HTTPError, ValueError, UnicodeError):
            return None
        passages = relevant_passages(text, query, chars, today)
        return {**res, "title": res.get("title") or title, "text": passages} if len(passages) > 80 else None

    read = await asyncio.gather(*(one(r) for r in candidates))
    return [r for r in read if r][:pages]
