"""web_search: search the web (DuckDuckGo or SearXNG, no API key) and read the top results.
fetch_page: read one web page, e.g. a search result or a link the user mentions."""

from __future__ import annotations

import re
from urllib.parse import urlparse

import httpx

from ai_buddy.tools import Tool, ToolContext, ToolResult
from ai_buddy.web import SearchError, fetch_page, is_live, read_pages, relevant_passages, search

# "search the web for X", "look up X online", "google X"
_EXPLICIT = re.compile(r"\b(?:search (?:the web|online|the internet) for|look up|google)\s+(?P<q>[^?.!]{3,120})", re.I)
# "what's the weather (going to be like) in Calgary tomorrow?"
_WEATHER = re.compile(r"\bweather(?:\s+(?:going to be|gonna be|be))?(?:\s+like)?\s+(?:in|for|at)\s+"
                      r"(?P<place>(?-i:[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,2}))(?P<when>[^?.!]{0,30})", re.I)
_TAIL = re.compile(r"\s+and\s+(?:then\s+)?(?:tell|let|give|show|read|summari[sz]e|explain)\b.*$", re.I)
_WHEN = re.compile(r"\b(today|tonight|tomorrow|this weekend|this week|on (?:mon|tues|wednes|thurs|fri|satur|sun)day)\b",
                   re.I)
LIVE_NOTE = ("Live data: go by the most recent source (as_of) and say how old it is if it isn't from today. If the "
             "sources disagree, say so and give both.")
NO_PAGES = "No page could be read, only search snippets. If they don't answer the question, say you couldn't find it."


class WebSearch(Tool):
    name = "web_search"
    description = ("Search the web and read the top results. Use it for anything current or factual you are not sure "
                   "about: weather, news, prices, exchange rates, sports, people, products, opening hours, events.")
    parameters = {"query": {"type": "string",
                            "description": "A short search query, like you would type into a search engine."},
                  "double_check": {"type": "boolean",
                                   "description": "True when checking again a fact the user questioned: reads one "
                                                  "more source."}}
    required = ["query"]
    label = "Searching the web"
    filler = ["Let me look that up.", "Let me check online.", "One sec, I'll look that up."]

    def describe(self, args: dict) -> str:
        return args.get("query", "")

    def recheck(self, args: dict) -> dict | None:
        return {**args, "double_check": True}      # a second look reads one more source

    def route(self, text: str, ctx: ToolContext) -> dict | None:
        if (m := _EXPLICIT.search(text)) and not m["q"].lower().startswith("to "):
            query = _TAIL.sub("", m["q"].strip(" ,"))
            query = re.sub(r"^(?:the|a)\s+", "", query, flags=re.I)
            query = re.sub(r"\s+(?:online|on the web|on the internet|for me)$", "", query, flags=re.I)
            return {"query": query} if query else None
        if m := _WEATHER.search(text):
            place = m["place"].strip()
            if place.split()[0].lower() in {"the", "my", "this", "that", "here", "there", "general"}:
                return None
            when = _WHEN.search(m["when"] or "")
            return {"query": f"{place} weather {when[1].lower() if when else 'today'}"}
        return None

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        t = ctx.cfg.tools
        query = str(args["query"]).strip()
        try:
            results = await search(ctx.http, query, backend=t.search_backend, searxng_url=t.searxng_url,
                                   n=t.search_results, timeout=t.fetch_timeout_s)
        except SearchError as e:
            return ToolResult({"query": query, "error": f"Search failed: {e}"}, summary=str(e), ok=False)
        if not results:
            return ToolResult({"query": query, "results": [], "note": "No results. Say you couldn't find it."},
                              summary="No results")
        now = ctx.now()
        pages = t.pages_to_read + bool(args.get("double_check"))
        read = await read_pages(ctx.http, results, query, pages=pages, chars=t.page_chars,
                                timeout=t.fetch_timeout_s, today=ctx.today(), now=now)
        read_urls = {r["url"] for r in read}
        items = [{"title": r["title"], "url": r["url"], **({"as_of": r["as_of_text"]} if r["as_of_text"] else {}),
                  "text": r["text"]} for r in read]
        items += [{"title": r["title"], "url": r["url"], "snippet": r["snippet"][:200]}
                  for r in results if r["url"] not in read_urls]
        sources = [{"title": r["title"] or urlparse(r["url"]).netloc, "url": r["url"], "read": r["url"] in read_urls}
                   for r in results]
        out = {"query": query, "checked_at": now.strftime("%a %b %-d, %-I:%M %p"), "results": items}
        if not read:
            out["note"] = NO_PAGES
        elif is_live(query):
            out["note"] = LIVE_NOTE
        return ToolResult(out, summary=f"Read {len(read)} of {len(results)} results", sources=sources)


class FetchPage(Tool):
    name = "fetch_page"
    description = "Read a web page, for example one of your search results or a link the user mentions."
    parameters = {"url": {"type": "string", "description": "The full address, starting with http."},
                  "question": {"type": "string", "description": "What you want to find on the page (optional)."}}
    required = ["url"]
    label = "Reading a page"
    filler = "Let me read that page."

    def describe(self, args: dict) -> str:
        return args.get("url", "")

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        url = str(args["url"]).strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            page = await fetch_page(ctx.http, url, ctx.cfg.tools.fetch_timeout_s, ctx.now())
        except (httpx.HTTPError, ValueError, UnicodeError) as e:
            return ToolResult({"url": url, "error": f"Could not read the page: {e or type(e).__name__}"},
                              summary="Could not read the page", ok=False)
        budget = ctx.cfg.tools.page_chars * 2
        question = str(args.get("question") or "")
        body = relevant_passages(page.text, question, budget, ctx.today()) if question else page.text[:budget]
        as_of = {"as_of": page.as_of_text()} if page.as_of else {}
        return ToolResult({"url": url, "title": page.title, **as_of, "text": body}, summary=page.title or url,
                          sources=[{"title": page.title or url, "url": url, "read": True}])
