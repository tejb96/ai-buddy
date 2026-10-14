"""web_search: search the web (DuckDuckGo or SearXNG, no API key) and read the top results.
fetch_page: read one web page, e.g. a search result or a link the user mentions."""

from __future__ import annotations

import re
from urllib.parse import urlparse

import httpx

from ai_buddy.tools import Tool, ToolContext, ToolResult
from ai_buddy.web import SearchError, fetch_page, read_pages, relevant_passages, search

# "search the web for X", "look up X online", "google X"
_EXPLICIT = re.compile(r"\b(?:search (?:the web|online|the internet) for|look up|google)\s+(?P<q>[^?.!]{3,120})", re.I)
# "what's the weather (going to be like) in Calgary tomorrow?"
_WEATHER = re.compile(r"\bweather(?:\s+(?:going to be|gonna be|be))?(?:\s+like)?\s+(?:in|for|at)\s+"
                      r"(?P<place>(?-i:[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,2}))(?P<when>[^?.!]{0,30})", re.I)
_TAIL = re.compile(r"\s+and\s+(?:then\s+)?(?:tell|let|give|show|read|summari[sz]e|explain)\b.*$", re.I)
_WHEN = re.compile(r"\b(today|tonight|tomorrow|this weekend|this week|on (?:mon|tues|wednes|thurs|fri|satur|sun)day)\b",
                   re.I)


class WebSearch(Tool):
    name = "web_search"
    description = ("Search the web and read the top results. Use it for anything current or factual you are not sure "
                   "about: weather, news, prices, exchange rates, sports, people, products, opening hours, events.")
    parameters = {"query": {"type": "string",
                            "description": "A short search query, like you would type into a search engine."}}
    required = ["query"]
    label = "Searching the web"
    filler = ["Let me look that up.", "Let me check online.", "One sec, I'll look that up."]

    def describe(self, args: dict) -> str:
        return args.get("query", "")

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
            return ToolResult({"query": query, "results": [], "note": "No results."}, summary="No results")
        read = await read_pages(ctx.http, results, query, pages=t.pages_to_read, chars=t.page_chars,
                                timeout=t.fetch_timeout_s, today=ctx.today())
        read_urls = {r["url"] for r in read}
        items = [{"title": r["title"], "url": r["url"], "text": r["text"]} for r in read]
        items += [{"title": r["title"], "url": r["url"], "snippet": r["snippet"][:200]}
                  for r in results if r["url"] not in read_urls]
        sources = [{"title": r["title"] or urlparse(r["url"]).netloc, "url": r["url"], "read": r["url"] in read_urls}
                   for r in results]
        return ToolResult({"query": query, "results": items},
                          summary=f"Read {len(read)} of {len(results)} results", sources=sources)


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
            title, text = await fetch_page(ctx.http, url, ctx.cfg.tools.fetch_timeout_s)
        except (httpx.HTTPError, ValueError, UnicodeError) as e:
            return ToolResult({"url": url, "error": f"Could not read the page: {e or type(e).__name__}"},
                              summary="Could not read the page", ok=False)
        budget = ctx.cfg.tools.page_chars * 2
        question = str(args.get("question") or "")
        body = relevant_passages(text, question, budget, ctx.today()) if question else text[:budget]
        return ToolResult({"url": url, "title": title, "text": body}, summary=title or url,
                          sources=[{"title": title or url, "url": url, "read": True}])
