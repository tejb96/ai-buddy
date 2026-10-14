"""web_search: search the web (DuckDuckGo or SearXNG, no API key) and read the top results."""

from __future__ import annotations

from urllib.parse import urlparse

from . import Tool, ToolContext, ToolResult
from ._web import SearchError, read_pages, search


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
