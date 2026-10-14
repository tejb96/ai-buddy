"""fetch_page: read one web page, e.g. a search result or a link the user mentions."""

from __future__ import annotations

import httpx

from . import Tool, ToolContext, ToolResult
from ._web import fetch_page, relevant_passages


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
