"""search_notes: look things up in the user's knowledge folder (notes, talking points, resume, docs)."""

from __future__ import annotations

import re

from ai_buddy.tools import Tool, ToolContext, ToolResult

# "What does my resume say about...", "Remind me what I wrote about the budget", "my talking points for..."
_MINE = re.compile(r"\b(?:my|in my|from my)\s+(?:own\s+)?(?:notes?|résumé|resume|cv|talking points|documents?|"
                   r"knowledge folder)\b|\bwhat (?:did )?i (?:wrote|write|noted|jotted)\b", re.I)
_ASKING = re.compile(r"^\s*(?:(?:um|uh|so|okay|ok|hey kira|kira)[,\s]+)*(?:what|which|where|when|who|how|remind|find|"
                     r"check|look|search|tell|show|read|pull|get|does|did|do|is|are|can|could|would)\b|\?\s*$", re.I)


class SearchNotes(Tool):
    name = "search_notes"
    description = "Search the user's own notes, talking points, resume and documents in their knowledge folder."
    parameters = {"topic": {"type": "string",
                            "description": "What to look for, e.g. 'Henderson project talking points'."}}
    required = ["topic"]
    label = "Checking your notes"
    filler = "Let me check your notes."

    def describe(self, args: dict) -> str:
        return args.get("topic", "")

    def route(self, text: str, ctx: ToolContext) -> dict | None:
        if ctx.knowledge is None or not _MINE.search(text) or not _ASKING.search(text):
            return None
        topic = re.sub(r"^\s*(?:(?:um|uh|so|okay|ok|hey kira|kira)[,\s]+)+", "", text.strip(), flags=re.I)
        return {"topic": topic.rstrip("?.! ")}

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        kb = ctx.knowledge
        topic = str(args["topic"]).strip()
        if kb is None or not kb.folder:
            return ToolResult({"topic": topic, "error": "No knowledge folder is set up."},
                              summary="No knowledge folder set", ok=False)
        hits = await kb.search(topic, ctx.cfg.rag.top_k)
        if not hits:
            note = "Nothing in the notes matches." if kb.status()["chunks"] else "The knowledge folder is empty."
            return ToolResult({"topic": topic, "results": [], "note": note}, summary="Nothing found")
        results = [{"file": h.file, "section": h.heading or None, "page": h.page, "text": h.text} for h in hits]
        sources, seen = [], set()
        for h in hits:
            title = h.file + (f" › {h.heading}" if h.heading else "") + (f" (p. {h.page})" if h.page else "")
            if title not in seen:
                seen.add(title)
                sources.append({"title": title, "path": h.path})
        return ToolResult({"topic": topic, "results": results},
                          summary=f"{len(hits)} passages from {len({h.file for h in hits})} files", sources=sources)
