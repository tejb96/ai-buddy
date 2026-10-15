"""Make the right tool call in code when the user's intent is unambiguous.

The model is good at choosing tools, but not perfect: after a tool turn the 35B sometimes answered "what
time is it in Tokyo" from memory (and got it wrong), and Gemma searched Paris' weather but skipped the clock
for "and what time is it there". Forcing the call through the API isn't possible: FLM ignores
`tool_choice` (tested on FLM 1.0.6). So the app makes such calls itself, without extra LLM calls: Whisper's
transcript is routed before the model runs, and the routed calls and their results go into the history before
the first request, so the model answers from them in one round instead of two.

Each tool decides what it claims (`Tool.route`); patterns are strict, so anything less clear is left to the
model and its instructions.
"""

from __future__ import annotations

import json
import logging

from .tools import ToolCall, ToolContext, ToolRegistry

log = logging.getLogger(__name__)


def route(text: str, registry: ToolRegistry, ctx: ToolContext) -> list[ToolCall]:
    """The tool calls this request unambiguously needs (usually none)."""
    calls = []
    for tool in registry.tools.values():
        try:
            args = tool.route(text, ctx)
        except Exception:
            log.exception("routing with %s failed", tool.name)
            continue
        if args is not None:
            calls.append(ToolCall(id=f"route_{tool.name}_{len(calls)}", name=tool.name, args=args,
                                  raw_arguments=json.dumps(args), routed=True))
    return calls
