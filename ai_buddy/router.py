"""Make the right tool call in code when the user's intent is unambiguous.

The model is good at choosing tools, but not perfect: after a tool turn the 35B sometimes answered "what
time is it in Tokyo" from memory (and got it wrong), and Gemma searched Paris' weather but skipped the clock
for "and what time is it there". Forcing the call through the API isn't possible: FLM ignores
`tool_choice` (tested on FLM 1.0.6). So the app makes such calls itself, without extra LLM calls:

- when the transcript is known before the model runs (Whisper first, text-only models), the routed calls
  and their results go into the history before the first request, and the model answers from them in one
  round instead of two;
- when the model hears the audio itself, the transcript (its <heard> block, or Whisper's, alongside) is
  routed as soon as it arrives; if the model then calls tools, missing routed calls are added to that round,
  and if it starts answering without them, the answer is stopped before it is spoken and the calls are made.

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


def missing(routed: list[ToolCall], made: list[ToolCall]) -> list[ToolCall]:
    """Routed calls the model didn't make itself (compared by tool name)."""
    names = {c.name for c in made if not c.error}
    return [c for c in routed if c.name not in names]
