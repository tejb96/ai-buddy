"""A tiny MCP server for tests: a shared to-do list with a read-only and a state-changing tool.

Run over stdio: python mcp_notes_server.py <file>   (the list is kept in <file>, one item per line)
"""
from __future__ import annotations

import sys
from pathlib import Path

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

LIST = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("todo.txt")
server = MCPServer("todo-test", instructions="A to-do list for tests.")


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def list_items() -> str:
    """List the to-do items."""
    items = LIST.read_text().splitlines() if LIST.exists() else []
    return "\n".join(f"{i}. {t}" for i, t in enumerate(items, 1)) or "The list is empty."


@server.tool()
def add_item(text: str) -> str:
    """Add an item to the to-do list."""
    with LIST.open("a") as f:
        f.write(text.strip() + "\n")
    return f"Added: {text.strip()}"


@server.tool()
def fail(reason: str) -> str:
    """Always fails (for tests)."""
    raise ValueError(reason)


if __name__ == "__main__":
    server.run("stdio")
