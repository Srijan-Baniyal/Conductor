from __future__ import annotations
from ..base import tool, ToolContext


def _store(ctx):
    if ctx is None or ctx.notes is None:
        raise RuntimeError("notes are not available")
    return ctx.notes


@tool("add_note", doc={"text": "The fact to remember, self-contained.", "tag": "Optional short label, e.g. 'experiment-a'."})
def add_note(text: str, tag: str = "", ctx: ToolContext = None) -> str:
    """Save a durable note that survives across turns and sessions."""
    n = _store(ctx).add(text, tag)
    return f"Saved note #{n['id']}"


@tool("search_notes", doc={"query": "Keywords to look for.", "limit": "Max results."})
def search_notes(query: str, limit: int = 5, ctx: ToolContext = None) -> str:
    """Find saved notes by keyword."""
    hits = _store(ctx).search(query, limit)
    return "\n".join(f"#{n['id']} [{n['tag']}] {n['text']}" for n in hits) or "No matching notes."


@tool("list_notes")
def list_notes(ctx: ToolContext = None) -> str:
    """List every saved note."""
    notes = _store(ctx).all()
    return "\n".join(f"#{n['id']} [{n['tag']}] {n['text']}" for n in notes) or "No notes saved."
