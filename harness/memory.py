from __future__ import annotations
import json, os, re, tempfile, threading
from pathlib import Path
from .types import Message


def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 4


def msg_tokens(m: Message) -> int:
    n = estimate_tokens(m.content)
    for c in m.tool_calls:
        n += estimate_tokens(c.name + json.dumps(c.arguments))
    return n


class ConversationMemory:
    """Append-only history with a token-budgeted view.

    Nothing is deleted. window() returns the newest whole turns that fit the budget plus the
    prefix that was dropped, so callers can summarize it. A turn starts at a user message and
    includes its assistant/tool messages, so tool calls are never split from their results."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.messages: list[Message] = []
        self.summary: str = ""
        self.summarized_upto: int = 0
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.messages.append(Message.from_dict(json.loads(line)))

    def add(self, m: Message) -> None:
        self.messages.append(m)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(m.to_dict()) + "\n")

    def reset(self) -> None:
        self.messages.clear()
        self.summary, self.summarized_upto = "", 0
        if self.path and self.path.exists():
            self.path.unlink()

    def window(self, budget: int) -> tuple[list[Message], list[Message]]:
        turns: list[list[Message]] = []
        for m in self.messages:
            if m.role == "user" or not turns:
                turns.append([m])
            else:
                turns[-1].append(m)
        cut, used = len(turns), 0
        for i in range(len(turns) - 1, -1, -1):
            cost = sum(msg_tokens(m) for m in turns[i])
            if cut < len(turns) and used + cost > budget:
                break  # newest turn is always kept, even if oversized
            used += cost
            cut = i
        dropped = [m for t in turns[:cut] for m in t]
        kept = [m for t in turns[cut:] for m in t]
        return kept, dropped


class NotesStore:
    """Durable notes with keyword search. Atomic writes. Swap for a vector store by
    keeping the same three methods."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._notes: list[dict] = []
        if self.path.exists():
            self._notes = json.loads(self.path.read_text(encoding="utf-8"))

    def add(self, text: str, tag: str = "") -> dict:
        with self._lock:
            note = {"id": len(self._notes) + 1, "text": text.strip(), "tag": tag.strip()}
            self._notes.append(note)
            self._save()
            return note

    def all(self) -> list[dict]:
        return list(self._notes)

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = set(re.findall(r"\w+", query.lower()))
        scored = []
        for n in self._notes:
            words = set(re.findall(r"\w+", (n["text"] + " " + n["tag"]).lower()))
            s = len(q & words)
            if s:
                scored.append((s, n))
        scored.sort(key=lambda p: (-p[0], -p[1]["id"]))
        return [n for _, n in scored[:k]]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self._notes, f, indent=1)
        os.replace(tmp, self.path)
