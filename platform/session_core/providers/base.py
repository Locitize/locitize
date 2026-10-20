"""Provider protocol and shared, side-effect-free read utilities.

The :class:`Provider` protocol is the keystone of v2: each local AI tool is a
plugin that discovers, loads, previews, deletes, and builds a resume command
for its sessions. The UI and ``resume.launch()`` consume the protocol only.

Shared bounded-read helpers live here so every provider reads large JSONL
histories with the same byte/line caps v1 used - no unbounded memory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Protocol

from ..config import (
    MAX_JSON_LINE_CHARS,
    MAX_METADATA_SCAN_BYTES,
    MAX_PREVIEW_MESSAGE_CHARS,
)
from ..models import Preview, ResumeCommand, Session, ThreadMessage

# Per-session caps for the heavier full-content passes (search index + thread).
MAX_INDEX_BYTES = 4 * 1024 * 1024      # search index: 4 MiB per session
MAX_THREAD_CHARS = 200_000             # thread viewer: cap rendered text


# -- Bounded JSONL reading (pure) --------------------------------------------
def iter_jsonl_records(fp: Path, max_bytes: int | None = None) -> Iterator[dict]:
    """Yield parsed JSON records from a JSONL file, capped by byte budget.

    Lines longer than ``MAX_JSON_LINE_CHARS`` and unparseable lines are
    skipped. ``max_bytes`` bounds total bytes read so a huge history cannot
    pin memory during a refresh.
    """
    budget = MAX_INDEX_BYTES if max_bytes is None else max_bytes
    try:
        with open(fp, "rb") as f:
            if max_bytes is None:
                f.seek(0, 2)
                start = max(0, f.tell() - budget)
                f.seek(start)
                if start:
                    partial = f.readline(min(MAX_JSON_LINE_CHARS + 1, budget))
                    budget -= len(partial)
                    while partial and not partial.endswith(b"\n") and budget > 0:
                        partial = f.readline(min(MAX_JSON_LINE_CHARS + 1, budget))
                        budget -= len(partial)
            while budget > 0:
                line = f.readline(min(MAX_JSON_LINE_CHARS + 1, budget))
                if not line:
                    break
                budget -= len(line)
                if len(line) > MAX_JSON_LINE_CHARS:
                    while line and not line.endswith(b"\n") and budget > 0:
                        line = f.readline(min(MAX_JSON_LINE_CHARS + 1, budget))
                        budget -= len(line)
                    continue
                try:
                    record = json.loads(line.decode("utf-8", errors="replace"))
                    if isinstance(record, dict):
                        yield record
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def clip_preview_text(text: str, limit: int = MAX_PREVIEW_MESSAGE_CHARS) -> str:
    text = " ".join((text or "").split())
    return text[:limit]


def remember_first_last(first, last, count, text):
    text = clip_preview_text(text)
    if not text:
        return first, last, count
    if first is None:
        first = text
    else:
        last = text
    return first, last, count + 1


def keep_thread_tail(msgs: list[ThreadMessage], total_chars: int) -> int:
    """Keep the newest thread messages within the render character cap."""
    while msgs and total_chars > MAX_THREAD_CHARS:
        removed = msgs.pop(0)
        total_chars -= len(removed.text)
    return total_chars


def path_is_under(path: Path, root: Path) -> bool:
    """Return True when ``path`` resolves inside ``root``."""
    try:
        path.expanduser().resolve(strict=False).relative_to(
            root.expanduser().resolve(strict=False)
        )
        return True
    except ValueError:
        return False


# -- Label helpers (pure) ----------------------------------------------------
def model_group_label(source: str, model: str = "") -> str:
    model = (model or "").strip()
    if model.startswith("<") and model.endswith(">"):
        model = ""
    if source == "claude":
        lower = model.lower()
        if "opus" in lower:
            return "Claude Code / Opus"
        if "sonnet" in lower:
            return "Claude Code / Sonnet"
        if "haiku" in lower:
            return "Claude Code / Haiku"
        return f"Claude Code / {model}" if model else "Claude Code / Unknown"
    if source == "codex":
        return f"OpenAI / {model}" if model else "OpenAI / Codex"
    return model or "Unknown"


def session_model_label(session: Session) -> str:
    from ..config import provider_label
    model = (session.model or "").strip()
    if not model or (model.startswith("<") and model.endswith(">")):
        model = "Unknown"
    return f"{provider_label(session.provider)} / {model}"


# -- Provider protocol -------------------------------------------------------
class Provider(Protocol):
    """A resumable local AI session source.

    Implementations build :class:`ResumeCommand` values; they never execute
    them. ``delete`` takes one session and removes only its own files.
    """

    key: str
    label: str

    def detected(self) -> bool: ...
    def load_sessions(self) -> list[Session]: ...
    def preview(self, session: Session) -> Preview: ...
    def delete(self, session: Session) -> None: ...
    def resume_command(self, session: Session) -> ResumeCommand: ...

    # Full-content passes (default: derive from preview). Providers override
    # to walk the whole file once, reusing their existing extractors.
    def collect_messages(self, session: Session) -> list[str]:
        """All human message texts for full-text search."""
        preview = self.preview(session)
        return [t for t in (preview.first, preview.last) if t]

    def collect_thread(self, session: Session) -> list[ThreadMessage]:
        """Ordered user/assistant messages for the read-only viewer."""
        preview = self.preview(session)
        msgs: list[ThreadMessage] = []
        if preview.first:
            msgs.append(ThreadMessage("user", preview.first))
        if preview.last and preview.last != preview.first:
            msgs.append(ThreadMessage("user", preview.last))
        return msgs


__all__ = [
    "Provider",
    "iter_jsonl_records",
    "clip_preview_text",
    "remember_first_last",
    "keep_thread_tail",
    "model_group_label",
    "session_model_label",
    "MAX_METADATA_SCAN_BYTES",
    "MAX_PREVIEW_MESSAGE_CHARS",
    "MAX_INDEX_BYTES",
    "MAX_THREAD_CHARS",
]
