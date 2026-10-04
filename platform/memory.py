"""Conversation memory store for the LOCITIZE assistant (Data Model 9.3, Architecture M8.3).

Deliberately small and honest: a local, append-only, per-session JSONL transcript
store -- NOT an embeddings/vector database (that is explicitly future; no torch,
faiss, or sentence-transformers here). One object per line under the memory dir:
`{session_id, ts, role, text}`.

M7 wired the assistant to EMIT its transcript through `append()` so every turn is
persisted; M8 completes the recall surface: `list_sessions()` (what conversations
exist), `recall(limit)` (load the most recent conversation's tail), and
`search(keyword, limit)` (case-insensitive substring scan across all transcripts).
Recall is explicit, injected into assistant context only on demand (the `/recall`
command or the settings.memory.auto_recall opt-in), never automatically every turn.
The store never leaves the user's data root: the memory dir is resolved from
settings.data_dir like logs/ and the benchmark reports (DEC-M14-9), and a
session id that would escape that dir is rejected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class MemoryEntry:
    """The in-memory mirror of one JSONL line (Data Model 9.3)."""

    session_id: str
    ts: str
    role: str
    text: str


# Roles the store accepts. A conversation entry is either the owner's utterance or
# the assistant's reply; anything else is a programming error, not user input.
_VALID_ROLES = ("user", "assistant")

# A limit large enough to mean "read every entry" for search()'s per-session scan.
# Transcripts are a single owner's local conversation, never anywhere near this size.
_ALL_ENTRIES = 1_000_000_000


def resolve_memory_dir(data_root: Path | str, configured: str) -> Path:
    """Resolve the memory directory under the data root, refusing any escape.

    `data_root` is settings.data_dir - the user's single data folder, NEVER the
    install directory (DEC-M14-9, defect NEW-QA-M14-8: transcripts were landing
    in the install tree, so a reinstall destroyed them and the documented
    "back up one folder" story was false). The escape-refusal below is anchored
    on that same root.

    `configured` is settings.memory.dir (default "memory"). An absolute or
    parent-traversing value that would land outside the root is rejected,
    because conversation transcripts are the most personal data LOCITIZE holds and
    must stay inside the folder the user backs up (Data Model 9.3 invariant,
    Permission Matrix M8).
    """
    base = Path(data_root).resolve()
    candidate = (base / (configured or "memory")).resolve()
    # Confirm candidate is base itself or a descendant; otherwise fall back to the
    # safe default under base rather than honoring a path that escapes the tree.
    if candidate != base and base not in candidate.parents:
        return base / "memory"
    return candidate


def _safe_session_stem(session_id: str) -> str:
    """Return a filesystem-safe stem for a session id (no path separators).

    A session id becomes a filename, so any path-significant character is replaced
    with an underscore -- this is a second guard (after resolve_memory_dir) so a
    crafted id can never write outside the memory dir.
    """
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in session_id)
    return safe or "session"


class ConversationMemory:
    """Append-only per-session conversation transcript store (Data Model 9.3).

    Constructed with the resolved memory dir (via resolve_memory_dir). Each append
    writes exactly one JSON line to `<dir>/<session_id>.jsonl`, created on first
    write; existing lines are never rewritten. `enabled=False` makes every method a
    no-op so an owner can disable persistence without changing call sites.
    """

    def __init__(self, memory_dir: Path | str, enabled: bool = True) -> None:
        self._dir = Path(memory_dir)
        self._enabled = enabled

    @property
    def directory(self) -> Path:
        return self._dir

    def path_for(self, session_id: str) -> Path:
        """The JSONL file path for a session (its id is the filename stem)."""
        return self._dir / f"{_safe_session_stem(session_id)}.jsonl"

    def append(self, session_id: str, role: str, text: str) -> None:
        """Append one {session_id, ts, role, text} line. Append-only, never rewrite.

        No-op when disabled or when text is empty (an empty utterance is not a
        transcript entry). The parent dir is created on demand. Content is written
        as compact UTF-8 JSON, one object per line, newest last.
        """
        if not self._enabled or not text:
            return
        if role not in _VALID_ROLES:
            raise ValueError(f"memory role must be one of {_VALID_ROLES}, got {role!r}")
        self._dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "session_id": session_id,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "role": role,
            "text": text,
        }
        line = json.dumps(entry, ensure_ascii=False)
        # Mode "a" appends; the file handle never truncates an existing file, so a
        # prior line is physically impossible to overwrite here.
        with open(self.path_for(session_id), "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def list_sessions(self) -> list[str]:
        """Return session ids (JSONL stems) newest-first by file mtime (M8.3 recall).

        The recall surface lists conversations so an owner can see what is stored
        and load the most recent. Newest-first (by last-modified time) matches "what
        were we just talking about". Missing dir -> empty list (honest, not an error).
        """
        if not self._dir.exists():
            return []
        files = [p for p in self._dir.glob("*.jsonl") if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.stem for p in files]

    def recall(self, limit: int) -> list[MemoryEntry]:
        """Return up to `limit` most-recent entries across the most recent session.

        The "load recent" recall verb (Data Model 9.1 recall_recent_n): it reads the
        single most-recently-modified session file and returns its trailing entries,
        oldest-first, so a recalled preamble reads in conversation order. Empty store
        -> empty list.
        """
        if limit <= 0:
            return []
        sessions = self.list_sessions()
        if not sessions:
            return []
        return self.recent(sessions[0], limit)

    def search(self, keyword: str, limit: int) -> list[MemoryEntry]:
        """Case-insensitive substring search across every stored transcript (M8.3).

        Deliberately a simple linear scan -- NOT an embedding/vector query (that is
        explicitly future, no heavy deps). Scans every session file, matching the
        keyword against entry text case-insensitively, and returns up to `limit`
        matches newest-first (most-recent session first, newest line first within a
        session) so the freshest relevant context surfaces. An empty keyword or a
        non-positive limit returns nothing rather than everything.
        """
        keyword = (keyword or "").strip()
        if not keyword or limit <= 0:
            return []
        needle = keyword.casefold()
        matches: list[MemoryEntry] = []
        # list_sessions() is newest-first; scanning each session's lines in reverse
        # gives newest-line-first, so the first `limit` hits are the freshest.
        for session_id in self.list_sessions():
            for entry in reversed(self.recent(session_id, limit=_ALL_ENTRIES)):
                if needle in entry.text.casefold():
                    matches.append(entry)
                    if len(matches) >= limit:
                        return matches
        return matches

    def recent(self, session_id: str, limit: int) -> list[MemoryEntry]:
        """Return up to `limit` most-recent entries for a session, oldest-first.

        Reads the append-only file top to bottom (chronological order) and returns
        the trailing `limit` entries. Missing file or a malformed line degrades to
        an honest partial result rather than crashing the assistant.
        """
        if limit <= 0:
            return []
        path = self.path_for(session_id)
        if not path.exists():
            return []
        entries: list[MemoryEntry] = []
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except ValueError:
                    continue
                entries.append(
                    MemoryEntry(
                        session_id=str(obj.get("session_id", session_id)),
                        ts=str(obj.get("ts", "")),
                        role=str(obj.get("role", "")),
                        text=str(obj.get("text", "")),
                    )
                )
        return entries[-limit:]
