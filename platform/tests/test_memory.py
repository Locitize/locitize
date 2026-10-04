"""Conversation-memory tests (keyword 'memory_store', Data Model 9.3, M8.3 hook).

Filesystem-only and deterministic (a temp dir per test), no real assistant session.
M7 wires the assistant to EMIT transcript records through append(); these tests
prove append-only behavior, the exact JSONL schema, chronological recent() reads,
and the path-safety guard that keeps transcripts inside the platform tree. The
richer search() surface is finished in M8.
"""

from __future__ import annotations

import json

from memory import ConversationMemory, MemoryEntry, resolve_memory_dir


def test_memory_store_append_writes_schema(tmp_path):
    """append() writes one JSONL line with exactly {session_id, ts, role, text}."""
    store = ConversationMemory(tmp_path / "memory")
    store.append("sess-1", "user", "what did we decide")

    path = store.path_for("sess-1")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert set(obj.keys()) == {"session_id", "ts", "role", "text"}
    assert obj["session_id"] == "sess-1"
    assert obj["role"] == "user"
    assert obj["text"] == "what did we decide"


def test_memory_store_is_append_only(tmp_path):
    """A second append never rewrites the first line (append-only, Data Model 9.3)."""
    store = ConversationMemory(tmp_path / "memory")
    store.append("sess-1", "user", "first")
    first_line = store.path_for("sess-1").read_text(encoding="utf-8").splitlines()[0]

    store.append("sess-1", "assistant", "second")
    lines = store.path_for("sess-1").read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    # The originally written line is byte-identical after the second append.
    assert lines[0] == first_line


def test_memory_store_recent_is_chronological(tmp_path):
    """recent() returns the trailing N entries oldest-first."""
    store = ConversationMemory(tmp_path / "memory")
    for i in range(5):
        store.append("s", "user", f"msg {i}")

    recent = store.recent("s", limit=3)
    assert [e.text for e in recent] == ["msg 2", "msg 3", "msg 4"]
    assert all(isinstance(e, MemoryEntry) for e in recent)


def test_memory_store_search_matches_case_insensitive(tmp_path):
    """search() finds entries by case-insensitive substring across sessions (M8.3)."""
    store = ConversationMemory(tmp_path / "memory")
    store.append("s1", "user", "what did we decide about the 27B target")
    store.append("s1", "assistant", "the 27B campaign was deferred by the owner")
    store.append("s2", "user", "unrelated chatter about the weather")

    hits = store.search("27b", limit=5)
    texts = [h.text for h in hits]
    assert any("27B target" in t for t in texts)
    assert any("27B campaign" in t for t in texts)
    assert all("weather" not in t for t in texts)


def test_memory_store_search_respects_limit(tmp_path):
    """search() returns at most `limit` matches, never more."""
    store = ConversationMemory(tmp_path / "memory")
    for i in range(6):
        store.append("s", "user", f"apples entry {i}")
    hits = store.search("apples", limit=3)
    assert len(hits) == 3


def test_memory_store_search_empty_keyword_returns_nothing(tmp_path):
    """An empty keyword matches nothing (not everything)."""
    store = ConversationMemory(tmp_path / "memory")
    store.append("s", "user", "something")
    assert store.search("", limit=5) == []


def test_memory_store_recall_loads_recent_session_tail(tmp_path):
    """recall() returns the most-recent session's trailing entries, oldest-first."""
    store = ConversationMemory(tmp_path / "memory")
    for i in range(4):
        store.append("only-session", "user", f"line {i}")
    recalled = store.recall(limit=2)
    assert [e.text for e in recalled] == ["line 2", "line 3"]


def test_memory_store_list_sessions(tmp_path):
    """list_sessions() reports the session ids that have a transcript file."""
    store = ConversationMemory(tmp_path / "memory")
    store.append("alpha", "user", "hi")
    store.append("beta", "user", "yo")
    assert set(store.list_sessions()) == {"alpha", "beta"}


def test_memory_store_disabled_is_noop(tmp_path):
    """enabled=False persists nothing (no file created)."""
    store = ConversationMemory(tmp_path / "memory", enabled=False)
    store.append("s", "user", "ignored")
    assert not store.path_for("s").exists()


def test_memory_store_dir_cannot_escape_base(tmp_path):
    """resolve_memory_dir refuses a path that would escape the platform tree."""
    base = tmp_path / "platform"
    base.mkdir()
    escaped = resolve_memory_dir(base, "../../etc")
    # The escape attempt falls back to the safe default under base.
    assert escaped == base / "memory"
    # A normal relative dir resolves under base as expected.
    assert resolve_memory_dir(base, "memory") == (base / "memory").resolve()
