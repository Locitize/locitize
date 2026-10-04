"""Session integration acceptance: local I/O, isolation, recovery, native controls."""
import json
import os
import time
from types import SimpleNamespace

import pytest

from session_core.models import Session, ThreadMessage
from session_launch import build_local_launch
from session_service import SessionService
from session_store import SessionStore


class FixtureProvider:
    key = "codex"
    label = "Codex"

    def __init__(self, root):
        self.root = root

    def detected(self):
        return True

    def load_sessions(self):
        return [Session("session-1", "codex", str(self.root), display="Build a calculator", timestamp=1700000000000)]

    def collect_thread(self, session):
        return [ThreadMessage("user", "Calculate 2 + 2"), ThreadMessage("assistant", "4\n```\n[link](https://example.invalid)")]


def test_metadata_identity_is_provider_scoped_and_persistent(tmp_path):
    store = SessionStore(tmp_path)
    store.update("codex", "same", title="Code", pinned=True)
    store.update("claude", "same", title="Other")
    store.update("codex", "same", notes="keep this", archived=True)
    rows = SessionStore(tmp_path).all()
    assert rows[("codex", "same")]["notes"] == "keep this"
    assert rows[("claude", "same")] == {"title": "Other"}
    assert rows[("codex", "same")]["pinned"]


def test_backup_restore_is_non_destructive_and_idempotent(tmp_path):
    source, target = SessionStore(tmp_path / "a"), SessionStore(tmp_path / "b")
    source.update("codex", "one", notes="saved")
    source.update("codex", "two", pinned=True)
    target.update("codex", "one", notes="newer")
    backup = tmp_path / "backup.json"
    assert source.backup(backup) == 2
    assert target.restore(backup) == 1
    assert target.restore(backup) == 0
    assert target.all()[("codex", "one")]["notes"] == "newer"
    assert target.path.with_suffix(".before-import.sqlite3").is_file()


def test_bad_backup_never_partially_imports(tmp_path):
    store = SessionStore(tmp_path)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 1, "records": [
        {"provider": "codex", "id": "one", "metadata": {"title": "ok"}},
        {"provider": "codex", "id": "two", "metadata": {"command": "untrusted"}}]}))
    with pytest.raises(ValueError):
        store.restore(path)
    assert store.all() == {}


def test_future_schema_is_not_overwritten(tmp_path):
    import sqlite3
    store = SessionStore(tmp_path)
    store.update("codex", "one", title="kept")
    with sqlite3.connect(store.path) as db:
        db.execute("PRAGMA user_version=999")
    before = store.path.read_bytes()
    with pytest.raises(ValueError, match="newer"):
        store.all()
    assert store.path.read_bytes() == before


def test_legacy_import_preserves_sources_and_ambiguous_ids(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    rename = legacy / "renames.json"
    rename.write_text(json.dumps({"one": "Imported", "same": "Ambiguous"}))
    (legacy / "session_meta.json").write_text(json.dumps({"one": {"note": "A note", "tags": ["work"], "pinned": True}}))
    before = rename.read_bytes()
    sessions = [Session("one", "codex", ""), Session("same", "codex", ""), Session("same", "claude", "")]
    store = SessionStore(tmp_path / "data")
    assert store.import_portal(legacy, sessions) == 1
    assert store.import_portal(legacy, sessions) == 0
    assert rename.read_bytes() == before
    assert store.all()[("codex", "one")]["notes"] == "A note"


def test_scan_is_local_and_tolerates_a_broken_provider(tmp_path):
    class Broken(FixtureProvider):
        key = "broken"
        def load_sessions(self):
            raise ValueError("bad metadata")
    service = SessionService(tmp_path, [FixtureProvider(tmp_path), Broken(tmp_path)])
    sessions, errors = service.scan()
    assert len(sessions) == 1 and len(errors) == 1
    from session_core.providers.registry import PROVIDERS
    assert "amp" not in [p.key for p in PROVIDERS]


def test_export_fences_untrusted_markdown_and_preserves_source(tmp_path):
    service = SessionService(tmp_path, [FixtureProvider(tmp_path)])
    s = service.scan()[0][0]
    source = tmp_path / "original.jsonl"
    source.write_text("original")
    s.source_file = str(source)
    with pytest.raises(ValueError):
        service.export(s, source)
    destination = tmp_path / "export.md"
    service.export(s, destination)
    assert "````text" in destination.read_text()
    assert source.read_text() == "original"


@pytest.mark.parametrize("provider", ["codex", "claude", "opencode"])
def test_local_resume_uses_only_process_configuration(tmp_path, provider):
    before = set(tmp_path.iterdir())
    argv, env = build_local_launch(provider, str(tmp_path), "local-model", 8080, "session-123")
    assert "session-123" in argv
    assert "127.0.0.1:8080" in " ".join(argv) + str(env)
    assert set(tmp_path.iterdir()) == before
    assert not any(x in argv for x in ("--dangerously-skip-permissions", "--yolo"))


@pytest.mark.parametrize("sid", ["x;echo bad", "--last", "x\ny", "x&y"])
def test_local_resume_rejects_untrusted_identity(tmp_path, sid):
    with pytest.raises(ValueError):
        build_local_launch("codex", str(tmp_path), "local", 8080, sid)


def test_local_launch_rejects_missing_project(tmp_path):
    with pytest.raises(ValueError, match="folder"):
        build_local_launch("codex", str(tmp_path / "missing"), "local", 8080)


def test_jsonl_reader_caps_large_lines_and_skips_non_objects(tmp_path, monkeypatch):
    from session_core.providers import base
    monkeypatch.setattr(base, "MAX_JSON_LINE_CHARS", 64)
    path = tmp_path / "history.jsonl"
    path.write_bytes(b'[]\n' + b'x' * 200 + b'\n{"ok":true}\n')
    assert list(base.iter_jsonl_records(path)) == [{"ok": True}]
    assert list(base.iter_jsonl_records(path, max_bytes=80)) == []


def test_jsonl_reader_tail_does_not_parse_a_partial_record(tmp_path, monkeypatch):
    from session_core.providers import base
    monkeypatch.setattr(base, "MAX_INDEX_BYTES", 32)
    path = tmp_path / "history.jsonl"
    path.write_bytes(b'{"long":"' + b'x' * 100 + b'"}\n{"new":1}\n')
    assert list(base.iter_jsonl_records(path)) == [{"new": 1}]


def test_qt_session_search_edit_archive_reopen(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from sessions_ui import SessionsPage
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    service = SessionService(tmp_path, [FixtureProvider(tmp_path)])
    page = SessionsPage(SimpleNamespace(), service)
    page.show()
    page.refresh_btn.click()
    deadline = time.monotonic() + 10
    while page._scanning and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert page.table.rowCount() == 1
    page.table.selectRow(0)
    while page._jobs and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert "Calculate 2 + 2" in page.preview.toPlainText()
    page.search.setText("Calculate 2 + 2")
    assert page.table.rowCount() == 0
    page.content_btn.click()
    while page._search_cancel is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert page.table.rowCount() == 1
    page.search.clear()
    page.table.selectRow(0)
    page.notes.setPlainText("release-check")
    page.pinned.setChecked(True)
    page.save_btn.click()
    page.search.setText("release-check")
    assert page.table.rowCount() == 1
    page.table.selectRow(0)
    page.archive_btn.click()
    assert page.table.rowCount() == 0
    page.archived.setChecked(True)
    assert page.table.rowCount() == 1
    page.table.selectRow(0)
    page.archive_btn.click()
    assert not SessionStore(tmp_path).all()[("codex", "session-1")]["archived"]
    while page._jobs and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    page.close()
