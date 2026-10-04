"""Versioned, transactional LOCITIZE annotations. Never edits provider histories."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1
MAX_BACKUP_BYTES = 8 * 1024 * 1024
FIELDS = {"title", "tags", "notes", "pinned", "archived", "model_id", "project"}


def validate_metadata(value):
    if not isinstance(value, dict) or set(value) - FIELDS:
        raise ValueError("Unrecognized session metadata fields")
    for key, item in value.items():
        if key in {"pinned", "archived"}:
            if not isinstance(item, bool):
                raise ValueError(f"{key} must be true or false")
        elif not isinstance(item, str) or len(item) > 16000:
            raise ValueError(f"Invalid {key}")
    return value


class SessionStore:
    def __init__(self, data_dir: Path):
        self.path = Path(data_dir).resolve() / "sessions" / "annotations.sqlite3"

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=5)
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            db.close()
            raise ValueError("Session data was created by a newer LOCITIZE. Update the app.")
        db.execute("CREATE TABLE IF NOT EXISTS annotations "
                   "(provider TEXT NOT NULL, id TEXT NOT NULL, metadata TEXT NOT NULL, "
                   "PRIMARY KEY(provider,id))")
        db.execute("PRAGMA user_version=1")
        db.commit()
        return db

    def all(self):
        db = self._connect()
        try:
            return {(p, sid): validate_metadata(json.loads(raw)) for p, sid, raw in
                    db.execute("SELECT provider,id,metadata FROM annotations")}
        finally:
            db.close()

    def update(self, provider: str, sid: str, **changes):
        validate_metadata(changes)
        if not provider or not sid or len(sid) > 512:
            raise ValueError("Invalid session identity")
        db = self._connect()
        try:
            with db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT metadata FROM annotations WHERE provider=? AND id=?",
                                 (provider, sid)).fetchone()
                metadata = validate_metadata(json.loads(row[0])) if row else {}
                metadata.update(changes)
                db.execute("INSERT OR REPLACE INTO annotations VALUES (?,?,?)",
                           (provider, sid, json.dumps(metadata)))
            return metadata
        finally:
            db.close()

    def backup(self, destination: Path):
        from config import _atomic_write

        records = [{"provider": p, "id": sid, "metadata": data}
                   for (p, sid), data in self.all().items()]
        destination = Path(destination).resolve()
        if destination == self.path:
            raise ValueError("Choose a different backup destination")
        _atomic_write(destination, json.dumps({"version": 1, "records": records}, indent=2))
        return len(records)

    def restore(self, source: Path):
        """Merge a validated backup without replacing existing annotations.

        Validation completes before any mutation. An SQLite recovery copy is
        created before the single import transaction. Repeated imports are safe.
        """
        source = Path(source).resolve()
        if source.stat().st_size > MAX_BACKUP_BYTES:
            raise ValueError("Session backup is too large")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("Unsupported session backup version")
        records = payload.get("records")
        if not isinstance(records, list) or len(records) > 20000:
            raise ValueError("Invalid session backup")
        parsed = []
        for row in records:
            if not isinstance(row, dict):
                raise ValueError("Invalid session backup record")
            p, sid = row.get("provider"), row.get("id")
            if not all(isinstance(s, str) and 0 < len(s) <= 512 for s in (p, sid)):
                raise ValueError("Invalid session identity")
            parsed.append((p, sid, json.dumps(validate_metadata(row.get("metadata")))))
        db = self._connect()
        try:
            recovery = sqlite3.connect(self.path.with_suffix(".before-import.sqlite3"))
            try:
                db.backup(recovery)
            finally:
                recovery.close()
            before = db.total_changes
            with db:
                db.executemany("INSERT OR IGNORE INTO annotations VALUES (?,?,?)", parsed)
            return db.total_changes - before
        finally:
            db.close()

    def import_portal(self, folder: Path, sessions):
        """Import legacy annotations by matching IDs to known local providers.

        Ambiguous IDs are skipped, never applied to a different tool's session.
        Legacy files remain untouched; existing LOCITIZE metadata wins.
        """
        folder = Path(folder).resolve()
        inputs = {}
        for name in ("renames.json", "session_meta.json", "hidden_sessions.json"):
            path = folder / name
            if path.is_file():
                if path.stat().st_size > MAX_BACKUP_BYTES:
                    raise ValueError("Legacy metadata is too large")
                item = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(item, dict):
                    raise ValueError(f"Invalid {name}")
                inputs[name] = item
        if not inputs:
            raise ValueError("No Session Portal annotation files found in this folder")
        ids = {}
        for s in sessions:
            ids.setdefault(s.id, set()).add(s.provider)
        records = []
        for s in sessions:
            data = {}
            if len(ids[s.id]) == 1:
                title = inputs.get("renames.json", {}).get(s.id)
                if isinstance(title, str):
                    data["title"] = title
                meta = inputs.get("session_meta.json", {}).get(s.id, {})
                if isinstance(meta, dict):
                    if isinstance(meta.get("note"), str):
                        data["notes"] = meta["note"]
                    for key in ("pinned", "notes", "tags"):
                        if key in meta:
                            data[key] = ", ".join(meta[key]) if key == "tags" and isinstance(meta[key], list) else meta[key]
            if s.id in inputs.get("hidden_sessions.json", {}).get(s.provider, []):
                data["archived"] = True
            if data:
                records.append({"provider": s.provider, "id": s.id, "metadata": validate_metadata(data)})
        from config import _atomic_write

        snapshot = self.path.parent / "portal-import.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(snapshot, json.dumps({"version": 1, "records": records}))
        return self.restore(snapshot)
