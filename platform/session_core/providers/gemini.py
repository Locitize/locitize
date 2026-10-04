"""Gemini CLI session provider.

Layout (verified against google-gemini/gemini-cli source, mid-2026):
``~/.gemini/tmp/<project-id>/chats/session-*.jsonl`` where <project-id> is a
registry slug (current) or a sha256 hex (legacy). A ``.project_root`` marker
file inside each project dir names the owning project path. Line 1 of each
JSONL file is a metadata record; message records follow. Subagent transcripts
live in SUBDIRECTORIES of chats/ and are excluded. Legacy sessions are a
single ``session-*.json`` object.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..config import GEMINI_DIR, GEMINI_TMP_DIR
from ..models import Preview, ResumeCommand, Session, ThreadMessage, Tokens
from ..resume import ps_single_quote
from .base import (
    MAX_INDEX_BYTES,
    MAX_METADATA_SCAN_BYTES,
    clip_preview_text,
    iter_jsonl_records,
    keep_thread_tail,
    remember_first_last,
)

PROJECT_ROOT_MARKER = ".project_root"
MAX_LEGACY_JSON_BYTES = 8 * 1024 * 1024


def _content_text(content) -> str:
    """Gemini `content` is a string or a genai part list [{"text": ...}]."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts).strip()
    return ""


def _iso_to_ms(raw: str) -> int:
    try:
        return int(datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp() * 1000)
    except (ValueError, TypeError):
        return 0


class GeminiProvider:
    key = "gemini"
    label = "Gemini CLI"

    def detected(self) -> bool:
        return GEMINI_TMP_DIR.exists() or GEMINI_DIR.exists()

    # -- discovery -----------------------------------------------------------
    def _project_for_dir(self, project_dir: Path) -> str:
        marker = project_dir / PROJECT_ROOT_MARKER
        try:
            if marker.is_file():
                root = marker.read_text(encoding="utf-8", errors="replace").strip()
                if root:
                    return root
        except OSError:
            pass
        return ""

    def _session_files(self) -> list[tuple[Path, str]]:
        """(file, project_path) for every main-session chat file."""
        out: list[tuple[Path, str]] = []
        if not GEMINI_TMP_DIR.exists():
            return out
        for project_dir in GEMINI_TMP_DIR.iterdir():
            if not project_dir.is_dir():
                continue
            chats = project_dir / "chats"
            if not chats.is_dir():
                continue
            project = self._project_for_dir(project_dir)
            for fp in chats.iterdir():
                # Subagent transcripts are in subdirectories - skip them.
                if fp.is_file() and fp.name.startswith("session-") and fp.suffix in (".jsonl", ".json"):
                    out.append((fp, project))
        return out

    def _records(self, fp: Path, max_bytes: int | None = None):
        """Yield records from JSONL, or from a legacy single-JSON file."""
        if fp.suffix == ".jsonl":
            yield from iter_jsonl_records(fp, max_bytes=max_bytes)
            return
        try:
            if fp.stat().st_size > MAX_LEGACY_JSON_BYTES:
                return
            data = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, dict):
            meta = {k: v for k, v in data.items() if k != "messages"}
            yield meta
            for rec in data.get("messages", []):
                if isinstance(rec, dict):
                    yield rec

    def _metadata(self, fp: Path) -> dict:
        """Line-1 metadata merged with any later {"$set": {...}} overrides."""
        meta: dict = {}
        for rec in self._records(fp, max_bytes=MAX_METADATA_SCAN_BYTES):
            if not meta and rec.get("sessionId"):
                meta = dict(rec)
            override = rec.get("$set")
            if isinstance(override, dict):
                meta.update({k: v for k, v in override.items() if k != "messages"})
        return meta

    # -- protocol ------------------------------------------------------------
    def load_sessions(self) -> list[Session]:
        result: list[Session] = []
        for fp, project in self._session_files():
            meta = self._metadata(fp)
            if meta.get("kind") == "subagent":
                continue
            sid = str(meta.get("sessionId") or fp.stem)
            ts = _iso_to_ms(meta.get("lastUpdated") or meta.get("startTime") or "")
            if not ts:
                try:
                    ts = int(fp.stat().st_mtime * 1000)
                except OSError:
                    ts = 0
            model = ""
            display = str(meta.get("summary") or "").strip()
            for rec in self._records(fp, max_bytes=MAX_METADATA_SCAN_BYTES):
                if rec.get("type") == "gemini" and isinstance(rec.get("model"), str):
                    model = rec["model"]
                if not display and rec.get("type") == "user":
                    display = clip_preview_text(_content_text(rec.get("content")))
            result.append(Session(
                id=sid,
                provider="gemini",
                project=project,
                model=model,
                model_group=f"Gemini CLI / {model}" if model else "Gemini CLI / Unknown",
                display=display,
                timestamp=ts,
                resumable=True,
                source_file=str(fp),
            ))
        return result

    def preview(self, session: Session) -> Preview:
        fp = Path(session.source_file) if session.source_file else None
        if not fp or not fp.exists():
            return Preview()
        first = last = None
        count = 0
        tokens = Tokens()
        for rec in self._records(fp):
            if rec.get("type") == "user":
                text = _content_text(rec.get("content"))
                if text:
                    first, last, count = remember_first_last(first, last, count, text)
            elif rec.get("type") == "gemini":
                usage = rec.get("tokens")
                if isinstance(usage, dict):
                    tokens.input += int(usage.get("input", 0) or 0)
                    tokens.output += int(usage.get("output", 0) or 0)
                    tokens.cache_read += int(usage.get("cached", 0) or 0)
        return Preview(first=first, last=last, message_count=count, tokens=tokens)

    def collect_messages(self, session: Session) -> list[str]:
        fp = Path(session.source_file) if session.source_file else None
        if not fp or not fp.exists():
            return []
        out = []
        for rec in self._records(fp, max_bytes=MAX_INDEX_BYTES):
            if rec.get("type") == "user":
                text = _content_text(rec.get("content"))
                if text:
                    out.append(clip_preview_text(text, limit=2000))
        return out

    def collect_thread(self, session: Session) -> list[ThreadMessage]:
        fp = Path(session.source_file) if session.source_file else None
        if not fp or not fp.exists():
            return []
        msgs: list[ThreadMessage] = []
        total = 0
        for rec in self._records(fp, max_bytes=MAX_INDEX_BYTES):
            kind = rec.get("type")
            if kind not in ("user", "gemini"):
                continue
            text = _content_text(rec.get("content"))
            if not text:
                continue
            text = clip_preview_text(text, limit=4000)
            role = "user" if kind == "user" else "assistant"
            msgs.append(ThreadMessage(role, text, rec.get("model", "") if kind == "gemini" else ""))
            total += len(text)
            total = keep_thread_tail(msgs, total)
        return msgs

    def delete(self, session: Session) -> None:
        if session.source_file:
            try:
                Path(session.source_file).unlink(missing_ok=True)
            except OSError:
                pass

    def resume_command(self, session: Session) -> ResumeCommand:
        # Resume is project-scoped: must run from the recorded project dir.
        cwd = session.project if session.project and Path(session.project).exists() else str(Path.home())
        return ResumeCommand(cwd=cwd, shell_command=f"gemini --resume {ps_single_quote(session.id)}")
