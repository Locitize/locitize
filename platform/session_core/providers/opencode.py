"""opencode session provider.

Layout (verified against anomalyco/opencode source, mid-2026): current
releases store sessions in SQLite at ``~/.local/share/opencode/opencode.db``
(xdg-basedir uses the Unix-style path even on Windows). Tables: ``session``
(id, directory, title, model JSON, time_created/updated epoch-ms, tokens_*),
``message`` (data JSON with role), ``part`` (data JSON; text parts carry the
message text). Older installs used JSON trees under ``project/<slug>/storage/
session/info/*.json`` - also read here.

opencode's database belongs to opencode: this provider opens it strictly
READ-ONLY and never deletes from it. Delete in Session Portal therefore hides
the row locally (``hide_only``), exactly like AMP's server-backed threads.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..config import OPENCODE_DATA_DIR, OPENCODE_DB_FILE
from ..logging_setup import get_logger
from ..models import Preview, ResumeCommand, Session, ThreadMessage, Tokens
from ..resume import ps_single_quote
from .base import clip_preview_text, keep_thread_tail, remember_first_last

logger = get_logger(__name__)

MAX_DB_MESSAGES = 2000


def _connect_readonly(db_file: Path) -> sqlite3.Connection:
    uri = f"file:{db_file.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=1.0)


def _model_name(raw) -> str:
    """session.model is JSON like {"id": ..., "providerID": ...}."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return ""
    if isinstance(data, dict):
        return str(data.get("id") or "")
    return ""


class OpencodeProvider:
    key = "opencode"
    label = "OpenCode"
    # Sessions live in opencode's own database; Session Portal never mutates
    # it. Delete hides the row locally instead (same pathway as AMP).
    hide_only = True

    def detected(self) -> bool:
        return OPENCODE_DATA_DIR.exists()

    # -- SQLite (current releases) -------------------------------------------
    def _load_db_sessions(self) -> list[Session]:
        if not OPENCODE_DB_FILE.exists():
            return []
        result: list[Session] = []
        try:
            con = _connect_readonly(OPENCODE_DB_FILE)
            try:
                rows = con.execute(
                    "SELECT id, directory, title, model, time_created, time_updated,"
                    " tokens_input, tokens_output, tokens_cache_read, tokens_cache_write"
                    " FROM session WHERE parent_id IS NULL"
                ).fetchall()
            finally:
                con.close()
        except sqlite3.Error:
            logger.exception("Failed to read opencode.db session table")
            return []
        for (sid, directory, title, model_raw, created, updated,
             tok_in, tok_out, tok_cr, tok_cw) in rows:
            model = _model_name(model_raw)
            tokens = Tokens(
                input=int(tok_in or 0), output=int(tok_out or 0),
                cache_read=int(tok_cr or 0), cache_write=int(tok_cw or 0),
            )
            result.append(Session(
                id=str(sid),
                provider="opencode",
                project=str(directory or ""),
                model=model,
                model_group=f"OpenCode / {model}" if model else "OpenCode / Unknown",
                display=str(title or ""),
                timestamp=int(updated or created or 0),
                resumable=True,
                source_file=str(OPENCODE_DB_FILE),
                tokens=tokens,
            ))
        return result

    def _db_messages(self, session_id: str) -> list[tuple[str, str]]:
        """(role, text) pairs for one session, oldest first, from the DB."""
        if not OPENCODE_DB_FILE.exists():
            return []
        try:
            con = _connect_readonly(OPENCODE_DB_FILE)
            try:
                rows = con.execute(
                    "SELECT m.id, m.data,"
                    " (SELECT group_concat(p.data, char(10)) FROM part p"
                    "   WHERE p.message_id = m.id) AS parts"
                    " FROM message m WHERE m.session_id = ?"
                    " ORDER BY m.time_created LIMIT ?",
                    (session_id, MAX_DB_MESSAGES),
                ).fetchall()
            finally:
                con.close()
        except sqlite3.Error:
            logger.exception("Failed to read opencode.db messages for %s", session_id)
            return []
        out: list[tuple[str, str]] = []
        for _mid, mdata_raw, parts_raw in rows:
            try:
                mdata = json.loads(mdata_raw) if mdata_raw else {}
            except json.JSONDecodeError:
                continue
            role = str(mdata.get("role") or "")
            texts = []
            for chunk in (parts_raw or "").split("\n"):
                try:
                    pdata = json.loads(chunk) if chunk else {}
                except json.JSONDecodeError:
                    continue
                if isinstance(pdata, dict) and pdata.get("type") == "text" and isinstance(pdata.get("text"), str):
                    texts.append(pdata["text"])
            text = " ".join(texts).strip()
            if role in ("user", "assistant") and text:
                out.append((role, text))
        return out

    # -- Legacy JSON layout (pre-SQLite installs) ----------------------------
    def _load_legacy_sessions(self) -> list[Session]:
        result: list[Session] = []
        info_globs = [
            OPENCODE_DATA_DIR / "project",   # layout 1: project/<slug>/storage/session/info/*.json
        ]
        for root in info_globs:
            if not root.exists():
                continue
            for info in root.glob("*/storage/session/info/*.json"):
                try:
                    data = json.loads(info.read_text(encoding="utf-8", errors="replace"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(data, dict) or not data.get("id"):
                    continue
                time_info = data.get("time") or {}
                result.append(Session(
                    id=str(data["id"]),
                    provider="opencode",
                    project=info.parents[3].name,  # project slug; real path unrecorded
                    model="",
                    model_group="OpenCode / Unknown",
                    display=str(data.get("title") or ""),
                    timestamp=int(time_info.get("updated") or time_info.get("created") or 0),
                    resumable=True,
                    source_file=str(info),
                ))
        return result

    # -- protocol ------------------------------------------------------------
    def load_sessions(self) -> list[Session]:
        db_sessions = self._load_db_sessions()
        if db_sessions:
            return db_sessions
        return self._load_legacy_sessions()

    def preview(self, session: Session) -> Preview:
        pairs = self._db_messages(session.id)
        first = last = None
        count = 0
        for role, text in pairs:
            if role == "user":
                first, last, count = remember_first_last(first, last, count, text)
        return Preview(first=first, last=last, message_count=count,
                       tokens=session.tokens or Tokens())

    def collect_messages(self, session: Session) -> list[str]:
        return [clip_preview_text(text, limit=2000)
                for role, text in self._db_messages(session.id) if role == "user"]

    def collect_thread(self, session: Session) -> list[ThreadMessage]:
        msgs: list[ThreadMessage] = []
        total = 0
        for role, text in self._db_messages(session.id):
            text = clip_preview_text(text, limit=4000)
            msgs.append(ThreadMessage(role, text, session.model if role == "assistant" else ""))
            total += len(text)
            total = keep_thread_tail(msgs, total)
        return msgs

    def delete(self, session: Session) -> None:
        # Never reached for hide_only providers; kept as a hard guarantee.
        raise OSError("opencode sessions are stored in opencode's own database; Session Portal only hides them locally")

    def resume_command(self, session: Session) -> ResumeCommand:
        cwd = session.project if session.project and Path(session.project).exists() else str(Path.home())
        return ResumeCommand(cwd=cwd, shell_command=f"opencode --session {ps_single_quote(session.id)}")
