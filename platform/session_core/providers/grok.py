"""Grok CLI session provider (ported faithfully from v1)."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

from ..config import GROK_SESSIONS_DIR
from ..models import Preview, ResumeCommand, Session, ThreadMessage
from ..resume import find_grok_exe, ps_single_quote
from .base import (
    MAX_INDEX_BYTES,
    clip_preview_text,
    iter_jsonl_records,
    keep_thread_tail,
    path_is_under,
    remember_first_last,
)


class GrokProvider:
    key = "grok"
    label = "Grok"

    def detected(self) -> bool:
        return GROK_SESSIONS_DIR.exists()

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _parse_iso_ms(value: str) -> int:
        if not value:
            return 0
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except ValueError:
            return 0

    @staticmethod
    def _decode_cwd(encoded: str) -> str:
        return unquote(encoded)

    @staticmethod
    def _extract_user_text(content) -> str:
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            text = " ".join(parts).strip()
        else:
            return ""
        if "<user_query>" in text and "</user_query>" in text:
            text = text.split("<user_query>", 1)[1].split("</user_query>", 1)[0].strip()
        if text.startswith("<user_info>") or text.startswith("<rules>") or text.startswith("<system-reminder>"):
            return ""
        return text

    def _preview_from_chat(self, fp: Path):
        first = None
        last = None
        count = 0
        for rec in iter_jsonl_records(fp):
            if rec.get("type") == "user":
                text = self._extract_user_text(rec.get("content", ""))
                if text:
                    first, last, count = remember_first_last(first, last, count, text)
        return first, last, count

    # -- protocol ------------------------------------------------------------
    @staticmethod
    def _extract_assistant_text(content) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            return " ".join(parts).strip()
        return ""

    def collect_messages(self, session: Session) -> list[str]:
        fp = Path(session.source_file) if session.source_file else None
        if not fp or not fp.exists() or fp.name != "chat_history.jsonl":
            return []
        out: list[str] = []
        for rec in iter_jsonl_records(fp):
            if rec.get("type") == "user":
                text = self._extract_user_text(rec.get("content", ""))
                if text:
                    out.append(clip_preview_text(text, limit=2000))
        return out

    def collect_thread(self, session: Session) -> list[ThreadMessage]:
        fp = Path(session.source_file) if session.source_file else None
        if not fp or not fp.exists() or fp.name != "chat_history.jsonl":
            return []
        msgs: list[ThreadMessage] = []
        total = 0
        for rec in iter_jsonl_records(fp, max_bytes=MAX_INDEX_BYTES):
            rtype = rec.get("type")
            if rtype == "user":
                text = self._extract_user_text(rec.get("content", ""))
                if text:
                    clipped = clip_preview_text(text, limit=4000)
                    msgs.append(ThreadMessage("user", clipped))
                    total += len(clipped)
                    total = keep_thread_tail(msgs, total)
            elif rtype == "assistant":
                text = self._extract_assistant_text(rec.get("content", ""))
                if text:
                    clipped = clip_preview_text(text, limit=4000)
                    msgs.append(ThreadMessage("assistant", clipped, session.model))
                    total += len(clipped)
                    total = keep_thread_tail(msgs, total)
        return msgs

    def load_sessions(self) -> list[Session]:
        if not GROK_SESSIONS_DIR.exists():
            return []
        entries: list[Session] = []
        for session_dir in GROK_SESSIONS_DIR.rglob("*"):
            if not session_dir.is_dir():
                continue
            summary_file = session_dir / "summary.json"
            chat_file = session_dir / "chat_history.jsonl"
            if not summary_file.exists() and not chat_file.exists():
                continue
            sid = session_dir.name
            if len(sid) != 36 or sid.count("-") != 4:
                continue
            summary = {}
            if summary_file.exists():
                try:
                    summary = json.loads(summary_file.read_text(encoding="utf-8"))
                except Exception:
                    summary = {}
            info = summary.get("info", {}) if isinstance(summary, dict) else {}
            cwd = info.get("cwd", "")
            if not cwd:
                cwd = self._decode_cwd(session_dir.parent.name)
            model = summary.get("current_model_id", "") or ""
            display = summary.get("generated_title") or summary.get("session_summary") or ""
            if not display and chat_file.exists():
                first, _, _ = self._preview_from_chat(chat_file)
                display = first or ""
            ts = (
                self._parse_iso_ms(summary.get("last_active_at", ""))
                or self._parse_iso_ms(summary.get("updated_at", ""))
                or int(session_dir.stat().st_mtime * 1000)
            )
            entries.append(Session(
                id=sid,
                provider="grok",
                project=cwd,
                model=model,
                model_group=f"Grok / {model}" if model else "Grok / Unknown",
                display=display,
                timestamp=ts,
                resumable=True,
                source_file=str(chat_file if chat_file.exists() else summary_file),
                session_dir=str(session_dir),
            ))
        return entries

    def preview(self, session: Session) -> Preview:
        fp = Path(session.source_file) if session.source_file else None
        if not fp or not fp.exists():
            return Preview()
        if fp.name == "chat_history.jsonl":
            first, last, count = self._preview_from_chat(fp)
            return Preview(first=first, last=last, message_count=count)
        try:
            text = fp.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            text = ""
        return Preview(first=text[:4000] or None, message_count=len(text.splitlines()))

    def delete(self, session: Session) -> None:
        session_dir = Path(session.session_dir or "")
        if session_dir.exists() and session_dir.is_dir() and path_is_under(session_dir, GROK_SESSIONS_DIR):
            try:
                shutil.rmtree(session_dir)
            except OSError:
                pass

    def resume_command(self, session: Session) -> ResumeCommand:
        exe = find_grok_exe()
        sid = session.id
        if "\\" in exe:
            command = f"& {ps_single_quote(exe)} --resume {ps_single_quote(sid)}"
        else:
            command = f"{exe} --resume {ps_single_quote(sid)}"
        cwd = session.project or str(Path.home())
        return ResumeCommand(cwd=cwd, shell_command=command)
