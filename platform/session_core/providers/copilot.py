"""GitHub Copilot CLI session provider (ported faithfully from v1)."""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from ..config import COPILOT_SESSIONS_DIR
from ..models import Preview, ResumeCommand, Session, ThreadMessage
from ..resume import ps_single_quote
from .base import (
    MAX_INDEX_BYTES,
    clip_preview_text,
    iter_jsonl_records,
    keep_thread_tail,
    path_is_under,
    remember_first_last,
)


class CopilotProvider:
    key = "copilot"
    label = "Copilot"

    def detected(self) -> bool:
        return COPILOT_SESSIONS_DIR.exists()

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
    def _read_simple_yaml(fp: Path) -> dict:
        data: dict[str, str] = {}
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or ":" not in stripped:
                        continue
                    key, value = stripped.split(":", 1)
                    data[key.strip()] = value.strip().strip('"').strip("'")
        except OSError:
            pass
        return data

    @staticmethod
    def _extract_user_text(rec: dict) -> str:
        if rec.get("type") != "user.message":
            return ""
        data = rec.get("data", {})
        if not isinstance(data, dict):
            return ""
        text = data.get("content", "")
        if not isinstance(text, str):
            return ""
        if "<current_datetime>" in text:
            parts = text.split("\n\n", 1)
            text = parts[1] if len(parts) > 1 else text
        if "<system_reminder>" in text:
            text = text.split("<system_reminder>", 1)[0]
        return text.strip()

    def _preview_from_events(self, fp: Path, max_bytes: int | None = None):
        first = None
        last = None
        count = 0
        model = ""
        for rec in iter_jsonl_records(fp, max_bytes=max_bytes):
            if rec.get("type") == "session.model_change":
                data = rec.get("data", {})
                if isinstance(data, dict) and data.get("newModel"):
                    model = str(data.get("newModel", "")).strip()
            elif rec.get("type") == "assistant.message":
                data = rec.get("data", {})
                if isinstance(data, dict) and data.get("model"):
                    model = str(data.get("model", "")).strip()
            text = self._extract_user_text(rec)
            if text:
                first, last, count = remember_first_last(first, last, count, text)
        return first, last, count, model

    # -- protocol ------------------------------------------------------------
    @staticmethod
    def _extract_assistant_text(rec: dict) -> tuple[str, str]:
        if rec.get("type") != "assistant.message":
            return "", ""
        data = rec.get("data", {})
        if not isinstance(data, dict):
            return "", ""
        text = data.get("content", "")
        if not isinstance(text, str):
            return "", ""
        return text.strip(), (data.get("model") or "")

    def collect_messages(self, session: Session) -> list[str]:
        fp = Path(session.source_file) if session.source_file else None
        if not fp or not fp.exists() or fp.name != "events.jsonl":
            return []
        out: list[str] = []
        for rec in iter_jsonl_records(fp):
            text = self._extract_user_text(rec)
            if text:
                out.append(clip_preview_text(text, limit=2000))
        return out

    def collect_thread(self, session: Session) -> list[ThreadMessage]:
        fp = Path(session.source_file) if session.source_file else None
        if not fp or not fp.exists() or fp.name != "events.jsonl":
            return []
        msgs: list[ThreadMessage] = []
        total = 0
        for rec in iter_jsonl_records(fp, max_bytes=MAX_INDEX_BYTES):
            user_text = self._extract_user_text(rec)
            if user_text:
                clipped = clip_preview_text(user_text, limit=4000)
                msgs.append(ThreadMessage("user", clipped))
                total += len(clipped)
                total = keep_thread_tail(msgs, total)
                continue
            atext, model = self._extract_assistant_text(rec)
            if atext:
                clipped = clip_preview_text(atext, limit=4000)
                msgs.append(ThreadMessage("assistant", clipped, model or session.model))
                total += len(clipped)
                total = keep_thread_tail(msgs, total)
        return msgs

    def load_sessions(self) -> list[Session]:
        if not COPILOT_SESSIONS_DIR.exists():
            return []
        entries: list[Session] = []
        for session_dir in COPILOT_SESSIONS_DIR.iterdir():
            if not session_dir.is_dir():
                continue
            sid = session_dir.name
            if len(sid) != 36 or sid.count("-") != 4:
                continue
            workspace_file = session_dir / "workspace.yaml"
            events_file = session_dir / "events.jsonl"
            if not workspace_file.exists():
                continue
            workspace = self._read_simple_yaml(workspace_file)
            cwd = workspace.get("cwd", "")
            display = workspace.get("name", "")
            model = ""
            count = 0
            if events_file.exists():
                first, _, count, model = self._preview_from_events(events_file)
                if not display:
                    display = first or ""
            ts = (
                self._parse_iso_ms(workspace.get("updated_at", ""))
                or self._parse_iso_ms(workspace.get("created_at", ""))
                or int(session_dir.stat().st_mtime * 1000)
            )
            entries.append(Session(
                id=sid,
                provider="copilot",
                project=cwd,
                model=model or "Copilot",
                model_group=f"Copilot / {model}" if model else "Copilot / Unknown",
                display=display,
                timestamp=ts,
                resumable=True,
                source_file=str(events_file if events_file.exists() else workspace_file),
                session_dir=str(session_dir),
                message_count=count,
            ))
        return entries

    def preview(self, session: Session) -> Preview:
        fp = Path(session.source_file) if session.source_file else None
        if not fp or not fp.exists():
            return Preview()
        if fp.name == "events.jsonl":
            first, last, count, _model = self._preview_from_events(fp)
            return Preview(first=first, last=last, message_count=count)
        text = fp.read_text(encoding="utf-8", errors="replace").strip()
        return Preview(
            first=text[:4000] or None,
            message_count=session.message_count or 0,
        )

    def delete(self, session: Session) -> None:
        session_dir = Path(session.session_dir or "")
        if session_dir.exists() and session_dir.is_dir() and path_is_under(session_dir, COPILOT_SESSIONS_DIR):
            try:
                shutil.rmtree(session_dir)
            except OSError:
                pass

    def resume_command(self, session: Session) -> ResumeCommand:
        cwd = session.project or str(Path.home())
        sid = session.id
        command = f"copilot --resume={ps_single_quote(sid)}"
        return ResumeCommand(cwd=cwd, shell_command=command)
