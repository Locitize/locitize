"""Claude Code session provider (ported faithfully from v1)."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import CLAUDE_DIR, HISTORY_FILE, PROJECTS_DIR
from ..models import Preview, ResumeCommand, Session, ThreadMessage, Tokens
from ..resume import find_claude_exe, ps_single_quote
from .base import (
    MAX_INDEX_BYTES,
    MAX_METADATA_SCAN_BYTES,
    clip_preview_text,
    iter_jsonl_records,
    keep_thread_tail,
    model_group_label,
    remember_first_last,
)


class ClaudeProvider:
    key = "claude"
    label = "Claude Code"

    def detected(self) -> bool:
        return PROJECTS_DIR.exists() or CLAUDE_DIR.exists()

    def _scan_files(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        if not PROJECTS_DIR.exists():
            return index
        for proj_dir in PROJECTS_DIR.iterdir():
            if not proj_dir.is_dir():
                continue
            for path in proj_dir.iterdir():
                if path.suffix == ".jsonl" and path.is_file():
                    sid = path.stem
                    if len(sid) == 36 and sid.count("-") == 4:
                        index[sid] = path
        return index

    @staticmethod
    def _decode_project_dir(path: Path) -> str:
        name = path.name
        if len(name) > 3 and name[1:3] == "--" and name[0].isalpha():
            return f"{name[0].upper()}:\\" + name[3:].replace("-", "\\")
        return str(path)

    def _get_cwd(self, path: Path) -> str:
        for record in iter_jsonl_records(path, max_bytes=MAX_METADATA_SCAN_BYTES):
            cwd = record.get("cwd")
            if cwd and Path(cwd).exists():
                return cwd
        decoded = self._decode_project_dir(path.parent)
        return decoded if Path(decoded).exists() else str(path.parent)

    def _get_ai_title(self, path: Path) -> str:
        title = ""
        for record in iter_jsonl_records(path, max_bytes=MAX_METADATA_SCAN_BYTES):
            if record.get("type") == "ai-title":
                value = record.get("aiTitle") or record.get("title")
                if isinstance(value, str) and value.strip():
                    title = value.strip()
        return title

    def _get_model(self, path: Path) -> str:
        model = ""
        for record in iter_jsonl_records(path, max_bytes=MAX_METADATA_SCAN_BYTES):
            message = record.get("message")
            if record.get("type") == "assistant" and isinstance(message, dict):
                value = message.get("model")
                if isinstance(value, str) and value.strip():
                    model = value.strip()
        return model

    @staticmethod
    def _looks_like_transcript_context(text: str) -> bool:
        compact = " ".join(text.split())
        return (
            "[system] Vault working directory:" in compact
            and " You:" in compact
            and " PeriCode:" in compact
        )

    def _extract_human_prompt(self, record: dict) -> str:
        if record.get("type") != "user":
            return ""
        if record.get("isMeta") or record.get("toolUseResult") or record.get("attachment"):
            return ""
        message = record.get("message")
        if not isinstance(message, dict) or message.get("role") != "user":
            return ""
        content = message.get("content", "")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "tool_result" or part.get("tool_use_id"):
                        return ""
                    if part.get("type") == "text":
                        parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            text = " ".join(parts).strip()
        else:
            return ""
        if not text:
            return ""
        if text.startswith("<system-reminder>") or text.startswith("<local-command"):
            return ""
        if self._looks_like_transcript_context(text):
            return ""
        return text

    def _extract_assistant_text(self, record: dict) -> tuple[str, str]:
        if record.get("type") != "assistant":
            return "", ""
        message = record.get("message")
        if not isinstance(message, dict):
            return "", ""
        model = message.get("model") or ""
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip(), model
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
            return " ".join(parts).strip(), model
        return "", model

    def collect_messages(self, session: Session) -> list[str]:
        path = Path(session.source_file) if session.source_file else None
        if not path or not path.exists():
            return []
        result = []
        for record in iter_jsonl_records(path):
            text = self._extract_human_prompt(record)
            if text:
                result.append(clip_preview_text(text, limit=2000))
        return result

    def collect_thread(self, session: Session) -> list[ThreadMessage]:
        path = Path(session.source_file) if session.source_file else None
        if not path or not path.exists():
            return []
        messages: list[ThreadMessage] = []
        total = 0
        for record in iter_jsonl_records(path, max_bytes=MAX_INDEX_BYTES):
            user_text = self._extract_human_prompt(record)
            if user_text:
                text = clip_preview_text(user_text, limit=4000)
                messages.append(ThreadMessage("user", text))
                total += len(text)
                total = keep_thread_tail(messages, total)
                continue
            assistant_text, model = self._extract_assistant_text(record)
            if assistant_text:
                text = clip_preview_text(assistant_text, limit=4000)
                messages.append(ThreadMessage("assistant", text, model))
                total += len(text)
                total = keep_thread_tail(messages, total)
        return messages

    def load_sessions(self) -> list[Session]:
        file_index = self._scan_files()
        history: dict[str, dict] = {}
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    sid = record.get("sessionId")
                    if not sid:
                        continue
                    timestamp = record.get("timestamp", 0)
                    if sid not in history or timestamp > history[sid]["timestamp"]:
                        history[sid] = record

        result: list[Session] = []
        for sid, path in file_index.items():
            model = self._get_model(path)
            historic = history.get(sid)
            timestamp = (
                historic.get("timestamp", int(path.stat().st_mtime * 1000))
                if historic else int(path.stat().st_mtime * 1000)
            )
            display = self._get_ai_title(path) or (historic or {}).get("display", "") or ""
            result.append(Session(
                id=sid,
                provider="claude",
                project=self._get_cwd(path),
                model=model,
                model_group=model_group_label("claude", model),
                display=display,
                timestamp=timestamp,
                resumable=True,
                source_file=str(path),
            ))

        resumable_ids = set(file_index)
        for sid, record in history.items():
            if sid in resumable_ids:
                continue
            model = record.get("model", "")
            result.append(Session(
                id=sid,
                provider="claude",
                project=record.get("project", ""),
                model=model,
                model_group=model_group_label("claude", model),
                display=record.get("display", ""),
                timestamp=record.get("timestamp", 0),
                resumable=False,
            ))
        return result

    def preview(self, session: Session) -> Preview:
        path = Path(session.source_file) if session.source_file else None
        if not path or not path.exists():
            return Preview()
        first = None
        last = None
        count = 0
        tokens = Tokens()
        for record in iter_jsonl_records(path):
            text = self._extract_human_prompt(record)
            if text:
                first, last, count = remember_first_last(first, last, count, text)
                continue
            if record.get("type") == "assistant" and isinstance(record.get("message"), dict):
                usage = record["message"].get("usage", {})
                if usage:
                    tokens.input += usage.get("input_tokens", 0)
                    tokens.output += usage.get("output_tokens", 0)
                    tokens.cache_read += usage.get("cache_read_input_tokens", 0)
                    tokens.cache_write += usage.get("cache_creation_input_tokens", 0)
        return Preview(first=first, last=last, message_count=count, tokens=tokens)

    def delete(self, session: Session) -> None:
        if session.source_file:
            try:
                Path(session.source_file).unlink(missing_ok=True)
            except OSError:
                pass
        if not HISTORY_FILE.exists():
            return
        lines = []
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        if json.loads(stripped).get("sessionId") != session.id:
                            lines.append(stripped)
                    except json.JSONDecodeError:
                        lines.append(stripped)
            from config import _atomic_write

            _atomic_write(
                HISTORY_FILE, "\n".join(lines) + ("\n" if lines else "")
            )
        except OSError:
            pass

    def resume_command(self, session: Session) -> ResumeCommand:
        exe = find_claude_exe()
        sid = session.id
        if "\\" in exe:
            command = f"& {ps_single_quote(exe)} --no-chrome --resume {ps_single_quote(sid)}"
        else:
            command = f"{exe} --no-chrome --resume {ps_single_quote(sid)}"
        return ResumeCommand(cwd=session.project or str(Path.home()), shell_command=command)
