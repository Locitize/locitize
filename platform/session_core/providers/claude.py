"""Claude Code session provider (ported faithfully from v1)."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import (
    CLAUDE_DIR,
    HISTORY_FILE,
    PROJECTS_DIR,
)
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

    # -- discovery -----------------------------------------------------------
    def _scan_files(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        if not PROJECTS_DIR.exists():
            return index
        for proj_dir in PROJECTS_DIR.iterdir():
            if not proj_dir.is_dir():
                continue
            for f in proj_dir.iterdir():
                if f.suffix == ".jsonl" and f.is_file():
                    sid = f.stem
                    if len(sid) == 36 and sid.count("-") == 4:
                        index[sid] = f
        return index

    @staticmethod
    def _decode_project_dir(path: Path) -> str:
        name = path.name
        if len(name) > 3 and name[1:3] == "--" and name[0].isalpha():
            decoded = f"{name[0].upper()}:\\" + name[3:].replace("-", "\\")
            return decoded
        return str(path)

    def _get_cwd(self, fp: Path) -> str:
        for rec in iter_jsonl_records(fp, max_bytes=MAX_METADATA_SCAN_BYTES):
            cwd = rec.get("cwd")
            if cwd and Path(cwd).exists():
                return cwd
        decoded = self._decode_project_dir(fp.parent)
        return decoded if Path(decoded).exists() else str(fp.parent)

    def _get_ai_title(self, fp: Path) -> str:
        title = ""
        for rec in iter_jsonl_records(fp, max_bytes=MAX_METADATA_SCAN_BYTES):
            if rec.get("type") == "ai-title":
                ai_title = rec.get("aiTitle") or rec.get("title")
                if isinstance(ai_title, str) and ai_title.strip():
                    title = ai_title.strip()
        return title

    def _get_model(self, fp: Path) -> str:
        model = ""
        for rec in iter_jsonl_records(fp, max_bytes=MAX_METADATA_SCAN_BYTES):
            message = rec.get("message")
            if rec.get("type") == "assistant" and isinstance(message, dict):
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

    def _extract_human_prompt(self, rec: dict) -> str:
        if rec.get("type") != "user":
            return ""
        if rec.get("isMeta") or rec.get("toolUseResult") or rec.get("attachment"):
            return ""
        message = rec.get("message")
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

    # -- protocol ------------------------------------------------------------
    def _extract_assistant_text(self, rec: dict) -> tuple[str, str]:
        """Return (text, model) for an assistant record, or ('', '')."""
        if rec.get("type") != "assistant":
            return "", ""
        message = rec.get("message")
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
        fp = Path(session.source_file) if session.source_file else None
        if not fp or not fp.exists():
            return []
        out: list[str] = []
        for rec in iter_jsonl_records(fp):
            text = self._extract_human_prompt(rec)
            if text:
                out.append(clip_preview_text(text, limit=2000))
        return out

    def collect_thread(self, session: Session) -> list[ThreadMessage]:
        fp = Path(session.source_file) if session.source_file else None
        if not fp or not fp.exists():
            return []
        msgs: list[ThreadMessage] = []
        total = 0
        for rec in iter_jsonl_records(fp, max_bytes=MAX_INDEX_BYTES):
            user_text = self._extract_human_prompt(rec)
            if user_text:
                text = clip_preview_text(user_text, limit=4000)
                msgs.append(ThreadMessage("user", text))
                total += len(text)
                total = keep_thread_tail(msgs, total)
            else:
                atext, model = self._extract_assistant_text(rec)
                if atext:
                    text = clip_preview_text(atext, limit=4000)
                    msgs.append(ThreadMessage("assistant", text, model))
                    total += len(text)
                    total = keep_thread_tail(msgs, total)
        return msgs

    def load_sessions(self) -> list[Session]:
        file_index = self._scan_files()

        history: dict[str, dict] = {}
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        sid = rec.get("sessionId")
                        if not sid:
                            continue
                        ts = rec.get("timestamp", 0)
                        if sid not in history or ts > history[sid]["timestamp"]:
                            history[sid] = rec
                    except json.JSONDecodeError:
                        continue

        result: list[Session] = []
        for sid, fp in file_index.items():
            ai_title = self._get_ai_title(fp)
            model = self._get_model(fp)
            cwd = self._get_cwd(fp)
            hist = history.get(sid)
            if hist:
                ts = hist.get("timestamp", int(fp.stat().st_mtime * 1000))
            else:
                ts = int(fp.stat().st_mtime * 1000)
            display = ai_title or (hist or {}).get("display", "") or ""
            result.append(Session(
                id=sid,
                provider="claude",
                project=cwd,
                model=model,
                model_group=model_group_label("claude", model),
                display=display,
                timestamp=ts,
                resumable=True,
                source_file=str(fp),
            ))

        resumable_ids = set(file_index.keys())
        for sid, rec in history.items():
            if sid in resumable_ids:
                continue
            model = rec.get("model", "")
            result.append(Session(
                id=sid,
                provider="claude",
                project=rec.get("project", ""),
                model=model,
                model_group=model_group_label("claude", model),
                display=rec.get("display", ""),
                timestamp=rec.get("timestamp", 0),
                resumable=False,
            ))
        return result

    def preview(self, session: Session) -> Preview:
        fp = Path(session.source_file) if session.source_file else None
        if not fp or not fp.exists():
            return Preview()
        first = None
        last = None
        count = 0
        tokens = Tokens()
        for rec in iter_jsonl_records(fp):
            text = self._extract_human_prompt(rec)
            if text:
                first, last, count = remember_first_last(first, last, count, text)
                continue
            if rec.get("type") == "assistant" and isinstance(rec.get("message"), dict):
                usage = rec["message"].get("usage", {})
                if usage:
                    tokens.input += usage.get("input_tokens", 0)
                    tokens.output += usage.get("output_tokens", 0)
                    tokens.cache_read += usage.get("cache_read_input_tokens", 0)
                    tokens.cache_write += usage.get("cache_creation_input_tokens", 0)
        return Preview(first=first, last=last, message_count=count, tokens=tokens)

    def delete(self, session: Session) -> None:
        sid = session.id
        if session.source_file:
            try:
                Path(session.source_file).unlink(missing_ok=True)
            except OSError:
                pass
        if HISTORY_FILE.exists():
            lines = []
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            rec = json.loads(stripped)
                            if rec.get("sessionId") != sid:
                                lines.append(stripped)
                        except json.JSONDecodeError:
                            lines.append(stripped)
                with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + ("\n" if lines else ""))
            except OSError:
                pass

    def resume_command(self, session: Session) -> ResumeCommand:
        exe = find_claude_exe()
        sid = session.id
        if "\\" in exe:
            command = f"& {ps_single_quote(exe)} --no-chrome --resume {ps_single_quote(sid)}"
        else:
            command = f"{exe} --no-chrome --resume {ps_single_quote(sid)}"
        cwd = session.project or str(Path.home())
        return ResumeCommand(cwd=cwd, shell_command=command)
