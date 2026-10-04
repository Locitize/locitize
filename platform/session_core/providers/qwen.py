"""Qwen Code session provider.

Layout (verified against QwenLM/qwen-code source, mid-2026): current builds
write Claude-style JSONL records to ``~/.qwen/projects/<sanitized-cwd>/chats/
<sessionId>.jsonl``; older builds used ``~/.qwen/tmp/<sha256>/chats/``. Every
line is one record: ``{uuid, parentUuid, sessionId, timestamp, type, cwd,
message: {role, parts: [{text}]}, model?, usageMetadata?}``. Custom titles are
``type:'system', subtype:'custom_title'`` records. Sidecar
``<sessionId>.runtime.json`` files are not sessions.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..config import QWEN_DIR, QWEN_PROJECTS_DIR, QWEN_TMP_DIR
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


def _parts_text(message) -> str:
    if not isinstance(message, dict):
        return ""
    parts = message.get("parts")
    if not isinstance(parts, list):
        return ""
    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)]
    return " ".join(t for t in texts if t).strip()


def _iso_to_ms(raw: str) -> int:
    try:
        return int(datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp() * 1000)
    except (ValueError, TypeError):
        return 0


class QwenProvider:
    key = "qwen"
    label = "Qwen Code"

    def detected(self) -> bool:
        return QWEN_PROJECTS_DIR.exists() or QWEN_TMP_DIR.exists() or QWEN_DIR.exists()

    # -- discovery -----------------------------------------------------------
    def _session_files(self) -> list[Path]:
        out: list[Path] = []
        for root in (QWEN_PROJECTS_DIR, QWEN_TMP_DIR):
            if not root.exists():
                continue
            for project_dir in root.iterdir():
                chats = project_dir / "chats"
                if not chats.is_dir():
                    continue
                # *.jsonl only: the <sid>.runtime.json sidecars end in .json.
                out.extend(f for f in chats.glob("*.jsonl") if f.is_file())
        return out

    # -- protocol ------------------------------------------------------------
    def load_sessions(self) -> list[Session]:
        result: list[Session] = []
        for fp in self._session_files():
            sid = fp.stem
            project = ""
            model = ""
            display = ""
            first_ts = last_ts = 0
            for rec in iter_jsonl_records(fp, max_bytes=MAX_METADATA_SCAN_BYTES):
                ts = _iso_to_ms(rec.get("timestamp", ""))
                if ts:
                    first_ts = first_ts or ts
                    last_ts = max(last_ts, ts)
                if not project and isinstance(rec.get("cwd"), str):
                    project = rec["cwd"]
                if rec.get("type") == "assistant" and isinstance(rec.get("model"), str):
                    model = rec["model"]
                if rec.get("type") == "system" and rec.get("subtype") == "custom_title":
                    payload = rec.get("systemPayload")
                    if isinstance(payload, str) and payload.strip():
                        display = payload.strip()
                    elif isinstance(payload, dict) and isinstance(payload.get("title"), str):
                        display = payload["title"].strip()
                if not display and rec.get("type") == "user":
                    text = _parts_text(rec.get("message"))
                    if text:
                        display = clip_preview_text(text)
            ts = last_ts or first_ts
            if not ts:
                try:
                    ts = int(fp.stat().st_mtime * 1000)
                except OSError:
                    ts = 0
            result.append(Session(
                id=sid,
                provider="qwen",
                project=project,
                model=model,
                model_group=f"Qwen Code / {model}" if model else "Qwen Code / Unknown",
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
        for rec in iter_jsonl_records(fp):
            if rec.get("type") == "user":
                text = _parts_text(rec.get("message"))
                if text:
                    first, last, count = remember_first_last(first, last, count, text)
            elif rec.get("type") == "assistant":
                usage = rec.get("usageMetadata")
                if isinstance(usage, dict):
                    tokens.input += int(usage.get("promptTokenCount", 0) or 0)
                    tokens.output += int(usage.get("candidatesTokenCount", 0) or 0)
                    tokens.cache_read += int(usage.get("cachedContentTokenCount", 0) or 0)
        return Preview(first=first, last=last, message_count=count, tokens=tokens)

    def collect_messages(self, session: Session) -> list[str]:
        fp = Path(session.source_file) if session.source_file else None
        if not fp or not fp.exists():
            return []
        out = []
        for rec in iter_jsonl_records(fp, max_bytes=MAX_INDEX_BYTES):
            if rec.get("type") == "user":
                text = _parts_text(rec.get("message"))
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
            kind = rec.get("type")
            if kind not in ("user", "assistant"):
                continue
            text = _parts_text(rec.get("message"))
            if not text:
                continue
            text = clip_preview_text(text, limit=4000)
            model = rec.get("model", "") if kind == "assistant" else ""
            msgs.append(ThreadMessage(kind, text, model))
            total += len(text)
            total = keep_thread_tail(msgs, total)
        return msgs

    def delete(self, session: Session) -> None:
        if not session.source_file:
            return
        fp = Path(session.source_file)
        try:
            fp.unlink(missing_ok=True)
            # Remove the runtime sidecar so the tool does not resurrect the row.
            fp.with_name(f"{fp.stem}.runtime.json").unlink(missing_ok=True)
        except OSError:
            pass

    def resume_command(self, session: Session) -> ResumeCommand:
        cwd = session.project if session.project and Path(session.project).exists() else str(Path.home())
        return ResumeCommand(cwd=cwd, shell_command=f"qwen --resume {ps_single_quote(session.id)}")
