"""Codex session provider (ported faithfully from v1)."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

from ..config import (
    CODEX_INDEX_FILE,
    CODEX_SESSIONS_DIR,
)
from ..models import Preview, ResumeCommand, Session, ThreadMessage, Tokens
from ..resume import find_codex_exe, ps_single_quote
from . import base as provider_base
from .base import (
    MAX_METADATA_SCAN_BYTES,
    clip_preview_text,
    iter_jsonl_records,
    keep_thread_tail,
    model_group_label,
    remember_first_last,
)

# Re-exported as a provider-level test hook for bounded-read regressions.
MAX_INDEX_BYTES = provider_base.MAX_INDEX_BYTES


class CodexProvider:
    key = "codex"
    label = "Codex"

    def detected(self) -> bool:
        return CODEX_SESSIONS_DIR.exists() or CODEX_INDEX_FILE.exists()

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _is_human_text(text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        lowered = text.lower()
        if text.startswith("<"):
            return False
        if lowered.startswith("# agents.md instructions"):
            return False
        if "<instructions>" in lowered or "</instructions>" in lowered:
            return False
        if "## mcp-first rule" in lowered or "## open requests" in lowered:
            return False
        if "[system]" in lowered or "vault working directory:" in lowered:
            return False
        return True

    @staticmethod
    def _clean_display(text: str) -> str:
        text = (text or "").strip()
        if not CodexProvider._is_human_text(text):
            return ""
        return clip_preview_text(text)

    def _first_message(self, fp: Path) -> str:
        for rec in iter_jsonl_records(fp, max_bytes=MAX_METADATA_SCAN_BYTES):
            if rec.get("type") == "response_item":
                payload = rec.get("payload", {})
                if (isinstance(payload, dict)
                        and payload.get("type") == "message"
                        and payload.get("role") == "user"):
                    for part in payload.get("content", []):
                        if isinstance(part, dict) and part.get("type") == "input_text":
                            text = part.get("text", "").strip()
                            if self._is_human_text(text):
                                return clip_preview_text(text)
        return ""

    def _get_model(self, fp: Path) -> str:
        model = ""
        for rec in iter_jsonl_records(fp, max_bytes=MAX_METADATA_SCAN_BYTES):
            payload = rec.get("payload")
            if not isinstance(payload, dict):
                continue
            value = payload.get("model")
            if isinstance(value, str) and value.strip():
                model = value.strip()
            settings = payload.get("collaboration_mode", {}).get("settings", {})
            value = settings.get("model") if isinstance(settings, dict) else ""
            if isinstance(value, str) and value.strip():
                model = value.strip()
        return model

    def _get_cwd(self, fp: Path) -> str:
        for rec in iter_jsonl_records(fp, max_bytes=MAX_METADATA_SCAN_BYTES):
            if rec.get("type") == "turn_context":
                return rec.get("payload", {}).get("cwd", "")
        return ""

    def _scan_files(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        if not CODEX_SESSIONS_DIR.exists():
            return index
        for f in CODEX_SESSIONS_DIR.rglob("rollout-*.jsonl"):
            stem = f.stem
            sid = stem[-36:]
            if len(sid) == 36 and sid.count("-") == 4:
                index[sid] = f
        return index

    # -- protocol ------------------------------------------------------------
    def _message_texts(self, payload: dict) -> tuple[str, str]:
        """Return (text, role) from a Codex response_item message payload."""
        if not isinstance(payload, dict):
            return "", ""
        if payload.get("type") != "message":
            return "", ""
        role = payload.get("role", "")
        kind = "input_text" if role == "user" else "output_text"
        parts = []
        for part in payload.get("content", []):
            if isinstance(part, dict) and part.get("type") == kind:
                parts.append(part.get("text", ""))
        return " ".join(parts).strip(), role

    def collect_messages(self, session: Session) -> list[str]:
        fp = Path(session.source_file) if session.source_file else None
        if not fp or not fp.exists():
            return []
        out: list[str] = []
        for rec in iter_jsonl_records(fp):
            if rec.get("type") != "response_item":
                continue
            text, role = self._message_texts(rec.get("payload", {}))
            if role == "user" and text and self._is_human_text(text):
                out.append(clip_preview_text(text, limit=2000))
        return out

    def collect_thread(self, session: Session) -> list[ThreadMessage]:
        fp = Path(session.source_file) if session.source_file else None
        if not fp or not fp.exists():
            return []
        msgs: list[ThreadMessage] = []
        total = 0
        # Thread view is opened for one selected session, so stream the whole
        # file and keep only the newest rendered tail in memory. Codex rollouts
        # can be much larger than the search-index cap, and resumed CLI output
        # lives at the end of the file.
        for rec in iter_jsonl_records(fp):
            if rec.get("type") != "response_item":
                continue
            text, role = self._message_texts(rec.get("payload", {}))
            if not text:
                continue
            if role == "user" and not self._is_human_text(text):
                continue
            clipped = clip_preview_text(text, limit=4000)
            msgs.append(ThreadMessage("user" if role == "user" else "assistant", clipped,
                                     session.model if role != "user" else ""))
            total += len(clipped)
            total = keep_thread_tail(msgs, total)
        return msgs

    def load_sessions(self) -> list[Session]:
        file_index = self._scan_files()

        index: dict[str, dict] = {}
        if CODEX_INDEX_FILE.exists():
            with open(CODEX_INDEX_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        sid = rec.get("id")
                        if sid:
                            index[sid] = rec
                    except json.JSONDecodeError:
                        continue

        result: list[Session] = []
        all_ids = set(file_index.keys()) | set(index.keys())
        for sid in all_ids:
            rec = index.get(sid, {})
            fp = file_index.get(sid)

            ts = 0
            updated_at = rec.get("updated_at", "")
            if updated_at:
                try:
                    dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    ts = int(dt.timestamp() * 1000)
                except ValueError:
                    pass
            if not ts and fp:
                ts = int(fp.stat().st_mtime * 1000)

            cwd = self._get_cwd(fp) if fp else ""
            model = self._get_model(fp) if fp else ""

            display = self._clean_display(rec.get("thread_name", ""))
            if not display and fp:
                display = self._clean_display(self._first_message(fp))

            project = cwd or (str(fp.parent) if fp else "")
            result.append(Session(
                id=sid,
                provider="codex",
                project=project,
                model=model,
                model_group=model_group_label("codex", model),
                display=display,
                timestamp=ts,
                resumable=fp is not None,
                source_file=str(fp) if fp else None,
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
            rtype = rec.get("type")
            payload = rec.get("payload", {})
            if rtype == "response_item" and isinstance(payload, dict):
                if payload.get("type") == "message" and payload.get("role") == "user":
                    for part in payload.get("content", []):
                        if isinstance(part, dict) and part.get("type") == "input_text":
                            text = part.get("text", "").strip()
                            if self._is_human_text(text):
                                first, last, count = remember_first_last(first, last, count, text)
            elif rtype == "event_msg" and isinstance(payload, dict):
                if payload.get("type") == "token_count":
                    info = payload.get("info") or {}
                    usage = info.get("last_token_usage") or info.get("total_token_usage") or {}
                    tokens.input += usage.get("input_tokens", 0)
                    tokens.output += usage.get("output_tokens", 0)
                    tokens.cache_read += usage.get("cached_input_tokens", 0)
        return Preview(first=first, last=last, message_count=count, tokens=tokens)

    def delete(self, session: Session) -> None:
        sid = session.id
        exe = find_codex_exe()
        if exe:
            try:
                subprocess.run([exe, "delete", sid], capture_output=True, timeout=10)
            except Exception:
                pass
        if session.source_file:
            try:
                Path(session.source_file).unlink(missing_ok=True)
            except OSError:
                pass

    def resume_command(self, session: Session) -> ResumeCommand:
        exe = find_codex_exe()
        cwd = session.project or str(Path.home())
        sid = session.id
        if exe:
            cmd = f"& {ps_single_quote(exe)} resume --cd {ps_single_quote(cwd)} {ps_single_quote(sid)}"
        else:
            cmd = f"codex resume --cd {ps_single_quote(cwd)} {ps_single_quote(sid)}"
        return ResumeCommand(cwd=cwd, shell_command=cmd)
