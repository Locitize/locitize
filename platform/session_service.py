"""Local provider discovery and bounded plain-text session reading."""
from __future__ import annotations

from pathlib import Path
from session_core.providers.registry import PROVIDERS, get_provider
from session_core.models import Session
from session_store import SessionStore


class SessionService:
    def __init__(self, data_dir: Path, providers=None):
        self.store = SessionStore(data_dir)
        self.providers = PROVIDERS if providers is None else providers

    def scan(self):
        sessions, errors = [], []
        for provider in self.providers:
            try:
                if provider.detected():
                    sessions.extend(provider.load_sessions())
            except Exception as exc:
                errors.append(f"{provider.label}: unable to read sessions ({type(exc).__name__})")
        unique = {(s.provider, s.id): s for s in sessions}
        return sorted(unique.values(), key=lambda s: s.timestamp, reverse=True), errors

    def transcript(self, session: Session):
        provider = next((p for p in self.providers if p.key == session.provider), None)
        if provider is None:
            raise ValueError("This session provider is unavailable")
        messages = provider.collect_thread(session)
        return "\n\n".join(f"{m.role.upper()}\n{m.text}" for m in messages)[-200000:]

    def original_command(self, session: Session, project=""):
        from dataclasses import replace
        import re
        import shutil
        from session_core.resume import find_codex_exe, find_grok_exe

        provider = get_provider(session.provider)
        folder = project or session.project
        if provider is None or not session.resumable:
            raise ValueError("This session cannot be resumed")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,199}", session.id):
            raise ValueError("Unsupported session identity")
        if not folder or not Path(folder).is_dir():
            raise ValueError("The project folder is missing. Choose its location using Resume with local model.")
        exe = find_codex_exe() if session.provider == "codex" else find_grok_exe() if session.provider == "grok" else shutil.which(session.provider)
        if not exe or not (Path(exe).is_file() or shutil.which(exe)):
            raise ValueError(f"{session.provider} is not installed")
        return provider.resume_command(replace(session, project=folder))

    def export(self, session: Session, destination: Path):
        from config import _atomic_write

        destination = Path(destination).resolve()
        if session.source_file and destination == Path(session.source_file).resolve():
            raise ValueError("Export cannot overwrite the original session")
        text = self.transcript(session)
        # Fence length exceeds any run in untrusted text; nothing becomes a link
        # or active Markdown instruction in the exported transcript.
        import re

        fence = "`" * max(3, 1 + max((len(x) for x in re.findall(r"`+", text)), default=0))
        _atomic_write(destination, f"# LOCITIZE session export\n\n{fence}text\n{text}\n{fence}\n")
        return destination
