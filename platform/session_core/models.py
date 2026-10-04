"""Immutable data shapes shared across providers, UI, and tests.

Every provider normalizes its on-disk records into one :class:`Session`.
Resume is modeled as a :class:`ResumeCommand` value object - providers build
it, ``resume.launch()`` executes it. That split makes argv/command
construction unit-testable without spawning a subprocess.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Tokens:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def total(self) -> int:
        return self.input + self.output


@dataclass
class Preview:
    first: Optional[str] = None
    last: Optional[str] = None
    message_count: int = 0
    tokens: Tokens = field(default_factory=Tokens)
    extra: dict = field(default_factory=dict)


@dataclass
class ThreadMessage:
    """One rendered message for the read-only thread viewer."""

    role: str            # "user" | "assistant"
    text: str
    model: str = ""


@dataclass(frozen=True)
class ResumeCommand:
    """A resume action ready to execute.

    ``cwd`` is the validated working directory (falls back to the user's home
    if the recorded one is gone). ``shell_command`` is the exact PowerShell
    line v1 runs, preserved verbatim so resume behavior is identical.
    """

    cwd: str
    shell_command: str


@dataclass
class Session:
    """One resumable local session, normalized across providers."""

    id: str
    provider: str                      # claude | codex | grok | copilot
    project: str                        # recorded cwd (string, may not exist)
    model: str = ""
    model_group: str = ""
    display: str = ""
    timestamp: int = 0
    resumable: bool = True
    source_file: Optional[str] = None   # primary file for preview/delete
    session_dir: Optional[str] = None    # grok/copilot directory for rmtree
    message_count: Optional[int] = None  # cached count, lazily computed
    search_blob: str = ""                # lowercased full-content index (lazy + persistent)
    tokens: Optional["Tokens"] = None    # cached preview tokens (lazy, for cost)
