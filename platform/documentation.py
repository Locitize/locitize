"""Append-only development journal for the LOCITIZE platform.

Records one dated block per interactive launch to docs/development_journal.md
(Architecture section 9.2, Data Model section 3). The writer only ever appends;
it never edits or deletes prior entries, preserving history as the spec requires.

Testability: DevelopmentJournal takes an explicit path, so tests point it at a
temp file and assert append-only behavior (existing content preserved, header
written exactly once).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Written once when the journal file is first created (Data Model 3.1).
_HEADER = (
    "# locitize Development Journal\n\n"
    "Append-only. Each entry records one platform launch. Never edit or delete "
    "prior entries.\n"
)


@dataclass
class JournalEntry:
    """One journal block (Data Model section 3.3). Empty fields render as 'none'."""

    date: str  # local timestamp "YYYY-MM-DD HH:MM"
    completed: str
    issues: str
    fixes: str
    next_steps: str

    @staticmethod
    def now(
        completed: str, issues: str, fixes: str = "none", next_steps: str = "none"
    ) -> "JournalEntry":
        """Build an entry stamped with the current local time."""
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return JournalEntry(
            date=stamp,
            completed=completed or "none",
            issues=issues or "none",
            fixes=fixes or "none",
            next_steps=next_steps or "none",
        )

    def render(self) -> str:
        """Render the Markdown block in the fixed field order (Data Model 3.2)."""
        return (
            f"\n## {self.date} - launch\n\n"
            f"- Completed: {self.completed or 'none'}\n"
            f"- Issues: {self.issues or 'none'}\n"
            f"- Fixes: {self.fixes or 'none'}\n"
            f"- Next steps: {self.next_steps or 'none'}\n"
        )


class DevelopmentJournal:
    """Append-only writer for the development journal file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def append(self, entry: JournalEntry) -> None:
        """Append one entry, creating the file with a header if it is new.

        Opens in append mode ("a") so existing content is never truncated
        (Permission Matrix section 4: journal is append-only).
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self._path.exists()
        with self._path.open("a", encoding="utf-8") as handle:
            if new_file:
                handle.write(_HEADER)
            handle.write(entry.render())
