"""Development-journal tests: append-only behavior and header-once semantics."""

from __future__ import annotations

from documentation import DevelopmentJournal, JournalEntry


def test_first_append_writes_header_and_entry(tmp_path):
    path = tmp_path / "journal.md"
    journal = DevelopmentJournal(path)
    journal.append(JournalEntry.now(completed="started", issues="none"))
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# LOCITIZE Development Journal")
    assert "- Completed: started" in text


def test_second_append_preserves_prior_entry(tmp_path):
    """Appending a second entry never rewrites the first (history preserved)."""
    path = tmp_path / "journal.md"
    journal = DevelopmentJournal(path)
    journal.append(JournalEntry.now(completed="first", issues="none"))
    journal.append(JournalEntry.now(completed="second", issues="none"))
    text = path.read_text(encoding="utf-8")
    assert "- Completed: first" in text
    assert "- Completed: second" in text
    # Header appears exactly once.
    assert text.count("# LOCITIZE Development Journal") == 1


def test_empty_fields_render_as_none(tmp_path):
    """Empty fields render as 'none', never blank, per the Data Model."""
    path = tmp_path / "journal.md"
    entry = JournalEntry(date="2026-07-18 10:00", completed="", issues="", fixes="", next_steps="")
    DevelopmentJournal(path).append(entry)
    text = path.read_text(encoding="utf-8")
    assert "- Completed: none" in text
    assert "- Next steps: none" in text


def test_render_field_order(tmp_path):
    """Fields render in the fixed order Completed, Issues, Fixes, Next steps."""
    entry = JournalEntry(
        date="2026-07-18 10:00",
        completed="c",
        issues="i",
        fixes="f",
        next_steps="n",
    )
    rendered = entry.render()
    order = [
        rendered.index("Completed"),
        rendered.index("Issues"),
        rendered.index("Fixes"),
        rendered.index("Next steps"),
    ]
    assert order == sorted(order)
