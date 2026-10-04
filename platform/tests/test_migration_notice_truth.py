"""Round 8: every claim the migration notice makes must be one LOCITIZE measured.

WHY THIS FILE EXISTS
--------------------
The one-time notice (DEC-M14-9 item 7) is the only account a user ever gets of
where their conversation history went. Round 7's repair to the marker rule made
it possible for that notice to appear on a machine where nothing had EVER been
copied, telling the user their conversations "were copied" and that "the old
copy is still at <install>" when neither was true and the store was gone.

So the notice is tested here as a set of factual claims rather than as a string:

  "your data was copied"       only when the marker's audit trail names a copy
  "the old copy is still at X" only when the originals were actually found there
  the sweep deletes only ours  only a directory carrying the authorship sentinel

Kept in its own file rather than appended to test_data_root_migration.py because
these are claim-truth tests, and because a file per concern is what let two
people work on this milestone at once without overwriting each other.

Keyword: migration_notice_truth
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

import migration


def never_running(_port: int) -> bool:
    """Open WebUI is not running - the shipped default (DEC-M14-1)."""
    return False


def always_running(_port: int) -> bool:
    """Open WebUI holds its sqlite database open, so the store must be deferred."""
    return True


def _spoken_by_the_launcher(outcome, data: Path) -> list[str]:
    """What the terminal half of the notice surface actually says for an outcome."""
    from launcher import Launcher

    lines: list[str] = []
    Launcher(deps={"output_fn": lines.append})._log_migration(
        logging.getLogger("test-migration-notice"), outcome, str(data)
    )
    return lines


def test_migration_notice_truth_is_withheld_when_nothing_was_ever_copied(tmp_path):
    """The round-8 HIGH, reproduced end to end.

    The machine has ONLY an Open WebUI store, and Open WebUI is running, so run 1
    copies nothing at all and defers that single item - which still writes a
    marker, because a deferral has to be recorded. The user then deletes the old
    install folder, exactly as the product's own notice invites them to. Run 2
    finds no source, defers nothing, and legitimately completes.

    Completion is honest there: there really is nothing left to do. The NOTICE is
    not: no byte was ever copied, so "your conversations were copied" and "the
    old copy is still at <install>" are both false, and the data root holds
    nothing but the marker itself. This is the run on which the product could
    tell someone their history is safe while it is gone.
    """
    install = tmp_path / "install"
    (install / "webui-data").mkdir(parents=True)
    (install / "webui-data" / "webui.db").write_bytes(b"the only thing they have")
    data = tmp_path / "locitize-data"
    data.mkdir()

    first = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=always_running
    )
    assert first.ok is True
    assert first.copied == []
    assert [i.label for i in first.deferred] == ["Open WebUI store"]
    assert (data / migration.MARKER_NAME).is_file(), "the deferral was not recorded"

    # The user acts on the notice's own invitation and removes the old install.
    shutil.rmtree(install / "webui-data")

    second = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )

    # The job really is over - that part was never the defect.
    assert second.ok is True and second.deferred == []
    assert second.complete is True
    assert migration.already_migrated(data) is True
    payload = json.loads((data / migration.MARKER_NAME).read_text(encoding="utf-8"))
    assert payload["complete"] is True
    assert all(row["action"] != migration.ACTION_COPIED for row in payload["items"])

    # And this is the claim that must not be made.
    assert migration.pending_notice(data) == "", (
        "the notice says the user's conversations were copied, and nothing was "
        "ever copied on any run"
    )
    assert second.copied == []
    assert [p.name for p in data.iterdir()] == [migration.MARKER_NAME], (
        "the data root holds nothing but the marker, so there is nothing to "
        "claim was migrated"
    )
    assert _spoken_by_the_launcher(second, data) == [], (
        "the launcher told a headless user their data moved when it did not"
    )


def test_migration_notice_truth_is_still_offered_when_a_copy_really_happened(tmp_path):
    """The control, and it is what stops the fix above from being 'say nothing'.

    Same shape - a deferral cleared on a later run by the user removing the
    install-side original - but this machine also had logs and transcripts, and
    those really were copied on run 1. The notice is owed here, and withholding
    it is the round-7 defect that this must not reintroduce: a user who cannot
    see that their history moved will conclude it was lost.
    """
    install = tmp_path / "install"
    (install / "logs").mkdir(parents=True)
    (install / "logs" / "launcher.log").write_text("old log\n", encoding="utf-8")
    (install / "memory").mkdir()
    (install / "memory" / "2026-08-01.jsonl").write_text('{"t": 1}\n', encoding="utf-8")
    (install / "webui-data").mkdir()
    (install / "webui-data" / "webui.db").write_bytes(b"sqlite-ish")
    data = tmp_path / "locitize-data"
    data.mkdir()

    first = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=always_running
    )
    assert [i.label for i in first.copied] == ["logs", "conversation transcripts"]
    assert migration.pending_notice(data) == "", "offered while an item is deferred"

    shutil.rmtree(install / "webui-data")
    second = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )

    assert second.copied == [] and second.complete is True
    notice = migration.pending_notice(data)
    assert notice, "the user's transcripts moved and nothing tells them"
    assert str(install) in notice and str(data) in notice
    spoken = "\n".join(_spoken_by_the_launcher(second, data))
    assert second.notice in spoken


def test_migration_notice_truth_does_not_claim_an_old_folder_that_is_gone(tmp_path):
    """The second false claim: "Nothing was deleted: the old copy is still at X".

    True of what LOCITIZE did - it never deletes - but not of what the USER did. The
    previous sentence tells them to remove that folder, and on the run that
    finally completes the migration it can already be gone. A notice that names a
    folder which is not there sends someone looking for their history in an empty
    place.
    """
    install = tmp_path / "install"
    (install / "logs").mkdir(parents=True)
    (install / "logs" / "launcher.log").write_text("old log\n", encoding="utf-8")
    (install / "webui-data").mkdir()
    (install / "webui-data" / "webui.db").write_bytes(b"sqlite-ish")
    data = tmp_path / "locitize-data"
    data.mkdir()

    first = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=always_running
    )
    # While the originals are there, the wording is the one the spec pins.
    assert f"the old copy is still at {install}" in first.notice
    assert "Nothing was deleted" in first.notice

    # The user removes the whole old install, as invited.
    shutil.rmtree(install)
    install.mkdir()

    second = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )
    assert second.complete is True
    notice = migration.pending_notice(data)
    assert notice, "a real copy happened on run 1, so the notice is owed"
    assert f"the old copy is still at {install}" not in notice, (
        "the notice points the user at a folder that no longer exists"
    )
    assert "no longer on this machine" in notice
    assert "deleted nothing" in notice
    assert str(data) in notice


def test_migration_notice_truth_sweep_deletes_only_what_it_can_prove_it_wrote(tmp_path):
    """MEDIUM (round 8): a NAME is not proof of authorship.

    Both directories below carry the exact name _copy_atomically authors, sit at
    the exact depth it stages at, and are aged past the sweep threshold. Every
    constraint the sweep had before round 8 is satisfied by both. The only thing
    that tells them apart is the sentinel file the copy writes at creation - and
    the one without it is a user's own directory, whose contents the sweep used
    to delete while claiming it could not lose anything.
    """
    install = tmp_path / "install"
    (install / "logs").mkdir(parents=True)
    (install / "logs" / "launcher.log").write_text("old log\n", encoding="utf-8")
    data = tmp_path / "locitize-data"
    data.mkdir()
    stale = time.time() - migration.STALE_STAGING_AGE_S - 60

    ours = data / "memory.migrating-4242-deadbeef"
    ours.mkdir()
    (ours / migration._STAGING_SENTINEL).write_text("staging\n", encoding="utf-8")
    (ours / "half-copied.jsonl").write_text("partial\n", encoding="utf-8")
    os.utime(ours, (stale, stale))

    theirs = data / "memory.migrating-1234-abcdef01"
    theirs.mkdir()
    (theirs / "their-notes.md").write_text("a person wrote this\n", encoding="utf-8")
    os.utime(theirs, (stale, stale))

    migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )

    assert not ours.exists(), "the sweep no longer removes its own orphans"
    assert theirs.is_dir(), "a user directory was deleted on the strength of its name"
    assert (theirs / "their-notes.md").read_text(encoding="utf-8") == (
        "a person wrote this\n"
    )


def test_migration_notice_truth_a_completed_copy_leaves_no_sentinel_behind(tmp_path):
    """The sentinel proves authorship; it must never become part of user data.

    It is written into the staging directory before any user data goes in and
    unlinked immediately before the rename, so a successful migration leaves the
    user's folders exactly as they were in the install tree - no LOCITIZE bookkeeping
    file sitting inside their transcripts.
    """
    install = tmp_path / "install"
    (install / "memory").mkdir(parents=True)
    (install / "memory" / "2026-08-01.jsonl").write_text('{"t": 1}\n', encoding="utf-8")
    data = tmp_path / "locitize-data"
    data.mkdir()

    outcome = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )

    assert [i.label for i in outcome.copied] == ["conversation transcripts"]
    copied_names = sorted(p.name for p in (data / "memory").iterdir())
    assert copied_names == ["2026-08-01.jsonl"], copied_names
    assert not (data / "memory" / migration._STAGING_SENTINEL).exists()
    # And the source is untouched, sentinel included (rule 1: copy, never move).
    assert sorted(p.name for p in (install / "memory").iterdir()) == [
        "2026-08-01.jsonl"
    ]
