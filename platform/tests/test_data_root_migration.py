"""AC-M14-27 (c): the install-tree -> data-root migration, rule by rule.

DEC-M14-9's migration is the part of this milestone that can cost someone real
conversation history, so every rule it names is tested against a real filesystem
here - copies are made, hashes are compared, and the failure case is a genuine
OS error rather than a mocked one:

  copy, never move          the source tree is byte-identical afterwards
  never merge               a non-empty destination is skipped whole
  no half-populated state   a failed copy leaves the destination absent
  marker in the data root   and nowhere else, ever
  no marker on failure      an honest failure beats a silent half-state
  idempotent                a second run copies nothing and changes nothing
  live sqlite               webui-data is skipped while Open WebUI is running
  the user is told once     the notice names both real paths, then stops

Keyword: data_root_migration (see AC-M14-27's verification command).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import migration


def tree_digest(root: Path) -> dict[str, str]:
    """relative path -> sha256, for proving a tree was not touched."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def build_install(tmp_path: Path) -> tuple[Path, Path]:
    """A pre-M14 install tree with real user data in it, plus an empty data root."""
    install = tmp_path / "install"
    (install / "logs").mkdir(parents=True)
    (install / "logs" / "launcher.log").write_text("old log line\n", encoding="utf-8")
    (install / "memory").mkdir()
    (install / "memory" / "2026-08-01.jsonl").write_text(
        '{"role": "user", "text": "the conversation that must not be lost"}\n',
        encoding="utf-8",
    )
    (install / "webui-data").mkdir()
    (install / "webui-data" / "webui.db").write_bytes(b"sqlite-ish bytes")
    (install / "docs").mkdir()
    (install / "docs" / "benchmark_results.jsonl").write_text("{}\n", encoding="utf-8")
    (install / "docs" / "development_journal.md").write_text("# journal\n", encoding="utf-8")
    (install / "docs" / "chat.md").write_text("shipped doc, not user data\n", encoding="utf-8")
    (install / "Caddyfile").write_text("locitize.local {\n}\n", encoding="utf-8")

    data = tmp_path / "locitize-data"
    data.mkdir()
    return install, data


def never_running(_port: int) -> bool:
    """Open WebUI is not running - the shipped default (DEC-M14-1)."""
    return False


def test_data_root_migration_copies_and_leaves_the_originals_byte_identical(tmp_path):
    """Rule 1: COPY. The install-side originals are untouched afterwards."""
    install, data = build_install(tmp_path)
    before = tree_digest(install)

    outcome = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )

    assert outcome.ran is True and outcome.ok is True
    assert tree_digest(install) == before, "the migration modified the source tree"
    # Every relocated item now exists in the data root, with identical content.
    assert (data / "logs" / "launcher.log").read_text(encoding="utf-8") == "old log line\n"
    assert "must not be lost" in (data / "memory" / "2026-08-01.jsonl").read_text(
        encoding="utf-8"
    )
    assert (data / "webui-data" / "webui.db").read_bytes() == b"sqlite-ish bytes"
    assert (data / "reports" / "benchmark_results.jsonl").is_file()
    assert (data / "reports" / "development_journal.md").is_file()
    assert (data / "Caddyfile").is_file()
    # A shipped doc is product, not user data, and is NOT copied.
    assert not (data / "reports" / "chat.md").exists()


def test_data_root_migration_treats_a_fresh_clone_as_nothing_to_copy(tmp_path):
    """A clone ships logs/.gitkeep only; that is not user data and earns no notice."""
    install = tmp_path / "install"
    (install / "logs").mkdir(parents=True)
    (install / "logs" / ".gitkeep").write_text("", encoding="utf-8")
    data = tmp_path / "locitize-data"
    data.mkdir()

    outcome = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )

    assert not outcome.copied
    assert all(i.action == migration.ACTION_NO_SOURCE for i in outcome.items)
    assert not (data / "logs" / ".gitkeep").exists()


def test_data_root_migration_skips_a_non_empty_destination_without_merging(tmp_path):
    """Rule 2: never merge two histories. The newer data root wins, untouched."""
    install, data = build_install(tmp_path)
    (data / "memory").mkdir()
    (data / "memory" / "2026-08-18.jsonl").write_text(
        '{"role": "user", "text": "newer history"}\n', encoding="utf-8"
    )

    outcome = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )

    memory_item = next(i for i in outcome.items if i.label == "conversation transcripts")
    assert memory_item.action == migration.ACTION_DESTINATION_IN_USE
    # The install-side transcript was NOT merged in, and the newer one is intact.
    assert sorted(p.name for p in (data / "memory").iterdir()) == ["2026-08-18.jsonl"]
    # Both real paths are recorded, because that is the only way a user can check.
    assert memory_item.source == str(install / "memory")
    assert memory_item.destination == str(data / "memory")


def test_data_root_migration_writes_its_marker_only_in_the_data_root(tmp_path):
    """Rule 4: the marker lives beside the settings, never in the install tree."""
    install, data = build_install(tmp_path)
    migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )

    marker = data / migration.MARKER_NAME
    assert marker.is_file()
    assert not (install / migration.MARKER_NAME).exists()
    assert list(install.rglob(migration.MARKER_NAME)) == []

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["migrated_from_install"] is True
    assert payload["install_dir"] == str(install)
    assert payload["data_dir"] == str(data)
    assert payload["notice_shown"] is False


def test_data_root_migration_writes_no_marker_on_partial_failure(tmp_path):
    """Rule 5: an honest failure, never a silent half-state.

    The failure is real, not injected: `reports` already exists as a FILE, so
    creating the parent directory for the benchmark results genuinely raises.
    """
    install, data = build_install(tmp_path)
    (data / "reports").write_text("a file where a folder must go\n", encoding="utf-8")

    outcome = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )

    assert outcome.ok is False
    assert outcome.errors, "a failure must be reported, with a reason"
    assert not (data / migration.MARKER_NAME).exists(), "a failed run claimed success"
    # The items that DID succeed are still copied - the user's transcripts are
    # not held hostage by an unrelated failure - and the originals are intact.
    assert (data / "memory" / "2026-08-01.jsonl").is_file()
    assert (install / "docs" / "benchmark_results.jsonl").is_file()
    # And no half-populated destination was left behind for rule 2 to mistake
    # for real data on the next run.
    assert not (data / "reports").is_dir()
    assert not list(data.glob("*.migrating-*"))


def test_data_root_migration_is_idempotent_on_a_second_run(tmp_path):
    """Rule 4's second guard: a re-run copies nothing and changes nothing."""
    install, data = build_install(tmp_path)
    first = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )
    assert first.copied

    after_first = tree_digest(data)
    # The user then adds a new conversation in the data root.
    (data / "memory" / "2026-08-19.jsonl").write_text("new\n", encoding="utf-8")
    expected = tree_digest(data)

    second = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )
    assert second.ran is False, "the marker did not stop a second migration"
    assert second.copied == []
    assert tree_digest(data) == expected
    assert set(after_first) <= set(expected)


def test_data_root_migration_retries_after_a_failure_without_duplicating(tmp_path):
    """No marker means the next run retries - and rule 2 stops a double copy."""
    install, data = build_install(tmp_path)
    (data / "reports").write_text("blocker\n", encoding="utf-8")
    first = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )
    assert first.ok is False

    # The user clears the blocker and starts LOCITIZE again.
    (data / "reports").unlink()
    second = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )
    assert second.ok is True
    assert (data / "reports" / "benchmark_results.jsonl").is_file()
    assert (data / migration.MARKER_NAME).is_file()
    # The transcripts copied by the first run were skipped, not copied twice.
    memory_item = next(i for i in second.items if i.label == "conversation transcripts")
    assert memory_item.action == migration.ACTION_DESTINATION_IN_USE
    assert sorted(p.name for p in (data / "memory").iterdir()) == ["2026-08-01.jsonl"]


def test_data_root_migration_defers_webui_data_while_open_webui_is_running(tmp_path):
    """Rule 5: a hot copy of a live sqlite database is a corrupt copy.

    The skip is a DEFERRAL. The previous version of this test stopped at
    `outcome.ok is True` and so encoded the shipped defect as intended
    behaviour: the marker was written, `already_migrated` short-circuited every
    later run, and the user's Open WebUI history stayed in the install tree
    forever while the app told them it had been copied (round 6, HIGH-1). The
    question that matters is the second one asked here - is the deferred item
    ever actually copied?
    """
    install, data = build_install(tmp_path)

    outcome = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=lambda _port: True
    )

    item = next(i for i in outcome.items if i.label == "Open WebUI store")
    assert item.action == migration.ACTION_SERVICE_RUNNING
    assert not (data / "webui-data").exists()
    # Nothing went wrong, so ok stays True - but the migration is NOT complete.
    assert outcome.ok is True
    assert outcome.complete is False
    assert [i.label for i in outcome.deferred] == ["Open WebUI store"]
    assert (data / "memory" / "2026-08-01.jsonl").is_file()

    # The marker records the deferral instead of claiming completion, and the
    # notice - which says the user's conversations "were copied" - is withheld
    # until that is true.
    payload = json.loads((data / migration.MARKER_NAME).read_text(encoding="utf-8"))
    assert payload["complete"] is False
    assert payload["deferred"] == ["Open WebUI store"]
    assert migration.already_migrated(data) is False
    assert migration.pending_notice(data) == ""


def test_data_root_migration_retries_a_deferred_webui_store_and_finishes_it(tmp_path):
    """HIGH-1's real question: the deferred store IS copied, on a later run."""
    install, data = build_install(tmp_path)
    migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=lambda _port: True
    )
    assert not (data / "webui-data").exists()

    # The user closes Open WebUI and starts LOCITIZE again.
    second = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )

    assert second.ran is True, "a deferred item was never re-attempted"
    assert second.complete is True
    assert (data / "webui-data" / "webui.db").read_bytes() == b"sqlite-ish bytes"
    # Already-copied items are skipped by rule 2, not copied twice.
    memory_item = next(i for i in second.items if i.label == "conversation transcripts")
    assert memory_item.action == migration.ACTION_DESTINATION_IN_USE
    assert sorted(p.name for p in (data / "memory").iterdir()) == ["2026-08-01.jsonl"]

    # Only now is the migration over, the notice owed, and the marker final.
    payload = json.loads((data / migration.MARKER_NAME).read_text(encoding="utf-8"))
    assert payload["complete"] is True and payload["deferred"] == []
    assert migration.already_migrated(data) is True
    assert migration.pending_notice(data) == second.notice
    # The marker still records that THIS tool copied the transcripts, even
    # though the second attempt saw them as already present.
    actions = {row["label"]: row["action"] for row in payload["items"]}
    assert actions["conversation transcripts"] == migration.ACTION_COPIED
    assert actions["Open WebUI store"] == migration.ACTION_COPIED

    # And a third run finally short-circuits.
    third = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )
    assert third.ran is False


def test_data_root_migration_clears_a_deferral_that_resolves_without_a_copy(tmp_path):
    """HIGH-1r: a deferral resolves however it resolves, and the marker learns it.

    The marker used to be rewritten only when a run COPIED or DEFERRED something.
    A run that cleared the outstanding deferral any other way wrote nothing, so
    the incomplete marker survived permanently: already_migrated stayed False
    forever, the whole relocation loop re-ran on every launch for the life of the
    install, and - the real harm - the one-time notice was never shown even
    though the user's logs and transcripts had genuinely moved. "A user who
    cannot see that their history moved will conclude it was lost."

    Both ordinary triggers are driven here, because they arrive by completely
    different routes and only the marker rule is shared:

      destination-gains-content  Open WebUI is started against the data root and
                                 writes <data>/webui-data/webui.db itself
      source-goes-away           the user removes the install-side original,
                                 exactly as the notice invites them to
    """
    for trigger in ("destination gains content", "source removed"):
        root = tmp_path / trigger.replace(" ", "-")
        root.mkdir()
        install, data = build_install(root)

        first = migration.migrate_install_data(
            install, data, webui_port=8081, service_is_running=lambda _port: True
        )
        # Asserted rather than assumed: if the first run had failed an item (a
        # locked file, an antivirus hold), the marker would be withheld for THAT
        # reason and the rest of this test would fail somewhere far away from
        # the cause.
        assert first.ok is True, [
            f"{i.label}: {i.error}" for i in first.items if i.error
        ]
        assert [i.label for i in first.deferred] == ["Open WebUI store"], trigger
        assert migration.already_migrated(data) is False, trigger
        assert migration.pending_notice(data) == "", trigger

        if trigger == "destination gains content":
            (data / "webui-data").mkdir()
            (data / "webui-data" / "webui.db").write_bytes(b"live sqlite")
            expected = migration.ACTION_DESTINATION_IN_USE
        else:
            shutil.rmtree(install / "webui-data")
            expected = migration.ACTION_NO_SOURCE

        second = migration.migrate_install_data(
            install, data, webui_port=8081, service_is_running=never_running
        )

        # Nothing was copied on this run - that is the whole point of the case.
        assert second.copied == [], trigger
        webui_item = next(i for i in second.items if i.label == "Open WebUI store")
        assert webui_item.action == expected, trigger
        assert second.complete is True, trigger

        payload = json.loads((data / migration.MARKER_NAME).read_text(encoding="utf-8"))
        assert payload["deferred"] == [], f"{trigger}: the deferral was never cleared"
        assert payload["complete"] is True, trigger
        assert migration.already_migrated(data) is True, trigger
        assert migration.pending_notice(data) == second.notice, (
            f"{trigger}: the user's data moved and they are never told"
        )
        # The audit trail still records what THIS tool copied on the first run.
        actions = {row["label"]: row["action"] for row in payload["items"]}
        assert actions["logs"] == migration.ACTION_COPIED, trigger
        assert actions["conversation transcripts"] == migration.ACTION_COPIED, trigger

        # And the migration is finally over: the next launch does no work at all.
        third = migration.migrate_install_data(
            install, data, webui_port=8081, service_is_running=never_running
        )
        assert third.ran is False, f"{trigger}: still re-running after completion"


def test_data_root_migration_a_failing_run_never_clears_an_existing_deferral(tmp_path):
    """The other half of HIGH-1r's rule: `ok` still gates every marker write.

    Widening the write condition must not let a FAILED run claim completion. The
    run staged here is the one where that would be invisible: the outstanding
    deferral IS resolved (Open WebUI has been closed and the store copies fine)
    while a different item errors. If `ok` stopped gating the write, the marker
    would flip to complete on a run that half failed, the notice would go out,
    and the failed item would never be retried.
    """
    install, data = build_install(tmp_path)
    migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=lambda _port: True
    )
    marker = data / migration.MARKER_NAME

    # A real OSError, not a mocked one: the user removes <data>/reports and a
    # FILE ends up in its place, so the benchmark item's parent cannot be made.
    shutil.rmtree(data / "reports")
    (data / "reports").write_text("a file where a folder must go\n", encoding="utf-8")

    second = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )

    assert second.ok is False and second.errors
    # The deferral really was resolved on this run - so only `ok` can be what
    # withholds the marker.
    assert [i.label for i in second.copied] == ["Open WebUI store"]
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["deferred"] == ["Open WebUI store"], (
        "a failing run rewrote the marker and cleared a deferral"
    )
    assert payload["complete"] is False
    assert migration.already_migrated(data) is False
    assert migration.pending_notice(data) == ""


def test_data_root_migration_the_launcher_announces_a_deferral_it_finally_clears(
    tmp_path,
):
    """HIGH-1r's other silent surface: the terminal and the launcher log.

    `_log_migration` spoke the notice only when THIS run copied something, so on
    the run that finally completed a deferred migration it said nothing at all -
    and a headless or terminal user has no in-app notice to fall back on. Their
    transcripts had moved and nothing anywhere told them.
    """
    import logging

    from launcher import Launcher

    install, data = build_install(tmp_path)
    first = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=lambda _port: True
    )
    # The deferral resolves without a copy: Open WebUI made the folder itself.
    (data / "webui-data").mkdir()
    (data / "webui-data" / "webui.db").write_bytes(b"live sqlite")
    second = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )
    assert second.copied == [] and second.complete is True

    lines: list[str] = []
    launcher = Launcher(deps={"output_fn": lines.append})
    launcher._log_migration(logging.getLogger("test-migration"), second, str(data))

    spoken = "\n".join(lines)
    assert str(install) in spoken and str(data) in spoken, (
        f"the run that completed the migration said nothing useful: {lines}"
    )
    assert second.notice in spoken

    # ...and a machine with nothing to migrate is NOT told its data was copied.
    quiet_root = tmp_path / "clean"
    (quiet_root / "install").mkdir(parents=True)
    (quiet_root / "locitize-data").mkdir()
    nothing = migration.migrate_install_data(
        quiet_root / "install",
        quiet_root / "locitize-data",
        webui_port=8081,
        service_is_running=never_running,
    )
    quiet_lines: list[str] = []
    Launcher(deps={"output_fn": quiet_lines.append})._log_migration(
        logging.getLogger("test-migration"), nothing, str(quiet_root / "locitize-data")
    )
    assert quiet_lines == [], f"claimed a copy that never happened: {quiet_lines}"
    assert first.notice  # the notice text itself is unconditional, by design


def test_data_root_migration_sweep_spares_user_paths_that_merely_look_like_staging(
    tmp_path,
):
    """MEDIUM-1r: the only routine that deletes inside the backup folder.

    The sweep used to rglob the ENTIRE data root for `*.migrating-*` and rmtree
    every match, so a reviewer's `notes.migrating-plan.md` and
    `archive.migrating-2025` - ordinary user files in the folder the product
    tells people to back up - were both destroyed by a helper whose docstring
    says it cannot lose anything.

    Every survivor below is aged past the sweep threshold, so age is not what
    saves it: the name pattern, the location and the kind are.
    """
    import os
    import time

    install, data = build_install(tmp_path)
    stale = time.time() - migration.STALE_STAGING_AGE_S - 60

    def age(path: Path) -> None:
        os.utime(path, (stale, stale))

    # Ours: the exact shape _copy_atomically authors, left by a killed process.
    # The sentinel is part of that shape - _copy_atomically writes it right
    # after the mkdir and only unlinks it once the copy completed, so a process
    # killed mid-copy always leaves one. Staging it without the sentinel would
    # simulate something the product cannot produce, and the sweep would be
    # right to spare it.
    orphan = data / "memory.migrating-4242-deadbeef"
    orphan.mkdir()
    (orphan / migration._STAGING_SENTINEL).write_text("staging\n", encoding="utf-8")
    (orphan / "half-copied.jsonl").write_text("partial\n", encoding="utf-8")
    age(orphan)

    # Theirs, and all of it must survive.
    nested = data / "models" / "my-backups"
    nested.mkdir(parents=True)
    user_file_deep = nested / "notes.migrating-plan.md"
    user_file_deep.write_text("my notes\n", encoding="utf-8")
    age(user_file_deep)

    user_dir_deep = data / "reports" / "archive.migrating-2025"
    user_dir_deep.mkdir(parents=True)
    (user_dir_deep / "keep.md").write_text("keep this\n", encoding="utf-8")
    age(user_dir_deep)

    # The hardest cases: user paths at the EXACT depth the migration stages at,
    # carrying a destination name as their prefix. Only the authored
    # <pid>-<hex> suffix tells these apart from a real staging path.
    user_file_top = data / "logs.migrating-notes.txt"
    user_file_top.write_text("a file at the staging depth\n", encoding="utf-8")
    age(user_file_top)
    user_dir_top = data / "webui-data.migrating-backup"
    user_dir_top.mkdir()
    (user_dir_top / "mine.txt").write_text("mine\n", encoding="utf-8")
    age(user_dir_top)

    migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )

    assert not orphan.exists(), "the genuine staging orphan was left behind"
    assert user_file_deep.read_text(encoding="utf-8") == "my notes\n"
    assert (user_dir_deep / "keep.md").read_text(encoding="utf-8") == "keep this\n"
    assert user_file_top.read_text(encoding="utf-8") == "a file at the staging depth\n"
    assert (user_dir_top / "mine.txt").read_text(encoding="utf-8") == "mine\n"


def test_data_root_migration_copies_into_an_existing_empty_destination(tmp_path):
    """The shape _has_content's docstring promises to tolerate, proved.

    `os.replace` onto an existing directory raises PermissionError [WinError 5]
    on Windows whether or not that directory is empty, so before the fix this
    failed on every launch forever, telling the user to "close anything using
    those files" when nothing had them open (round 6, HIGH-2). An empty
    <data>/logs is not exotic: configure_logging creates it moments after the
    migration runs, so any deferred-then-retried machine hits exactly this.
    """
    install, data = build_install(tmp_path)
    (data / "logs").mkdir()
    (data / "memory").mkdir()

    outcome = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )

    assert outcome.ok is True, [f"{i.label}: {i.error}" for i in outcome.items if i.error]
    assert outcome.complete is True
    logs_item = next(i for i in outcome.items if i.label == "logs")
    assert logs_item.action == migration.ACTION_COPIED
    assert (data / "logs" / "launcher.log").read_text(encoding="utf-8") == "old log line\n"
    assert "must not be lost" in (data / "memory" / "2026-08-01.jsonl").read_text(
        encoding="utf-8"
    )
    assert not list(data.glob("*.migrating-*"))


def test_data_root_migration_never_deletes_a_destination_that_has_content(tmp_path):
    """The empty-destination fix must not become a delete-the-destination bug.

    The guarded removal uses rmdir, which REFUSES a non-empty directory. This
    proves the refusal by handing it the one shape that could be catastrophic:
    a destination holding the user's newer history.
    """
    install, data = build_install(tmp_path)
    (data / "memory").mkdir()
    (data / "memory" / "2026-08-18.jsonl").write_text("newer history\n", encoding="utf-8")

    migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )

    assert (data / "memory" / "2026-08-18.jsonl").read_text(encoding="utf-8") == (
        "newer history\n"
    )
    assert sorted(p.name for p in (data / "memory").iterdir()) == ["2026-08-18.jsonl"]


def test_data_root_migration_clearing_a_destination_refuses_anything_with_content(
    tmp_path,
):
    """The empty-destination clearance must be incapable of deleting user data.

    Tested on the helper directly, because in the normal flow rule 2 has already
    skipped a destination with content - so a recursive delete here would look
    identical from the outside while being catastrophic in the one case that
    matters: a destination that GAINS content between rule 2's check and the
    rename (another LOCITIZE window, a sync client). `rmdir` refuses; the failure
    is recorded and every byte survives. That refusal is the whole safety
    property, so it is asserted rather than assumed.
    """
    import pytest

    destination = tmp_path / "memory"
    destination.mkdir()
    (destination / "2026-08-18.jsonl").write_text("newer history\n", encoding="utf-8")

    with pytest.raises(OSError):
        migration._clear_empty_destination(destination)

    assert (destination / "2026-08-18.jsonl").read_text(encoding="utf-8") == (
        "newer history\n"
    )

    # And it does clear the shape it exists for.
    empty = tmp_path / "logs"
    empty.mkdir()
    migration._clear_empty_destination(empty)
    assert not empty.exists()


def test_data_root_migration_two_attempts_in_one_process_do_not_destroy_each_other(
    tmp_path,
):
    """MEDIUM-3: staging paths are unique per ATTEMPT, not per process.

    Two migrations racing inside one process used to share
    `<dest>.migrating-<pid>`, so each one's cleanup deleted the other's
    in-flight copy and NOTHING was migrated. Threads are used because that is
    the only way to produce the same-pid case the defect needed.
    """
    import threading

    install, data = build_install(tmp_path)
    outcomes: list[migration.MigrationOutcome] = []
    lock = threading.Lock()

    def attempt() -> None:
        result = migration.migrate_install_data(
            install, data, webui_port=8081, service_is_running=never_running
        )
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(outcomes) == 2
    # The data is there exactly once, whichever attempt won each item.
    assert (data / "logs" / "launcher.log").read_text(encoding="utf-8") == "old log line\n"
    assert sorted(p.name for p in (data / "memory").iterdir()) == ["2026-08-01.jsonl"]
    assert (data / "webui-data" / "webui.db").read_bytes() == b"sqlite-ish bytes"
    # No staging path survived, and the originals are untouched.
    assert not list(data.rglob("*.migrating-*"))
    assert (install / "memory" / "2026-08-01.jsonl").is_file()


def test_data_root_migration_sweeps_a_staging_orphan_from_a_killed_process(tmp_path):
    """A process killed mid-copy must not leave a partial copy of transcripts.

    `_remove` only ever cleared the CURRENT attempt's staging, so an orphan from
    a killed run lived forever inside the folder the product tells the user to
    back up. Aged past the sweep threshold here, because a fresh staging path
    might belong to a migration running right now in another LOCITIZE process and
    must never be touched.
    """
    import os
    import time

    install, data = build_install(tmp_path)
    # The sentinel is what proves we authored this directory; _copy_atomically
    # writes it before the copy and unlinks it after, so a killed process
    # always leaves one behind. Omitting it here would test a shape the product
    # never creates.
    orphan = data / "memory.migrating-4242-deadbeef"
    orphan.mkdir()
    (orphan / migration._STAGING_SENTINEL).write_text("staging\n", encoding="utf-8")
    (orphan / "half-copied.jsonl").write_text("partial\n", encoding="utf-8")
    stale = time.time() - migration.STALE_STAGING_AGE_S - 60
    os.utime(orphan, (stale, stale))

    fresh = data / "logs.migrating-4243-cafebabe"
    fresh.mkdir()
    (fresh / migration._STAGING_SENTINEL).write_text("staging\n", encoding="utf-8")

    migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )

    assert not orphan.exists(), "a stale staging orphan was left behind"
    assert fresh.exists(), "the sweep deleted a staging path that could be in flight"


def test_data_root_migration_reads_a_pre_deferral_marker_as_complete(tmp_path):
    """A marker written before the deferral field existed still means "done".

    Not hypothetical: the owner's machine already carries one, written at
    2026-08-19T14:55:05 with six items copied and no `deferred` key at all. If
    the new reader treated a missing key as "incomplete", his next launch would
    re-walk a migration that finished days ago, and if it treated an unreadable
    marker as absent it would do so forever. Both shapes are pinned here.
    """
    install, data = build_install(tmp_path)
    legacy = {
        "migrated_from_install": True,
        "install_dir": str(install),
        "data_dir": str(data),
        "when": "2026-08-19T14:55:05",
        "notice": migration.notice_text(install, data),
        "notice_shown": False,
        "items": [{"label": "logs", "action": "copied", "source": "", "destination": ""}],
    }
    (data / migration.MARKER_NAME).write_text(json.dumps(legacy), encoding="utf-8")

    assert migration.already_migrated(data) is True
    # The notice is still owed - the flag, not the schema, decides that.
    assert migration.pending_notice(data) == legacy["notice"]
    outcome = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )
    assert outcome.ran is False, "a legacy marker no longer stops a re-migration"

    # A corrupt marker is also treated as "a migration ran here", never as absent:
    # rule 2's non-empty check still protects the data, and re-running on every
    # launch forever would be the noisier failure.
    (data / migration.MARKER_NAME).write_text("{not json", encoding="utf-8")
    assert migration.already_migrated(data) is True
    assert migration.pending_notice(data) == ""


def test_data_root_migration_requires_the_webui_port_to_be_supplied(tmp_path):
    """MEDIUM-5: the live-sqlite guard cannot be disabled by omission.

    `webui_port` had a default of 0 and the guard read `webui_port and ...`, so
    the UNSAFE behaviour - hot-copying a live database - was what a caller got
    for forgetting an argument. It is now a required keyword, and a caller that
    genuinely passes 0 gets the documented default port checked instead of
    nothing.
    """
    import inspect

    install, data = build_install(tmp_path)
    parameter = inspect.signature(migration.migrate_install_data).parameters["webui_port"]
    assert parameter.default is inspect.Parameter.empty, (
        "webui_port has a default again; a safety guard whose default is "
        "'do not check' is not a guard"
    )

    checked: list[int] = []

    def record(port: int) -> bool:
        checked.append(port)
        return True

    outcome = migration.migrate_install_data(
        install, data, webui_port=0, service_is_running=record
    )
    assert checked == [migration.DEFAULT_WEBUI_PORT]
    assert outcome.complete is False, "a 0 port silently hot-copied the live store"


def test_data_root_migration_does_nothing_when_the_data_root_is_the_install(tmp_path):
    """Portable/dev mode and every test: source and destination are one folder."""
    install, _data = build_install(tmp_path)
    outcome = migration.migrate_install_data(
        install, install, webui_port=0, service_is_running=never_running
    )
    assert outcome.ran is False and outcome.ok is True
    assert not (install / migration.MARKER_NAME).exists()


def test_data_root_migration_notice_names_both_paths_and_shows_exactly_once(tmp_path):
    """Rule 7: the user is told, in words they can act on, one time.

    A product that relocates someone's chat history without a word has done
    something they cannot audit, so the wording is pinned here rather than left
    to a documentation pass.
    """
    install, data = build_install(tmp_path)
    outcome = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )

    notice = outcome.notice
    assert str(install) in notice and str(data) in notice
    # The required wording, asserted as the whole clause rather than as two
    # loose path substrings: a notice that said only where the data went would
    # still contain both paths (the last sentence names the old folder too) and
    # would still leave the user unable to tell WHERE it came from.
    assert f"copied from {install} to {data}" in notice
    assert f"the old copy is still at {install}" in notice
    assert "Nothing was deleted" in notice
    assert "reads and writes only the data folder" in notice
    # No Python repr of a path: the separators the user reads are single, the
    # way the filesystem writes them, not the doubled ones a repr produces.
    assert "\\\\" not in notice

    assert migration.pending_notice(data) == notice
    migration.mark_notice_shown(data)
    assert migration.pending_notice(data) == "", "the notice repeated itself"
    # Flipping the flag must not lose the record of what happened.
    payload = json.loads((data / migration.MARKER_NAME).read_text(encoding="utf-8"))
    assert payload["migrated_from_install"] is True and payload["notice_shown"] is True


def test_data_root_migration_notice_is_absent_when_nothing_was_copied(tmp_path):
    """A fresh install has nothing to migrate, so it owes the user no notice."""
    install = tmp_path / "install"
    install.mkdir()
    data = tmp_path / "locitize-data"
    data.mkdir()

    outcome = migration.migrate_install_data(
        install, data, webui_port=8081, service_is_running=never_running
    )
    assert outcome.ran is True and outcome.ok is True and outcome.copied == []
    assert not (data / migration.MARKER_NAME).exists()
    assert migration.pending_notice(data) == ""
