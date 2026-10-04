"""One-time relocation of user data from the install tree into the data root.

WHAT THIS IS (DEC-M14-9)
------------------------
Until M14, LOCITIZE wrote the user's logs, conversation transcripts, Open WebUI
chat database, benchmark reports and generated Caddyfile inside its own install
directory. DEC-M14-9 moved every one of those to the data root, so that "back up
one folder and you have backed up LOCITIZE" is true and a reinstall cannot destroy
anyone's chat history.

This module is what happens to the data that is ALREADY in the install tree on
the machine of someone who upgrades. The rules are deliberately timid, because
the artefact at stake is a person's conversation history:

1. COPY, never move. The originals are left byte-identical, so a mistake here
   costs the user nothing and an older LOCITIZE on the same machine keeps working.
2. A destination that already exists and is NOT empty is skipped entirely.
   Nothing is merged, overwritten or "reconciled" - merging two histories
   automatically is how someone loses a conversation.
3. Each item is copied to a temporary sibling first and renamed into place only
   once the whole copy succeeded, so a failure can never leave a half-populated
   destination that rule 2 would then mistake for real data.
4. The marker that says "this machine has been migrated" lives in the DATA ROOT
   and claims completion only when every item either succeeded or had nothing to
   copy. A partial failure is reported, not silently marked done - an honest
   failure beats a silent half-state, and an incomplete marker is what lets the
   next run retry.
5. webui-data is a live sqlite database, so it is copied only when Open WebUI is
   not running. A hot copy of sqlite is a corrupt copy. Being busy DEFERS the
   item; it never completes it. A deferred item is written into the marker by
   name, the marker is flagged incomplete, and the next run re-attempts exactly
   those items before it is allowed to short-circuit. Treating "busy" as "done"
   would abandon the user's chat history permanently while telling them it moved
   (review round 6, HIGH-1). A deferral is resolved however it resolves - the
   copy finally happens, or the destination has meanwhile gained content, or the
   install-side original is gone - and the marker records that resolution in all
   three cases, or it would name a deferral that nothing will ever clear
   (review round 7, HIGH-1r).
6. The user is told once, in the app, in a notice that names BOTH real paths.
   A product that relocates someone's chat history without a word has done
   something they cannot audit. Every sentence of that notice must be something
   this module MEASURED: it is offered only when the marker's audit trail names
   a real copy, and its closing sentence claims the old folder still exists only
   when the install-side originals were actually found there (round 8, HIGH).
   A notice is the user's only account of where their history went, so a false
   one is worse than none.

The install-side originals are never deleted by LOCITIZE. After migration LOCITIZE
reads and writes only the data root, and the old copy is the user's to remove.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# The marker file, in the data root beside settings.yaml. A COMPLETE marker is
# the first of the two independent guards (the other is rule 2's non-empty
# check) that stop a reinstall from copying stale install-side data over newer
# data. An INCOMPLETE marker (one naming deferred items) records progress
# without ending the migration.
MARKER_NAME = "data-migration.json"

# The port Open WebUI is served on by default (settings.default.yaml ->
# ports.openwebui). Used only when a caller cannot supply the configured port:
# rule 5's guard must fall back to checking the documented port, never to
# checking nothing, because "nothing to check" means a hot copy of a live
# sqlite database (review round 6, MEDIUM-5).
DEFAULT_WEBUI_PORT = 8096

# A staging directory older than this is an orphan from a process that was
# killed mid-copy (power loss, task kill), not a live copy: even a multi-GB
# webui store finishes in minutes. Swept at the start of each migration so a
# partial copy of someone's transcripts does not live forever inside the one
# folder the product tells them to back up (review round 6, MEDIUM-3).
STALE_STAGING_AGE_S = 24 * 60 * 60

# The infix every staging path carries, so the sweep above can recognise one and
# nothing else. Kept as a constant because two spellings of it would mean the
# sweep silently stopped matching what the copy creates.
_STAGING_MARK = ".migrating-"

# The proof of authorship the sweep requires before deleting anything. A NAME is
# not proof: a user directory called exactly "memory.migrating-1234-abcdef01" is
# improbable but perfectly legal, and the sweep used to delete it while its
# docstring claimed it only removed "names this module can prove it authored"
# (review round 8, MEDIUM). _copy_atomically writes this file into every staging
# directory it creates, and the sweep refuses to remove a directory that does not
# contain it - so "probably ours" became "provably ours".
_STAGING_SENTINEL = ".locitize-staging"

# What moves, in the order it is attempted. Each entry is
# (label, install-relative source, data-relative destination).
# Directories and files are handled by the same rules; the kind is detected from
# the source on disk rather than declared, so a user who somehow has a file
# where a directory was expected still gets a correct copy rather than a crash.
RELOCATIONS: tuple[tuple[str, str, str], ...] = (
    ("logs", "logs", "logs"),
    ("conversation transcripts", "memory", "memory"),
    ("Open WebUI store", "webui-data", "webui-data"),
    ("benchmark results", "docs/benchmark_results.jsonl", "reports/benchmark_results.jsonl"),
    ("benchmark report", "docs/benchmark_results.md", "reports/benchmark_results.md"),
    ("development journal", "docs/development_journal.md", "reports/development_journal.md"),
    ("proxy configuration", "Caddyfile", "Caddyfile"),
)

# The one item whose copy is unsafe while a service holds it open.
_LIVE_SERVICE_ITEM = "webui-data"

# Action values recorded per item. Only "copied" changes anything on disk.
ACTION_COPIED = "copied"
ACTION_NO_SOURCE = "skipped: nothing to copy"
ACTION_DESTINATION_IN_USE = "skipped: the data folder already has this"
ACTION_SERVICE_RUNNING = "skipped: Open WebUI is running"
ACTION_FAILED = "failed"


@dataclass(frozen=True)
class MigrationItem:
    """What happened to one relocated item, for logging, tests and the marker."""

    label: str
    action: str
    source: str
    destination: str
    error: str = ""


@dataclass(frozen=True)
class MigrationOutcome:
    """The whole result of one migration attempt.

    `ran` is False when there was nothing to do at all (already marked, or the
    data root IS the install tree). `ok` is False when any item failed, which is
    exactly the condition under which no marker is written.
    """

    ran: bool
    ok: bool
    items: list[MigrationItem] = field(default_factory=list)
    notice: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def copied(self) -> list[MigrationItem]:
        """The items whose bytes actually moved - what the notice is about."""
        return [item for item in self.items if item.action == ACTION_COPIED]

    @property
    def deferred(self) -> list[MigrationItem]:
        """Items that were refused for now and MUST be re-attempted later.

        Only rule 5's "the service is holding it open" produces one. It is kept
        distinct from both a success and a failure on purpose: nothing went
        wrong, but nothing is finished either, and conflating it with success is
        exactly how the Open WebUI store was abandoned (HIGH-1).
        """
        return [item for item in self.items if item.action == ACTION_SERVICE_RUNNING]

    @property
    def complete(self) -> bool:
        """True when no item failed and none is still waiting to be re-attempted.

        This - not `ok` - is the condition under which the migration is over.
        """
        return self.ok and not self.deferred


def notice_text(
    install_dir: Path | str,
    data_dir: Path | str,
    *,
    originals_remain: bool = True,
) -> str:
    """The required first-run wording (DEC-M14-9 item 7), with real paths.

    Kept in one function because the same sentence has to appear in the app and
    in the launcher log, and two copies of a sentence drift.

    The closing sentence is the only part that varies, and it varies because it
    is the only part that makes a claim about a folder LOCITIZE does not own. "The
    old copy is still at <install>" is true of what LOCITIZE did (it never deletes)
    but not necessarily true of what the USER did: the previous sentence
    explicitly invites them to remove that folder, and on a later run - the run
    that finally clears a deferred item - the folder can already be gone. So the
    caller measures whether install-side originals are still there and this
    function states only what was measured (review round 8, HIGH). Every claim
    in this notice has to be one the product checked, because the notice is the
    only thing standing between the user and a false belief about where their
    conversation history lives.
    """
    if originals_remain:
        closing = (
            "Nothing was deleted: the old copy is still at "
            f"{install_dir} and you can remove it once you have checked the new "
            "folder."
        )
    else:
        closing = (
            "LOCITIZE deleted nothing - it only ever copied - and the old folder at "
            f"{install_dir} is no longer on this machine."
        )
    return (
        "LOCITIZE moved to a single data folder. Your existing conversations, logs "
        f"and settings were copied from {install_dir} to {data_dir}. LOCITIZE now "
        f"reads and writes only the data folder. {closing}"
    )


def marker_path(data_dir: Path | str) -> Path:
    """Where the 'already migrated' record lives - in the data root, only."""
    return Path(data_dir) / MARKER_NAME


def _read_marker(data_dir: Path | str) -> dict[str, Any] | None:
    """The marker's payload, or None when there is no readable marker."""
    marker = marker_path(data_dir)
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # An unreadable marker still asserts "a migration ran on this machine".
        # Treated as complete rather than as absent: rule 2's non-empty check
        # independently protects the data, and re-running forever on a corrupt
        # file would be the noisier failure.
        return {}
    return payload if isinstance(payload, dict) else {}


def already_migrated(data_dir: Path | str) -> bool:
    """True once a COMPLETE migration has been recorded for this data root.

    A marker that names deferred items (rule 5: Open WebUI was holding its
    database open) is progress, not completion, so it deliberately does NOT
    short-circuit the next run - that run is how the deferred item finally
    gets copied.
    """
    payload = _read_marker(data_dir)
    if payload is None:
        return False
    return not payload.get("deferred")


def migrate_install_data(
    install_dir: Path | str,
    data_dir: Path | str,
    *,
    webui_port: int,
    service_is_running: Callable[[int], bool] | None = None,
) -> MigrationOutcome:
    """Copy pre-M14 user data out of the install tree into the data root.

    Returns a MigrationOutcome describing every item, so the caller can log it
    honestly and show the user the one-time notice. Never raises for a per-item
    failure: a failure is recorded, the marker is withheld, and the caller keeps
    running on the data root (never falling back to the install directory).

    `webui_port` is required rather than defaulted: it gates rule 5's live-sqlite
    guard, and a guard whose default is "do not check" is not a guard. A caller
    that genuinely has no configured port passes 0 and gets the documented
    default port checked instead (MEDIUM-5).

    `service_is_running` is the seam that keeps rule 5 testable without starting
    Open WebUI; it defaults to a loopback connect on `webui_port`.
    """
    install = Path(install_dir)
    data = Path(data_dir)

    # Nothing to do when source and destination are the same folder: a portable
    # or dev checkout whose data root IS the install directory, and every test
    # that hands Config.load an explicit base_dir.
    if install.resolve() == data.resolve():
        return MigrationOutcome(ran=False, ok=True)
    if already_migrated(data):
        return MigrationOutcome(ran=False, ok=True)

    is_running = service_is_running or _port_is_open
    checked_port = int(webui_port) or DEFAULT_WEBUI_PORT
    items: list[MigrationItem] = []
    errors: list[str] = []

    _sweep_stale_staging(data)

    for label, rel_source, rel_dest in RELOCATIONS:
        source = install / rel_source
        destination = data / rel_dest
        item = _relocate_one(
            label, source, destination, checked_port, is_running, rel_source
        )
        items.append(item)
        if item.action == ACTION_FAILED:
            errors.append(f"{label}: {item.error}")

    # Measured, not assumed: does the install tree still hold any of the data
    # this notice is about? The user may already have removed it (the notice
    # invites exactly that), and the closing sentence must not claim otherwise.
    originals_remain = any(
        _has_content(install / rel_source) for _label, rel_source, _rel in RELOCATIONS
    )

    outcome = MigrationOutcome(
        ran=True,
        ok=not errors,
        items=items,
        notice=notice_text(install, data, originals_remain=originals_remain),
        errors=errors,
    )

    if _should_write_marker(data, outcome):
        _write_marker(data, install, outcome)
    return outcome


def _should_write_marker(data: Path, outcome: MigrationOutcome) -> bool:
    """Rule 4: does THIS run's result belong in the marker file?

    Extracted into a named predicate because three separate defects have now
    landed on this one condition (review rounds 6 and 7), and a growing boolean
    inside migrate_install_data was not something anyone could test directly.

    Two different claims share one file:
      * a marker with an empty "deferred" list says "this machine is migrated",
        short-circuits every later run, and puts the one-time notice on offer;
      * a marker naming deferred items says "this much is done, come back for
        the rest" - it records progress and the pending notice, and
        already_migrated() deliberately does not accept it.

    A partial FAILURE earns neither: no marker at all is written, so the next
    run retries the failed items (the ones that succeeded are then skipped by
    rule 2's non-empty check, so a retry cannot duplicate anything).

    The third clause is the round-7 fix (HIGH-1r). A run that CLEARS a deferral
    any way other than by copying - Open WebUI has since created
    <data>/webui-data itself, or the user removed the install-side original as
    the notice invites them to - copies nothing and defers nothing, so the first
    two clauses are both false and the incomplete marker used to survive
    forever. The consequences were all user-visible: the notice was withheld
    although the transcripts really had moved, already_migrated never
    short-circuited, and nothing was left that would ever retry the store. So
    when a marker already exists, this run's result is recorded too - that is
    what lets `deferred` finally go empty and `complete` finally go True.
    """
    if not outcome.ok:
        return False
    return bool(outcome.copied or outcome.deferred) or _read_marker(data) is not None


def _staging_name_pattern(destination_name: str) -> re.Pattern[str]:
    """The exact names _copy_atomically can author for one destination.

    Anchored at both ends and spelled out in full - "<destination>.migrating-
    <pid>-<8 hex digits>" - because the sweep below DELETES what this matches.
    A substring glob would also match a user's own `notes.migrating-plan.md` or
    `archive.migrating-2025`, and a reviewer destroyed exactly those two with the
    earlier version of this sweep (review round 7, MEDIUM-1r).
    """
    return re.compile(
        rf"^{re.escape(destination_name)}{re.escape(_STAGING_MARK)}\d+-[0-9a-f]{{8}}$"
    )


def _is_link_or_junction(path: Path) -> bool:
    """True for a symlink or a Windows junction - or an unreadable path.

    Nothing this module creates is ever a link, so a link wearing a staging name
    was authored by someone else and is not ours to delete. Failing to read the
    path counts as "yes, leave it alone": the sweep's only job is tidying, and
    tidying must never be the thing that removes a person's data.
    """
    try:
        info = path.lstat()
    except OSError:
        return True
    return path.is_symlink() or bool(getattr(info, "st_reparse_tag", 0))


def _bears_staging_sentinel(path: Path) -> bool:
    """True only when this directory carries the file _copy_atomically writes.

    The sweep's proof of authorship. Anything unreadable counts as "not ours":
    the sweep only tidies, so every uncertainty must resolve toward leaving the
    path alone.
    """
    try:
        return (path / _STAGING_SENTINEL).is_file()
    except OSError:
        return False


def _sweep_stale_staging(data: Path) -> None:
    """Delete staging directories left behind by a process that died mid-copy.

    Five independent constraints, because this is the only routine in LOCITIZE that
    deletes anything inside the folder users are told to back up:

    1. LOCATION - only the directories _copy_atomically stages in, which is
       `destination.parent` for each entry in RELOCATIONS (the data root itself
       and <data>/reports). Never recursive: the old `data.rglob` reached user
       paths at any depth, and it also walked the entire model store - hundreds
       of GB - on every attempt.
    2. NAME - only names _copy_atomically can author, matched against the full
       anchored pattern above rather than a `*.migrating-*` substring. This
       narrows the candidates; it does not prove authorship, which is what
       constraint 5 is for.
    3. KIND - directories only, and never a link or junction. A staging FILE
       orphan (the shape left by a killed copy of Caddyfile or a report) is
       deliberately left in place: it is a few kilobytes of clutter that blocks
       nothing, and no file-shaped rule is worth the risk of unlinking something
       a person wrote.
    4. AGE - only paths older than STALE_STAGING_AGE_S, so a migration running
       right now in another LOCITIZE process can never have its in-flight copy
       deleted by this one.
    5. PROOF - only a directory containing the _STAGING_SENTINEL file that
       _copy_atomically writes at creation. A name is not proof: a user directory
       named exactly `memory.migrating-1234-abcdef01` satisfies constraints 1-4
       and was deleted with its contents (review round 8, MEDIUM). A file this
       module wrote is proof, and nothing else in this routine claims to be.

    What survives all five was written by this module and holds nothing but a
    partial copy of data that still exists intact in the install tree, so
    removing it loses nothing.
    """
    cutoff = time.time() - STALE_STAGING_AGE_S
    for _label, _rel_source, rel_dest in RELOCATIONS:
        destination = data / rel_dest
        pattern = _staging_name_pattern(destination.name)
        try:
            candidates = list(
                destination.parent.glob(f"{destination.name}{_STAGING_MARK}*")
            )
        except OSError:
            continue
        for path in candidates:
            if not pattern.match(path.name):
                continue
            if _is_link_or_junction(path) or not path.is_dir():
                continue
            if not _bears_staging_sentinel(path):
                continue
            try:
                if path.stat().st_mtime > cutoff:
                    continue
            except OSError:
                continue
            _remove(path)


def _relocate_one(
    label: str,
    source: Path,
    destination: Path,
    webui_port: int,
    is_running: Callable[[int], bool],
    rel_source: str,
) -> MigrationItem:
    """Apply the copy rules to exactly one item. Never raises."""
    def record(action: str, error: str = "") -> MigrationItem:
        """Every exit below reports the same five facts, so they are built once."""
        return MigrationItem(label, action, str(source), str(destination), error)

    if not _has_content(source):
        return record(ACTION_NO_SOURCE)
    if _has_content(destination):
        # Rule 2. Both paths are recorded so the log line names what was kept
        # and what was left behind, which is the only way a user can check.
        return record(ACTION_DESTINATION_IN_USE)
    # Rule 5. No `webui_port and ...` short-circuit: the caller always hands a
    # port here (migrate_install_data substitutes the documented default), so
    # the guard cannot be disabled by omission (MEDIUM-5). The resulting skip is
    # a DEFERRAL - migrate_install_data will not call the migration complete
    # while it stands.
    if rel_source == _LIVE_SERVICE_ITEM and is_running(webui_port):
        return record(ACTION_SERVICE_RUNNING)

    try:
        _copy_atomically(source, destination)
    except OSError as exc:
        # strerror, not str(exc): str(OSError) carries the errno and a repr of
        # the filename, which is the unreadable shape NEW-QA-M14-9 was about.
        return record(ACTION_FAILED, (exc.strerror or "the copy failed").strip())
    return record(ACTION_COPIED)


def _copy_atomically(source: Path, destination: Path) -> None:
    """Copy source to destination via a temporary sibling, then rename it in.

    Rule 3. Copying straight into the destination would leave a half-populated
    folder behind if the copy died half way, and rule 2 would then read that
    folder as "the user already has data here" and skip it forever. Renaming a
    finished copy into place means the destination either does not exist or is
    complete, with no third state.

    The staging name carries a random suffix as well as the pid, because two
    attempts inside ONE process (two windows, a retry racing a launch) would
    otherwise share a staging path and each one's cleanup would delete the
    other's in-flight copy, leaving nothing migrated (MEDIUM-3).

    A directory staging path also gets the _STAGING_SENTINEL file written into it
    BEFORE any user data is copied in, because that file is what later proves to
    _sweep_stale_staging that this module authored the directory it is about to
    delete (round 8, MEDIUM). It is removed again immediately before the rename,
    so the sentinel never lands inside the user's migrated data. The two windows
    that leaves - killed before the sentinel is written, or killed between the
    unlink and the rename - both produce an orphan the sweep will REFUSE to
    touch. That is the safe direction to fail in: a few stray megabytes of a copy
    whose original still exists, rather than a deletion this code cannot justify.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / (
        f"{destination.name}{_STAGING_MARK}{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    _remove(staging)
    try:
        if source.is_dir():
            staging.mkdir()
            (staging / _STAGING_SENTINEL).write_text(
                "Created by LOCITIZE while copying user data into the data folder. "
                "Safe to delete along with this directory.\n",
                encoding="utf-8",
            )
            # dirs_exist_ok because the sentinel means the directory is already
            # there; copytree refuses an existing destination otherwise.
            shutil.copytree(source, staging, dirs_exist_ok=True)
        else:
            shutil.copy2(source, staging)
        _clear_empty_destination(destination)
        if source.is_dir():
            (staging / _STAGING_SENTINEL).unlink()
        os.replace(staging, destination)
    except BaseException:
        _remove(staging)
        raise


def _clear_empty_destination(destination: Path) -> None:
    """Remove an EMPTY destination directory so the rename can land on it.

    On Windows `os.replace` raises PermissionError [WinError 5] when the target
    is an existing directory - empty or not - so without this the copy fails on
    every launch, forever, with a diagnosis ("close anything using those files")
    that the user cannot act on (review round 6, HIGH-2). The shape is ordinary:
    a partly-restored backup, a sync client, or LOCITIZE's own configure_logging
    creating <data>/logs on a run where the item was deferred.

    `rmdir` is used rather than any recursive delete precisely because it
    REFUSES to remove a directory with anything in it. That refusal is the
    safety property: this function can never delete a byte of user data, and a
    non-empty destination is rule 2's business (it was already skipped upstream).
    A file destination needs nothing - os.replace overwrites a file happily.
    """
    if destination.is_dir():
        destination.rmdir()


def _remove(path: Path) -> None:
    """Delete a staging path if it exists. Only ever called on our own staging."""
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def _has_content(path: Path) -> bool:
    """True when path is a non-empty file or a directory holding anything.

    An empty directory is treated as absent on BOTH sides on purpose: an empty
    logs/ is nothing to copy, and an empty destination folder (the shape a
    previous run's mkdir leaves) must not block the copy. A directory holding
    only the repository's .gitkeep placeholder counts as empty, so a fresh
    clone is not told its (nonexistent) data was moved.
    """
    if path.is_dir():
        return any(child.name != ".gitkeep" for child in path.iterdir())
    if path.is_file():
        return path.stat().st_size > 0
    return False


def _write_marker(data: Path, install: Path, outcome: MigrationOutcome) -> None:
    """Record the migration's state, and that the notice is still owed.

    `deferred` is the load-bearing field: a non-empty list means this marker is
    a progress record, not a completion claim, so already_migrated() refuses it
    and pending_notice() withholds a notice that would otherwise tell the user
    their conversations were copied when one item was only skipped.
    """
    deferred = [item.label for item in outcome.deferred]
    previous = _read_marker(data) or {}
    payload: dict[str, Any] = {
        "migrated_from_install": True,
        "complete": not deferred,
        "deferred": deferred,
        "install_dir": str(install),
        "data_dir": str(data),
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notice": outcome.notice,
        # The notice is owed to the user, not to this file: the flag flips only
        # once a surface has actually shown it (see mark_notice_shown). A rewrite
        # must never un-show a notice the user already read.
        "notice_shown": bool(previous.get("notice_shown")),
        "items": _merge_items(previous.get("items"), outcome.items),
    }
    try:
        marker_path(data).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    except OSError:
        # A marker that cannot be written is not a reason to fail the run: the
        # data is already copied, and rule 2's non-empty check alone still
        # prevents a second copy. The next run simply re-reports.
        pass


def _merge_items(previous: Any, current: list[MigrationItem]) -> list[dict[str, str]]:
    """This attempt's per-item records, keeping an earlier attempt's "copied".

    The marker is the only audit trail the user has of where their data went. A
    second attempt reports the already-copied items as "the data folder already
    has this" - true, but it would erase the record that THIS tool copied them,
    so the earlier "copied" line wins for that label.
    """
    kept: dict[str, dict[str, str]] = {}
    if isinstance(previous, list):
        for row in previous:
            if isinstance(row, dict) and row.get("action") == ACTION_COPIED:
                kept[str(row.get("label", ""))] = {
                    str(k): str(v) for k, v in row.items()
                }
    merged: list[dict[str, str]] = []
    for item in current:
        earlier = kept.get(item.label)
        if earlier is not None and item.action != ACTION_COPIED:
            merged.append(earlier)
            continue
        merged.append(
            {
                "label": item.label,
                "action": item.action,
                "source": item.source,
                "destination": item.destination,
            }
        )
    return merged


def _marker_records_a_copy(payload: dict[str, Any]) -> bool:
    """True when the marker's own audit trail names at least one copied item.

    The marker is the only durable evidence that bytes moved: `_merge_items`
    keeps an earlier attempt's "copied" line for the life of the file, so this
    question can be asked on any later run, not just on the run that copied.
    """
    items = payload.get("items")
    if not isinstance(items, list):
        return False
    return any(
        isinstance(row, dict) and row.get("action") == ACTION_COPIED for row in items
    )


def pending_notice(data_dir: Path | str) -> str:
    """The one-time notice text if it is still owed, else an empty string.

    Two independent conditions withhold it, and both exist because the notice
    makes a factual claim about a person's chat history:

    1. A marker naming DEFERRED items. The notice says the user's conversations
       "were copied"; saying that while the Open WebUI store is still sitting in
       the install tree waiting for a retry would be false (round 6, HIGH-1).
    2. A marker whose merged item list records NO copy at all. Marker existence
       is not evidence that anything moved: a run that defers the one item
       present copies nothing yet still writes a marker, and if the user then
       removes the install folder the next run legitimately completes - deferred
       empty, complete True - with the data root holding nothing but this marker.
       Offering the notice there told the user their conversations "were copied"
       and that "the old copy is still at <install>" when neither had ever been
       true and the store was gone (round 8, HIGH). So the notice is gated on the
       audit trail, not on the file's existence.

    Completion is a separate question from the claim: `already_migrated` may
    legitimately be True on a run like that - the job really is over, there is
    nothing left to copy - while this function stays silent, because there is
    nothing truthful to say.
    """
    payload = _read_marker(data_dir)
    if not payload:
        return ""
    if payload.get("deferred") or payload.get("notice_shown"):
        return ""
    if not _marker_records_a_copy(payload):
        return ""
    return str(payload.get("notice") or "")


def mark_notice_shown(data_dir: Path | str) -> None:
    """Record that the user has now seen the notice, so it shows exactly once."""
    marker = marker_path(data_dir)
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return
        payload["notice_shown"] = True
        marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError):
        # Failing to flip the flag means the notice appears again next start.
        # Repeating a true statement is the harmless direction to fail in.
        pass


def _port_is_open(port: int) -> bool:
    """True when something is listening on loopback:port (Open WebUI, here).

    A connect attempt is the only honest local answer to "is the service up?"
    that does not depend on LOCITIZE having started it itself - the user may have
    launched Open WebUI by hand.

    IT CANNOT TELL OPEN WEBUI FROM ANY OTHER PROCESS ON THAT PORT, and that is a
    deliberate choice rather than an oversight (review round 6, HIGH-1). The two
    ways to be wrong are not comparable:

      * false positive (a dev server, an unrelated app) - the store is deferred
        and re-attempted on the next launch. Nothing is copied, nothing is
        deleted, the original is untouched, and the launcher says so out loud
        every run until it succeeds.
      * false negative (Open WebUI is up but unreachable on this port) - a HOT
        COPY of a live sqlite database, i.e. a corrupt copy of someone's entire
        chat history, presented to them as their migrated data.

    So the guard is kept maximally sensitive on purpose. Probing the HTTP health
    route instead would identify the service more precisely but would still not
    be conclusive, and it would trade an unmistakable failure direction for a
    subtler one. The residual risk is that a process permanently bound to
    `ports.openwebui` leaves the store un-migrated indefinitely; that is why the
    deferral is reported on every launch rather than kept silent.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", int(port))) == 0
