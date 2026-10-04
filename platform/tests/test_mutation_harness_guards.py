"""The mutation harness's own two safety guards, tested like product code.

WHY THIS FILE EXISTS
--------------------
`scripts/mutation_harness.py` rewrites real source files. Two of its guards are
therefore data-integrity mechanisms, not conveniences, and both failed silently
in review round 7:

  THE LOCK      Two harnesses on one tree is corruption, not noise. The second
                one snapshots the first one's MUTATED bytes as its "original"
                and writes them back permanently in its finally block, shipping
                a disabled safety rule that no test can see. A reviewer came one
                command away from doing exactly that on 2026-08-19.

  THE RESIDUE   A killed harness leaves a mutation applied. `--verify-anchors`
  CHECK         exists to find that, but it asked "is the OLD text still there?"
                first - which is always true for an append-style mutation - so
                the five additive mutations could sit applied while it reported
                the tree clean. The 18:54 run crashed mid-write to
                gui_controller.py, which is exactly that path.

Nothing here touches the real tree: the lock is exercised in tmp_path and the
residue check is pointed at a synthetic mutation over a temporary file.

Keyword: mutation_harness_guards
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PLATFORM_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLATFORM_DIR / "scripts"))

import mutation_harness  # noqa: E402

# --------------------------------------------------------------------------- #
# The exclusive lock
# --------------------------------------------------------------------------- #


def test_mutation_harness_guards_a_second_harness_refuses_to_start(tmp_path):
    """A live lock is respected, and the refusal names who holds it."""
    lock = tmp_path / "harness.lock"

    token = mutation_harness.acquire_lock(lock)
    assert lock.is_file()
    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert payload["token"] == token
    # This process is genuinely alive, so the second attempt cannot break it.
    assert payload["pid"] > 0

    with pytest.raises(mutation_harness.HarnessLocked) as refused:
        mutation_harness.acquire_lock(lock)
    message = str(refused.value)
    assert str(payload["pid"]) in message, message
    assert "harness" in message.lower()

    mutation_harness.release_lock(token, lock)
    assert not lock.exists(), "the lock outlived the run that took it"


def test_mutation_harness_guards_a_crashed_run_does_not_block_the_tree(tmp_path):
    """A lock whose owning process is gone is broken, loudly, and re-taken.

    Stale-safety is the other half of the guard: a harness killed by a closed
    session must not leave a lockfile that refuses every future run until a human
    notices and deletes it by hand.
    """
    lock = tmp_path / "harness.lock"
    # A pid that cannot be running: pid 0 is never a user process, so this is a
    # dead owner recorded on THIS host, which is what makes it breakable.
    dead = {
        "pid": 999999999,
        "host": mutation_harness.socket.gethostname(),
        "token": "someone-elses-token",
        "started": 0.0,
        "started_at": "long ago",
    }
    lock.write_text(json.dumps(dead), encoding="utf-8")

    token = mutation_harness.acquire_lock(lock)
    assert token != "someone-elses-token"
    assert json.loads(lock.read_text(encoding="utf-8"))["token"] == token


def test_mutation_harness_guards_release_never_deletes_someone_elses_lock(tmp_path):
    """A run whose lock was broken open must not delete its successor's."""
    lock = tmp_path / "harness.lock"
    mine = mutation_harness.acquire_lock(lock)
    lock.write_text(
        json.dumps({"pid": 1, "host": "another-machine", "token": "theirs"}),
        encoding="utf-8",
    )

    mutation_harness.release_lock(mine, lock)

    assert lock.is_file(), "released a lock this run no longer owned"
    assert json.loads(lock.read_text(encoding="utf-8"))["token"] == "theirs"


def test_mutation_harness_guards_main_writes_nothing_while_a_lock_is_held(
    tmp_path, monkeypatch, capsys
):
    """The guard where it matters: `main` mutates nothing when the tree is taken.

    `run_mutations` is the only thing in this script that writes to a source
    file, so proving it is never entered proves no byte was written. It is
    replaced here by a recorder rather than stubbed out silently, so the test can
    also prove the LOCK PATH works in the ordinary case - a guard that refused
    everything would pass a one-sided test.
    """
    lock = tmp_path / "harness.lock"
    monkeypatch.setattr(mutation_harness, "LOCK_PATH", lock)
    entered: list[str] = []
    held: list[bool] = []

    # 1. Nobody holds the tree: the run proceeds, and the lock is held while it
    #    does and released afterwards.
    monkeypatch.setattr(
        mutation_harness,
        "run_mutations",
        lambda selected: (held.append(lock.is_file()), entered.append("ran"), 0)[-1],
    )
    assert mutation_harness.main(["D8"]) == 0
    assert entered == ["ran"] and held == [True]
    assert not lock.exists()

    # 2. Another harness holds it: refuse, exit 3, and do not enter the writer.
    foreign = mutation_harness.acquire_lock(lock)
    assert mutation_harness.main(["D8"]) == 3
    assert entered == ["ran"], "a second harness started mutating the tree anyway"
    printed = capsys.readouterr().out
    assert "REFUSED" in printed, printed
    assert lock.is_file(), "the refused run deleted the live lock on its way out"
    mutation_harness.release_lock(foreign, lock)


# --------------------------------------------------------------------------- #
# --verify-anchors: leftover ADDITIVE mutations
# --------------------------------------------------------------------------- #


def _synthetic_tree(tmp_path: Path, monkeypatch, text: str, mutation) -> None:
    """Point the harness at one temporary file holding `text`."""
    (tmp_path / mutation.relpath).write_text(text, encoding="utf-8")
    monkeypatch.setattr(mutation_harness, "PLATFORM_DIR", tmp_path)
    monkeypatch.setattr(mutation_harness, "MUTATIONS", [mutation])
    monkeypatch.setattr(mutation_harness, "LOCK_PATH", tmp_path / "no-such.lock")


ADDITIVE = mutation_harness.Mutation(
    "T-ADD",
    "an append-style mutation: the replacement KEEPS the anchor and adds a line",
    "probe.py",
    "    guard()",
    "    guard()\n    disable_the_guard()",
)


def test_mutation_harness_guards_verify_anchors_sees_a_leftover_additive_mutation(
    tmp_path, monkeypatch, capsys
):
    """The exact shape a crashed run leaves behind for 5 of the 91 mutations."""
    _synthetic_tree(
        tmp_path,
        monkeypatch,
        "def f():\n    guard()\n    disable_the_guard()\n",
        ADDITIVE,
    )

    result = mutation_harness.verify_anchors()

    printed = capsys.readouterr().out
    assert result == 1, "a mutated tree was reported clean"
    assert "RESIDUE" in printed and "T-ADD" in printed, printed


def test_mutation_harness_guards_verify_anchors_stays_quiet_on_a_clean_tree(
    tmp_path, monkeypatch, capsys
):
    """The other half: the same additive mutation, NOT applied, is silent.

    Without this, "report residue" could be satisfied by a check that reports
    residue always - which would be the same blindness in the opposite
    direction, and would train everyone to ignore the output.
    """
    _synthetic_tree(tmp_path, monkeypatch, "def f():\n    guard()\n", ADDITIVE)

    result = mutation_harness.verify_anchors()

    printed = capsys.readouterr().out
    assert result == 0, printed
    assert "RESIDUE" not in printed and "STALE" not in printed, printed


def test_mutation_harness_guards_verify_anchors_still_reports_stale_and_replaced(
    tmp_path, monkeypatch, capsys
):
    """The two verdicts round 8 must not have broken while fixing the third."""
    replacing = mutation_harness.Mutation(
        "T-REP",
        "an ordinary replacement mutation",
        "probe.py",
        "    strict = True",
        "    strict = False",
    )

    # Applied: the anchor is gone and the replacement is present.
    _synthetic_tree(tmp_path, monkeypatch, "def f():\n    strict = False\n", replacing)
    assert mutation_harness.verify_anchors() == 1
    assert "RESIDUE" in capsys.readouterr().out

    # Reworded: neither text is present, so the mutation now tests nothing.
    _synthetic_tree(tmp_path, monkeypatch, "def f():\n    careful = True\n", replacing)
    assert mutation_harness.verify_anchors() == 1
    assert "STALE?" in capsys.readouterr().out


def test_mutation_harness_guards_every_shipped_anchor_is_classified_honestly():
    """The real mutation table, read (never applied): is each one decidable?

    An additive mutation is only detectable because `new` contains `old`. This
    asserts the classifier agrees with the table as shipped, so that adding a
    mutation whose shape the residue check cannot see is a test failure here
    rather than a silent hole discovered by the next crashed run.
    """
    additive = [m.name for m in mutation_harness.MUTATIONS
                if mutation_harness._is_additive(m)]
    # Measured, not assumed: these are the append-style entries in the table.
    assert set(additive) == {"M0a", "M21", "W2", "R5", "E1"}, additive
    for mutation in mutation_harness.MUTATIONS:
        assert mutation.old != mutation.new, mutation.name
        text = (PLATFORM_DIR / mutation.relpath).read_text(encoding="utf-8")
        assert mutation_harness._contains(text, mutation.old), (
            f"{mutation.name}: its anchor no longer matches {mutation.relpath}, so "
            "it tests nothing"
        )
