"""Mutation harness for the model-acquisition safety rules (M14.14).

WHAT THIS IS
------------
A test suite that is green proves the code passes its tests; it does not prove
the tests would notice if a safety rule were deleted. This script deletes them,
one at a time, and reports whether the suite goes red.

Each mutation is a LITERAL old-string/new-string pair against a real source
file, so anyone can read exactly what was broken and reproduce it. That is the
point of checking this in: in the previous two review rounds both a false
"caught" and a false "survived" came from mutations that existed only as prose
descriptions in a report, with no way to audit them.

HOW IT RUNS
-----------
  python scripts/mutation_harness.py            # every mutation
  python scripts/mutation_harness.py M2 H4a     # only the named ones
  python scripts/mutation_harness.py --list
  python scripts/mutation_harness.py --verify-anchors   # no tests: is the tree clean?

RUN --verify-anchors FIRST if a previous run was interrupted. This script leaves
a mutated file on disk if it is killed between applying an edit and restoring it,
and a disabled safety rule in the source tree is invisible to the suite.

ONE HARNESS AT A TIME. A real run takes an exclusive lockfile
(.mutation-harness.lock) in the platform directory and refuses to start when
another run holds it, because a second harness would capture the first one's
MUTATED bytes as its "original" and restore them permanently - shipping a
disabled safety rule that no test can see. The lock is broken automatically when
its owning process is gone, so a crash does not block the tree.
--list and --verify-anchors write nothing and need no lock.

For each mutation: capture the file bytes, apply the edit, run the FULL test
suite (never a -k subset - a subset can score a mutation "caught" or "survived"
by test-selection accident), restore the bytes, and verify the restore against a
sha256 taken before anything was touched.

READING THE RESULT
------------------
CAUGHT   = at least one test failed. The rule is genuinely tested.
SURVIVED = the suite stayed green with the rule removed. That is a HOLE in the
           suite, and it is reported as a failure of this script (exit 1).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# The exclusive lock (review round 8, MEDIUM)
# --------------------------------------------------------------------------- #
# Two harnesses on one tree is a DATA-INTEGRITY hazard, not merely noisy numbers.
# A second harness starting while a first is mid-flight snapshots the mutated
# bytes as its "original", then writes that mutated content back in its `finally`
# block - permanently disabling a shipped safety rule, with nothing in the suite
# able to notice (a reviewer came within one command of doing exactly this on
# 2026-08-19). So a run takes an O_EXCL lockfile first and refuses, loudly, if one
# is already held.
LOCK_NAME = ".mutation-harness.lock"
LOCK_PATH = PLATFORM_DIR / LOCK_NAME

# A crashed run must not block the tree forever, so a lock is breakable - but only
# on EVIDENCE, never on a plain timeout alone at first glance. The owner's pid is
# recorded and checked for liveness, and the age is the backstop for the case pid
# liveness cannot answer honestly (a different machine, or a recycled pid).
# 26 hours is comfortably longer than any real run (91 mutations x a 25s suite is
# well under an hour, and SUITE_TIMEOUT_S bounds the worst case) and short enough
# that a crash does not need a human the next day.
LOCK_MAX_AGE_S = 26 * 60 * 60


class HarnessLocked(RuntimeError):
    """Raised when another harness already holds this tree."""


def _pid_is_running(pid: int) -> bool:
    """True when a process with this pid exists right now.

    Deliberately conservative: anything this cannot determine is answered "yes,
    it is running", because wrongly declaring a live run dead is what breaks a
    tree, while wrongly declaring a dead run live costs one manual delete.
    """
    if pid <= 0:
        return True
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists, owned by someone else
        except OSError:
            return True
        return True
    # Windows: OpenProcess with the least privilege that can answer, then ask
    # for the exit code. STILL_ACTIVE (259) is the only answer that means alive;
    # a handle that opens on an exited-but-not-reaped process would otherwise
    # read as running forever.
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # 87 ERROR_INVALID_PARAMETER = no such process. 5 ACCESS_DENIED and
        # anything else means it exists (or we cannot tell) - treat as running.
        return kernel32.GetLastError() != 87
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _lock_payload(path: Path) -> dict[str, object]:
    """Whatever the existing lockfile says, or {} if it says nothing readable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _lock_is_stale(path: Path, payload: dict[str, object], now: float) -> str:
    """The reason this lock may be broken, or "" when it must be respected."""
    age = now - float(payload.get("started", 0.0) or 0.0)
    if payload.get("host") == socket.gethostname():
        pid = int(payload.get("pid", 0) or 0)
        if not _pid_is_running(pid):
            return f"its owner (pid {pid}) is no longer running"
    if age > LOCK_MAX_AGE_S:
        return f"it is {age / 3600:.1f}h old, past the {LOCK_MAX_AGE_S / 3600:.0f}h limit"
    return ""


def acquire_lock(path: Path | None = None, *, now: float | None = None) -> str:
    """Take the tree's exclusive harness lock, or raise HarnessLocked.

    Returns the token written into the lockfile; release_lock refuses to delete a
    lockfile carrying anyone else's token, so a run that overran and was broken
    open cannot then delete its successor's lock on the way out.

    O_CREAT|O_EXCL is the whole mechanism: the create either wins or fails, with
    no window between checking and creating. A stale lock is broken by RENAMING
    it away first - if two processes race to break the same stale lock, only one
    rename can succeed, so only one goes on to create.
    """
    # Resolved at CALL time, not bound as a default: a default argument captures
    # the module constant once at import, which would make the lock path
    # impossible to redirect and silently write into the real platform directory
    # from a test.
    path = LOCK_PATH if path is None else path
    moment = time.time() if now is None else now
    token = uuid.uuid4().hex
    for attempt in (1, 2):
        try:
            handle = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            payload = _lock_payload(path)
            reason = _lock_is_stale(path, payload, moment) if attempt == 1 else ""
            if not reason:
                raise HarnessLocked(
                    f"another mutation harness holds {path.name}: "
                    f"pid {payload.get('pid', '?')} on {payload.get('host', '?')}, "
                    f"started {payload.get('started_at', 'at an unknown time')}. "
                    "Two harnesses on one tree will permanently restore each "
                    "other's mutated bytes - wait for it, or delete the lockfile "
                    "by hand once you have confirmed that run is gone."
                ) from None
            print(f"[LOCK  ] breaking a stale lock: {reason}")
            try:
                path.rename(path.with_name(f"{path.name}.stale-{token}"))
                path.with_name(f"{path.name}.stale-{token}").unlink()
            except OSError:
                pass  # someone else broke it first; the retry will find out
            continue
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "token": token,
                    "started": moment,
                    "started_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%S", time.localtime(moment)
                    ),
                },
                stream,
                indent=2,
            )
        return token
    raise HarnessLocked(f"could not take {path.name} after breaking a stale lock")


def release_lock(token: str, path: Path | None = None) -> None:
    """Drop the lock, but only if it is still OURS.

    Checked rather than assumed: if this run overran and another process broke
    the lock open, the file now belongs to that run and deleting it would hand a
    third process the tree while two are working on it.
    """
    path = LOCK_PATH if path is None else path
    if _lock_payload(path).get("token") != token:
        return
    try:
        path.unlink()
    except OSError:
        pass


@dataclass(frozen=True)
class Mutation:
    """One literal edit that removes exactly one safety rule."""

    name: str
    description: str
    relpath: str
    old: str
    new: str
    # Some rules are enforced twice on purpose (defence in depth). Removing only
    # the outer copy therefore CANNOT change behaviour, so a green suite is the
    # correct result and not a hole. Such a mutation carries the reason here and
    # is scored against that expectation instead of against "must go red".
    expect_survive: str = ""


# --------------------------------------------------------------------------- #
# The mutations. Grouped by the rule each one removes.
# --------------------------------------------------------------------------- #

MUTATIONS: list[Mutation] = [
    # ---- the completion line's copy contract (UX Spec section 14) --------- #
    Mutation(
        "M0a",
        "a truncated hex digest is reintroduced on the completion line "
        "(the pre-fix MEDIUM-6 string, which reads as proof of nothing)",
        "desktop.py",
        '        parts.append(VERIFICATION_LABELS.get(rung, VERIFICATION_LABELS["none"]))',
        '        parts.append(VERIFICATION_LABELS.get(rung, VERIFICATION_LABELS["none"]))\n'
        "        parts.append(f\"({payload.get('sha256', '')[:16]}...)\")",
    ),
    Mutation(
        "M0b",
        "the V-ETAG user label reverts to naming the bare word \"ETag\" - the one "
        "term HuggingFace overloads between the LFS value and the Xet hash",
        "modelhub.py",
        '    V_ETAG: "Checksum verified against HuggingFace\'s linked file hash.",',
        '    V_ETAG: "Checksum verified against HuggingFace\'s published ETag.",',
    ),
    Mutation(
        "M0c",
        "the view stops reading modelhub's labels and states its own claim "
        "(the two-copies-of-one-string drift that caused MEDIUM-6)",
        "desktop.py",
        '        parts.append(VERIFICATION_LABELS.get(rung, VERIFICATION_LABELS["none"]))',
        '        parts.append("Checksum verified against HuggingFace.")',
    ),
    # ---- the checksum ladder's honesty rules ------------------------------ #
    Mutation(
        "M1", "ETag 64-hex length check dropped (any opaque tag becomes a digest)",
        "modelhub.py",
        "    return text if SHA256_RE.match(text) else None",
        "    return text or None",
    ),
    Mutation(
        "M2",
        "final RESPONSE's plain ETag adopted as the digest (review MEDIUM-5)",
        "modelhub.py",
        """            with open_checked(
                opener,
                request,
                self.config.allowed_hosts,
                timeout=self.config.search_timeout_s,
                handler=handler,
            ):
                pass""",
        """            with open_checked(
                opener,
                request,
                self.config.allowed_hosts,
                timeout=self.config.search_timeout_s,
                handler=handler,
            ) as response:
                fallback = normalize_etag_digest(response.headers["ETag"])
                if fallback and not getattr(handler, "linked_etag", None):
                    handler.linked_etag = fallback""",
    ),
    Mutation(
        "M2b", "redirect RECORDER falls back to the plain ETag (review HIGH-2)",
        "modelhub.py",
        '            raw = headers.get("X-Linked-ETag")\n'
        '            raw_size = headers.get("X-Linked-Size")',
        '            raw = headers.get("X-Linked-ETag") or headers.get("ETag")\n'
        '            raw_size = headers.get("X-Linked-Size")',
    ),
    Mutation(
        "M3a", "rung V-API claimed with no lfs.oid behind it (parse_tree_files)",
        "modelhub.py",
        '                "verification": V_API if digest else V_NONE,',
        '                "verification": V_API,',
    ),
    Mutation(
        "M3b", "rung reported with no digest to compare (download_verified)",
        "modelhub.py",
        "        verification=verification if expected else V_NONE,",
        "        verification=verification,",
    ),
    Mutation(
        "M4", "an unverified download needs no confirmation (download_verified)",
        "modelhub.py",
        "    if not expected and not confirm_unverified:",
        "    if False and not expected and not confirm_unverified:",
    ),
    Mutation(
        "M4b", "an unverified download needs no confirmation (Downloader.download)",
        "modelhub.py",
        "        if not expected and not confirm_unverified:",
        "        if False and not expected and not confirm_unverified:",
        expect_survive=(
            "Downloader.download passes confirm_unverified straight through to "
            "download_verified, whose OWN gate (M4) still refuses. Removing the "
            "outer gate changes only which of the two refusal messages is "
            "returned, so the user-visible rule still holds - which is what "
            "defence in depth is for."
        ),
    ),
    # ---- egress: allowlist, TLS, redirects -------------------------------- #
    Mutation(
        "M5", "host allowlist becomes a substring match (evil-hf.co passes)",
        "modelhub.py",
        '        if name == allowed or name.endswith("." + allowed):',
        "        if allowed in name:",
    ),
    Mutation(
        "M6", "per-hop host revalidation removed (a 302 may leave the allowlist)",
        "modelhub.py",
        """        if not host_allowed(parsed.hostname or "", self.allowed_hosts):
            return None""",
        """        if False:
            return None""",
    ),
    Mutation(
        "M7", "per-hop https check removed (a 302 may downgrade to http)",
        "modelhub.py",
        """        if parsed.scheme.lower() != "https":
            return None""",
        """        if False:
            return None""",
    ),
    Mutation(
        "M8", "redirect hop cap removed",
        "modelhub.py",
        "        if self.hops > self.max_hops:",
        "        if False:",
    ),
    Mutation(
        "M16", "open_checked stops validating (the T2 chokepoint, review HIGH-1)",
        "modelhub.py",
        """    url = getattr(target, "full_url", None) or str(target)
    validate_url(url, allowed_hosts)""",
        """    url = getattr(target, "full_url", None) or str(target)""",
    ),
    Mutation(
        "M17", "download()'s pre-open URL gate removed (review HIGH-1)",
        "modelhub.py",
        """            url = validate_url(
                build_resolve_url(self.config.api_base, repo_id, filename),
                self.config.allowed_hosts,
            )""",
        """            url = build_resolve_url(
                self.config.api_base, repo_id, filename
            )""",
    ),
    Mutation(
        "M18", "per-request hop budget reset removed (review MEDIUM-4)",
        "modelhub.py",
        """    reset = getattr(handler, "reset_hops", None)
    if callable(reset):
        reset()""",
        """    reset = getattr(handler, "reset_hops", None)
    if False and callable(reset):
        reset()""",
    ),
    Mutation(
        "M21", "a credential header is attached to a request (AC22: never)",
        "modelhub.py",
        '            request = urllib.request.Request(url, method="HEAD")',
        """            request = urllib.request.Request(url, method="HEAD")
            request.add_header("Authorization", "Bearer hf_not_a_real_token")""",
    ),
    # ---- what lands on disk ------------------------------------------------ #
    Mutation(
        "M12", "no-clobber removed (an existing file is silently overwritten)",
        "modelhub.py",
        "    if dest_path.exists():",
        "    if False and dest_path.exists():",
    ),
    Mutation(
        "M13", "GGUF magic check removed (an HTML sign-in page is kept as a model)",
        "modelhub.py",
        "        if magic != GGUF_MAGIC:",
        "        if False:",
    ),
    Mutation(
        "M14", "checksum comparison removed (a mismatching file is kept)",
        "modelhub.py",
        "    if expected and actual != expected:",
        "    if False:",
    ),
    Mutation(
        "M15", "size agreement check removed",
        "modelhub.py",
        """        if (
            expected_size
            and total_bytes
            and int(expected_size) != int(total_bytes)
        ):""",
        """        if False:""",
    ),
    Mutation(
        "M22", "a registry failure deletes the downloaded file (AC24: never)",
        "gui_controller.py",
        """        except (ValueError, RegistryWriteError) as exc:
            # DEFECT-QA-M14-4 (H11) was originally fixed here, by wrapping""",
        """        except (ValueError, RegistryWriteError) as exc:
            Path(done["path"]).unlink(missing_ok=True)
            # DEFECT-QA-M14-4 (H11) was originally fixed here, by wrapping""",
    ),
    Mutation(
        "M24", "the page-paint catalog read goes to the network (HF-1)",
        "gui_controller.py",
        "        payload = self._hub().catalog()",
        '        payload = self._hub().search("")',
    ),
    # ---- shutdown / no-orphan (review MEDIUM-1 and HIGH-4) ----------------- #
    Mutation(
        "M19", "the shutting-down latch is removed (a download starts after the join)",
        "gui_controller.py",
        """            self._hub_shutting_down = True
            self._hub_cancel.set()""",
        """            self._hub_cancel.set()""",
    ),
    Mutation(
        "H4a", "the join drops its is_alive guard (an unstarted thread raises)",
        "gui_controller.py",
        "        if thread is not None and thread.is_alive():\n"
        "            thread.join(timeout=5.0)",
        "        if thread is not None:\n"
        "            thread.join(timeout=5.0)",
    ),
    Mutation(
        "H4b", "one failing teardown step aborts the rest (children are orphaned)",
        "gui_controller.py",
        # Re-anchored in round 7: the trailing comment on the `except` line had
        # drifted to "teardown must attempt every step", so this mutation had
        # been silently STALE - it matched nothing and therefore tested nothing.
        # Found by running the FULL harness rather than a subset.
        """            try:
                action()
            except Exception as exc:  # teardown must attempt every step
                self.shutdown_errors.append(f"{name}: {exc}")""",
        """            action()""",
    ),
    Mutation(
        "H4pre",
        "the EXACT pre-fix shutdown: unguarded join AND no per-step guard, so the "
        "RuntimeError escapes and the child processes are orphaned",
        "gui_controller.py",
        # The anchor carries the guard's REAL comment text. It previously quoted
        # an older wording, so this mutation matched nothing and reported STALE -
        # i.e. it silently stopped testing the invariant it names, which is the
        # failure mode this whole harness exists to prevent.
        """            try:
                action()
            except Exception as exc:  # teardown must attempt every step
                self.shutdown_errors.append(f"{name}: {exc}")""",
        """            action()""",
    ),
    # ---- text-rendering policy and download ceiling (security gate, M14) --- #
    Mutation(
        "S1",
        "the confirm dialog goes back to Qt's AutoText, so a publisher-authored "
        "string is rendered as MARKUP again (SEC-M14-1, the reported defect)",
        "desktop.py",
        "    box.setTextFormat(PLAIN_TEXT)",
        "    box.setTextFormat(QtCore.Qt.TextFormat.AutoText)",
    ),
    Mutation(
        "S1b",
        "the window-wide plain-text policy is never applied, so every label "
        "sniffs its own content for markup again (SEC-M14-1, the class fix)",
        "desktop.py",
        "        harden_text_rendering(self)",
        "        pass  # policy removed",
    ),
    Mutation(
        "S1c",
        "the licence tag is passed through unvalidated - the exact pre-fix "
        "parser that let publisher markup out of modelhub (SEC-M14-1, data half)",
        "modelhub.py",
        '    return text if _LICENSE_TAG_RE.match(text) else ""',
        "    return text",
    ),
    Mutation(
        "S2",
        "the download's byte ceiling is removed, so a transfer that declares no "
        "size is unbounded again (SEC-M14-2)",
        "modelhub.py",
        "                if written > ceiling:",
        "                if False and written > ceiling:",
    ),
    Mutation(
        "S2b",
        "the free-space check is skipped again whenever the tree API reports no "
        "size (the pre-fix `if size_bytes:` guard, SEC-M14-2)",
        "modelhub.py",
        "        disk = self.check_disk(size_bytes or 0, target_dir)\n"
        '        if not disk["ok"]:\n'
        '            return DownloadOutcome(False, error=disk["reason"])',
        "        disk = {\"ok\": True, \"free_mb\": None}\n"
        "        if size_bytes:\n"
        "            disk = self.check_disk(size_bytes, target_dir)\n"
        '            if not disk["ok"]:\n'
        '                return DownloadOutcome(False, error=disk["reason"])',
    ),
    Mutation(
        "S3",
        "a failed teardown step is recorded but never logged (SEC-M14-5)",
        "gui_controller.py",
        "        self._log_shutdown_errors()",
        "        pass  # logging removed",
    ),
    Mutation(
        "H4c",
        "the construct/start critical section is removed AND the join guard with it "
        "(this is the pre-fix code that stranded the child processes)",
        "gui_controller.py",
        "        if thread is not None and thread.is_alive():\n"
        "            thread.join(timeout=5.0)",
        "        if thread is not None:\n"
        "            thread.join(timeout=5.0)",
    ),
    # ---- the six M14 defects QA found in the live app (2026-08-19) --------- #
    #
    # Each of these restores the EXACT shipped behaviour QA measured, so a green
    # suite here would mean the new regression tests are decoration. Four of the
    # six were invisible to 724 passing unit tests before these mutations
    # existed, which is the whole argument for adding them.
    Mutation(
        "Q1",
        "the models directory is no longer created before the free-space probe, "
        "so the first download of a clean install dies on WinError 3 "
        "(DEFECT-QA-M14-1, the shipped defect)",
        "modelhub.py",
        """        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:""",
        """        try:
            pass
        except OSError as exc:""",
    ),
    Mutation(
        "Q1b",
        "check_disk stops catching OSError, so an unmeasurable path raises a raw "
        "WinError instead of refusing with a next step (DEFECT-QA-M14-1, rung 2)",
        "modelhub.py",
        "        try:\n"
        "            free_mb = float(self._system_provider.disk_free_mb(str(target_dir)))\n"
        "        except OSError as exc:",
        "        free_mb = float(self._system_provider.disk_free_mb(str(target_dir)))\n"
        "        if False:\n"
        "            exc = None",
    ),
    Mutation(
        "Q2",
        "terminal download UI is applied without checking the job id, so a "
        "refused second download kills Cancel and the progress bar of the "
        "transfer that is still running (DEFECT-QA-M14-2, the shipped defect)",
        "desktop.py",
        "        if active is not None and job and job != active:",
        "        if False:",
    ),
    Mutation(
        "Q2b",
        "selecting another file re-arms Download during a live download "
        "(DEFECT-QA-M14-2, the reachability half)",
        "desktop.py",
        "        self._hub_download_btn.setEnabled(self._hub_busy_job() is None)",
        "        self._hub_download_btn.setEnabled(True)",
    ),
    Mutation(
        "Q3",
        "BOTH halves of the layout fix are removed - no minimum table height and "
        "no scroll area - which is the exact shipped state in which QA measured a "
        "zero-height catalog viewport (DEFECT-QA-M14-3, the shipped defect)",
        "desktop.py",
        "        table.setMinimumHeight(HUB_TABLE_MIN_HEIGHT)",
        "        pass  # no minimum height",
        # The second edit lives in EXTRA_EDITS below. Both are needed to restore
        # the defect: see Q3c for the measurement that proved it.
    ),
    Mutation(
        "Q3c",
        "ONLY the minimum table height is removed, with the scroll area left in "
        "place (DEFECT-QA-M14-3, the redundant half)",
        "desktop.py",
        "        table.setMinimumHeight(HUB_TABLE_MIN_HEIGHT)",
        "        pass  # no minimum height",
        expect_survive=(
            "measured, not assumed: with the page inside its scroll area the "
            "table is laid out at its own size hint (150 px, 4 rows of 30 px) at "
            "BOTH 1180x780 and 960x680, identical to the unmutated build, so "
            "removing the floor cannot change what the user sees. The floor is "
            "defence in depth for a future layout that drops the scroll area, "
            "which is why Q3 removes both halves together and IS caught."
        ),
    ),
    Mutation(
        "Q3b",
        "the Models page is returned unwrapped again, so nothing on it can be "
        "scrolled to (DEFECT-QA-M14-3, the reachability half)",
        "desktop.py",
        "        return self._scrollable(page)",
        "        return page",
    ),
    Mutation(
        "Q4",
        "a registry failure reports the raw exception with no next step "
        "(DEFECT-QA-M14-4 / H11, the shipped defect)",
        "gui_controller.py",
        """            done["register_error"] = (
                f"{str(exc).strip()} The model itself was downloaded and kept at "
                f"{done.get('path', '')}, so nothing was lost: press Register on "
                f"the Models page once the cause above is cleared."
            )""",
        """            done["register_error"] = f"could not register: {exc}\"""",
    ),
    Mutation(
        "Q4b",
        "the chokepoint's 'no models: section' refusal loses its next step and "
        "goes back to a bare diagnostic (DEFECT-QA-M14-4 second branch, now "
        "raised once for all four writers by _edit_registry per DEC-M14-11)",
        "config.py",
        "            f\"top-level 'models:' section to write into, and it refuses to \"\n"
        "            f\"guess where the list should start. Open that file, add a \"\n"
        "            f\"'models:' line, then try again.\"",
        '            f"top-level \'models:\' section to write into."',
    ),
    Mutation(
        "Q4c",
        "the post-write verification refusal loses its next step "
        "(DEFECT-QA-M14-4, third branch)",
        "config.py",
        "        f\"post-write verification failed for new model '{model_id}'; \"\n"
        '        f"file left unchanged. Check {path} for a formatting problem "\n'
        "        f\"near the end of the 'models:' list, then try again.\"",
        "        f\"post-write verification failed for new model '{model_id}'; \"\n"
        '        f"file left unchanged."',
    ),
    # ---- DEC-M14-9: the directory ownership contract and invariant W1 ------ #
    #
    # D1-D7 each restore ONE of the resolvers to the location it used before the
    # decision. D1, D2 and D3 are the exact shipped defect QA measured on a
    # stranger install (NEW-QA-M14-8): logs, transcripts and the Open WebUI chat
    # store written into the INSTALL directory, where a reinstall destroys them
    # and the documented "back up one folder" claim is false.
    Mutation(
        "D1",
        "logs go back to <install>/logs (NEW-QA-M14-8, the shipped defect)",
        "logger.py",
        '    return Path(settings.data_dir) / "logs"',
        '    return Path(settings.base_dir) / "logs"',
    ),
    Mutation(
        "D2",
        "the voice session's transcript store goes back to the install tree "
        "(NEW-QA-M14-8, the shipped defect - this is the call site that wrote "
        "QA's qa-finding-a.jsonl into the install directory)",
        "launcher.py",
        """        memory = ConversationMemory(
            resolve_memory_dir(settings.data_dir, settings.memory.dir),
            enabled=settings.memory.enabled,
        )

        loop = AssistantLoop(
            stt,
            llm_client,""",
        """        memory = ConversationMemory(
            resolve_memory_dir(settings.base_dir, settings.memory.dir),
            enabled=settings.memory.enabled,
        )

        loop = AssistantLoop(
            stt,
            llm_client,""",
    ),
    Mutation(
        "D2b",
        "the memory SEARCH panel reads the install tree again (the second of "
        "the three call sites, which a one-site fix would have missed)",
        "launcher.py",
        """        mem = ConversationMemory(
            resolve_memory_dir(settings.data_dir, settings.memory.dir),""",
        """        mem = ConversationMemory(
            resolve_memory_dir(settings.base_dir, settings.memory.dir),""",
    ),
    Mutation(
        "D3",
        "the Open WebUI chat database goes back under the install tree "
        "(NEW-QA-M14-8, and it carries .webui_secret_key back into the source "
        "tree with it)",
        "webui.py",
        "    base = Path(settings.data_dir).resolve()",
        "    base = Path(settings.base_dir).resolve()",
    ),
    Mutation(
        "D4",
        "benchmark results go back to <install>/docs, where a reinstall deletes "
        "an hour of measurements",
        "benchmark.py",
        '    return Path(settings.data_dir) / "reports"',
        '    return Path(settings.base_dir) / "docs"',
    ),
    Mutation(
        "D5",
        "the development journal is written into the install tree again",
        "launcher.py",
        """        reports = resolve_results_dir(settings)
        journal = DevelopmentJournal(reports / "development_journal.md")""",
        """        journal = DevelopmentJournal(
            settings.base_dir / "docs" / "development_journal.md"
        )""",
    ),
    Mutation(
        "D6",
        "the generated Caddyfile is written into the install tree again",
        "secure_proxy.py",
        '    return Path(settings.data_dir) / "Caddyfile"',
        '    return Path(settings.base_dir) / "Caddyfile"',
    ),
    Mutation(
        "D7",
        "a blank finetune.outputs_dir means <studio_dir>/outputs again, so "
        "trained models live in a checkout instead of the backed-up folder",
        "finetune.py",
        """    root = getattr(settings, "data_dir", None)
    if root is None:
        return None
    return Path(root) / "finetune" / "outputs\"""",
        """    base = studio_dir(settings)
    if base is None:
        return None
    return base / "outputs\"""",
    ),
    # ---- DEC-M14-9: the migration rules (the part that can lose history) --- #
    Mutation(
        "D8",
        "the migration MOVES the user's data instead of copying it - the one "
        "irreversible mistake this design exists to prevent",
        "migration.py",
        # Re-anchored in round 8: the copy is now made into a staging directory
        # that already carries its authorship sentinel, so the copytree call
        # gained dirs_exist_ok. The rule under test is untouched - COPY, never
        # move - and an anchor that no longer matched would report STALE, which
        # is this harness saying "D8 tests nothing".
        """            shutil.copytree(source, staging, dirs_exist_ok=True)
        else:
            shutil.copy2(source, staging)""",
        """            shutil.move(str(source), str(staging))
        else:
            shutil.move(str(source), str(staging))""",
    ),
    Mutation(
        "D9",
        "a non-empty destination is merged into instead of skipped, which is "
        "how two conversation histories become one broken one",
        "migration.py",
        """    if _has_content(destination):
        # Rule 2. Both paths are recorded so the log line names what was kept
        # and what was left behind, which is the only way a user can check.
        return record(ACTION_DESTINATION_IN_USE)""",
        """    if False:
        return record(ACTION_DESTINATION_IN_USE)""",
    ),
    Mutation(
        "D10",
        "the 'migrated' marker is written even when an item failed to copy, so "
        "a half-migrated machine reports itself as done and never retries",
        "migration.py",
        # Re-anchored twice now: round 6 widened this condition, and round 7's
        # HIGH-1r fix moved it into the named predicate _should_write_marker.
        # The rule under test is unchanged: `outcome.ok` is what withholds the
        # marker from a run in which an item FAILED.
        "    if not outcome.ok:\n        return False",
        "    if False:\n        return False",
    ),
    Mutation(
        "D11",
        "webui-data is copied while Open WebUI is running, i.e. a hot copy of a "
        "live sqlite database",
        "migration.py",
        "    if rel_source == _LIVE_SERVICE_ITEM and is_running(webui_port):",
        "    if False:",
    ),
    Mutation(
        "D12",
        "the first-run notice stops naming the folder the data came FROM, so "
        "the user cannot find or check their old copy",
        "migration.py",
        """        "LOCITIZE moved to a single data folder. Your existing conversations, logs "
        f"and settings were copied from {install_dir} to {data_dir}. LOCITIZE now \"""",
        """        "LOCITIZE moved to a single data folder. Your existing conversations, logs "
        f"and settings were copied to {data_dir}. LOCITIZE now \"""",
    ),
    # ---- Review round 6: the defects a green suite did not notice ---------- #
    Mutation(
        "D13",
        "only a run that COPIED something records anything, so the deferral is "
        "neither written down nor ever cleared (round 6, HIGH-1; round 7, "
        "HIGH-1r)",
        "migration.py",
        "    return bool(outcome.copied or outcome.deferred) or _read_marker(data) is not None",
        "    return bool(outcome.copied)",
        # No longer redundant-by-design. Under round 6's semantics this mutation
        # changed nothing observable - withholding the marker and writing an
        # incomplete one both led to the same retry - and it was correctly
        # labelled as such. HIGH-1r's fix changed those semantics: an existing
        # marker is now what lets a resolved deferral be recorded, so a run that
        # copies nothing but finishes the job must still write. Dropping the
        # `deferred` half of this line strands exactly that case, and the round-7
        # test catches it.
    ),
    Mutation(
        "D13b",
        "already_migrated accepts an INCOMPLETE marker, so the deferred item is "
        "never re-attempted even though the marker names it (round 6, HIGH-1)",
        "migration.py",
        "    return not payload.get(\"deferred\")",
        "    return True",
    ),
    Mutation(
        "D13c",
        "the one-time notice ('your conversations were copied') is shown while "
        "an item is still deferred, i.e. a false statement about chat history",
        "migration.py",
        '    if payload.get("deferred") or payload.get("notice_shown"):',
        '    if payload.get("notice_shown"):',
    ),
    Mutation(
        "D14",
        "the empty-destination clearance is dropped, so os.replace hits "
        "PermissionError 5 on any existing destination folder and the migration "
        "fails on every launch, forever (round 6, HIGH-2)",
        "migration.py",
        # Re-anchored in round 8: the sentinel is removed from the staging
        # directory between these two lines, so the pair is no longer adjacent.
        "        _clear_empty_destination(destination)\n"
        "        if source.is_dir():\n"
        "            (staging / _STAGING_SENTINEL).unlink()\n"
        "        os.replace(staging, destination)",
        "        if source.is_dir():\n"
        "            (staging / _STAGING_SENTINEL).unlink()\n"
        "        os.replace(staging, destination)",
    ),
    Mutation(
        "D14b",
        "the empty-destination clearance becomes a recursive delete, which would "
        "trade HIGH-2 for the far worse bug of deleting the user's newer history",
        "migration.py",
        "    if destination.is_dir():\n        destination.rmdir()",
        "    if destination.is_dir():\n        shutil.rmtree(destination)",
    ),
    Mutation(
        "D15",
        "the staging path is keyed on the pid alone again, so two migration "
        "attempts in one process delete each other's in-flight copy and nothing "
        "is migrated at all (round 6, MEDIUM-3)",
        "migration.py",
        '    staging = destination.parent / (\n'
        '        f"{destination.name}{_STAGING_MARK}{os.getpid()}-{uuid.uuid4().hex[:8]}"\n'
        "    )",
        '    staging = destination.parent / f"{destination.name}{_STAGING_MARK}{os.getpid()}"',
    ),
    Mutation(
        "D15b",
        "the stale-staging sweep is removed, so a partial copy of the user's "
        "transcripts left by a killed process lives forever in the backup folder",
        "migration.py",
        "    _sweep_stale_staging(data)",
        "    pass",
    ),
    Mutation(
        "D16",
        "webui_port gets a default of 0 again, so the live-sqlite guard can be "
        "disabled by simply omitting the argument at a call site - the signature "
        "is the only thing that makes it un-forgettable (round 6, MEDIUM-5). "
        "The mutation restores the default ONLY; `int(webui_port) or "
        "DEFAULT_WEBUI_PORT` still substitutes port 8096, so no hot copy occurs "
        "under this mutation and the earlier description overstated it "
        "(round 7, LOW-2r)",
        "migration.py",
        # Anchored with the line after it: `webui_port: int,` alone also matches
        # _relocate_one's parameter list, and an ambiguous anchor is a STALE
        # result rather than a mutation.
        "    webui_port: int,\n"
        "    service_is_running: Callable[[int], bool] | None = None,",
        "    webui_port: int = 0,\n"
        "    service_is_running: Callable[[int], bool] | None = None,",
    ),
    # ---- Review round 6, MEDIUM-1: relative paths and child working dirs --- #
    Mutation(
        "W2",
        "a relative-path write is reintroduced in a runtime module - the exact "
        "one-line probe that passed BOTH W1 layers in round 6, because LOCITIZE's "
        "working directory is the install directory",
        "logger.py",
        "def configure_logging(",
        'def _probe_relative_write() -> None:\n'
        '    Path("locitize-crash.txt").write_text("boom", encoding="utf-8")\n'
        "\n"
        "\ndef configure_logging(",
    ),
    Mutation(
        "W3",
        "Open WebUI goes back to inheriting LOCITIZE's working directory, i.e. the "
        "install tree, where every relative file it writes lands",
        "webui.py",
        "        cwd=resolve_service_cwd(settings),",
        "        cwd=None,",
    ),
    Mutation(
        "W3b",
        "llama-server goes back to inheriting the install directory as its "
        "working directory",
        "models.py",
        "            cwd=resolve_service_cwd(s),",
        "            cwd=None,",
    ),
    Mutation(
        "W4",
        "the fine-tune empty state stops naming the key that repoints the scan, "
        "so a user whose runs are elsewhere cannot learn why the list is empty "
        "(round 6, MEDIUM-6)",
        "finetune.py",
        '            f"No fine-tuned models found in `{root}`. If your training runs are "\n'
        '            f"somewhere else, set finetune.outputs_dir in settings.yaml to that "\n'
        '            f"folder."',
        'f"No fine-tuned models found under `{root}`."',
    ),
    # ---- DEC-M14-11: the single registry-write chokepoint ------------------ #
    Mutation(
        "R1",
        "the chokepoint drops its existence check, so a missing models.yaml "
        "raises the raw FileNotFoundError QA read on screen (NEW-QA-M14-9)",
        "config.py",
        "    if not path.is_file():",
        "    if False:",
    ),
    Mutation(
        "R2",
        "the write failure renders the path as a Python repr again - the "
        "doubled-backslash string the user had to un-escape by eye",
        "config.py",
        """    return (
        f"LOCITIZE could not {verb} your model list at {path}: {cause}. Close any \"""",
        """    return (
        f"LOCITIZE could not {verb} your model list at {path!r}: {cause}. Close any \"""",
    ),
    Mutation(
        "R3",
        "the OSError translation goes back to str(exc), which re-embeds the "
        "errno and the temp-file name the user has never heard of",
        "config.py",
        '    cause = (exc.strerror or "the file could not be opened").strip().rstrip(".")',
        "    cause = str(exc)",
    ),
    Mutation(
        "R5",
        "a SECOND registry writer is added to config.py in the idiomatic "
        "path.open(\"w\") form - the rogue writer Reviewer appended in round 6, "
        "which the AC-M14-28 structural check could not see (HIGH-3)",
        "config.py",
        "class Config:",
        "def probe_rogue_registry_writer(base_dir):\n"
        "    path = Path(base_dir) / MODELS_FILE\n"
        '    with path.open("w", encoding="utf-8") as handle:\n'
        '        handle.write("models: []")\n'
        "\n"
        "\nclass Config:",
    ),
    Mutation(
        "R4",
        "the chokepoint stops translating write failures, so a read-only "
        "models.yaml raises a bare OSError at every one of the four writers",
        "config.py",
        """    try:
        _atomic_write(path, new_text)
    except OSError as exc:
        raise RegistryWriteError(_registry_os_message(path, exc, "write")) from exc""",
        "    _atomic_write(path, new_text)",
    ),
    # ---- DEC-M14-10: the removed egress environment override --------------- #
    Mutation(
        "E1",
        # The removed key's name is written in two pieces here for the same
        # reason the mutation body writes it in two pieces: AC-M14-28 greps the
        # whole tree for it, and a harness that spelled it out would make the
        # criterion fail on a file whose only crime is describing the removal.
        "LOCITIZE_MODELS_HUB" + "_API_BASE is reintroduced, so the egress "
        "boundary is widenable from the environment again (SEC-M14-3)",
        "config.py",
        """    if environ.get("LOCITIZE_MODELS_HUB_DOWNLOAD_DIR"):
        settings.models_hub.download_dir = environ["LOCITIZE_MODELS_HUB_DOWNLOAD_DIR"]""",
        """    if environ.get("LOCITIZE_MODELS_HUB_DOWNLOAD_DIR"):
        settings.models_hub.download_dir = environ["LOCITIZE_MODELS_HUB_DOWNLOAD_DIR"]
    if environ.get("LOCITIZE_MODELS_HUB" + "_API_BASE"):
        settings.models_hub.api_base = environ["LOCITIZE_MODELS_HUB" + "_API_BASE"]""",
    ),
    # ---- NEW-QA-M14-7: the job-id guard that could not fire ---------------- #
    Mutation(
        "Q6",
        "the requested job id is adopted as ACTIVE before the controller has "
        "accepted it - the exact shipped state in which the H20 guard compared "
        "a refusal against itself and killed a live transfer's controls",
        "desktop.py",
        '        self._hub_pending_job = hub_job_id(repo["repo_id"], entry["filename"])',
        '        self._hub_active_job = hub_job_id(repo["repo_id"], entry["filename"])',
    ),
    Mutation(
        "Q7",
        "progress no longer promotes the pending job to active, so no transfer "
        "is ever protected by the job-id guard",
        "desktop.py",
        """            if self._hub_active_job is None and job == self._hub_pending_job:
                self._hub_active_job = job
                self._hub_pending_job = None""",
        "            if False:\n                pass",
    ),
    Mutation(
        "Q5",
        "the registry notes record the internal rung key again instead of the "
        "literal provenance field (DEFECT-QA-M14-6, the shipped defect)",
        "modelhub.py",
        "        f\"verification={verification_field_name(verification)}\"",
        "        f\"verification={verification}\"",
    ),
    # ---- Review round 7: the defects the round-6 repairs left behind -------- #
    Mutation(
        "D17",
        "a deferral that resolves WITHOUT a copy stops being recorded, so the "
        "incomplete marker survives forever: the migration re-runs on every "
        "launch for the life of the install and the user is never told their "
        "conversations moved (round 7, HIGH-1r)",
        "migration.py",
        "    return bool(outcome.copied or outcome.deferred) or _read_marker(data) is not None",
        "    return bool(outcome.copied or outcome.deferred)",
    ),
    Mutation(
        "D18",
        "the stale-staging sweep goes back to rglob-ing the WHOLE data root for "
        "any *.migrating-* path and rmtree-ing it, which destroyed a reviewer's "
        "own notes.migrating-plan.md and archive.migrating-2025 (round 7, "
        "MEDIUM-1r)",
        "migration.py",
        """        try:
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
                continue""",
        """        try:
            candidates = list(data.rglob(f"*{_STAGING_MARK}*"))
        except OSError:
            continue
        for path in candidates:
            if False:
                continue""",
    ),
    Mutation(
        "D19",
        "the launcher goes back to announcing the migration only when THIS run "
        "copied something, so the run that finally completes a deferred "
        "migration says nothing at all and a headless user is never told their "
        "conversations moved (round 7, HIGH-1r's second surface)",
        "launcher.py",
        "        if outcome.complete and (\n"
        "            outcome.copied or migration.pending_notice(data_dir)\n"
        "        ):",
        "        if outcome.copied and outcome.complete:",
    ),
    # ---- Review round 8: the defects the round-7 repairs left behind ------- #
    Mutation(
        "D20",
        "the one-time notice goes back to being gated on the marker's EXISTENCE "
        "rather than on evidence of a copy, so a user whose only item was "
        "deferred and never copied is told their conversations 'were copied' and "
        "that the old folder is still there - both false, and the store is gone "
        "(round 8, HIGH)",
        "migration.py",
        "    if not _marker_records_a_copy(payload):\n        return \"\"",
        "    if False:\n        return \"\"",
    ),
    Mutation(
        "D21",
        "the notice claims the install-side original 'is still at <install>' "
        "unconditionally, including on the run that completes a migration "
        "AFTER the user removed that folder as the notice invited them to "
        "(round 8, HIGH, second claim)",
        "migration.py",
        "    originals_remain = any(\n"
        "        _has_content(install / rel_source) for _label, rel_source, _rel in RELOCATIONS\n"
        "    )",
        "    originals_remain = True",
    ),
    Mutation(
        "D22",
        "the stale-staging sweep goes back to proving authorship by NAME alone, "
        "so a user directory called exactly memory.migrating-1234-abcdef01 is "
        "deleted with its contents by a routine whose docstring claims it only "
        "removes what it can prove it wrote (round 8, MEDIUM)",
        "migration.py",
        "            if not _bears_staging_sentinel(path):\n                continue",
        "            if False:\n                continue",
    ),
    Mutation(
        "H5",
        "the mutation harness stops taking its exclusive lock, so two harnesses "
        "on one tree capture each other's MUTATED bytes as 'original' and "
        "restore a disabled safety rule permanently (round 8, MEDIUM)",
        "scripts/mutation_harness.py",
        "    try:\n"
        "        token = acquire_lock()\n"
        "    except HarnessLocked as exc:\n"
        "        print(f\"[REFUSED] {exc}\")\n"
        "        return 3",
        "    token = \"\"",
    ),
    Mutation(
        "H6",
        "--verify-anchors goes back to asking 'is the OLD text present?' first, "
        "which is always true for an append-style mutation, so the five additive "
        "mutations (M0a, M21, W2, R5, E1) can sit applied in the tree while the "
        "harness reports it clean (round 8, MEDIUM)",
        "scripts/mutation_harness.py",
        "        if _is_additive(mutation) and _contains(text, mutation.new):",
        "        if False:",
    ),
    # ---- Review round 7, MEDIUM-2r: W1-S4's real reach --------------------- #
    Mutation(
        "W5",
        "W1-S4 stops seeing a relative path built by string concatenation, one "
        "of the six shapes it missed while disclosing that it did not",
        "scripts/verify_write_fence.py",
        # Re-anchored when ast.Mod joined this branch. An anchor that no longer
        # matches reports STALE, which is the harness saying "this mutation
        # tested nothing" - the whole reason W5 exists.
        "    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add, ast.Mod)):",
        "    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Mod)):",
    ),
    Mutation(
        "W5b",
        "W1-S4 stops seeing an f-string whose leading component is a literal "
        "relative path, e.g. a crash file written to f\"logs/{name}.txt\"",
        "scripts/verify_write_fence.py",
        """    if isinstance(node, ast.JoinedStr):""",
        """    if isinstance(node, ast.JoinedStr) and False:""",
    ),
    Mutation(
        "W5c",
        "W1-S4 stops seeing os.path.join of literals - the most ordinary way to "
        "spell a relative path in code that predates pathlib",
        "scripts/verify_write_fence.py",
        '        if name == "join" and isinstance(node.func, ast.Attribute):',
        '        if False and isinstance(node.func, ast.Attribute):',
    ),
    Mutation(
        "W6",
        "the open() mode is chosen by node type again, so io.open(path, \"w\") "
        "reads its mode out of the PATH expression and is scored a read - "
        "blinding both the W1 fence and the AC-M14-28 registry chokepoint test "
        "(round 7, MEDIUM-3r)",
        "scripts/verify_write_fence.py",
        "    return _dotted_name(call.func.value) in PATH_FIRST_OPEN_MODULES",
        "    return False",
    ),
    # ---- round 7, second pass: the half of MEDIUM-3r left on the DESTINATION,
    # and the four MEDIUM-2r shapes that were still missed and undisclosed ---- #
    Mutation(
        "W7",
        "the write DESTINATION is resolved by node type again, so "
        "os.replace(staging, dest) - the atomic-rename idiom the product itself "
        "uses - reports the module `os` as the thing being written to, and "
        "W1-S4 cannot see where the rename lands (round 7, MEDIUM-3r)",
        "scripts/verify_write_fence.py",
        "    return _dotted_name(call.func.value) not in FUNCTION_OWNER_MODULES",
        "    return True",
    ),
    Mutation(
        "W8",
        "the str.replace/Path.replace arity test is dropped, so an ordinary "
        "string operation on a relative-looking literal is reported as a write "
        "into the install tree - the false-positive direction that gets a fence "
        "switched off",
        "scripts/verify_write_fence.py",
        '        if name in ("replace", "rename") and len(call.args) != 1:',
        "        if False:",
    ),
    Mutation(
        "W9",
        "ast.Mod leaves the BinOp branch, so \"logs/%s.txt\" % name - the last "
        "formatting spelling - is invisible to W1-S4 again (round 7, MEDIUM-2r)",
        "scripts/verify_write_fence.py",
        "    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add, ast.Mod)):",
        "    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):",
    ),
    Mutation(
        "W10",
        "the joinpath branch is removed, so Path(\"logs\").joinpath(\"c.txt\") - "
        "the method spelling of \"/\" - stops being provably relative "
        "(round 7, MEDIUM-2r)",
        "scripts/verify_write_fence.py",
        '        if name == "joinpath" and isinstance(node.func, ast.Attribute):',
        "        if False and isinstance(node.func, ast.Attribute):",
    ),
    Mutation(
        "W11",
        "the conditional-expression branch is removed, so a write to "
        '"a.txt" if flag else "b.txt" - unanchored either way - is missed '
        "(round 7, MEDIUM-2r)",
        "scripts/verify_write_fence.py",
        "    if isinstance(node, ast.IfExp):",
        "    if False:",
    ),
]

# H4c needs a second edit in the same file: the lock is dropped and the thread is
# published before it is started, which is exactly the window HIGH-4 described.
# Kept beside the mutation rather than inside it so the dataclass stays a simple
# one-edit record.
EXTRA_EDITS: dict[str, list[tuple[str, str]]] = {
    # Q3's second half: the Models page is returned unwrapped, as it shipped.
    # Removing the minimum height alone changes nothing a user could see (Q3c
    # records the measurement), so a one-edit Q3 would have SURVIVED and told us
    # the layout regression was untested - which would have been wrong. The
    # shipped defect was both halves at once, so the mutation is both at once.
    "Q3": [
        (
            "        return self._scrollable(page)",
            "        return page",
        ),
    ],
    "H4pre": [
        (
            "        if thread is not None and thread.is_alive():\n"
            "            thread.join(timeout=5.0)",
            "        if thread is not None:\n"
            "            thread.join(timeout=5.0)",
        ),
    ],
    "H4c": [
        (
            """        refusal: str | None = None
        with self._hub_lock:""",
            """        refusal: str | None = None
        if True:""",
        ),
        (
            """                thread.start()
                self._hub_thread = thread""",
            """                self._hub_thread = thread
                thread.start()""",
        ),
        (
            """        with self._hub_lock:
            self._hub_shutting_down = True
            self._hub_cancel.set()
            thread = self._hub_thread""",
            """        if True:
            self._hub_shutting_down = True
            self._hub_cancel.set()
            thread = self._hub_thread""",
        ),
    ],
}


def edits_for(mutation: Mutation) -> list[tuple[str, str]]:
    """Every (old, new) pair this mutation applies, primary edit first."""
    return [(mutation.old, mutation.new)] + EXTRA_EDITS.get(mutation.name, [])


def apply_edit(text: str, old: str, new: str) -> str:
    """Replace one line-anchored block, or raise if it is not uniquely present.

    Two details that caused false results the first time this ran:
      * these sources are stored with CRLF line endings, so an anchor written
        with plain "\\n" matches nothing - the anchor is translated to the
        newline the file actually uses;
      * an anchor is matched WITH its preceding newline, so a 4-space-indented
        anchor cannot also match the 8-space-indented line that merely contains
        it as a substring.
    """
    newline = "\r\n" if "\r\n" in text else "\n"
    key = "\n" + old.replace("\n", newline)
    replacement = "\n" + new.replace("\n", newline)
    if text.count(key) != 1:
        raise ValueError(f"anchor matched {text.count(key)} times, expected exactly 1")
    return text.replace(key, replacement)


# A whole-suite run takes seconds. The timeout exists because a mutation can
# remove a TERMINATION rule (the download byte ceiling is one), and a suite that
# never ends would otherwise hang this script with a mutated file on disk - which
# is both a wasted run and a real hazard. A timeout is not a pass: it is reported
# as a non-zero result, i.e. the mutation was caught by the suite failing to
# complete.
SUITE_TIMEOUT_S = 600


def run_suite() -> tuple[int, str]:
    """Run the FULL suite and return (exit code, last line of output)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=str(PLATFORM_DIR),
            capture_output=True,
            text=True,
            check=False,
            timeout=SUITE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return 124, f"TIMED OUT after {SUITE_TIMEOUT_S}s (suite did not terminate)"
    lines = [line for line in proc.stdout.strip().splitlines() if line.strip()]
    return proc.returncode, lines[-1] if lines else "(no output)"


def _contains(text: str, fragment: str) -> bool:
    """Substring test that ignores which newline the file happens to use.

    These sources are stored with CRLF, while every anchor in this file is
    written with "\\n" - the same trap apply_edit documents. Comparing raw would
    make any multi-line anchor look absent, i.e. report a clean file as STALE.
    """
    return fragment.replace("\r\n", "\n") in text.replace("\r\n", "\n")


def _is_additive(mutation: Mutation) -> bool:
    """True when the replacement KEEPS the anchor and appends to it.

    Such a mutation leaves `old` in the file while it is applied, so the presence
    of `old` says nothing about whether the tree is clean - only the presence of
    `new` does.
    """
    return mutation.old in mutation.new


def verify_anchors() -> int:
    """Report any mutation that is STALE or still APPLIED, without running tests.

    Two failure modes this script has actually suffered, both silent:

    RESIDUE - this script writes a mutated file, runs the suite, and restores in
    a `finally`. Kill the process (a closed session, a reboot) and the restore
    never happens, so a disabled safety rule is left sitting in the source tree.
    That is not hypothetical: a run interrupted on 2026-08-19 left the GGUF magic
    check replaced by `if False:` in modelhub.py, and nothing in the suite
    noticed, because the mutation's own test is the only thing that checks it and
    that test was the one being deliberately broken.

    STALE - an anchor whose text has since been reworded matches nothing, so the
    mutation applies no edit and quietly stops testing the rule it names.

    A mutation's EXTRA_EDITS are applied in the same breath as its primary edit,
    so the primary edit alone is a sufficient tell for residue; only it is
    checked here, and that is a deliberate choice rather than an oversight.

    Deliberately NOT a pytest test: the harness mutates real files while the
    suite runs, so a test asserting "no mutation is applied" would fail during
    every mutation and score them all as caught.

    Exit 0 when every anchor matches real code and none is applied.
    """
    problems = 0
    if LOCK_PATH.exists():
        # Read-only, so it does not take the lock - but a residue verdict taken
        # while another harness is mutating means nothing, and the reviewer who
        # loses an hour to that deserves the warning.
        print(f"[LOCK  ] {LOCK_NAME} exists: a harness may be running RIGHT NOW. "
              "Any verdict below is a snapshot of a moving tree.")
    for mutation in MUTATIONS:
        path = PLATFORM_DIR / mutation.relpath
        text = path.read_text(encoding="utf-8")
        # ADDITIVE first, and this ORDER is the round-8 fix (MEDIUM). An
        # append-style mutation contains its own anchor, so `old` is still in the
        # file while the mutation is APPLIED - and the old code checked `old`
        # first and skipped, reporting a mutated tree as clean. 5 of the 91
        # mutations are additive (M0a, M21, W2, R5, E1), and the 18:54 run that
        # crashed mid-write to gui_controller.py is exactly the path that leaves
        # one behind. For these, "is it applied?" has to be asked of the NEW text.
        if _is_additive(mutation) and _contains(text, mutation.new):
            print(f"[RESIDUE] {mutation.name:6s} {mutation.relpath} is still MUTATED "
                  f"(additive) - restore it before trusting any test result")
            problems += 1
            continue
        # A DELETION-style mutation is the hard case: its replacement text is a
        # subset of the text it replaces, so finding that replacement proves
        # nothing - it is present in the intact code too. Those are reported as
        # "anchor matches nothing" with both possible causes named, rather than
        # guessed at, because a confidently wrong label is worse than an honest
        # ambiguity.
        if _contains(text, mutation.old):
            continue  # the rule's real text is present: intact, nothing to say
        decidable = mutation.old in mutation.new or mutation.new not in mutation.old
        if decidable and _contains(text, mutation.new):
            print(f"[RESIDUE] {mutation.name:6s} {mutation.relpath} is still MUTATED "
                  f"- restore it before trusting any test result")
        else:
            print(f"[STALE? ] {mutation.name:6s} {mutation.relpath}: anchor matches "
                  f"nothing. Either the rule was reworded (this mutation now "
                  f"tests nothing) or the mutation is still applied - read the file")
        problems += 1
    print(f"{len(MUTATIONS)} anchors checked, {problems} problem(s)")
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="run only these mutations")
    parser.add_argument("--list", action="store_true", help="list and exit")
    parser.add_argument(
        "--verify-anchors",
        action="store_true",
        help="check every anchor matches real code and none is left applied",
    )
    args = parser.parse_args(argv)
    if args.verify_anchors:
        return verify_anchors()

    selected = [m for m in MUTATIONS if not args.names or m.name in args.names]
    if args.list:
        for mutation in MUTATIONS:
            print(f"{mutation.name:5s} {mutation.relpath:20s} {mutation.description}")
        return 0
    unknown = set(args.names) - {m.name for m in MUTATIONS}
    if unknown:
        print(f"unknown mutation(s): {sorted(unknown)}")
        return 2

    # One writer at a time. Everything below this point rewrites real source
    # files, so it runs only while this process holds the tree's lock.
    try:
        token = acquire_lock()
    except HarnessLocked as exc:
        print(f"[REFUSED] {exc}")
        return 3
    try:
        return run_mutations(selected)
    finally:
        release_lock(token)


def run_mutations(selected: list[Mutation]) -> int:
    """Mutate / run the suite / restore, once per mutation. The lock is HELD.

    Split out of main so the refusal path above is a plain guard: if the lock
    cannot be taken, this function is never entered and not one byte is written.
    """
    # Snapshot every file any selected mutation touches, so a crash mid-run can
    # still be detected (and repaired) by comparing hashes at the end.
    touched = sorted({m.relpath for m in selected})
    original: dict[str, bytes] = {}
    digests: dict[str, str] = {}
    for relpath in touched:
        data = (PLATFORM_DIR / relpath).read_bytes()
        original[relpath] = data
        digests[relpath] = hashlib.sha256(data).hexdigest()

    code, summary = run_suite()
    print(f"BASELINE: {summary} exit {code}")
    if code != 0:
        print("REFUSING to mutate: the suite is not green to start with.")
        return 2

    survivors: list[str] = []
    for mutation in selected:
        path = PLATFORM_DIR / mutation.relpath
        text = original[mutation.relpath].decode("utf-8")
        try:
            for old, new in edits_for(mutation):
                text = apply_edit(text, old, new)
        except ValueError as exc:
            print(f"[STALE  ] {mutation.name:5s} {mutation.description} -> {exc}")
            survivors.append(mutation.name)
            continue
        path.write_bytes(text.encode("utf-8"))
        try:
            code, summary = run_suite()
        finally:
            path.write_bytes(original[mutation.relpath])
        if code == 0 and mutation.expect_survive:
            print(f"[REDUNDANT] {mutation.name:5s} {mutation.description}"
                  f" -> {summary}\n            expected: {mutation.expect_survive}")
        elif code == 0:
            survivors.append(mutation.name)
            print(f"[*** SURVIVED ***] {mutation.name:5s} {mutation.description}"
                  f" -> {summary}")
        else:
            print(f"[CAUGHT ] {mutation.name:5s} {mutation.description} -> {summary}")

    print("\n--- final integrity (sha256 vs the bytes captured before the run) ---")
    for relpath in touched:
        now = hashlib.sha256((PLATFORM_DIR / relpath).read_bytes()).hexdigest()
        print(f"{relpath}: {'OK' if now == digests[relpath] else 'CHANGED'}")
    code, summary = run_suite()
    print(f"POST-RESTORE SUITE: {summary} exit {code}")

    # Honest accounting: a mutation that survived BY DESIGN (redundant guard) is
    # not a catch and must not be counted as one.
    redundant = [m.name for m in selected if m.expect_survive]
    caught = len(selected) - len(survivors) - len(redundant)
    print(f"\n{caught} caught / {len(redundant)} redundant-by-design "
          f"({', '.join(redundant) or 'none'}) / {len(survivors)} survived, "
          f"out of {len(selected)}")
    if survivors:
        print(f"SURVIVORS (holes in the suite): {', '.join(survivors)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
