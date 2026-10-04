"""Automatic per-model context-window tuning (owner request 2026-08-22).

What this replaces
------------------
Raising one model's context window used to be a manual afternoon: read the
model's real trained window out of a llama.cpp startup log, guess a higher
`--ctx-size`, run `launcher.py --smoke-start` by hand, guess again, find where it
stops loading, back off, work out the right YaRN rope-scaling numbers from the
trained window, and hand-edit models.yaml. This module does exactly that
sequence, for whichever ONE model the owner picked, when they press the button.

Deliberately opt-in per model (Models page -> "Auto-tune context"), never
automatic on selection or start: every probe is a real llama-server load that
costs 20-30 seconds and real VRAM, so doing it on every click would make normal
use slower for no reason. What is automatic is the MATH and the DECISIONS once
the owner triggers it - which values to try, when to stop, what YaRN parameters
the model's own trained window implies, and what to write back.

The five steps
--------------
1. Read the model's real native training context from its own GGUF header
   (gguf_meta.py) - milliseconds, no server start, no ML dependency.
2. Merge quantized KV cache flags into the model's server_args and write them,
   so the probe below measures the ceiling under the SAME conditions the final
   config will run with. q8_0 KV cache roughly halves the VRAM cost per context
   token, which is what makes a large context affordable at all.
3. Probe upward for the real ceiling on THIS machine with a bounded number of
   real trial starts, reusing launcher.py's existing --smoke-start path (start,
   health-check, guaranteed clean shutdown) rather than reimplementing any of it.
4. Compute YaRN rope-scaling parameters from the native window THIS model
   reported - never a hardcoded or copied number - and only when the chosen
   context actually exceeds that window.
5. Write context_size plus the merged server_args back into that one model's
   models.yaml entry through config.py's existing atomic, re-parse-verified
   registry writer, and confirm the written config really starts. If the
   confirmation fails, the previous values are restored rather than left broken.

Everything except step 3's subprocess call and the two registry writes is a pure
function, so the decision-making is unit-testable without a GPU.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

# --------------------------------------------------------------------------- #
# Policy constants - the tuning knobs, gathered here so they are arguable in one
# place rather than buried as literals in the search loop.
# --------------------------------------------------------------------------- #

# Never probe below this. A context under 8k is not worth extending and is below
# what any harness this platform launches (Codex CLI, Claude Code) can even use.
MIN_CONTEXT = 8192

# Probe candidates are rounded to this grid so the search spends its very limited
# trial budget on meaningfully different values instead of on 3000-token
# refinements that no user would notice.
CONTEXT_GRID = 4096

# Stop bisecting once the known-good and known-bad values are within this
# fraction of each other. This is also where the safety margin comes from: the
# chosen value is the highest one that ACTUALLY loaded, and the search stops
# while a real gap still separates it from the lowest one that actually failed -
# so the result is never sitting on the knife edge of the failure point. (The
# manual 2026-08-22 tuning landed on 400000 with 450000 known-failing: an 11%
# gap, which is what this 12% threshold is calibrated against.)
CEILING_TOLERANCE = 0.12

# Hard ceiling on how far past the trained window YaRN is allowed to stretch a
# model. Model publishers (Qwen among them) document YaRN as reliable to about
# 4x the trained window and degrading beyond it, so probing past 4x would only
# discover a number that is technically loadable and practically useless.
MAX_YARN_FACTOR = 4

# Real trial starts the probe may spend, not counting the final confirmation
# start. Each costs 20-30 seconds and a full VRAM allocation.
DEFAULT_MAX_TRIALS = 5

# Generous per-trial subprocess timeout. A 15GB model with a 400k KV cache took
# roughly 90 seconds end to end in the manual 2026-08-22 run; this leaves room
# for a slower disk without hanging the ops worker indefinitely.
TRIAL_TIMEOUT_S = 600

# How often the trial wait loop wakes to ask "has the owner pressed Stop?".
# A quarter second is far below human perception of "it ignored my click" and
# costs nothing next to a 25-second model load.
TRIAL_POLL_INTERVAL_S = 0.25

# How long to wait for the trial's whole process tree to actually die after a
# cancel before giving up and reporting the cancel as unclean. taskkill /F /T is
# usually instant, but a llama-server mid-VRAM-allocation can sit in an
# uninterruptible driver call for a moment (the same reality services.py's
# kill_confirm_timeout exists for).
CANCEL_KILL_TIMEOUT_S = 20.0

# Flags this tuner manages. Kept as (flag, value) pairs because merging into an
# existing flat server_args list is by flag name, not by position.
KV_CACHE_ARGS: tuple[tuple[str, str], ...] = (
    ("--cache-type-k", "q8_0"),
    ("--cache-type-v", "q8_0"),
)
ROPE_FLAGS = ("--rope-scaling", "--rope-scale", "--yarn-orig-ctx")


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass
class Trial:
    """One real --smoke-start attempt at one context size."""

    context_size: int
    ok: bool
    reason: str = ""
    elapsed_s: float = 0.0
    # Measured generation speed at this context (M15.4), or None when the
    # harness reported none. A context that LOADS but has spilled its KV cache
    # out of VRAM generates 10-15x slower - measured on this machine as 209.8
    # vs 11.8 tok/s - and only a real generation exposes that, so a load-only
    # trial is not enough to accept a context.
    tokens_per_second: float | None = None
    # True when this trial was stopped by the owner rather than by the machine.
    # It is deliberately NOT the same thing as `ok=False`: a canceled trial
    # learned nothing about the ceiling, so the search must discard it rather
    # than record it as a failure that would drag the answer downward.
    canceled: bool = False


@dataclass
class ProbeResult:
    """What the bounded search learned about this machine's real ceiling."""

    # Highest context size that actually loaded and served /health, or None when
    # nothing did.
    ceiling: int | None
    # Lowest context size that actually failed, or None when nothing failed
    # (i.e. the search stopped at the policy cap, not at a real limit).
    first_failure: int | None
    trials: list[Trial] = field(default_factory=list)
    # True when the search ended because the owner pressed Stop.
    canceled: bool = False


@dataclass
class AutotuneResult:
    """The full outcome, in the shape the GUI renders and the tests assert on."""

    model_id: str
    ok: bool
    # Real native training window read from the GGUF header.
    native_context: int | None = None
    architecture: str = ""
    previous_context: int | None = None
    chosen_context: int | None = None
    first_failure: int | None = None
    yarn_applied: bool = False
    rope_scale: int | None = None
    server_args: list[str] = field(default_factory=list)
    trials: list[Trial] = field(default_factory=list)
    detail: str = ""
    # True when the owner pressed Stop mid-run. The UI renders this differently
    # from a failure on purpose: nothing went wrong, and the model's previous
    # settings were restored exactly as on the failure path.
    canceled: bool = False


# --------------------------------------------------------------------------- #
# Pure logic: server_args merging and YaRN parameters
# --------------------------------------------------------------------------- #


# M15.4: fraction of the baseline generation speed a larger context must keep
# to count as usable. 0.85 tolerates run-to-run noise (measured ~2% here) while
# rejecting the spill cliff, which lands at ~6-8% of baseline, not 85%.
THROUGHPUT_FLOOR = 0.85


def normalize_server_args(args: Sequence[Any] | None) -> list[str]:
    """Return server_args as a clean list of strings.

    models.yaml is hand-edited, so a value can legitimately arrive as a YAML int
    (one row really does carry `["--parallel", 1]`). Everything downstream - flag
    lookup, subprocess argv, the written YAML - wants strings, and coercing once
    here keeps that from being every caller's problem.
    """
    return [str(a) for a in (args or [])]


def compute_yarn_args(native_context: int, target_context: int) -> list[str]:
    """Return the rope-scaling flags a target context needs, or [] if it needs none.

    YaRN is a way of stretching a model's positional encoding so it stays
    coherent past the window it was trained on. It is ONLY meaningful past that
    window: inside the trained range the model already handles positions
    correctly, and adding a scale factor there would distort a working encoding.
    So a target at or below the native window returns an empty list - no flags at
    all - which is the whole reason this is a function and not a template.

    Above the native window, the scale factor is ceil(target / native): the
    smallest whole multiple of the trained window that still contains the target.
    Using ceil rather than the exact ratio matters - a scale that lands just
    short of the target would leave the top of the context outside the properly
    interpolated range, which is exactly the unscaled extrapolation llama.cpp
    permits but which degrades output quality.

    `--yarn-orig-ctx` is always the model's OWN native window, read from its own
    GGUF header. Copying another model's number here was the specific mistake
    this whole feature exists to make impossible.
    """
    if native_context <= 0:
        raise ValueError(f"native_context must be positive (got {native_context!r})")
    if target_context <= 0:
        raise ValueError(f"target_context must be positive (got {target_context!r})")
    if target_context <= native_context:
        return []
    scale = math.ceil(target_context / native_context)
    return [
        "--rope-scaling",
        "yarn",
        "--rope-scale",
        str(scale),
        "--yarn-orig-ctx",
        str(native_context),
    ]


def merge_server_args(
    existing: Sequence[Any] | None,
    additions: Sequence[Any],
    *,
    drop_flags: Sequence[str] = (),
) -> list[str]:
    """Merge `additions` into a flat llama-server argv list without duplicating flags.

    server_args is a flat list (`["--parallel", "1", "--cache-type-k", "q8_0"]`),
    so "is this flag already set" is a scan for the flag token, and updating one
    means replacing the token AFTER it. Rules:

    - A flag in `additions` that is already present has its value REPLACED in
      place, keeping its original position. Position matters to nobody here, but
      preserving it keeps the owner's hand-ordered list recognisable.
    - A flag not present is APPENDED with its value.
    - Flags named in `drop_flags` are removed first, with their values. This is
      how a stale rope-scaling block is cleared before a freshly computed one is
      added: a re-tune that lowers the context below the native window must not
      leave the previous run's `--rope-scale` behind.
    - Every other argument, including flags this tuner knows nothing about, is
      preserved untouched.
    """
    merged = normalize_server_args(existing)

    for flag in drop_flags:
        while flag in merged:
            index = merged.index(flag)
            # Remove the flag and its value. The trailing-flag case (a flag with
            # no value at the end of the list) would make the slice remove only
            # the flag, which is the correct degrade rather than an IndexError.
            del merged[index : index + 2]

    pairs = _as_flag_pairs(additions)
    for flag, value in pairs:
        if flag in merged:
            index = merged.index(flag)
            if index + 1 < len(merged):
                merged[index + 1] = value
            else:  # pragma: no cover - malformed trailing flag; repair it
                merged.append(value)
        else:
            merged.extend([flag, value])
    return merged


def _as_flag_pairs(additions: Sequence[Any]) -> list[tuple[str, str]]:
    """Accept either a flat ["--flag", "value", ...] list or [(flag, value), ...]."""
    if not additions:
        return []
    if isinstance(additions[0], (tuple, list)):
        return [(str(f), str(v)) for f, v in additions]  # type: ignore[misc]
    flat = normalize_server_args(additions)
    if len(flat) % 2 != 0:
        raise ValueError(
            f"server_args additions must be flag/value pairs (got an odd number "
            f"of items: {flat!r})"
        )
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def plan_target_args(
    existing: Sequence[Any] | None,
    native_context: int,
    target_context: int,
) -> list[str]:
    """Build the final server_args for a chosen context: KV cache + YaRN, merged.

    The two halves are deliberately different in kind. The KV cache flags are
    unconditional (quantizing the cache is close to free and is what makes a big
    context fit at all), while the rope flags are conditional on the target
    genuinely exceeding the trained window - and when they are not wanted, any
    previously written rope flags are dropped rather than left to contradict the
    new context_size.
    """
    rope = compute_yarn_args(native_context, target_context)
    merged = merge_server_args(existing, KV_CACHE_ARGS)
    if rope:
        return merge_server_args(merged, rope)
    return merge_server_args(merged, (), drop_flags=ROPE_FLAGS)


# --------------------------------------------------------------------------- #
# Pure logic: the bounded search
# --------------------------------------------------------------------------- #


def _round_to_grid(value: int) -> int:
    """Snap a candidate to the nearest CONTEXT_GRID multiple, never below MIN_CONTEXT."""
    snapped = int(round(value / CONTEXT_GRID)) * CONTEXT_GRID
    return max(MIN_CONTEXT, snapped)


def probe_cap(native_context: int) -> int:
    """The highest context this tuner will even try, from the model's own window."""
    return _round_to_grid(native_context * MAX_YARN_FACTOR)


def first_candidate(native_context: int, current_context: int) -> int:
    """Pick the opening probe value: the more informative of native vs current.

    Starting at the model's own trained window is the single most valuable first
    data point - it answers "does this machine even fit what the model was built
    for" in one trial. When the model is already configured HIGHER than that
    (because a previous tune or a hand-edit raised it), that configured value is
    the better opening bid: re-discovering ground already known to work would
    waste a third of the trial budget.
    """
    return min(
        probe_cap(native_context),
        _round_to_grid(max(native_context, current_context, MIN_CONTEXT)),
    )


def probe_ceiling(
    trial_fn: Callable[[int], Trial],
    native_context: int,
    current_context: int,
    *,
    max_trials: int = DEFAULT_MAX_TRIALS,
    progress: Callable[[str], None] | None = None,
    cancel: threading.Event | None = None,
    floor_tokps: float | None = None,
) -> ProbeResult:
    """Find the highest context that really loads AND still performs.

    `floor_tokps` (M15.4) is the throughput floor: a trial that loads but
    generates below it is treated as a FAILURE for the search, because a
    context whose KV cache has spilled out of VRAM answers 10-15x slower while
    reporting a perfectly healthy /health. The floor is derived from a real
    baseline measurement by the caller; None (no baseline available) preserves
    the old load-only behaviour rather than inventing a threshold.

    Grow-then-bisect, which is the shape a human uses by hand: double upward
    while it keeps working (cheap way to bracket an unknown ceiling), then
    bisect the bracket once a failure gives you an upper bound. It stops early
    the moment another trial could not teach it anything worth 25 seconds:

    - the policy cap was reached and worked (there is nothing above to find);
    - the good/bad bracket is inside CEILING_TOLERANCE (further refinement
      would move the answer by a few percent);
    - the next candidate rounds onto a value already tried.

    `trial_fn` is injected so the decision-making is testable without a GPU; the
    real one starts an actual llama-server. `progress` receives one human line
    per trial so the UI can show which value is being tried rather than a frozen
    spinner - these trials are slow and the owner must be able to see that
    something is happening.

    `cancel` is the owner's Stop button. It is checked BEFORE each trial (so a
    cancel that lands between trials never pays for another 25-second load) and
    honoured again AFTER one (because run_smoke_trial can abort a trial that is
    already in flight and report it as canceled). A canceled trial is recorded
    for the record but is NOT allowed to set the known-bad bound: it failed
    because the owner stopped it, not because the machine could not fit it, and
    treating it as evidence would silently lower the answer.
    """
    cap = probe_cap(native_context)
    trials: list[Trial] = []
    low: int | None = None  # highest known-good
    high: int | None = None  # lowest known-bad
    candidate = first_candidate(native_context, current_context)
    tried: set[int] = set()
    canceled = False

    while len(trials) < max_trials:
        if cancel is not None and cancel.is_set():
            canceled = True
            break
        if candidate in tried:
            break
        tried.add(candidate)
        if progress is not None:
            progress(
                f"trial {len(trials) + 1} of at most {max_trials}: "
                f"starting at context {candidate} ..."
            )
        trial = trial_fn(candidate)
        trials.append(trial)
        if trial.canceled or (cancel is not None and cancel.is_set()):
            canceled = True
            if progress is not None:
                progress(f"context {candidate}: canceled by the owner")
            break
        if (
            trial.ok
            and floor_tokps is not None
            and trial.tokens_per_second is not None
            and trial.tokens_per_second < floor_tokps
        ):
            # The context loaded but the KV cache has left VRAM: for a ceiling
            # search that is a failure, and saying so here is the entire reason
            # this floor exists (the load-only search once chose a context that
            # generated at 6% of baseline speed).
            trial = replace(
                trial,
                ok=False,
                reason=(
                    f"loaded but generated {trial.tokens_per_second:.1f} tok/s, "
                    f"below the {floor_tokps:.1f} tok/s floor"
                ),
            )
            trials[-1] = trial
        if progress is not None:
            progress(
                f"context {candidate}: "
                + ("loaded" if trial.ok else f"failed ({trial.reason or 'no reason'})")
                + (
                    f" at {trial.tokens_per_second:.1f} tok/s"
                    if trial.ok and trial.tokens_per_second is not None
                    else ""
                )
            )

        if trial.ok:
            low = candidate
            if candidate >= cap:
                break  # nothing above the policy cap is worth finding
            nxt = min(candidate * 2, cap) if high is None else (candidate + high) // 2
        else:
            high = candidate
            if low is None:
                # Nothing has worked yet: halve downward to find any floor at all.
                nxt = max(MIN_CONTEXT, candidate // 2)
            else:
                nxt = (low + candidate) // 2

        if low is not None and high is not None:
            if high - low <= max(CONTEXT_GRID, int(low * CEILING_TOLERANCE)):
                break

        nxt = _round_to_grid(nxt)
        # A candidate that is not strictly inside the remaining unknown band
        # cannot teach us anything, so stop rather than burn a real start on it.
        if low is not None and nxt <= low:
            break
        if high is not None and nxt >= high:
            break
        candidate = nxt

    return ProbeResult(
        ceiling=low, first_failure=high, trials=trials, canceled=canceled
    )


# --------------------------------------------------------------------------- #
# The real trial: a thin wrapper over launcher.py --smoke-start
# --------------------------------------------------------------------------- #


def console_python() -> str:
    """The interpreter a trial subprocess must run under: a CONSOLE python.exe.

    Bug fixed 2026-08-22, and the reason this is a function rather than a bare
    `sys.executable`. LOCITIZE Desktop runs under `pythonw.exe`, a GUI-subsystem
    binary that Windows never gives a console to. A trial spawned with
    sys.executable therefore inherited that consoleless-ness, and inside it
    `ManagedProcess.stop()`'s graceful `CTRL_BREAK_EVENT` could not be delivered
    at all - it failed with WinError 6 and (see services.py) took the whole
    shutdown down with it, so every single trial came back "model became ready
    but did not shut down cleanly afterward" and the tune concluded that no
    context size would load. Running the trial from the terminal worked purely
    because a terminal's python.exe HAS a console. services.py now survives the
    consoleless case on its own; this makes the GUI's trial subprocess match the
    verified terminal path exactly instead of merely surviving a difference.

    Falls back to sys.executable unchanged whenever the sibling python.exe is not
    actually there, so a packaged or unusual interpreter layout degrades to
    today's behaviour rather than to a missing-file crash.
    """
    executable = sys.executable or ""
    if sys.platform != "win32" or not executable:
        return executable
    path = Path(executable)
    if path.name.lower() != "pythonw.exe":
        return executable
    console = path.with_name("python.exe")
    return str(console) if console.exists() else executable


def _terminate_trial_tree(process: subprocess.Popen[Any]) -> bool:
    """Kill a trial subprocess AND everything it started; True iff it is gone.

    A trial is three processes deep - this process spawned `launcher.py`, which
    spawned `llama-server.exe` holding many GB of VRAM. Killing only the middle
    one would orphan the expensive one, which is the exact failure mode the whole
    codebase's AC7 "no orphan" discipline exists to prevent, so this goes through
    `taskkill /F /T` (kill the tree, by a PID this process itself created - never
    a name pattern) rather than Popen.terminate().

    taskkill snapshots the tree before killing, so the grandchild is included
    even though it lives in its own process group (CREATE_NEW_PROCESS_GROUP).
    """
    if process.poll() is not None:
        return True
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError:
            # taskkill itself unavailable: fall through to the portable kill so
            # a cancel still does something rather than silently nothing.
            process.kill()
    else:  # pragma: no cover - the platform this ships on is Windows
        process.kill()
    try:
        process.wait(timeout=CANCEL_KILL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False
    return True


def run_smoke_trial(
    model_id: str,
    context_size: int,
    *,
    base_dir: Path | str | None = None,
    timeout_s: float = TRIAL_TIMEOUT_S,
    cancel: threading.Event | None = None,
) -> Trial:
    """Start MODEL_ID for real at `context_size` via the existing CLI, and report.

    Deliberately a thin wrapper and nothing more. `launcher.py --smoke-start`
    already owns the entire lifecycle - build the real controller, start the
    child, poll /health until ready or the model's own timeout, ALWAYS stop it,
    confirm no orphan was left, and print a JSON verdict. Reimplementing any of
    that here would mean a second, less-tested copy of the one code path that
    guarantees no llama-server is left holding 15GB of VRAM.

    This is Popen plus a wait loop rather than one blocking `subprocess.run`
    purely so the owner's Stop button can land mid-trial. `run()` cannot be
    interrupted, so a Stop during a 25-second model load used to be invisible
    until the trial finished on its own; here the loop wakes every
    TRIAL_POLL_INTERVAL_S, and a set `cancel` event kills the whole trial tree
    immediately so the GPU is actually released rather than held for the rest of
    the trial's own timeout.

    Child output goes to temp FILES, not pipes. Polling a process while its
    stdout pipe fills is the classic deadlock, and this codebase already refuses
    unread PIPEs for exactly that reason (see services.ManagedProcess).

    Like spawn_in_terminal elsewhere in this codebase, the subprocess call itself
    is left unexercised by the test suite (it needs a GPU and half a minute); the
    logic that decides WHICH context sizes to pass here is what the tests cover.
    """
    root = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    command = [
        console_python(),
        str(root / "launcher.py"),
        "--smoke-start",
        model_id,
        "--ctx-size",
        str(context_size),
        "--json",
    ]
    # CREATE_NO_WINDOW: this runs behind a GUI button, so a console window
    # flashing up once per trial would be noise, not information. The child still
    # GETS a console (it is a console-subsystem python.exe), which is what
    # console_python() above is for; this flag only stops that console being
    # given a visible window.
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0

    if cancel is not None and cancel.is_set():
        # Already canceled before we spent anything: do not start a load at all.
        return Trial(
            context_size=context_size,
            ok=False,
            reason="canceled before this trial started",
            canceled=True,
        )

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="locitize-trial-") as scratch:
        out_path = Path(scratch) / "stdout.txt"
        err_path = Path(scratch) / "stderr.txt"
        try:
            out_file = open(out_path, "wb")  # noqa: SIM115 - closed in finally
            err_file = open(err_path, "wb")  # noqa: SIM115 - closed in finally
        except OSError as exc:
            return Trial(
                context_size=context_size,
                ok=False,
                reason=f"could not prepare the trial output files: {exc}",
            )
        try:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(root),
                    stdout=out_file,
                    stderr=err_file,
                    creationflags=creationflags,
                )
            except OSError as exc:
                return Trial(
                    context_size=context_size,
                    ok=False,
                    reason=f"could not run the trial start: {exc}",
                )

            outcome = _await_trial(process, cancel, timeout_s)
            if outcome is not None:
                # Canceled or timed out: the tree is already dealt with inside
                # _await_trial, and there is no verdict worth parsing.
                return Trial(
                    context_size=context_size,
                    ok=False,
                    reason=outcome[1],
                    canceled=outcome[0],
                    elapsed_s=round(time.monotonic() - started, 2),
                )
            returncode = process.returncode
        finally:
            out_file.close()
            err_file.close()

        stdout = _read_text(out_path)
        stderr = _read_text(err_path)

    verdict = _parse_smoke_json(stdout)
    ok = returncode == 0 and verdict.get("outcome") == "ready"
    reason = str(verdict.get("reason") or "").strip()
    if not ok and not reason:
        # No JSON verdict to quote (the harness died before printing one): fall
        # back to its last stderr line, then to the bare exit code, so a failed
        # trial is never reported with an empty explanation.
        stderr_lines = [line for line in (stderr or "").splitlines() if line.strip()]
        reason = stderr_lines[-1].strip() if stderr_lines else f"exit code {returncode}"
    raw_tokps = verdict.get("tokens_per_second")
    return Trial(
        context_size=context_size,
        ok=ok,
        reason="" if ok else reason,
        elapsed_s=float(verdict.get("elapsed_s") or 0.0),
        tokens_per_second=(
            float(raw_tokps)
            if isinstance(raw_tokps, (int, float)) and not isinstance(raw_tokps, bool)
            else None
        ),
    )


def _await_trial(
    process: subprocess.Popen[Any],
    cancel: threading.Event | None,
    timeout_s: float,
) -> tuple[bool, str] | None:
    """Wait for a trial, watching for cancellation; None means it finished normally.

    Returns `(canceled, reason)` when the trial was ended by something other than
    itself - the owner's Stop, or the trial timeout - having already killed the
    process tree in both cases. Returning None means the process exited on its
    own and its exit code and JSON verdict are the real answer.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        if process.poll() is not None:
            return None
        if cancel is not None and cancel.is_set():
            killed = _terminate_trial_tree(process)
            return (
                True,
                "canceled by the owner"
                if killed
                else (
                    "canceled by the owner, but the trial process could not be "
                    "confirmed stopped within "
                    f"{int(CANCEL_KILL_TIMEOUT_S)}s"
                ),
            )
        if time.monotonic() >= deadline:
            _terminate_trial_tree(process)
            return (False, f"the trial start did not finish within {int(timeout_s)}s")
        # Sleeping on the event (when there is one) rather than on the clock
        # makes a cancel land within milliseconds instead of within a poll tick.
        if cancel is not None:
            cancel.wait(TRIAL_POLL_INTERVAL_S)
        else:
            time.sleep(TRIAL_POLL_INTERVAL_S)


def _read_text(path: Path) -> str:
    """Read a trial's captured output; never raise on a missing/undecodable file.

    This runs while assembling a trial verdict, so an unreadable scratch file
    must degrade to "no output to quote" rather than replace a real result with
    an IO traceback.
    """
    try:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _parse_smoke_json(stdout: str) -> dict[str, Any]:
    """Pull the JSON verdict out of --smoke-start's output.

    The harness prints exactly one JSON object, but it shares stdout with
    whatever the launcher's own logging emits, so the last parseable line wins
    rather than assuming the whole stream is JSON.
    """
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


# --------------------------------------------------------------------------- #
# The orchestrator
# --------------------------------------------------------------------------- #

# Single-sourced so every cancellation path tells the owner the same two things:
# it stopped because they asked, and their model was left exactly as it was.
_CANCEL_DETAIL = (
    "auto-tune canceled. This model's context_size and server_args were left "
    "unchanged."
)


def autotune_model_context(
    *,
    model_id: str,
    location: str,
    current_context: int,
    current_server_args: Sequence[Any] | None,
    write_tuning: Callable[[str, int, list[str]], None],
    trial_fn: Callable[[int], Trial],
    progress: Callable[[str], None] | None = None,
    max_trials: int = DEFAULT_MAX_TRIALS,
    cancel: threading.Event | None = None,
) -> AutotuneResult:
    """Run the whole tune for one model and return what really happened.

    Every side effect is injected (`write_tuning` performs the registry write,
    `trial_fn` performs a real start), which is what lets the entire sequence -
    including the two writes and the rollback - be tested without a GPU or a
    models.yaml.

    Ordering note: the KV-cache flags are written BEFORE probing on purpose. The
    probe measures how much context this machine can actually allocate, and that
    answer is only valid for the flags the model will really run with; probing
    with an unquantized cache and then writing a quantized one would report a
    ceiling for a configuration that never runs.

    Failure policy: if the confirmation start of the final written config does
    not come up, the model's previous context_size and server_args are written
    back. A tune that leaves the owner's model unable to start is worse than a
    tune that reports it could not find an improvement.

    Cancellation policy (owner request 2026-08-22): `cancel` is the Stop button,
    and it is treated as a first-class outcome, not as a failure. Whenever it is
    observed the model's previous context_size and server_args are restored - the
    SAME rollback the failure path performs, because a half-probed tune has
    exactly as little claim on the owner's config as a failed one - and the
    result comes back ok=False with canceled=True so the UI can say "canceled"
    instead of inventing a problem that did not happen.
    """
    from gguf_meta import GgufError, read_gguf_header

    emit = progress if progress is not None else (lambda _line: None)

    def canceled() -> bool:
        """True once the owner has pressed Stop (no cancel event = never)."""
        return cancel is not None and cancel.is_set()

    previous_args = normalize_server_args(current_server_args)
    result = AutotuneResult(
        model_id=model_id,
        ok=False,
        previous_context=current_context,
        server_args=list(previous_args),
    )

    # A cancel that lands before anything has been written has nothing to roll
    # back, so it returns straight away rather than performing a no-op write.
    if canceled():
        result.canceled = True
        result.detail = _CANCEL_DETAIL
        return result

    # ---- Step 1: the model's own trained window, from its own file ---------- #
    emit("reading the model's native context length from its GGUF header ...")
    try:
        header = read_gguf_header(location)
    except GgufError as exc:
        result.detail = str(exc)
        return result
    if header.context_length is None:
        result.architecture = header.architecture
        result.detail = (
            f"This model's GGUF header carries no "
            f"'{header.architecture}.context_length' value, so locitize cannot "
            f"tell what window it was trained for. Set context_size by hand."
        )
        return result

    native = int(header.context_length)
    result.native_context = native
    result.architecture = header.architecture
    emit(f"native context: {native} (architecture {header.architecture})")

    # ---- Step 2: probe under the flags the final config will use ----------- #
    probe_args = merge_server_args(previous_args, KV_CACHE_ARGS)
    if probe_args != previous_args:
        emit("adding quantized KV cache (q8_0) so the probe measures the real config ...")
        try:
            write_tuning(model_id, current_context, probe_args)
        except Exception as exc:  # noqa: BLE001 - boundary: report, never raise up
            result.detail = f"could not update this model's server_args: {exc}"
            return result
        result.server_args = list(probe_args)

    # ---- Step 3: a real throughput baseline at the current context --------- #
    # One extra start (~30s) buys the search its floor: anything that later
    # loads but generates far slower than TODAY's config is a spilled KV cache,
    # not a usable ceiling. A baseline that fails or carries no measurement
    # downgrades honestly to the old load-only search rather than blocking.
    floor_tokps: float | None = None
    emit(f"measuring throughput baseline at current context {current_context} ...")
    baseline = trial_fn(current_context)
    if baseline.canceled or canceled():
        _restore(write_tuning, model_id, current_context, previous_args, emit)
        result.server_args = list(previous_args)
        result.canceled = True
        result.detail = _CANCEL_DETAIL
        emit(result.detail)
        return result
    if baseline.ok and baseline.tokens_per_second is not None:
        floor_tokps = baseline.tokens_per_second * THROUGHPUT_FLOOR
        emit(
            f"baseline {baseline.tokens_per_second:.1f} tok/s; a larger context "
            f"must keep {floor_tokps:.1f} tok/s ({int(THROUGHPUT_FLOOR * 100)}%)"
        )
    else:
        emit(
            "baseline carried no throughput measurement; falling back to the "
            "load-only search (results may include a context that loads slowly)"
        )

    # ---- Step 4: the bounded real search ----------------------------------- #
    probe = probe_ceiling(
        trial_fn,
        native,
        current_context,
        max_trials=max_trials,
        progress=emit,
        cancel=cancel,
        floor_tokps=floor_tokps,
    )
    result.trials = probe.trials
    result.first_failure = probe.first_failure
    if probe.canceled or canceled():
        # Undo step 2's KV-cache write for the same reason the "nothing loaded"
        # branch below does: it was made to serve a probe that never finished,
        # and an abandoned operation must not leave an edit behind.
        _restore(write_tuning, model_id, current_context, previous_args, emit)
        result.server_args = list(previous_args)
        result.canceled = True
        result.detail = _CANCEL_DETAIL
        emit(result.detail)
        return result
    if probe.ceiling is None:
        # Nothing loaded at all. Put the server_args back the way they were: the
        # KV-cache change was made to serve a probe that found nothing, and
        # leaving an unrequested edit behind after a failed operation is not this
        # codebase's habit.
        _restore(write_tuning, model_id, current_context, previous_args, emit)
        result.server_args = list(previous_args)
        failed = probe.trials[0] if probe.trials else None
        result.detail = (
            "no context size tried would load on this machine"
            + (f" (lowest failure: {failed.reason})" if failed and failed.reason else "")
            + ". This model's current settings were left unchanged."
        )
        return result

    chosen = probe.ceiling
    result.chosen_context = chosen

    # ---- Step 4: YaRN from THIS model's real native window ------------------ #
    final_args = plan_target_args(previous_args, native, chosen)
    rope = compute_yarn_args(native, chosen)
    result.yarn_applied = bool(rope)
    result.rope_scale = int(rope[3]) if rope else None
    if rope:
        emit(
            f"chosen context {chosen} exceeds the trained window {native}: "
            f"applying YaRN at scale {result.rope_scale}"
        )
    else:
        emit(f"chosen context {chosen} is within the trained window: no YaRN needed")

    # ---- Step 5: write, then confirm the written config actually starts ----- #
    try:
        write_tuning(model_id, chosen, final_args)
    except Exception as exc:  # noqa: BLE001 - boundary: report, never raise up
        _restore(write_tuning, model_id, current_context, previous_args, emit)
        result.server_args = list(previous_args)
        result.detail = f"could not write the tuned settings: {exc}"
        return result
    result.server_args = list(final_args)

    emit(f"confirming the written configuration starts at context {chosen} ...")
    confirm = trial_fn(chosen)
    result.trials.append(confirm)
    if confirm.canceled or canceled():
        # The tuned values are already on disk at this point, but they were never
        # confirmed to start. Restoring is the honest choice and matches the
        # failed-confirmation branch immediately below: LOCITIZE only keeps a
        # tuned config it has actually watched come up.
        _restore(write_tuning, model_id, current_context, previous_args, emit)
        result.server_args = list(previous_args)
        result.chosen_context = None
        result.canceled = True
        result.detail = _CANCEL_DETAIL
        emit(result.detail)
        return result
    if not confirm.ok:
        _restore(write_tuning, model_id, current_context, previous_args, emit)
        result.server_args = list(previous_args)
        result.chosen_context = None
        result.detail = (
            f"the tuned configuration did not start on confirmation "
            f"({confirm.reason or 'no reason reported'}); this model's previous "
            f"settings have been restored."
        )
        return result

    result.ok = True
    result.detail = format_summary(result)
    emit(result.detail)
    return result


def _restore(
    write_tuning: Callable[[str, int, list[str]], None],
    model_id: str,
    context_size: int,
    server_args: list[str],
    emit: Callable[[str], None],
) -> None:
    """Put a model's previous context_size/server_args back after a failed tune.

    Best effort by design: this runs on an already-failing path, and a rollback
    that raised would replace an honest "could not tune" message with an
    unrelated stack trace. A rollback that itself fails is reported, not hidden.
    """
    try:
        write_tuning(model_id, context_size, server_args)
    except Exception as exc:  # noqa: BLE001 - boundary on a failure path
        emit(f"warning: could not restore the previous settings: {exc}")


def format_summary(result: AutotuneResult) -> str:
    """One-line human summary of a finished tune, for the GUI and the logs."""
    parts = [
        f"native context {result.native_context}",
        f"safe ceiling {result.chosen_context}",
    ]
    if result.first_failure is not None:
        parts.append(f"first failure {result.first_failure}")
    else:
        parts.append("no failure found below the 4x policy cap")
    parts.append(
        f"YaRN scale {result.rope_scale}" if result.yarn_applied else "no YaRN needed"
    )
    parts.append(
        f"context_size {result.previous_context} -> {result.chosen_context}"
    )
    parts.append(f"{len(result.trials)} real starts")
    return "; ".join(parts)
