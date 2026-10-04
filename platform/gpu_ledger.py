"""GPU occupancy ledger (M17.8): see what holds VRAM, free LOCITIZE's own.

A local-AI app lives and dies by VRAM, and the commonest "why is this slow / why
won't it load" cause is another program still holding the card - a previous LM
Studio session, a crashed llama-server, or LOCITIZE's own measurement probe. The
GPU is opaque: Windows shows no easy per-app VRAM view, so a user cannot tell
what is occupying it. This module answers two questions:

  1. What is on the GPU right now, and how much is LOCITIZE's own doing?
  2. Which of those can LOCITIZE safely free?

The safety line is deliberate and narrow (owner decision 2026-08-29): LOCITIZE
frees only its OWN processes - the llama-servers it launched, identified by their
executable living under LOCITIZE's bin directory. Another vendor's app (LM Studio,
a game, a notebook) is shown so the user knows it is there, but LOCITIZE never
terminates it; that is the user's call in that app. Freeing only what we started
cannot cost anyone unsaved work in a program LOCITIZE does not own.

Pure and dependency-free except one thin, injectable nvidia-smi call. The parse
and classification - the parts that decide what gets stopped - are pure and
tested; the shell-out is a seam a test replaces. Nothing here terminates a
process: this module only IDENTIFIES; the controller performs the stop through
the existing service supervisor. ASCII only.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# CREATE_NO_WINDOW: keep nvidia-smi from flashing a console (Windows only; the
# flag is harmlessly ignored where subprocess does not recognise it because the
# call site only runs on the platform that defines it).
_NO_WINDOW = 0x08000000


@dataclass
class GpuProcess:
    """One process holding the GPU, as nvidia-smi reports it."""

    pid: int
    name: str  # executable path (or "[Insufficient Permissions]" for a hidden one)
    used_mb: float | None  # None when the driver withholds per-process memory
    is_locitize: bool = False

    def short_name(self) -> str:
        """A display name: the executable's basename, or the raw string if it is
        a permissions placeholder rather than a path."""
        raw = self.name.strip()
        if raw.startswith("[") and raw.endswith("]"):
            return raw
        return Path(raw).name or raw


def query_compute_apps(
    run: Callable[..., Any] = subprocess.run, smi: str | None = None
) -> str | None:
    """Raw nvidia-smi compute-apps CSV, or None when unavailable. Never raises.

    `run` is injected so tests exercise the parser without a GPU. None means
    "cannot tell" (no nvidia-smi, or it errored) - the honest answer that leaves
    the caller showing "GPU status unavailable" rather than an empty list that
    would read as "nothing is running".
    """
    exe = smi or shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        proc = run(
            [
                exe,
                "--query-compute-apps=pid,used_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(proc, "returncode", 1) != 0:
        return None
    return getattr(proc, "stdout", None)


def parse_compute_apps(text: str | None) -> list[GpuProcess]:
    """Parse nvidia-smi compute-apps CSV into GpuProcess rows.

    Each line is "pid, used_memory, process_name". A row with an unparseable pid
    is skipped (it is not a real process we can act on); a used_memory the driver
    withholds ("[N/A]", needs elevation for per-process figures) becomes None
    rather than a fabricated zero, so the UI can say "unknown" honestly.
    """
    rows: list[GpuProcess] = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        mem_raw = parts[1]
        used: float | None = None
        if mem_raw and not (mem_raw.startswith("[") and mem_raw.endswith("]")):
            try:
                used = float(mem_raw)
            except ValueError:
                used = None
        # A Windows path has no commas, so the name is everything after the first
        # two fields rejoined - robust even if that ever changes.
        name = ",".join(parts[2:]).strip()
        rows.append(GpuProcess(pid=pid, name=name, used_mb=used))
    return rows


def mark_ours(
    procs: list[GpuProcess],
    *,
    bin_dir: str | Path | None,
    own_pids: Any = (),
) -> list[GpuProcess]:
    """Return the rows with is_locitize set for LOCITIZE's own processes.

    A process is LOCITIZE's own when its executable lives under `bin_dir` (the
    data root's bin/ where LOCITIZE puts llama-server) OR its pid is one the
    service supervisor is currently tracking. The path test is primary because it
    also catches servers the supervisor lost track of - a crashed session, or the
    measurement probe - which are exactly the ones a user needs to clear.
    """
    base = ""
    if bin_dir:
        base = str(Path(bin_dir)).replace("\\", "/").rstrip("/").lower()
    owned = {int(p) for p in (own_pids or ())}
    result: list[GpuProcess] = []
    for p in procs:
        norm = p.name.replace("\\", "/").lower()
        # A path segment match, not a substring: <data>/bin must not also
        # claim <data>/bin2 or <data>/binaries.
        is_ours = p.pid in owned or (bool(base) and (base + "/") in norm)
        result.append(GpuProcess(p.pid, p.name, p.used_mb, is_ours))
    return result


def freeable_pids(procs: list[GpuProcess]) -> list[int]:
    """The pids LOCITIZE may stop: only its own. Never a process it does not own."""
    return [p.pid for p in procs if p.is_locitize]


def terminate_pids(
    pids: Any, run: Callable[..., Any] = subprocess.run
) -> dict[int, bool]:
    """Force-terminate each pid via taskkill, returning {pid: succeeded}.

    Used only on pids freeable() has already confirmed are LOCITIZE's own, so
    this never reaches a foreign app. /T also ends the process's children (a
    llama-server can spawn helpers), /F forces it since a busy server may ignore
    a gentle close. Never raises: a pid that is already gone (the server exited
    between the scan and the click) simply reports False, which the caller treats
    as "nothing to do", not an error. The service supervisor's clean stop is
    preferred for a server it still tracks; this is the fallback for orphans it
    lost - a crashed session, or the measurement probe.
    """
    results: dict[int, bool] = {}
    for raw in pids or ():
        pid = int(raw)
        try:
            proc = run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                creationflags=_NO_WINDOW,
            )
            results[pid] = getattr(proc, "returncode", 1) == 0
        except (OSError, subprocess.SubprocessError):
            results[pid] = False
    return results


def summarize(procs: list[GpuProcess]) -> str:
    """One honest status line for a chip or log."""
    if not procs:
        return "no processes are holding the GPU."
    ours = [p for p in procs if p.is_locitize]
    known_mb = sum(p.used_mb for p in procs if p.used_mb is not None)
    noun = "process" if len(procs) == 1 else "processes"
    tail = f" ({known_mb:.0f} MB attributed)" if known_mb else ""
    return (
        f"{len(procs)} {noun} on the GPU, {len(ours)} owned by locitize{tail}."
    )


# --------------------------------------------------------------------------- #
# Where one process's GPU memory actually sits (owner report 2026-09-03, "when
# I switch models Open WebUI does not chat").
# --------------------------------------------------------------------------- #
# nvidia-smi withholds per-process memory on Windows (WDDM), and it could not
# answer the question that matters anyway: not how much a server holds, but how
# much of it OVERFLOWED the card. On Windows an allocation that exceeds the card
# does not fail - the driver pages it through system RAM - so a model loads,
# passes /health, and then generates at a few tokens per second with nothing
# saying why. Windows itself keeps the answer in two performance counters per
# process: dedicated (on the card) and shared (system RAM the driver is using as
# video memory). Measured on the reference card, 2026-09-03: Qwen3.8-27B at
# full offload showed 782 MB shared and 5 tok/s, Qwen3.6-35B 1128 MB and 11
# tok/s; fitted (backends.offload_value) they showed 158 and 280 MB and ran
# 24 and 112 tok/s. A fitted server carries 100-300 MB shared regardless
# (pinned host buffers), so the threshold below sits above that noise. What
# overflowed decides how much it hurts - gemma-4-26b ran 108 tok/s with 500 MB
# shared - so the line below reports the fact and a hint, not a verdict; the
# smoke start's real generation (tokens_per_second) is the speed measurement.

SPILL_THRESHOLD_MB = 400


@dataclass
class GpuPlacement:
    """One process's video memory: on the card vs. paged through system RAM."""

    pid: int
    dedicated_mb: int
    shared_mb: int

    @property
    def spilled(self) -> bool:
        return self.shared_mb >= SPILL_THRESHOLD_MB

    def describe(self, label: str = "the model") -> str:
        """One line for a log or chip: the two numbers, and past the threshold
        the lever to pull. A number, not a verdict: measured 2026-09-03, 472 MB
        shared ran a dense 27B at 29 tok/s and 904 MB a 35B MoE at 123, while
        664-1128 MB crawled."""
        if self.spilled:
            return (
                f"{label}: {self.dedicated_mb} MB on the GPU and {self.shared_mb} MB "
                f"paged through system RAM; if replies crawl, that is why - lower "
                f"this model's context_size or free VRAM."
            )
        return f"{label}: {self.dedicated_mb} MB on the GPU, {self.shared_mb} MB shared."


# One PowerShell call reads both counters for one pid. The counter instance
# names look like "pid_36212_luid_0x00000000_0x0000E3F4_phys_0"; a process may
# have several (one per adapter LUID), so the values are summed. Output is two
# integers, MB, "dedicated shared".
_PLACEMENT_SCRIPT = (
    "$ErrorActionPreference='Stop'; "
    "$d=(Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage').CounterSamples"
    " | Measure-Object CookedValue -Sum; "
    "$s=(Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Shared Usage').CounterSamples"
    " | Measure-Object CookedValue -Sum; "
    "if ($d.Count -eq 0) {{ exit 2 }}; "
    "'{{0}} {{1}}' -f [int64]($d.Sum/1MB), [int64]($s.Sum/1MB)"
)


def query_placement(pid: Any, run: Callable[..., Any] = subprocess.run) -> str | None:
    """Raw "dedicated shared" MB text for one pid, or None when it cannot be read.

    Windows-only by nature (the counters are WDDM's); on another platform, a
    missing PowerShell, a pid that has exited, or a counter set that is not
    installed all come back as None - "cannot tell", never a fabricated zero.
    """
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return None
    if pid_int <= 0 or not shutil.which("powershell"):
        return None
    try:
        proc = run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             _PLACEMENT_SCRIPT.format(pid=pid_int)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(proc, "returncode", 1) != 0:
        return None
    return getattr(proc, "stdout", None)


def parse_placement(pid: Any, text: str | None) -> GpuPlacement | None:
    """Turn the counter text into a GpuPlacement; None when it is not two ints."""
    parts = (text or "").split()
    if len(parts) != 2:
        return None
    try:
        dedicated, shared = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if dedicated < 0 or shared < 0:
        return None
    return GpuPlacement(pid=int(pid), dedicated_mb=dedicated, shared_mb=shared)


def placement_of(pid: Any, run: Callable[..., Any] = subprocess.run) -> GpuPlacement | None:
    """Measure one process's GPU placement; None when it cannot be measured."""
    try:
        return parse_placement(pid, query_placement(pid, run=run))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# The fit margin llama-server needs on THIS card, right now (owner rules
# 2026-09-03: "do not affect my tok/s", "load as much to the GPU").
#
# llama-server's --fit measures free device memory with cudaMemGetInfo and
# keeps layers on the CPU until the model fits under a margin (--fit-target,
# engine default 1024 MiB). On Windows that reading is not what it says: it
# came back 14923 MiB free in thirteen logged loads while the GPU voice engine
# held 0, 594 and 948 MB, and 13293 free while another llama-server held
# 14112 MB (nvidia-smi: 362 free). WDDM lets the driver page other processes'
# memory out, so the engine cannot see them, and whatever margin it is given
# stands in for "what everyone else holds" - a number that changes as the
# session goes on.
#
# So the margin is computed from two real readings taken just before launch:
#   engine_free  what fit will see          (llama-server --list-devices)
#   real_free    what the card actually has (nvidia-smi adapter used/total)
#   target       = engine_free - real_free + FIT_SAFETY_MIB, floored at 0
# i.e. fit's budget lands FIT_SAFETY_MIB under the card's nominal free memory.
# Either reading missing means None and the engine keeps its own default
# margin - the safe answer, never a guess.
#
# FIT_SAFETY_MIB is the part of the card the model cannot have even when
# nothing else holds it, plus one layer of slack. Measured on the dense 27B
# (66 layers, ~180 MiB each), 1580 MB held by other processes, margin swept
# in 128 MiB steps; "shared" is the process's memory paged through system RAM:
#
#   layers on GPU   dedicated   shared   gen tok/s
#   64              14157       822      4.6      crawl
#   63              14155       646      6.7      crawl
#   62              14087       536      27-29    fast, but see below
#   61              14088       344      25
#   60              14096       158      26       zero paging, card full
#   59              13900       158      24
#
# Three facts in that table:
# - dedicated never passes ~14100 MB: with 1580 held elsewhere the card's
#   usable ceiling is ~15700 MB of its 16303, the rest Windows keeps for
#   itself. fit does not know that either; its 1024 default happens to cover
#   it, 256 did not.
# - fit's own tally runs ~290 MiB under the process's real footprint (the CUDA
#   context and library workspaces are not in it), constant across the sweep.
# - the 62-layer row is fast because what got paged was the cold part of the
#   KV cache, and one layer more is a crawl. Paged memory never comes back:
#   the same 27B at 29 tok/s beside an idle voice engine ran 4.8 tok/s after
#   that engine spoke one paragraph, and still 4.8 twenty-five seconds after
#   the engine had shrunk again. A configuration one layer from the cliff is
#   not a configuration, so the target is the last zero-paging row plus one
#   layer for whatever the desktop allocates during the session:
#   ~600 (ceiling) + ~290 (tally) - ~150 (fit's own launch-time allocations,
#   already in its reading) + ~180 (one layer) = 920, rounded up to 1024 -
#   fit's own default, given the one number it cannot see.
#
# The GPU is as full as Windows allows in every row from 60 layers up; the
# only way to put more of a model on it is to free video memory - the voice
# engine's GPU build holds ~1.2 GB (see kokoro_server.py), a 49152-token
# context 1.6 GB of KV cache - and that is the owner's call, in models.yaml
# and the setup wizard, not a margin.
# --------------------------------------------------------------------------- #

FIT_SAFETY_MIB = 1024


@dataclass(frozen=True)
class FitBudget:
    """The two readings and the margin they give."""

    engine_total_mib: int
    engine_free_mib: int
    adapter_used_mib: int
    adapter_total_mib: int

    @property
    def real_free_mib(self) -> int:
        return self.adapter_total_mib - self.adapter_used_mib

    @property
    def target_mib(self) -> int:
        return max(0, self.engine_free_mib - self.real_free_mib + FIT_SAFETY_MIB)

    def describe(self) -> str:
        return (
            f"fit target {self.target_mib} MiB: the engine sees {self.engine_free_mib} MiB "
            f"free, the card has {self.real_free_mib} MiB ({self.adapter_used_mib} MiB held "
            f"by other processes)"
        )


def parse_adapter_memory(text: str | None) -> tuple[int, int] | None:
    """(used, total) MiB from `nvidia-smi --query-gpu=memory.used,memory.total
    --format=csv,noheader,nounits`; the first GPU when there are several."""
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            used, total = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if used < 0 or total <= 0 or used > total:
            return None
        return used, total
    return None


def query_adapter_memory(
    run: Callable[..., Any] = subprocess.run, smi: str | None = None
) -> str | None:
    """Raw adapter memory CSV from nvidia-smi, or None. Never raises."""
    exe = smi or shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        proc = run(
            [exe, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(proc, "returncode", 1) != 0:
        return None
    return getattr(proc, "stdout", None)


def parse_engine_free(text: str | None) -> tuple[int, int] | None:
    """(total, free) MiB for the first CUDA device in `llama-server --list-devices`
    output, whose lines read `CUDA0: <name> (16302 MiB, 14923 MiB free)`."""
    import re

    for line in (text or "").splitlines():
        if "CUDA" not in line:
            continue
        m = re.search(r"\((\d+) MiB, (\d+) MiB free\)", line)
        if m:
            total, free = int(m.group(1)), int(m.group(2))
            if free > total:
                return None
            return total, free
    return None


def query_engine_free(binary: Any, run: Callable[..., Any] = subprocess.run) -> str | None:
    """Raw `--list-devices` output from the server binary that is about to load
    the model, or None. The same binary, so the reading is the one fit will
    take a second later."""
    try:
        path = Path(str(binary))
    except (TypeError, ValueError):
        return None
    if not binary or not path.is_file():
        return None
    try:
        proc = run(
            [str(path), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            cwd=str(path.parent),
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(proc, "returncode", 1) != 0:
        return None
    return (getattr(proc, "stdout", None) or "") + (getattr(proc, "stderr", None) or "")


def fit_budget(binary: Any, run: Callable[..., Any] = subprocess.run) -> FitBudget | None:
    """Both readings, taken now; None when either cannot be taken."""
    engine = parse_engine_free(query_engine_free(binary, run=run))
    if engine is None:
        return None
    adapter = parse_adapter_memory(query_adapter_memory(run=run))
    if adapter is None:
        return None
    return FitBudget(
        engine_total_mib=engine[0],
        engine_free_mib=engine[1],
        adapter_used_mib=adapter[0],
        adapter_total_mib=adapter[1],
    )
