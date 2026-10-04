"""Structured PASS / WARNING / FAIL health checks for the LOCITIZE platform.

This is the platform's core observability layer (Architecture section 5). It runs
a roster of probes (python, virtual_env, gpu, cuda, vram, ram, disk_space, ports,
whisper, llama_cpp, kokoro, voice, models) and aggregates them into a HealthReport
with a worst-of overall status.

The key design constraint (Architecture section 5.2, AC6): probes never call
nvidia-smi / psutil / the filesystem inline. They receive provider objects that
are dependency-injected. In production the real providers are used; in tests fake
providers (e.g. a GpuInfoProvider that returns None) let the whole report be
asserted with NO GPU, llama.cpp, or model file present. That is how the platform
is proven testable on any machine, GPU or not.

A probe never raises for an expected-absent dependency; it catches its own error
and returns FAIL/WARNING with a remedy string, so the launcher never shows a raw
traceback for an expected failure (Architecture section 8).
"""

from __future__ import annotations

import json
import time
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from config import Model, Settings, model_vram_need_mb


class HealthStatus(Enum):
    """A probe outcome. The literal strings are the enum values (AC3 wants PASS)."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


# Ordering used to compute the worst-of overall status. Higher = worse.
_SEVERITY = {HealthStatus.PASS: 0, HealthStatus.WARNING: 1, HealthStatus.FAIL: 2}

# Owner request 2026-08-21 (black-flash-on-launch root cause): nvidia-smi.exe is
# a CONSOLE app; the desktop GUI runs under pythonw.exe, which has no console of
# its own. Without CREATE_NO_WINDOW, Windows allocates - and the default
# terminal host visibly shows - a brand-new console for nvidia-smi to attach to,
# every single time this probe runs (twice per health check: gpus() then
# _cuda_version()), even though its stdout/stderr are already being captured.
# That is the actual flash the owner was seeing on every launch, not a Qt
# render-order issue and not the OS window-open animation (both already
# investigated and ruled out first).
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

# Single-sourced ASCII status marks (Architecture section 5.4). No glyphs/emojis.
_MARKS = {
    HealthStatus.PASS: "[OK]",
    HealthStatus.WARNING: "[!!]",
    HealthStatus.FAIL: "[XX]",
}


def status_mark(status: HealthStatus) -> str:
    """Map a status to its canonical ASCII mark ([OK] / [!!] / [XX])."""
    return _MARKS[status]


@dataclass
class HealthResult:
    """The outcome of a single probe."""

    name: str
    status: HealthStatus
    detail: str
    remedy: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
            "remedy": self.remedy,
            "data": self.data,
        }


@dataclass
class HealthReport:
    """A full run of the probe roster with a computed overall status."""

    results: list[HealthResult]
    overall: HealthStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.value,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def worst_of(results: list[HealthResult]) -> HealthStatus:
    """Return the worst status across results (any FAIL -> FAIL, etc.)."""
    if not results:
        return HealthStatus.PASS
    return max((r.status for r in results), key=lambda s: _SEVERITY[s])


# --------------------------------------------------------------------------- #
# Injected system-fact providers (the testability seam)
# --------------------------------------------------------------------------- #


@dataclass
class GpuInfo:
    """One GPU's facts as reported by nvidia-smi (or a test stub)."""

    name: str
    vram_total_mb: float
    vram_free_mb: float
    cuda_version: str | None = None


@runtime_checkable
class SystemInfoProvider(Protocol):
    """Host CPU/RAM/disk/python facts. Real impl wraps sys + psutil."""

    def python_version(self) -> tuple[int, int, int]: ...
    def in_virtualenv(self) -> bool: ...
    def ram_total_mb(self) -> float: ...
    def ram_available_mb(self) -> float: ...
    def disk_free_mb(self, path: str) -> float: ...


@runtime_checkable
class GpuInfoProvider(Protocol):
    """GPU facts. gpus() returns None when there is no GPU/driver at all."""

    def gpus(self) -> list[GpuInfo] | None: ...


@runtime_checkable
class BinaryProbeProvider(Protocol):
    """Existence/runnability of external binaries (llama.cpp, Whisper, Kokoro)."""

    def exists(self, path: str) -> bool: ...
    def runnable(self, path: str) -> bool: ...


@runtime_checkable
class PortProbeProvider(Protocol):
    """Whether a loopback port is currently free to bind."""

    def is_free(self, port: int) -> bool: ...


# --------------------------------------------------------------------------- #
# Default (production) provider implementations
# --------------------------------------------------------------------------- #


class DefaultSystemInfoProvider:
    """Real host facts via sys + psutil. Instantiated only in production."""

    def python_version(self) -> tuple[int, int, int]:
        v = sys.version_info
        return (v.major, v.minor, v.micro)

    def in_virtualenv(self) -> bool:
        # base_prefix differs from prefix inside a venv/virtualenv.
        return sys.prefix != getattr(sys, "base_prefix", sys.prefix)

    def ram_total_mb(self) -> float:
        import psutil

        return psutil.virtual_memory().total / (1024 * 1024)

    def ram_available_mb(self) -> float:
        import psutil

        return psutil.virtual_memory().available / (1024 * 1024)

    def disk_free_mb(self, path: str) -> float:
        usage = shutil.disk_usage(path)
        return usage.free / (1024 * 1024)


class NvidiaSmiGpuInfoProvider:
    """GPU facts by parsing nvidia-smi CSV output.

    nvidia-smi ships with the NVIDIA driver, so this needs no pynvml dependency
    (Architecture section 13). Returns None when nvidia-smi is absent or errors,
    which the gpu/cuda/vram probes read as "no GPU" and report honestly.
    """

    # Perf audit 2026-08-31: one health run calls gpus() from three probes
    # back to back, and the sysmon loop calls it every 1.5s - each call
    # used to spawn nvidia-smi TWICE (the CSV query plus a bare run just
    # to parse the CUDA version). A short same-instance snapshot collapses
    # a probe ladder to one spawn pair, and the CUDA version - which
    # cannot change while the process runs - is fetched exactly once.
    _SNAPSHOT_TTL_S = 1.0

    def __init__(self) -> None:
        self._snapshot: tuple[float, list[GpuInfo] | None] | None = None
        self._cuda_cached = False
        self._cuda_value: str | None = None

    def gpus(self) -> list[GpuInfo] | None:
        snap = self._snapshot
        now = time.monotonic()
        if snap is not None and now - snap[0] < self._SNAPSHOT_TTL_S:
            return snap[1]
        result = self._gpus_uncached()
        self._snapshot = (now, result)
        return result

    def _gpus_uncached(self) -> list[GpuInfo] | None:
        smi = shutil.which("nvidia-smi")
        if smi is None:
            return None
        query = "name,memory.total,memory.free"
        try:
            proc = subprocess.run(
                [smi, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                creationflags=_NO_WINDOW,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        cuda_version = self._cuda_version(smi)
        gpus: list[GpuInfo] = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                gpus.append(
                    GpuInfo(
                        name=parts[0],
                        vram_total_mb=float(parts[1]),
                        vram_free_mb=float(parts[2]),
                        cuda_version=cuda_version,
                    )
                )
            except ValueError:
                # Skip a malformed row rather than failing the whole probe.
                continue
        return gpus or None

    def _cuda_version(self, smi: str) -> str | None:
        """The CUDA version, fetched once per provider (it cannot change).

        Reads nvidia-smi's default text output header on the first call
        only; every later call answers from the instance cache."""
        if self._cuda_cached:
            return self._cuda_value
        self._cuda_value = self._cuda_version_uncached(smi)
        self._cuda_cached = True
        return self._cuda_value

    def _cuda_version_uncached(self, smi: str) -> str | None:
        try:
            proc = subprocess.run(
                [smi], capture_output=True, text=True, timeout=10, check=False,
                creationflags=_NO_WINDOW,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        for token in proc.stdout.split():
            # The header reads "... CUDA Version: 12.4 ..."; capture the number.
            if token.replace(".", "", 1).isdigit() and "." in token:
                # Heuristic: only accept it if "CUDA" appears in the output.
                if "CUDA Version" in proc.stdout:
                    idx = proc.stdout.find("CUDA Version")
                    tail = proc.stdout[idx : idx + 40]
                    for tok in tail.replace(":", " ").split():
                        if tok.replace(".", "", 1).isdigit() and "." in tok:
                            return tok
                    return None
        return None


class DefaultBinaryProbeProvider:
    """Real filesystem/PATH checks for external binaries."""

    def exists(self, path: str) -> bool:
        if not path:
            return False
        # Accept either an absolute path or a name resolvable on PATH.
        return Path(path).exists() or shutil.which(path) is not None

    def runnable(self, path: str) -> bool:
        # Milestone 1 treats "resolvable and marked executable" as runnable; we
        # do not actually execute the binary here to avoid side effects during a
        # health check. Deeper --version probing is a later refinement.
        if not path:
            return False
        resolved = shutil.which(path)
        if resolved is not None:
            return True
        p = Path(path)
        return p.exists() and p.is_file()


class DefaultPortProbeProvider:
    """Real loopback free-port check via socket bind."""

    def is_free(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                # Bind to loopback only (Permission Matrix section 3).
                sock.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False


class DefaultPortOwnerProvider:
    """Which executable is listening on a loopback port, or "" when unknown.

    Owner-observed 2026-09-03: the ports probe reported "ports in use:
    llama_cpp:8080" while LOCITIZE was correctly serving a model on 8080. A
    port held by our OWN service is not a conflict, and reporting it as one
    made the health light amber whenever the platform was working.

    Returns "" for anything it cannot determine - another user's process, a
    permission refusal, psutil missing. The caller treats "" as FOREIGN, which
    keeps the old, stricter behaviour whenever ownership is not provable. A
    port wrongly called ours would hide a real conflict; a port wrongly called
    foreign only shows a warning the owner can read.
    """

    def listener_exe(self, port: int) -> str:
        """The listener's executable AND its command line, joined.

        Both, because not every LOCITIZE service is a binary under our bin/.
        Owner-observed 2026-09-03 (second pass): the probe still warned
        "ports in use: kokoro:8092" while Kokoro was correctly serving. Kokoro
        is a PYTHON service - the executable is the venv interpreter, which
        lives nowhere near locitize-data/bin - so an exe-only test could never
        recognise it. Its command line names kokoro_server.py and the model
        under the data root, which is what identifies it.
        """
        try:
            import psutil

            for conn in psutil.net_connections(kind="inet"):
                if (
                    conn.status == psutil.CONN_LISTEN
                    and conn.laddr
                    and conn.laddr.port == port
                    and conn.pid
                ):
                    proc = psutil.Process(conn.pid)
                    parts = [proc.exe() or ""]
                    try:
                        parts.extend(proc.cmdline() or [])
                    except Exception:  # noqa: BLE001 - exe alone still helps
                        pass
                    return " ".join(p for p in parts if p)
        except Exception:  # noqa: BLE001 - a probe never crashes the ladder
            return ""
        return ""


@dataclass
class HealthProviders:
    """Bundle of the providers so the checker takes one injectable object."""

    system: SystemInfoProvider
    gpu: GpuInfoProvider
    binary: BinaryProbeProvider
    port: PortProbeProvider
    # Who is listening on an occupied port. Injectable like the rest, and
    # defaulted rather than required so every existing caller is unchanged.
    #
    # It has to be injectable: PortsProbe consulting the REAL machine while the
    # port provider is a fake made test_ports_probe_warns_when_occupied pass or
    # fail depending on whether the developer happened to be serving a model on
    # 8080 - the "works on my machine" failure M14 exists to eliminate.
    port_owner: Any = None

    @staticmethod
    def defaults() -> "HealthProviders":
        """Build the production providers (real hardware/binary/port access)."""
        return HealthProviders(
            system=DefaultSystemInfoProvider(),
            gpu=NvidiaSmiGpuInfoProvider(),
            binary=DefaultBinaryProbeProvider(),
            port=DefaultPortProbeProvider(),
            port_owner=DefaultPortOwnerProvider(),
        )


# --------------------------------------------------------------------------- #
# Probe protocol and the roster
# --------------------------------------------------------------------------- #


@runtime_checkable
class Probe(Protocol):
    """A single named check. run() must never raise for an expected failure."""

    name: str

    def run(self) -> HealthResult: ...


def _min_version(spec: str) -> tuple[int, int]:
    """Parse a 'MAJOR.MINOR' string into a comparable tuple, defaulting to 3.11."""
    try:
        major, minor = (int(x) for x in spec.split(".")[:2])
        return (major, minor)
    except (ValueError, AttributeError):
        return (3, 11)


class PythonProbe:
    name = "python"

    def __init__(self, system: SystemInfoProvider, min_version: str) -> None:
        self._system = system
        self._min = _min_version(min_version)

    def run(self) -> HealthResult:
        major, minor, micro = self._system.python_version()
        version_str = f"{major}.{minor}.{micro}"
        data = {"version": version_str}
        # Hard floor is 3.10; 3.10 is a WARNING, below 3.10 is a FAIL.
        if (major, minor) >= self._min:
            return HealthResult(self.name, HealthStatus.PASS, f"Python {version_str}", None, data)
        if (major, minor) == (3, 10):
            return HealthResult(
                self.name,
                HealthStatus.WARNING,
                f"Python {version_str} is below the recommended {self._min[0]}.{self._min[1]}",
                "install Python 3.11+",
                data,
            )
        return HealthResult(
            self.name,
            HealthStatus.FAIL,
            f"Python {version_str} is too old",
            "install Python 3.11+",
            data,
        )


class VirtualEnvProbe:
    name = "virtual_env"

    def __init__(self, system: SystemInfoProvider) -> None:
        self._system = system

    def run(self) -> HealthResult:
        if self._system.in_virtualenv():
            return HealthResult(self.name, HealthStatus.PASS, "running inside a virtual environment")
        return HealthResult(
            self.name,
            HealthStatus.FAIL,
            "not running inside a virtual environment",
            "activate the project venv before launching",
        )


class GpuProbe:
    name = "gpu"

    def __init__(self, gpu: GpuInfoProvider) -> None:
        self._gpu = gpu

    def run(self) -> HealthResult:
        gpus = self._gpu.gpus()
        if not gpus:
            return HealthResult(
                self.name,
                HealthStatus.FAIL,
                "no CUDA-capable GPU or driver detected",
                "check the NVIDIA driver / nvidia-smi is on PATH",
            )
        names = ", ".join(g.name for g in gpus)
        return HealthResult(
            self.name,
            HealthStatus.PASS,
            f"{len(gpus)} GPU(s): {names}",
            None,
            {"count": len(gpus), "names": [g.name for g in gpus]},
        )


class CudaProbe:
    name = "cuda"

    def __init__(self, gpu: GpuInfoProvider) -> None:
        self._gpu = gpu

    def run(self) -> HealthResult:
        gpus = self._gpu.gpus()
        if not gpus:
            return HealthResult(
                self.name,
                HealthStatus.FAIL,
                "no CUDA runtime (no GPU/driver)",
                "install a CUDA-capable NVIDIA driver",
            )
        cuda_version = next((g.cuda_version for g in gpus if g.cuda_version), None)
        if cuda_version:
            return HealthResult(
                self.name,
                HealthStatus.PASS,
                f"CUDA {cuda_version} reported by driver",
                None,
                {"cuda_version": cuda_version},
            )
        # Driver present but the CUDA version could not be read: usable but unknown.
        return HealthResult(
            self.name,
            HealthStatus.WARNING,
            "GPU present but CUDA version could not be determined",
            "verify the driver exposes a CUDA version via nvidia-smi",
        )


class VramProbe:
    name = "vram"

    def __init__(
        self,
        gpu: GpuInfoProvider,
        models: list[Model],
        headroom_mb: int,
        reclaimable_mb_fn: Any = None,
    ) -> None:
        self._gpu = gpu
        self._models = models
        self._headroom = headroom_mb
        # VRAM held by LOCITIZE's OWN servers, which a model switch releases.
        # Owner-observed 2026-09-03: with a 12GB model correctly loaded and
        # serving, this probe reported "free VRAM 1996MB is tight for the
        # largest model" and the window sat on WARNING - the platform working
        # exactly as designed, reported as a fault. A health light that is
        # amber whenever the product is doing its job teaches the owner to
        # ignore it. None -> 0, so a caller that cannot tell ours from a
        # foreign app keeps the old strict behaviour rather than guessing.
        self._reclaimable_mb_fn = reclaimable_mb_fn

    def run(self) -> HealthResult:
        gpus = self._gpu.gpus()
        if not gpus:
            return HealthResult(
                self.name,
                HealthStatus.FAIL,
                "no GPU VRAM available",
                "install a GPU / driver, or run CPU-only models",
            )
        # Use the GPU with the most free VRAM as the working target.
        target = max(gpus, key=lambda g: g.vram_free_mb)
        # Smallest installed model's need is the minimum bar to clear. The need
        # comes from model_vram_need_mb (the on-disk .gguf size when the file is
        # there, else the row's estimate), NOT from vram_estimate_mb alone: a
        # scan-imported registry carries vram_estimate_mb: 0 on every row, which
        # emptied this list, pinned smallest/largest to 0, made the FAIL branch
        # below unreachable, and printed "tight for the largest model (0MB)".
        installed = [m for m in self._models if m.status == "installed"]
        estimates = [n for n in (model_vram_need_mb(m) for m in installed) if n > 0]
        smallest = min(estimates) if estimates else 0
        largest = max(estimates) if estimates else 0
        ours = 0.0
        ours_running = False
        if self._reclaimable_mb_fn is not None:
            try:
                ours, ours_running = self._reclaimable_mb_fn()
                ours = float(ours or 0.0)
            except Exception:  # noqa: BLE001 - a probe never crashes the ladder
                ours, ours_running = 0.0, False
        # What a model start can ACTUALLY have: free VRAM plus whatever
        # LOCITIZE is holding, because starting a model stops the current one
        # first (the M8.2 one-model-at-a-time discipline).
        basis = "free"
        if ours > 0:
            effective_free = target.vram_free_mb + ours
            basis = "free+ours"
        elif ours_running:
            # Our server IS on the GPU but nvidia-smi will not say how much it
            # holds - on this machine (RTX 5070 Ti under WDDM) every
            # per-process used-memory field comes back [N/A]. Judging against
            # CURRENT free VRAM would then permanently warn about the model we
            # ourselves loaded, which is the defect being fixed. Total is the
            # honest fallback: it slightly OVERSTATES (a foreign app may hold
            # some too), so the detail line says which basis was used rather
            # than presenting an assumption as a measurement.
            effective_free = target.vram_total_mb
            basis = "total (per-process VRAM unavailable)"
        else:
            effective_free = target.vram_free_mb
        data = {
            "vram_total_mb": target.vram_total_mb,
            "vram_free_mb": target.vram_free_mb,
            "locitize_held_mb": round(ours),
            "locitize_running": ours_running,
            "effective_free_mb": round(effective_free),
            "basis": basis,
            "smallest_model_mb": round(smallest),
            "largest_model_mb": round(largest),
        }
        if target.vram_total_mb < smallest:
            return HealthResult(
                self.name,
                HealthStatus.FAIL,
                f"total VRAM {target.vram_total_mb:.0f}MB is below the smallest model ({smallest:.0f}MB)",
                "use a smaller/more-quantized model",
                data,
            )
        # WARNING only when even RECLAIMING our own model would not fit the
        # largest one plus headroom - that is a real constraint the owner can
        # act on. Being full of our own model is not.
        held = f" ({ours:.0f}MB of it held by locitize)" if ours else ""
        if effective_free < largest + self._headroom:
            return HealthResult(
                self.name,
                HealthStatus.WARNING,
                f"usable VRAM {effective_free:.0f}MB{held} is tight for the "
                f"largest model ({largest:.0f}MB)",
                "close other GPU apps or pick a smaller model",
                data,
            )
        if ours or ours_running:
            extra = (
                f"plus {ours:.0f}MB held by locitize"
                if ours
                else "the rest held by locitize's own model"
            )
            return HealthResult(
                self.name,
                HealthStatus.PASS,
                f"free VRAM {target.vram_free_mb:.0f}MB of "
                f"{target.vram_total_mb:.0f}MB, {extra} (released on switch)",
                None,
                data,
            )
        return HealthResult(
            self.name,
            HealthStatus.PASS,
            f"free VRAM {target.vram_free_mb:.0f}MB of {target.vram_total_mb:.0f}MB",
            None,
            data,
        )


class RamProbe:
    name = "ram"

    def __init__(self, system: SystemInfoProvider, floor_mb: int, hard_min_mb: int) -> None:
        self._system = system
        self._floor = floor_mb
        self._hard_min = hard_min_mb

    def run(self) -> HealthResult:
        available = self._system.ram_available_mb()
        total = self._system.ram_total_mb()
        data = {"ram_available_mb": available, "ram_total_mb": total}
        if available < self._hard_min:
            return HealthResult(
                self.name,
                HealthStatus.FAIL,
                f"available RAM {available:.0f}MB below hard minimum {self._hard_min}MB",
                "close memory-heavy applications",
                data,
            )
        if available < self._floor:
            return HealthResult(
                self.name,
                HealthStatus.WARNING,
                f"available RAM {available:.0f}MB below recommended floor {self._floor}MB",
                "close memory-heavy applications",
                data,
            )
        return HealthResult(
            self.name, HealthStatus.PASS, f"available RAM {available:.0f}MB", None, data
        )


class DiskProbe:
    name = "disk_space"

    def __init__(
        self, system: SystemInfoProvider, path: str, floor_mb: int, hard_min_mb: int
    ) -> None:
        self._system = system
        self._path = path
        self._floor = floor_mb
        self._hard_min = hard_min_mb

    def run(self) -> HealthResult:
        try:
            free = self._system.disk_free_mb(self._path)
        except OSError as exc:
            return HealthResult(
                self.name,
                HealthStatus.FAIL,
                f"could not read disk usage for {self._path}: {exc}",
                "verify the platform directory is accessible",
            )
        data = {"disk_free_mb": free, "path": self._path}
        if free < self._hard_min:
            return HealthResult(
                self.name,
                HealthStatus.FAIL,
                f"free disk {free:.0f}MB below hard minimum {self._hard_min}MB",
                "free disk space",
                data,
            )
        if free < self._floor:
            return HealthResult(
                self.name,
                HealthStatus.WARNING,
                f"free disk {free:.0f}MB below recommended floor {self._floor}MB",
                "free disk space",
                data,
            )
        return HealthResult(
            self.name, HealthStatus.PASS, f"free disk {free:.0f}MB", None, data
        )


class PortsProbe:
    name = "ports"

    def __init__(
        self,
        port_provider: PortProbeProvider,
        ports: dict[str, int],
        owner_provider: Any = None,
        own_bin_dir: str = "",
    ) -> None:
        self._provider = port_provider
        self._ports = ports
        # Together these decide whether an occupied port is OURS. Same test
        # gpu_ledger.mark_ours uses: the executable lives under the data root's
        # bin/, where LOCITIZE puts llama-server. Absent -> every occupied port
        # is treated as foreign, the pre-2026-09-03 behaviour.
        self._owner_provider = owner_provider
        self._own_bin_dir = (
            str(own_bin_dir).replace(chr(92), "/").rstrip("/").lower()
            if own_bin_dir
            else ""
        )
        # The data ROOT (bin/ minus the trailing component). Every LOCITIZE
        # service references it somewhere in its argv, whether or not the
        # executable itself lives under bin/.
        self._own_root = (
            self._own_bin_dir.rsplit("/", 1)[0] if "/" in self._own_bin_dir
            else self._own_bin_dir
        )

    def _is_ours(self, port: int) -> bool:
        """True only when the listener is provably one of our own services.

        Matched against the DATA ROOT rather than just bin/, because a LOCITIZE
        service is not always a binary we ship: Kokoro runs on the venv
        interpreter and is identified by the kokoro_server.py and model paths
        on its command line, all of which sit under the data root. bin/ is a
        subdirectory of it, so whisper and llama-server still match.
        """
        if self._owner_provider is None or not self._own_root:
            return False
        try:
            described = self._owner_provider.listener_exe(port) or ""
        except Exception:  # noqa: BLE001 - a probe never crashes the ladder
            return False
        if not described:
            return False
        return self._own_root in described.replace(chr(92), "/").lower()

    def run(self) -> HealthResult:
        blocked = []
        ours = []
        for label, port in self._ports.items():
            if self._provider.is_free(port):
                continue
            if self._is_ours(port):
                ours.append(f"{label}:{port}")
            else:
                blocked.append(f"{label}:{port}")
        data = {"checked": self._ports, "blocked": blocked, "ours": ours}
        if not blocked:
            detail = f"all {len(self._ports)} reserved ports free"
            if ours:
                # Serving on a reserved port IS the healthy state; say so
                # rather than silently calling it "free".
                served = ", ".join(ours)
                detail = (
                    f"{len(self._ports)} reserved ports available; "
                    f"{served} served by locitize"
                )
            return HealthResult(self.name, HealthStatus.PASS, detail, None, data)
        # Ports being occupied is a WARNING under the default 'auto' policy: the
        # allocator can reassign within range. A strict-policy conflict surfaces
        # at service start, not here.
        return HealthResult(
            self.name,
            HealthStatus.WARNING,
            f"ports in use: {', '.join(blocked)}",
            "free the port(s) or rely on auto allocation within 8080-8099",
            data,
        )


class BinaryProbe:
    """Generic binary-presence probe shared by whisper / llama_cpp / kokoro."""

    def __init__(
        self,
        name: str,
        binary: BinaryProbeProvider,
        path: str,
        remedy: str,
        missing_status: HealthStatus = HealthStatus.FAIL,
        unconfigured_status: HealthStatus | None = None,
    ) -> None:
        self.name = name
        self._binary = binary
        self._path = path
        self._remedy = remedy
        # kokoro is optional, so its "missing" case is WARNING not FAIL.
        self._missing_status = missing_status
        # An optional component nobody has set up yet (empty path) is not the
        # same as a configured path that is broken; it defaults to missing_status.
        self._unconfigured_status = unconfigured_status or missing_status

    def run(self) -> HealthResult:
        if not self._path:
            return HealthResult(
                self.name,
                self._unconfigured_status,
                f"{self.name} path not configured",
                self._remedy,
            )
        if not self._binary.exists(self._path):
            return HealthResult(
                self.name,
                self._missing_status,
                f"{self.name} binary not found at configured path",
                self._remedy,
            )
        if self._binary.runnable(self._path):
            return HealthResult(self.name, HealthStatus.PASS, f"{self.name} present and runnable")
        return HealthResult(
            self.name,
            HealthStatus.WARNING,
            f"{self.name} found but not verified runnable",
            self._remedy,
        )


class KokoroProbe:
    """Capability probe for Kokoro text-to-speech (voice OUT, Architecture M6.4).

    Upgraded from the M1 bare binary-presence check to a real capability check:
    PASS only when the model checkpoint AND voices dir exist on disk AND the
    `kokoro` package is importable. The two facts are injected so the probe is
    unit-testable across present/absent cases with no torch import and no
    filesystem: `package_present` is the find_spec result (computed cheaply with
    importlib.util.find_spec in run_all, never importing torch), and `path_exists`
    answers whether a path is on disk. Kokoro missing is always a WARNING, never a
    FAIL -- voice OUT is a degradable feature (Architecture sections 5.3, 8).
    """

    name = "kokoro"

    def __init__(
        self,
        model_path: str,
        voices_dir: str,
        enabled: bool,
        package_present: bool,
        path_exists: Any,
    ) -> None:
        self._model_path = model_path
        self._voices_dir = voices_dir
        self._enabled = enabled
        self._package_present = package_present
        self._exists = path_exists

    def run(self) -> HealthResult:
        if not self._enabled:
            return HealthResult(
                self.name,
                HealthStatus.WARNING,
                "text-to-speech disabled (tts.enabled=false)",
                "set tts.enabled: true in settings.yaml to enable voice OUT",
            )
        weights_ok = (
            bool(self._model_path)
            and bool(self._voices_dir)
            and self._exists(self._model_path)
            and self._exists(self._voices_dir)
        )
        if not weights_ok:
            return HealthResult(
                self.name,
                HealthStatus.WARNING,
                "kokoro weights not found",
                "set paths.kokoro_model (kokoro-v1_0.pth) and paths.kokoro_voices "
                "(the dir of voice .pt files) in settings.yaml",
            )
        if not self._package_present:
            return HealthResult(
                self.name,
                HealthStatus.WARNING,
                "kokoro weights present but the kokoro package is not installed",
                "pip install -r requirements.txt (CPU torch + kokoro)",
            )
        return HealthResult(
            self.name,
            HealthStatus.PASS,
            "kokoro text-to-speech ready (CPU)",
        )


class VoiceProbe:
    """Composite probe: voice needs Whisper (STT) and Kokoro (TTS).

    Depends on the already-computed whisper/kokoro results so it does not re-run
    binary checks. Whisper missing is FAIL (no STT); Kokoro missing is WARNING
    (TTS degrades gracefully, Architecture section 8).
    """

    name = "voice"

    def __init__(self, whisper: HealthResult, kokoro: HealthResult) -> None:
        self._whisper = whisper
        self._kokoro = kokoro

    def run(self) -> HealthResult:
        if self._whisper.status == HealthStatus.FAIL:
            return HealthResult(
                self.name,
                HealthStatus.FAIL,
                "voice unavailable: Whisper (speech-to-text) is missing",
                "see the whisper probe remedy",
            )
        if (
            self._whisper.status == HealthStatus.WARNING
            or self._kokoro.status != HealthStatus.PASS
        ):
            return HealthResult(
                self.name,
                HealthStatus.WARNING,
                "voice partially available (text-to-speech may be disabled)",
                "see the whisper/kokoro probe remedies",
            )
        return HealthResult(self.name, HealthStatus.PASS, "voice pipeline ready")


class ModelsProbe:
    name = "models"

    def __init__(self, models: list[Model]) -> None:
        self._models = models

    def run(self) -> HealthResult:
        # Only installed models must resolve on disk; future models are declared.
        installed = [m for m in self._models if m.status == "installed"]
        if not installed:
            return HealthResult(
                self.name,
                HealthStatus.WARNING,
                "no installed models declared in models.yaml",
                "add at least one model with status: installed",
            )
        missing = [m.id for m in installed if not m.location or not Path(m.location).exists()]
        data = {"installed": [m.id for m in installed], "missing": missing}
        if len(missing) == len(installed):
            return HealthResult(
                self.name,
                HealthStatus.FAIL,
                "no installed model file resolves on disk",
                "set each model's 'location' in models.yaml (or LOCITIZE_MODEL_*)",
                data,
            )
        if missing:
            return HealthResult(
                self.name,
                HealthStatus.WARNING,
                f"some model files missing: {', '.join(missing)}",
                "set the missing model locations in models.yaml",
                data,
            )
        return HealthResult(
            self.name,
            HealthStatus.PASS,
            f"all {len(installed)} installed model files resolve",
            None,
            data,
        )



# ---------------------------------------------------------------------------
# Stack liveness (production install/health). Separate from the hardware
# ladder: RAM/VRAM can FAIL while every managed listener is healthy.
# ---------------------------------------------------------------------------

# Agent Portal is a separate optional app, but the stack health command still probes it.
PORTAL_DEFAULT_PORT = 4200
PORTAL_HEALTH_PATH = "/api/health"

STACK_ENDPOINTS: list[dict[str, Any]] = [
    {"name": "llama", "port_attr": "llama_cpp", "path": "/health", "required": True},
    {"name": "router", "port_attr": "router", "path": "/v1/models", "required": True},
    {"name": "openwebui", "port_attr": "openwebui", "path": "/", "required": True},
    {"name": "whisper", "port_attr": "whisper", "path": "", "required": True},
    {"name": "kokoro", "port_attr": "kokoro", "path": "", "required": True},
    {"name": "portal", "port": PORTAL_DEFAULT_PORT, "path": PORTAL_HEALTH_PATH, "required": True},
]


def _tcp_open(host: str, port: int, timeout: float = 1.5) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_probe(host: str, port: int, path: str, timeout: float = 2.0) -> tuple[bool, str]:
    """Best-effort HTTP GET. Returns (ok, detail). Never raises."""
    if not path:
        return True, "tcp only"
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}{path}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - loopback only
            code = getattr(resp, "status", None) or resp.getcode()
            return 200 <= int(code) < 500, f"HTTP {code}"
    except urllib.error.HTTPError as exc:
        # 401/403 still proves the service is up (portal may need login).
        if 400 <= int(exc.code) < 500:
            return True, f"HTTP {exc.code}"
        return False, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - probe never crashes the ladder
        return False, f"http error: {type(exc).__name__}"


def resolve_stack_endpoints(settings: Settings | None = None) -> list[dict[str, Any]]:
    """Concrete endpoint list from settings (ports) + portal default."""
    ports = getattr(settings, "ports", None) if settings is not None else None
    defaults = {
        "llama_cpp": 8080,
        "router": 8093,
        "openwebui": 8096,
        "whisper": 8091,
        "kokoro": 8092,
    }
    out: list[dict[str, Any]] = []
    for spec in STACK_ENDPOINTS:
        if "port" in spec:
            port = int(spec["port"])
        else:
            attr = spec["port_attr"]
            port = int(getattr(ports, attr)) if ports is not None else defaults[attr]
        out.append(
            {
                "name": spec["name"],
                "host": "127.0.0.1",
                "port": port,
                "path": spec.get("path") or "",
                "required": bool(spec.get("required", True)),
            }
        )
    return out


def check_stack_liveness(
    settings: Settings | None = None,
    *,
    host: str = "127.0.0.1",
) -> HealthReport:
    """PASS/FAIL per managed listener. Exit-code ready via overall status."""
    results: list[HealthResult] = []
    for ep in resolve_stack_endpoints(settings):
        port = int(ep["port"])
        name = str(ep["name"])
        path = str(ep["path"] or "")
        listening = _tcp_open(host, port)
        data = {"host": host, "port": port, "path": path, "tcp": listening}
        if not listening:
            results.append(
                HealthResult(
                    name,
                    HealthStatus.FAIL,
                    f"127.0.0.1:{port} not listening",
                    f"start locitize (llama -> router -> OWUI); portal separately on :{PORTAL_DEFAULT_PORT}",
                    data,
                )
            )
            continue
        http_ok, http_detail = _http_probe(host, port, path)
        data["http"] = http_detail
        if path and not http_ok:
            results.append(
                HealthResult(
                    name,
                    HealthStatus.FAIL,
                    f"listening on {port} but {path} failed ({http_detail})",
                    "service may still be starting; retry in a few seconds",
                    data,
                )
            )
            continue
        detail = f"127.0.0.1:{port} listening"
        if path:
            detail = f"{detail}; {path} -> {http_detail}"
        results.append(HealthResult(name, HealthStatus.PASS, detail, None, data))
    return HealthReport(results=results, overall=worst_of(results))


def format_stack_health_table(report: HealthReport) -> str:
    """Human-readable PASS/FAIL lines for scripts/health.ps1 and --stack-health."""
    lines = ["locitize stack health", "-" * 40]
    for r in report.results:
        mark = "PASS" if r.status is HealthStatus.PASS else "FAIL"
        lines.append(f"[{mark}] {r.name}: {r.detail}")
        if r.remedy and r.status is not HealthStatus.PASS:
            lines.append(f"       remedy: {r.remedy}")
    lines.append("-" * 40)
    lines.append(f"Overall: {report.overall.value}")
    return "\n".join(lines)


class HealthChecker:
    """Builds and runs the probe roster, aggregating a HealthReport.

    Constructed with a Settings, the model list, and a HealthProviders bundle.
    In production HealthProviders.defaults() is passed; tests pass fakes so the
    whole roster runs with no real hardware.
    """

    def __init__(
        self,
        settings: Settings,
        models: list[Model],
        providers: HealthProviders,
    ) -> None:
        self._settings = settings
        self._models = models
        self._providers = providers

    def run_all(self) -> HealthReport:
        """Run every probe (never aborting early) and compute the overall status."""
        s = self._settings
        p = self._providers
        thr = s.thresholds
        results: list[HealthResult] = []

        # Independent probes first.
        results.append(PythonProbe(p.system, thr.python_min).run())
        results.append(VirtualEnvProbe(p.system).run())
        results.append(GpuProbe(p.gpu).run())
        results.append(CudaProbe(p.gpu).run())
        # VRAM LOCITIZE itself is holding is reclaimable: starting another
        # model stops the current one first. Same ours-vs-foreign test the
        # Offload GPU button uses (gpu_ledger.mark_ours), so the two surfaces
        # cannot disagree about which processes are ours.
        own_bin_dir = str(Path(s.data_dir) / "bin")

        def _locitize_vram_mb() -> tuple[float, bool]:
            """(VRAM our servers hold, whether any of them is on the GPU).

            The two are separate because nvidia-smi does not always report
            per-process memory: on this maintainer's RTX 5070 Ti every
            used-memory field comes back [N/A] while the process list itself is
            complete. Summing those Nones raised a TypeError that the probe's
            boundary guard turned into a silent 0 - which looked exactly like
            "we hold nothing" and kept the false warning alive. The flag lets
            the probe tell "we hold nothing" from "we hold an unknown amount".
            """
            import gpu_ledger

            rows = gpu_ledger.mark_ours(
                gpu_ledger.parse_compute_apps(gpu_ledger.query_compute_apps()),
                bin_dir=own_bin_dir,
            )
            mine = [r for r in rows if r.is_locitize]
            held = float(sum(r.used_mb for r in mine if r.used_mb))
            return held, bool(mine)

        results.append(
            VramProbe(
                p.gpu, self._models, thr.vram_headroom_mb, _locitize_vram_mb
            ).run()
        )
        results.append(RamProbe(p.system, thr.ram_floor_mb, thr.ram_hard_min_mb).run())
        results.append(
            DiskProbe(
                p.system,
                str(s.base_dir),
                thr.disk_floor_mb,
                thr.disk_hard_min_mb,
            ).run()
        )
        results.append(
            PortsProbe(
                p.port,
                {
                    "llama_cpp": s.ports.llama_cpp,
                    "router": s.ports.router,
                    "whisper": s.ports.whisper,
                    "kokoro": s.ports.kokoro,
                    "openwebui": s.ports.openwebui,
                },
                p.port_owner,
                own_bin_dir,
            ).run()
        )

        # Binary probes; voice depends on whisper + kokoro results.
        # Voice is optional (added later via locitize.vbs --setup), so a whisper
        # that was never set up is a WARNING; a configured path that is missing
        # stays a FAIL.
        whisper = BinaryProbe(
            "whisper",
            p.binary,
            s.paths.whisper,
            "set whisper path in settings.yaml (or add voice via locitize.vbs --setup)",
            unconfigured_status=HealthStatus.WARNING,
        ).run()
        llama = BinaryProbe(
            "llama_cpp", p.binary, s.paths.llama_cpp, "set llama.cpp path in settings.yaml"
        ).run()
        # M6: kokoro is now a capability probe (weights on disk + package
        # importable), not a bare binary presence check. find_spec is a cheap
        # import-resolution query that never imports torch, so the health ladder
        # stays fast and GPU-free. Package presence is computed here and injected.
        import importlib.util

        kokoro_present = importlib.util.find_spec("kokoro") is not None
        kokoro = KokoroProbe(
            s.paths.kokoro_model,
            s.paths.kokoro_voices,
            s.tts.enabled,
            kokoro_present,
            p.binary.exists,
        ).run()
        results.append(whisper)
        results.append(llama)
        results.append(kokoro)
        results.append(VoiceProbe(whisper, kokoro).run())
        results.append(ModelsProbe(self._models).run())

        return HealthReport(results=results, overall=worst_of(results))
