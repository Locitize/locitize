"""Live system-resource sampling for the monitor view (M17.18).

A task-manager-style read of RAM and VRAM: how much is in use, how much is free,
and (over time) how that moves as a model loads and runs. The gathering is a
pure function over the SAME injected providers the rest of the app uses
(health.py's GPU/system seams), so the monitor's numbers are testable without a
real GPU or psutil, and degrade honestly - a machine with no NVIDIA GPU reports
None for VRAM rather than a fabricated figure.

Nothing here draws anything or starts a thread; desktop.py owns the sampling
cadence and the widgets. This module only answers "what are the numbers right
now". ASCII only.
"""

from __future__ import annotations

from dataclasses import dataclass

MB_PER_GB = 1024.0


@dataclass
class SystemSample:
    """One instantaneous read of memory pressure. VRAM fields are None when the
    machine has no readable NVIDIA GPU."""

    ram_used_mb: float
    ram_total_mb: float
    vram_used_mb: float | None
    vram_total_mb: float | None

    @property
    def ram_pct(self) -> float:
        return _pct(self.ram_used_mb, self.ram_total_mb)

    @property
    def ram_free_mb(self) -> float:
        return max(0.0, self.ram_total_mb - self.ram_used_mb)

    @property
    def vram_pct(self) -> float | None:
        if self.vram_used_mb is None or self.vram_total_mb is None:
            return None
        return _pct(self.vram_used_mb, self.vram_total_mb)

    @property
    def vram_free_mb(self) -> float | None:
        if self.vram_used_mb is None or self.vram_total_mb is None:
            return None
        return max(0.0, self.vram_total_mb - self.vram_used_mb)

    @property
    def has_gpu(self) -> bool:
        return self.vram_used_mb is not None and self.vram_total_mb is not None


def _pct(used: float, total: float) -> float:
    if not total or total <= 0:
        return 0.0
    return max(0.0, min(100.0, (used / total) * 100.0))


def gather(gpu_provider, sys_provider) -> SystemSample:
    """Read RAM (always) and VRAM (when an NVIDIA GPU is readable) right now.

    RAM used is total-minus-available (available, not free, so cache the OS can
    reclaim is not counted as "used" - the figure Task Manager shows). VRAM used
    is total-minus-free from the largest GPU, since a model runs on one card and
    the largest is the one it will use. Never raises: a provider that errors or
    returns nothing yields a sample with VRAM None (RAM is required and always
    available via psutil).
    """
    total = float(sys_provider.ram_total_mb())
    try:
        available = float(sys_provider.ram_available_mb())
    except Exception:  # noqa: BLE001 - RAM read must not break the monitor
        available = total
    used = max(0.0, total - available)

    vram_used: float | None = None
    vram_total: float | None = None
    try:
        gpus = gpu_provider.gpus() if gpu_provider is not None else None
    except Exception:  # noqa: BLE001 - honest degrade to "no GPU reading"
        gpus = None
    if gpus:
        primary = max(gpus, key=lambda g: getattr(g, "vram_total_mb", 0) or 0)
        vram_total = float(primary.vram_total_mb)
        vram_used = max(0.0, vram_total - float(primary.vram_free_mb))

    return SystemSample(
        ram_used_mb=used,
        ram_total_mb=total,
        vram_used_mb=vram_used,
        vram_total_mb=vram_total,
    )


def format_gb(mb: float | None) -> str:
    """'12.3 GB', or '-' when unknown."""
    if mb is None:
        return "-"
    return f"{mb / MB_PER_GB:.1f} GB"
