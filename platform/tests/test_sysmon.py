"""Tests for live system-resource sampling (sysmon.py, M17.18).

The monitor's numbers must be right and must degrade honestly: RAM is always
read, VRAM is None when there is no readable GPU, and a provider that throws
never crashes the sampler.
"""

from __future__ import annotations

import sysmon
from sysmon import SystemSample, gather


class _Sys:
    def __init__(self, total, available):
        self._total, self._available = total, available

    def ram_total_mb(self):
        return self._total

    def ram_available_mb(self):
        return self._available


class _Gpu:
    def __init__(self, gpus):
        self._gpus = gpus

    def gpus(self):
        return self._gpus


class _GpuInfo:
    def __init__(self, total, free):
        self.vram_total_mb, self.vram_free_mb = total, free


def test_ram_used_is_total_minus_available():
    s = gather(_Gpu(None), _Sys(total=32000, available=20000))
    assert s.ram_used_mb == 12000
    assert s.ram_total_mb == 32000
    assert round(s.ram_pct, 1) == 37.5
    assert s.ram_free_mb == 20000


def test_vram_used_is_total_minus_free_of_largest_gpu():
    s = gather(_Gpu([_GpuInfo(16000, 4000)]), _Sys(32000, 16000))
    assert s.vram_used_mb == 12000
    assert s.vram_total_mb == 16000
    assert round(s.vram_pct, 1) == 75.0
    assert s.vram_free_mb == 4000
    assert s.has_gpu is True


def test_no_gpu_yields_none_vram_not_zero():
    s = gather(_Gpu(None), _Sys(32000, 16000))
    assert s.vram_used_mb is None and s.vram_total_mb is None
    assert s.vram_pct is None and s.vram_free_mb is None
    assert s.has_gpu is False


def test_largest_gpu_is_chosen():
    s = gather(_Gpu([_GpuInfo(8000, 8000), _GpuInfo(24000, 6000)]), _Sys(32000, 16000))
    assert s.vram_total_mb == 24000
    assert s.vram_used_mb == 18000


def test_provider_errors_degrade_to_no_vram():
    class _Boom:
        def gpus(self):
            raise OSError("nvidia-smi gone")

    s = gather(_Boom(), _Sys(32000, 16000))
    assert s.vram_used_mb is None
    assert s.ram_used_mb == 16000  # RAM still read


def test_pct_is_clamped_and_safe_on_zero_total():
    s = SystemSample(ram_used_mb=5, ram_total_mb=0, vram_used_mb=None, vram_total_mb=None)
    assert s.ram_pct == 0.0  # no divide-by-zero


def test_format_gb():
    assert sysmon.format_gb(None) == "-"
    assert sysmon.format_gb(16303) == "15.9 GB"
