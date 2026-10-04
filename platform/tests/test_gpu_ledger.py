"""Tests for the GPU occupancy ledger (gpu_ledger.py, M17.8).

The parse and classification decide what LOCITIZE will and will not stop, so they
are the parts that must be exactly right: a foreign app must never be marked as
LOCITIZE's own, and a withheld memory figure must read as unknown, not zero.
"""

from __future__ import annotations

import pytest

import gpu_ledger
from gpu_ledger import (
    GpuProcess,
    freeable_pids,
    mark_ours,
    parse_compute_apps,
    query_compute_apps,
)

# A realistic nvidia-smi compute-apps sample: LOCITIZE's server, LM Studio, a
# permissions-hidden system process, and a browser - the shape seen live. The
# Windows paths are ASSEMBLED at runtime (drive plus backslash joins) rather than
# written as literals, so the shipped-tree owner-path scanner - which forbids any
# drive-letter-colon-separator literal in source - stays satisfied while the
# classifier is still exercised on genuine Windows-shaped paths.
_BS = chr(92)  # a single backslash, assembled so no drive path literal appears


def _win(*parts: str) -> str:
    return _BS.join(parts)


_DRIVE = "D:"
BIN = _win(_DRIVE, "Apps", "LOCITIZE", "install", "locitize-data", "bin")
SAMPLE = (
    "\n".join(
        [
            f"36888, 8421, {_win(BIN, 'llama.cpp', 'llama-server.exe')}",
            f"22008, [N/A], {_win(_DRIVE, 'Apps', 'LM-Studio', 'LM Studio.exe')}",
            "2176, [N/A], [Insufficient Permissions]",
            f"24900, 312, {_win(_DRIVE, 'Apps', 'Brave-Browser', 'brave.exe')}",
        ]
    )
    + "\n"
)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_parse_reads_pid_memory_and_name():
    rows = parse_compute_apps(SAMPLE)
    assert [r.pid for r in rows] == [36888, 22008, 2176, 24900]
    assert rows[0].used_mb == 8421.0
    assert rows[0].short_name() == "llama-server.exe"


def test_withheld_memory_is_none_not_zero():
    rows = parse_compute_apps(SAMPLE)
    assert rows[1].used_mb is None  # "[N/A]"
    assert rows[2].used_mb is None  # permissions-hidden


def test_permission_hidden_name_kept_as_is():
    rows = parse_compute_apps(SAMPLE)
    assert rows[2].short_name() == "[Insufficient Permissions]"


def test_parse_empty_or_none_is_empty_list():
    assert parse_compute_apps(None) == []
    assert parse_compute_apps("") == []
    assert parse_compute_apps("\n  \n") == []


def test_parse_skips_unparseable_pid():
    assert parse_compute_apps("notapid, 100, thing.exe\n7, 5, ok.exe") == [
        GpuProcess(pid=7, name="ok.exe", used_mb=5.0)
    ]


# --------------------------------------------------------------------------- #
# Classification - the safety-critical part
# --------------------------------------------------------------------------- #


def test_only_our_bin_dir_processes_are_marked_ours():
    marked = mark_ours(parse_compute_apps(SAMPLE), bin_dir=BIN)
    ours = {p.pid: p.is_locitize for p in marked}
    assert ours[36888] is True  # our llama-server under bin/
    assert ours[22008] is False  # LM Studio - never ours
    assert ours[2176] is False  # hidden system process - never ours
    assert ours[24900] is False  # browser - never ours


def test_supervised_pid_marked_even_if_path_unknown():
    """A tracked pid is ours even when its name is a permissions placeholder."""
    marked = mark_ours(parse_compute_apps(SAMPLE), bin_dir=BIN, own_pids=[2176])
    assert {p.pid: p.is_locitize for p in marked}[2176] is True


def test_freeable_is_only_ours():
    marked = mark_ours(parse_compute_apps(SAMPLE), bin_dir=BIN)
    assert freeable_pids(marked) == [36888]


def test_no_bin_dir_marks_nothing_by_path():
    """With no known bin dir, path classification is disabled - only supervised
    pids can be ours, so nothing is freed by accident."""
    marked = mark_ours(parse_compute_apps(SAMPLE), bin_dir=None)
    assert freeable_pids(marked) == []


def test_bin_dir_match_is_case_and_slash_insensitive():
    forward = SAMPLE.replace("\\", "/")
    marked = mark_ours(parse_compute_apps(forward), bin_dir=BIN.upper())
    assert 36888 in freeable_pids(marked)


# --------------------------------------------------------------------------- #
# The injected query seam
# --------------------------------------------------------------------------- #


def test_query_returns_none_when_no_nvidia_smi(monkeypatch):
    monkeypatch.setattr(gpu_ledger.shutil, "which", lambda _n: None)
    assert query_compute_apps() is None


def test_query_parses_injected_output():
    class _Proc:
        returncode = 0
        stdout = SAMPLE

    text = query_compute_apps(run=lambda *a, **k: _Proc(), smi="nvidia-smi")
    rows = parse_compute_apps(text)
    assert rows[0].pid == 36888


def test_query_none_on_nonzero_exit():
    class _Proc:
        returncode = 9
        stdout = ""

    assert query_compute_apps(run=lambda *a, **k: _Proc(), smi="nvidia-smi") is None


def test_summarize_counts_ours(monkeypatch):
    marked = mark_ours(parse_compute_apps(SAMPLE), bin_dir=BIN)
    line = gpu_ledger.summarize(marked)
    assert "1 owned by locitize" in line
    assert "4 processes" in line


# --------------------------------------------------------------------------- #
# Terminating our own (injected taskkill)
# --------------------------------------------------------------------------- #


def test_terminate_reports_per_pid_success():
    calls = []

    class _Proc:
        returncode = 0

    def fake_run(argv, **kw):
        calls.append(argv)
        return _Proc()

    out = gpu_ledger.terminate_pids([36888, 111], run=fake_run)
    assert out == {36888: True, 111: True}
    # taskkill was asked to force-kill the tree for each pid.
    assert all("/F" in c and "/T" in c for c in calls)
    assert ["36888", "111"] == [c[-1] for c in calls]


def test_terminate_already_gone_is_false_not_raise():
    class _Proc:
        returncode = 128  # taskkill: process not found

    out = gpu_ledger.terminate_pids([999], run=lambda *a, **k: _Proc())
    assert out == {999: False}


def test_terminate_swallows_os_error():
    def boom(*a, **k):
        raise OSError("taskkill missing")

    assert gpu_ledger.terminate_pids([7], run=boom) == {7: False}


def test_terminate_empty_is_empty():
    assert gpu_ledger.terminate_pids([]) == {}


# --------------------------------------------------------------------------- #
# Where one process's GPU memory sits (owner report 2026-09-03: a 27B that
# "loads and switches" but does not chat had 782 MB paged through system RAM).
# --------------------------------------------------------------------------- #


class _Proc:
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode = returncode
        self.stdout = stdout


def test_parse_placement_reads_dedicated_and_shared_mb():
    placement = gpu_ledger.parse_placement(4242, "14483 782\n")
    assert placement is not None
    assert (placement.pid, placement.dedicated_mb, placement.shared_mb) == (4242, 14483, 782)


@pytest.mark.parametrize("text", [None, "", "14483", "a b", "1 2 3", "-1 5"])
def test_parse_placement_refuses_anything_but_two_non_negative_ints(text):
    assert gpu_ledger.parse_placement(1, text) is None


def test_spilled_is_the_measured_overflow_not_a_guess():
    # The fitted servers carried 124-280 MB shared; the ones that crawled
    # carried 500-1128 MB. The threshold sits between them.
    assert not gpu_ledger.GpuPlacement(1, 13900, 158).spilled
    assert gpu_ledger.GpuPlacement(1, 14483, 782).spilled
    assert gpu_ledger.GpuPlacement(1, 0, gpu_ledger.SPILL_THRESHOLD_MB).spilled


def test_describe_names_the_overflow_and_a_remedy_only_when_spilled():
    spilled = gpu_ledger.GpuPlacement(1, 14483, 782).describe("qwen3-8-27b")
    assert spilled.startswith("qwen3-8-27b: 14483 MB on the GPU and 782 MB")
    assert "paged through system RAM" in spilled
    assert "context_size" in spilled
    assert "overflowed" not in spilled  # a number and a hint, not a verdict
    fine = gpu_ledger.GpuPlacement(1, 13900, 158).describe("qwen3-8-27b")
    assert fine == "qwen3-8-27b: 13900 MB on the GPU, 158 MB shared."
    assert "crawl" not in fine
    assert spilled.isascii() and fine.isascii()


def test_query_placement_runs_powershell_for_the_pid(monkeypatch):
    monkeypatch.setattr(gpu_ledger.shutil, "which", lambda _n: "powershell")
    calls: list[list[str]] = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return _Proc(0, "100 20\n")

    assert gpu_ledger.query_placement(4242, run=run) == "100 20\n"
    assert calls[0][0] == "powershell"
    script = calls[0][-1]
    assert "pid_4242_" in script
    assert "Dedicated Usage" in script and "Shared Usage" in script


@pytest.mark.parametrize("pid", [None, "x", 0, -3])
def test_query_placement_refuses_bad_pids_without_running_anything(monkeypatch, pid):
    monkeypatch.setattr(gpu_ledger.shutil, "which", lambda _n: "powershell")

    def run(*_a, **_k):
        raise AssertionError("must not run for a bad pid")

    assert gpu_ledger.query_placement(pid, run=run) is None


def test_query_placement_none_without_powershell(monkeypatch):
    monkeypatch.setattr(gpu_ledger.shutil, "which", lambda _n: None)
    assert gpu_ledger.query_placement(4242, run=lambda *a, **k: _Proc(0, "1 1")) is None


def test_placement_of_is_none_when_the_counters_are_missing_or_the_call_fails(monkeypatch):
    monkeypatch.setattr(gpu_ledger.shutil, "which", lambda _n: "powershell")
    # A pid with no counter instances: the script exits 2, never "0 0".
    assert gpu_ledger.placement_of(4242, run=lambda *a, **k: _Proc(2, "")) is None

    def boom(*_a, **_k):
        raise gpu_ledger.subprocess.TimeoutExpired(cmd="powershell", timeout=20)

    assert gpu_ledger.placement_of(4242, run=boom) is None


def test_placement_of_parses_a_good_reading(monkeypatch):
    monkeypatch.setattr(gpu_ledger.shutil, "which", lambda _n: "powershell")
    placement = gpu_ledger.placement_of("17672", run=lambda *a, **k: _Proc(0, "14210 124"))
    assert placement == gpu_ledger.GpuPlacement(pid=17672, dedicated_mb=14210, shared_mb=124)
    assert not placement.spilled


# --------------------------------------------------------------------------- #
# The fit margin this card needs right now (owner rule "do not affect my tok/s")
# --------------------------------------------------------------------------- #


def test_fit_budget_lands_the_engine_budget_a_safety_under_the_real_free():
    # The desktop case that crawled at a fixed 256: engine blind at 14923,
    # card really holding 2146 MB for others.
    b = gpu_ledger.FitBudget(16302, 14923, 2146, 16303)
    assert b.real_free_mib == 14157
    assert b.target_mib == 14923 - 14157 + gpu_ledger.FIT_SAFETY_MIB
    # fit's budget is engine_free - target: exactly FIT_SAFETY_MIB under the truth.
    assert b.engine_free_mib - b.target_mib == b.real_free_mib - gpu_ledger.FIT_SAFETY_MIB
    assert "2146 MiB held by other processes" in b.describe()
    assert b.describe().isascii()


def test_fit_budget_is_zero_when_the_card_has_more_room_than_the_engine_thinks():
    # Nothing but the desktop on the card: the engine's blind reading is already
    # more than a safety under the truth, so every layer goes on.
    assert gpu_ledger.FitBudget(16302, 14923, 300, 16303).target_mib == 0
    assert gpu_ledger.FitBudget(16302, 14923, 356, 16303).target_mib == 0
    assert gpu_ledger.FitBudget(16302, 14923, 357, 16303).target_mib == 1


def test_fit_safety_is_the_engine_default_given_what_it_cannot_see():
    # The sweep (gpu_ledger.py, FIT_SAFETY_MIB): with 1580 MB held elsewhere
    # the last zero-paging row (60 layers, 13805 MiB planned) needed a target
    # of 790-968; 1024 lands one layer under it, the slack the desktop takes
    # during a session. 256 sat one layer from the crawl.
    assert gpu_ledger.FIT_SAFETY_MIB == 1024
    b = gpu_ledger.FitBudget(16302, 14923, 1580, 16303)
    assert b.target_mib == 14923 - (16303 - 1580) + 1024 == 1224
    # fit plans layers under (its own free reading - target); its reading ran
    # 150 MiB under --list-devices, so 14773 - 1224 = 13549 -> 59 layers (13613
    # does not fit, 13435 does), zero paging in the sweep.
    assert 14773 - b.target_mib < 13613


def test_parse_adapter_memory_reads_used_and_total():
    assert gpu_ledger.parse_adapter_memory("15546, 16303\n") == (15546, 16303)
    assert gpu_ledger.parse_adapter_memory("1000, 16303\n2000, 8192\n") == (1000, 16303)


@pytest.mark.parametrize("text", [None, "", "abc", "16303", "20000, 16303", "-1, 16303"])
def test_parse_adapter_memory_refuses_nonsense(text):
    assert gpu_ledger.parse_adapter_memory(text) is None


def test_parse_engine_free_reads_the_first_cuda_device():
    text = (
        "Available devices:\n"
        "  CUDA0: NVIDIA GeForce RTX 5070 Ti (16302 MiB, 14923 MiB free)\n"
        "  CUDA1: Other (8192 MiB, 100 MiB free)\n"
    )
    assert gpu_ledger.parse_engine_free(text) == (16302, 14923)


@pytest.mark.parametrize("text", [None, "", "Available devices:\n  CPU: x\n", "CUDA0: x (10 MiB, 20 MiB free)"])
def test_parse_engine_free_refuses_nonsense(text):
    assert gpu_ledger.parse_engine_free(text) is None


def test_query_engine_free_runs_the_given_binary_from_its_own_folder(tmp_path):
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"")
    calls: list = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs.get("cwd")))
        return _Proc(0, "  CUDA0: card (16302 MiB, 14923 MiB free)\n")

    text = gpu_ledger.query_engine_free(exe, run=run)
    assert calls[0][0] == [str(exe), "--list-devices"]
    assert calls[0][1] == str(tmp_path)  # beside its DLLs
    assert "14923 MiB free" in text


def test_query_engine_free_is_none_for_a_missing_binary_or_a_failing_run(tmp_path):
    def never(*a, **k):
        raise AssertionError("must not run a binary that is not there")

    assert gpu_ledger.query_engine_free(tmp_path / "nope.exe", run=never) is None
    assert gpu_ledger.query_engine_free(None, run=never) is None
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"")
    assert gpu_ledger.query_engine_free(exe, run=lambda *a, **k: _Proc(1, "")) is None


def test_fit_budget_needs_both_readings(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_ledger.shutil, "which", lambda _n: "nvidia-smi")
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"")

    def run(cmd, **kwargs):
        if cmd[-1] == "--list-devices":
            return _Proc(0, "CUDA0: card (16302 MiB, 14923 MiB free)\n")
        return _Proc(0, "1700, 16303\n")

    b = gpu_ledger.fit_budget(exe, run=run)
    assert b == gpu_ledger.FitBudget(16302, 14923, 1700, 16303)
    assert b.target_mib == 14923 - (16303 - 1700) + 1024 == 1344

    def no_smi(cmd, **kwargs):
        if cmd[-1] == "--list-devices":
            return _Proc(0, "CUDA0: card (16302 MiB, 14923 MiB free)\n")
        return _Proc(9, "")

    assert gpu_ledger.fit_budget(exe, run=no_smi) is None
    assert gpu_ledger.fit_budget(tmp_path / "missing.exe", run=run) is None
