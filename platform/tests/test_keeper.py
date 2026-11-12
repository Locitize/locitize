"""The keeper starts the desktop again after a crash, and a new desktop clears the
model server a dead one left behind (owner report 2026-11-12: the desktop died in
the night and its llama-server kept the card)."""
from __future__ import annotations

import gpu_ledger
import keeper


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _keep(exits, uptimes):
    """Run keep() over scripted runs: each run lasts its uptime and exits with its code."""
    clock = FakeClock()
    runs, lines = [], []
    script = list(zip(exits, uptimes))

    def run(command):
        code, uptime = script[len(runs)]
        runs.append(command)
        clock.now += uptime
        return code

    def sleep(seconds):
        clock.now += seconds

    code = keeper.keep(["desktop"], run=run, clock=clock, sleep=sleep, log=lines.append)
    return code, len(runs), lines


def test_a_closed_desktop_is_not_started_again():
    assert _keep([0], [3600]) == (0, 1, [])


def test_a_crash_starts_it_again():
    code, runs, lines = _keep([1, 0], [3600, 60])
    assert (code, runs) == (0, 2)
    assert "starting it again" in lines[0]


def test_a_desktop_failing_as_it_starts_is_left_alone():
    code, runs, lines = _keep([2], [3])
    assert (code, runs) == (2, 1)
    assert "failing to start" in lines[0]


def test_it_gives_up_after_three_restarts_in_ten_minutes():
    code, runs, lines = _keep([1, 1, 1, 1], [60, 60, 60, 60])
    assert (code, runs) == (1, 4)
    assert "leaving it stopped" in lines[-1]


def test_crashes_hours_apart_are_each_restarted():
    code, runs, _lines = _keep([1, 1, 1, 1, 1, 0], [3600] * 6)
    assert (code, runs) == (0, 6)


def test_only_the_plain_desktop_is_kept():
    assert keeper.keeps(["--desktop"])
    assert keeper.keeps(["--desktop", "--desktop", "--start-hidden"])
    assert not keeper.keeps(["--desktop", "--terminal"])
    assert not keeper.keeps(["--health"])


class FakeProc:
    def __init__(self, pid, exe, parent_alive=True, readable=True):
        self.pid, self._exe, self._parent, self._readable = pid, exe, parent_alive, readable

    def exe(self):
        if not self._readable:
            raise PermissionError("access denied")
        return self._exe

    def parent(self):
        return object() if self._parent else None


def test_only_our_servers_whose_parent_is_gone_are_orphans(tmp_path):
    bin_dir = tmp_path / "data" / "bin"
    ours = str(bin_dir / "llama.cpp" / "llama-server.exe")
    procs = [
        FakeProc(10, ours, parent_alive=False),                      # the dead desktop's server
        FakeProc(11, ours, parent_alive=True),                       # a running desktop's server
        FakeProc(12, str(tmp_path / "other" / "llama-server.exe"), parent_alive=False),  # not ours
        FakeProc(13, str(tmp_path / "data" / "bin2" / "x.exe"), parent_alive=False),  # bin2 is not bin
        FakeProc(14, ours, parent_alive=False, readable=False),     # cannot tell: hands off
    ]
    assert gpu_ledger.orphaned_servers(bin_dir, procs) == [(10, ours)]
    assert gpu_ledger.orphaned_servers(None, procs) == []


def test_the_desktop_start_stops_the_orphans(monkeypatch, tmp_path):
    import launcher

    killed = []
    monkeypatch.setattr(gpu_ledger, "orphaned_servers", lambda bin_dir: [(10, "llama-server.exe")])
    monkeypatch.setattr(gpu_ledger, "terminate_pids", lambda pids: killed.extend(pids) or {p: True for p in pids})

    class Settings:
        data_dir = str(tmp_path)

    assert launcher.Launcher._clear_orphaned_servers(object.__new__(launcher.Launcher), Settings()) == [10]
    assert killed == [10]


def test_the_clean_up_never_fails_a_start(monkeypatch, tmp_path):
    import launcher

    def boom(bin_dir):
        raise RuntimeError("no")

    monkeypatch.setattr(gpu_ledger, "orphaned_servers", boom)

    class Settings:
        data_dir = str(tmp_path)

    assert launcher.Launcher._clear_orphaned_servers(object.__new__(launcher.Launcher), Settings()) == []
