"""Headless proof (no real binary/mic/GPU) that the GUI assistant mic wires on the
SHARED ServiceManager and that close()/stop_all() reaps it with no orphan (RM3/M10.4).

It drives the real _MicSttSource through a FAKE process factory (whisper-stream is a
FakeProcess), so it exercises the actual production wiring without touching hardware.

Run: Codebase/.venv/Scripts/python Codebase/platform/scripts/m10_shared_manager_proof.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the platform package importable when run from anywhere.
_PLATFORM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PLATFORM))
sys.path.insert(0, str(_PLATFORM / "tests"))

from threading import Event

from config import Settings
from fakes import FakeProcess, make_fake_launcher  # from tests/fakes.py
from launcher import Launcher, _MicSttSource
from services import ManagedProcess, ServiceManager, ServiceStatus


def _fake_stream_factory(created: dict):
    """A process factory that 'starts' whisper-stream as a FakeProcess (no binary)."""

    def factory(spec):
        proc = FakeProcess(poll_sequence=[None])
        mp = ManagedProcess(
            spec,
            launcher=make_fake_launcher(proc),
            readiness=lambda s, h: True,
            kill_runner=lambda pid: None,
            poll_interval_s=0.01,
        )
        created[spec.name] = (mp, proc)
        return mp

    return factory


def main() -> int:
    settings = Settings()
    settings.base_dir = _PLATFORM
    settings.paths.whisper_stream = "whisper-stream.exe"
    settings.paths.whisper_model = "ggml.bin"

    created: dict = {}
    launcher = Launcher(deps={"process_factory": _fake_stream_factory(created)})

    # --- Case A: the GUI path passes the SHARED, atexit-backstopped manager -------
    shared = ServiceManager()
    mic = _MicSttSource(launcher, settings, Event(), manager=shared)

    assert mic._owns_manager is False, "shared-manager path must NOT own the manager"
    assert mic._manager is shared, "the mic MUST reuse the shared manager (RM3), not a fresh one"
    assert mic._running is True, "whisper-stream should have started on the shared manager"
    # The whisper-stream child is registered on the SHARED manager (one service).
    monitored = shared.monitor()
    assert monitored, "the shared manager should now hold the whisper-stream service"
    print(f"[A] shared-manager mic wired: reused=True running=True services={list(monitored)}")

    # close() stops ONLY its own whisper controller and leaves the shared manager for
    # the GUI/atexit teardown; then the shared stop_all() reaps everything cleanly.
    _mp, proc = created["whisper_stream"]
    mic.close()
    assert proc.signals, "close() must stop the whisper-stream child (a stop signal)"
    shared.stop_all()  # the GUI's shutdown()/atexit backstop
    statuses = list(shared.monitor().values())
    assert all(s is ServiceStatus.STOPPED for s in statuses), f"orphan left: {statuses}"
    print(f"[A] no orphan after close()+stop_all: statuses={[s.value for s in statuses]}")

    # --- Case B: the CLI path (no manager) builds a FRESH, self-backstopped one ----
    created.clear()
    mic_cli = _MicSttSource(launcher, settings, Event())  # no manager passed
    assert mic_cli._owns_manager is True, "CLI path must OWN its fresh manager"
    assert mic_cli._manager is not shared, "CLI path must NOT reuse the GUI shared manager"
    print("[B] CLI mic wired: owns a fresh manager (its own atexit backstop), distinct from shared")
    mic_cli.close()

    print("PROOF OK: GUI assistant mic uses the SHARED manager; close()+stop_all leave no orphan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
