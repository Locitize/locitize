"""Launcher tests: the CI-observable main(["--health","--json"]) path.

Drives the launcher with injected collaborators (a fake config loader and a stub
health checker) and asserts the JSON shape and exit code for both an all-PASS
report and a with-FAIL report (Architecture section 12). Also proves import is
side-effect free (AC2) and the interactive menu handles an unavailable app.
"""

from __future__ import annotations

import json

from config import Model, ModelRegistryData, Settings
from fakes import FakeProcess, make_fake_launcher
from health import HealthReport, HealthResult, HealthStatus
from launcher import Launcher, main
from services import ManagedProcess, ServiceSpec, ServiceStatus


class _StubChecker:
    """Health checker stub returning a preset report."""

    def __init__(self, report: HealthReport) -> None:
        self._report = report

    def run_all(self) -> HealthReport:
        return self._report


def _models() -> ModelRegistryData:
    return ModelRegistryData(
        version=1,
        models=[
            Model(
                id="qwen3-14b",
                name="Qwen3 14B",
                description="d",
                location="",
                context_size=32768,
                gpu_layers=-1,
                status="installed",
            )
        ],
    )


def _config_load(_base=None, env=None):
    # Return valid settings/models with no fatal issues.
    return Settings(), _models(), []


def _deps(report: HealthReport, lines: list[str], input_fn=None):
    deps = {
        "config_load": _config_load,
        "health_checker": _StubChecker(report),
        "output_fn": lines.append,
    }
    if input_fn is not None:
        deps["input_fn"] = input_fn
    return deps


def _pass_report() -> HealthReport:
    results = [HealthResult("python", HealthStatus.PASS, "3.11")]
    return HealthReport(results=results, overall=HealthStatus.PASS)


def _fail_report() -> HealthReport:
    results = [
        HealthResult("python", HealthStatus.PASS, "3.11"),
        HealthResult("gpu", HealthStatus.FAIL, "no gpu", remedy="install driver"),
    ]
    return HealthReport(results=results, overall=HealthStatus.FAIL)


def test_import_launcher_is_side_effect_free():
    """Importing launcher must not run main() or produce output (AC2)."""
    import importlib

    import launcher as launcher_module

    # Reloading must not raise or print; the module only defines symbols.
    importlib.reload(launcher_module)
    assert hasattr(launcher_module, "main")


def test_listen_dedup_filters_surfaced_transcript_across_read_boundaries():
    """D-M3-1 wiring: the listen echo path de-duplicates whisper-stream's overlapping
    -window output even when a duplicate segment is split across two polled reads."""
    from whisper import TranscriptDeduplicator

    lines: list[str] = []
    launcher = Launcher(deps=_deps(_pass_report(), lines))
    dedup = TranscriptDeduplicator()
    # First read delivers one complete line plus a partial second line.
    pending = launcher._emit_listen_lines("Can you hear me?\nCan you hea", dedup, final=False)
    # Second read completes that second (duplicate) line; it must be suppressed.
    pending = launcher._emit_listen_lines(pending + "r me?\n", dedup, final=False)
    launcher._emit_listen_lines(pending, dedup, final=True)
    assert lines == ["Can you hear me?"]  # emitted once despite the split duplicate


def test_listen_drops_silence_hallucination_markers_in_vad_cadence():
    """D-M3-2 wiring: in VAD mode whisper-stream prints one utterance per detected
    speech burst and non-speech markers ('.', '[BLANK_AUDIO]') on silent windows.
    The listen echo path surfaces only the spoken utterance, even when a marker is
    split across two polled reads, and partial-line buffering still holds."""
    from whisper import TranscriptDeduplicator

    lines: list[str] = []
    launcher = Launcher(deps=_deps(_pass_report(), lines))
    dedup = TranscriptDeduplicator()
    # Read 1: a real utterance, then a silence-marker line, then a partial marker.
    pending = launcher._emit_listen_lines(
        "What time is it?\n.\n[BLANK_", dedup, final=False
    )
    # Read 2: completes the marker line and adds a hallucinated stock phrase window.
    pending = launcher._emit_listen_lines(
        pending + "AUDIO]\nThank you.\n", dedup, final=False
    )
    launcher._emit_listen_lines(pending, dedup, final=True)
    # Only real words survive; "." and the split "[BLANK_AUDIO]" are dropped. The
    # genuine-looking "Thank you." is NOT phrase-blocklisted here (VAD mode upstream
    # is what prevents it appearing on true silence), so it is surfaced as spoken.
    assert lines == ["What time is it?", "Thank you."]


def test_mic_startup_noise_gate_drops_engine_lines_only_speech_emits():
    """startup_noise (D-M7-1): the mic/listen path, fed a real whisper-stream log
    (engine boot lines, then '[Start speaking]', then '### Transcription' blocks with
    timestamped speech), surfaces ONLY the spoken words. The three owner-observed
    CUDA lines that were being sent to the LLM as fake user turns must NOT emit, and
    the timestamp prefixes are stripped off the real utterances."""
    from whisper import StartupNoiseGate, TranscriptDeduplicator

    # A faithful slice of the captured whisper-stream startup log, fed in two reads
    # with a segment split across the boundary to also prove partial-line buffering
    # still holds under the gate.
    read1 = (
        "ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16302 MiB):\n"
        "  Device 0: NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, VMM: yes, VRAM: 16302 MiB\n"
        "load_backend: loaded CUDA backend from /locitize-test/whisper/ggml-cuda.dll\n"
        "SDL_main: using VAD, will transcribe on speech activity\n"
        "\n"
        "[Start speaking]\n"
        "\n"
        "### Transcription 0 START | t0 = 0 ms | t1 = 2713 ms\n"
        "\n"
        "[00:00:00.000 --> 00:00:05.920]   What is the cap"  # split mid-utterance
    )
    read2 = (
        "ital of Kenya?\n"
        "\n"
        "### Transcription 0 END\n"
        "\n"
        "### Transcription 1 START | t0 = 0 ms | t1 = 8535 ms\n"
        "\n"
        "[00:00:00.000 --> 00:00:10.000]   .\n"  # silence placeholder -> dropped
        "\n"
        "### Transcription 1 END\n"
    )

    lines: list[str] = []
    launcher = Launcher(deps=_deps(_pass_report(), lines))
    dedup = TranscriptDeduplicator()
    gate = StartupNoiseGate()
    pending = launcher._emit_listen_lines(read1, dedup, final=False, emit=lines.append, gate=gate)
    pending = launcher._emit_listen_lines(
        pending + read2, dedup, final=False, emit=lines.append, gate=gate
    )
    launcher._emit_listen_lines(pending, dedup, final=True, emit=lines.append, gate=gate)

    # Only the real spoken utterance survives; every engine line, block marker, the
    # pre-banner preamble, and the "." silence placeholder are gone.
    assert lines == ["What is the capital of Kenya?"]
    for noise in (
        "ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16302 MiB):",
        "  Device 0: NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, VMM: yes, VRAM: 16302 MiB",
        "load_backend: loaded CUDA backend from /locitize-test/whisper/ggml-cuda.dll",
    ):
        assert noise not in lines


def test_build_service_controller_reuses_a_shared_manager():
    """H-2: passing an existing manager makes _build_service_controller reuse it (and
    the SingleServiceController register on it), so the whisper-stream child is torn
    down by the shared manager's stop_all -- the GUI's shutdown()/atexit backstop --
    rather than an unbackstopped local manager."""
    from services import ManagedProcess, ServiceManager

    def fake_factory(spec):
        return ManagedProcess(
            spec,
            launcher=make_fake_launcher(FakeProcess(poll_sequence=[None, 0])),
            readiness=lambda s, h: True,
            kill_runner=lambda pid: None,
            poll_interval_s=0.01,
        )

    lines: list[str] = []
    launcher = Launcher(
        deps={**_deps(_pass_report(), lines), "process_factory": fake_factory}
    )
    shared = ServiceManager()
    spec = ServiceSpec(
        name="whisper-stream",
        command=["fake", "--run"],
        port=None,
        health_path=None,
        ready_timeout_s=1.0,
        stop_timeout_s=0.5,
    )
    manager, controller = launcher._build_service_controller(
        Settings(), lambda: spec, manager=shared
    )
    assert manager is shared  # reused, not a fresh local manager
    controller.start()
    assert "whisper-stream" in shared._services  # registered on the SHARED manager
    shared.stop_all()  # the one backstop reaches the whisper child
    assert shared.monitor()["whisper-stream"] is ServiceStatus.STOPPED


def test_gui_listen_routes_through_the_shared_service_manager():
    """H-2: _gui_listen must build its whisper-stream controller on the GUI's shared
    manager (self._service_manager, atexit-backstopped in _run_gui), never a fresh
    local one, so a window close during Listen cannot orphan whisper-stream."""
    from services import ServiceManager

    lines: list[str] = []
    launcher = Launcher(deps=_deps(_pass_report(), lines))
    shared = ServiceManager()
    launcher._service_manager = shared
    captured: dict = {}

    class _FakeController:
        def start(self):
            return ServiceStatus.RUNNING

        def stop(self):
            captured["stopped"] = True
            return ServiceStatus.STOPPED

    def fake_build(settings, spec_builder, manager=None):
        captured["manager"] = manager
        return manager, _FakeController()

    # Stub the controller build and the log tail so the wiring is exercised without
    # a real whisper binary, file, or clock.
    launcher._build_service_controller = fake_build  # type: ignore[assignment]
    launcher._tail_listen = lambda log_path, seconds, emit=None: None  # type: ignore[assignment]

    ok = launcher._gui_listen(Settings(), 0.0, emit=lambda line: None)
    assert ok is True
    assert captured["manager"] is shared  # the shared manager, not a fresh one
    assert captured.get("stopped") is True  # clean stop on the normal path


def test_health_json_all_pass_exit_zero():
    """--health --json on an all-PASS report emits valid JSON and exits 0."""
    lines: list[str] = []
    code = Launcher(deps=_deps(_pass_report(), lines)).run(["--health", "--json"])
    assert code == 0
    data = json.loads("\n".join(lines))
    assert data["overall"] == "PASS"
    assert data["results"][0]["name"] == "python"


def test_health_json_with_fail_exit_one():
    """--health --json with a FAIL emits JSON and exits 1 (worst-of drives code)."""
    lines: list[str] = []
    code = Launcher(deps=_deps(_fail_report(), lines)).run(["--health", "--json"])
    assert code == 1
    data = json.loads("\n".join(lines))
    assert data["overall"] == "FAIL"
    names = {r["name"] for r in data["results"]}
    assert "gpu" in names


def test_health_table_mode_exit_code():
    """--health (no json) prints a table and still maps FAIL -> exit 1."""
    lines: list[str] = []
    code = Launcher(deps=_deps(_fail_report(), lines)).run(["--health"])
    assert code == 1
    text = "\n".join(lines)
    assert "[XX] gpu" in text  # ASCII fail mark rendered
    assert "Overall: FAIL" in text


def test_no_menu_renders_and_exits():
    """--no-menu renders banner/status/models/applications then exits 0."""
    lines: list[str] = []
    code = Launcher(deps=_deps(_pass_report(), lines)).run(["--no-menu"])
    assert code == 0
    text = "\n".join(lines)
    assert "locitize" in text
    assert "System Status" in text
    assert "Installed Models" in text
    assert "Applications" in text


def test_menu_launches_voice_assistant(tmp_path):
    """Selecting the Voice Assistant (now live, M7) runs the assistant session.

    The assistant_runner dep substitutes the real mic+LLM+Kokoro session so this
    stays hermetic (no model start, no audio); the test asserts the menu routed to
    it instead of the old 'not available' message.
    """
    lines: list[str] = []
    choices = iter(["voice-assistant", "q"])

    def fake_input(_prompt: str) -> str:
        return next(choices)

    settings = Settings()
    settings.base_dir = tmp_path  # journal writes under a temp dir, not the repo
    settings.launcher.auto_journal = False  # keep the test from touching docs/

    def config_load(_base=None, env=None):
        return settings, _models(), []

    ran = []

    deps = {
        "config_load": config_load,
        "health_checker": _StubChecker(_pass_report()),
        "output_fn": lines.append,
        "input_fn": fake_input,
        "assistant_runner": lambda s, m: ran.append((s, m)),
    }
    code = Launcher(deps=deps).run([])
    assert code == 0
    assert len(ran) == 1  # the menu launched the assistant session
    text = "\n".join(lines)
    assert "not available" not in text


def test_main_wrapper_returns_int():
    """main() returns an int exit code for the default non-interactive path."""
    # No TTY in the test harness, so main() defaults to JSON and returns 0/1.
    code = main(["--health", "--json"])
    assert code in (0, 1)


def test_force_utf8_output_lets_emoji_reply_print_on_cp1252_stdout():
    """DEF-QA-1: an assistant reply containing an emoji must print without crashing
    even when stdout is a non-UTF-8 (cp1252) piped/captured stream. Proves the fix by
    showing the identical emit raises UnicodeEncodeError on the raw cp1252 stream (the
    reported bug) and succeeds after _force_utf8_output reconfigures the stream."""
    import io
    import sys as _sys

    import pytest

    from launcher import _force_utf8_output

    reply = "locitize: All set \U0001f60a"  # U+1F60A, not encodable in cp1252
    original = _sys.stdout

    # Baseline (documents the reported crash): raw cp1252 stdout cannot encode it.
    bad = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")
    _sys.stdout = bad
    try:
        with pytest.raises(UnicodeEncodeError):
            print(reply)
    finally:
        _sys.stdout = original

    # With the fix: the same style of stream, reconfigured, prints the emoji as UTF-8.
    raw = io.BytesIO()
    fixed = io.TextIOWrapper(raw, encoding="cp1252", newline="")
    _sys.stdout = fixed
    try:
        _force_utf8_output()  # reconfigures sys.stdout (fixed) to utf-8/backslashreplace
        print(reply)  # the exact assistant emit path -- must not raise now
        fixed.flush()
    finally:
        _sys.stdout = original

    assert "\U0001f60a" in raw.getvalue().decode("utf-8")


def test_mic_stt_source_registers_atexit_no_orphan_backstop(monkeypatch):
    """Review M-1: _MicSttSource builds a FRESH ServiceManager (no manager passed to
    _build_service_controller), so it must register that manager's stop_all with
    atexit -- otherwise an abnormal exit skipping the assistant finally/close() would
    orphan the whisper-stream mic child, unlike the model and Kokoro managers."""
    import atexit as _atexit

    from launcher import _MicSttSource
    from services import ServiceManager

    registered: list = []
    monkeypatch.setattr(
        _atexit, "register", lambda fn, *a, **k: registered.append(fn) or fn
    )

    fresh = ServiceManager()

    class _FakeController:
        def start(self):
            return ServiceStatus.RUNNING

    launcher = Launcher(deps=_deps(_pass_report(), []))
    # Stub the controller build to hand back a fresh (unbackstopped) manager, exactly
    # the production shape that this fix must cover, without a real whisper binary.
    launcher._build_service_controller = (  # type: ignore[assignment]
        lambda settings, spec_builder, manager=None: (fresh, _FakeController())
    )

    source = _MicSttSource(launcher, Settings(), interrupt=None)

    assert source._manager is fresh
    assert fresh.stop_all in registered  # the fresh manager's backstop was registered


def _mic_source_with_gate(gate):
    """Build a _MicSttSource wired to a fake controller (no whisper binary) and the
    given half-duplex gate, for the D-M7-3 mic-drop test."""
    import atexit as _atexit

    from launcher import _MicSttSource
    from services import ServiceManager

    class _FakeController:
        def start(self):
            return ServiceStatus.RUNNING

    launcher = Launcher(deps=_deps(_pass_report(), []))
    launcher._build_service_controller = (  # type: ignore[assignment]
        lambda settings, spec_builder, manager=None: (ServiceManager(), _FakeController())
    )
    # Do not leave a real atexit backstop behind from this unit test.
    original = _atexit.register
    _atexit.register = lambda fn, *a, **k: fn  # type: ignore[assignment]
    try:
        return _MicSttSource(launcher, Settings(), interrupt=None, speaking_gate=gate)
    finally:
        _atexit.register = original  # type: ignore[assignment]


def test_mic_half_duplex_drops_segments_while_gate_speaking():
    """barge / half_duplex: while the shared gate is set (assistant speaking), the mic
    source drops captured segments -- the assistant's own TTS echoing back -- instead
    of queuing them as user turns. When the gate clears, real speech queues (D-M7-3).

    These direct _queue_segment calls carry no capture timestamp, so they exercise the
    poll-time flag fallback; the deterministic capture-time path is proven below."""
    from assistant import HalfDuplexGate

    gate = HalfDuplexGate(tail_s=0.8)
    source = _mic_source_with_gate(gate)

    gate.begin()  # assistant starts speaking -> mic gated
    source._queue_segment("We face with smiling eyes")  # the AC17 self-echo
    source._queue_segment("For example, I can answer questions")
    assert source._queue == []  # both dropped, never queued

    gate.clear()  # barge-in / speech done + tail expired
    source._queue_segment("what time is it")
    assert source._queue == ["what time is it"]  # real speech passes through


class _FakeClock:
    """A controllable monotonic clock shared by the mic and speaking gates."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_mic_capture_overlap_drops_late_read_self_echo():
    """capture_overlap / half_duplex (THE D-M7-3b race, end to end through the real
    mic poll path): whisper transcribes the assistant's trailing self-spoken sentence
    LONG after playback ends (past the 0.8s tail), but its CAPTURE window overlapped a
    recorded speaking interval, so the mic source drops it deterministically. A user
    utterance captured strictly after the assistant finished still queues."""
    from assistant import HalfDuplexGate
    from whisper import CAPTURE_BANNER, StartupNoiseGate

    clock = _FakeClock()
    speaking_gate = HalfDuplexGate(tail_s=0.8, clock=clock)
    source = _mic_source_with_gate(speaking_gate)
    # Share the fake clock so the startup gate's capture anchor and the speaking gate's
    # recorded intervals are measured on the same timeline (both monotonic in prod).
    source._gate = StartupNoiseGate(clock=clock)
    # The session-wide temp log is shared by every mic test; truncate so this source's
    # position-0 read sees only this test's feed, not another test's leftover lines.
    open(source._log_path, "wb").close()

    def feed(text: str) -> None:
        with open(source._log_path, "ab") as handle:
            handle.write(text.encode("utf-8"))

    # t=0.0: live capture begins; the "[Start speaking]" banner anchors the stream.
    clock.now = 0.0
    feed(CAPTURE_BANNER + "\n")
    source._poll_once()
    assert source._queue == []  # the banner surfaces nothing

    # The assistant speaks its reply from wall-clock 1.0 to 3.0 (recorded interval).
    clock.now = 1.0
    speaking_gate.begin()
    clock.now = 3.0
    speaking_gate.end()

    # t=10.0: 7s past playback and far beyond the 0.8s tail, whisper FINALLY emits the
    # trailing self-spoken sentence -- captured at 1.5s-2.5s (during playback) -- and a
    # genuine user utterance captured at 4.0s-5.0s (after the assistant finished).
    clock.now = 10.0
    assert speaking_gate.is_gated() is False  # poll-time flag reopened the mic long ago
    # D-M7-3c: the CAPTURE interval comes from the block HEADER cumulative ms, not the
    # per-line "[00:00:00 -->]" span (which resets each block). The echo's header says
    # it was captured 1.5s-2.5s (during playback); the real utterance's header 4s-5s.
    feed("### Transcription 8 START | t0 = 1500 ms | t1 = 2500 ms\n")
    feed("[00:00:00.000 --> 00:00:01.000]   we face with smiling eyes\n")
    feed("### Transcription 9 START | t0 = 4000 ms | t1 = 5000 ms\n")
    feed("[00:00:00.000 --> 00:00:01.000]   what time is it\n")
    source._poll_once()

    # The late-read echo is dropped by CAPTURE-time overlap; only the real utterance,
    # captured strictly after the assistant finished, queues. The D-M7-3 reopener race
    # (poll-time-only gate leaking the trailing echo) is closed.
    assert source._queue == ["what time is it"]


# ---- D-M7-6: push-to-talk over the real mic source (keyword push_to_talk) ---- #


def test_push_to_talk_mic_flush_discards_closed_window_capture():
    """push_to_talk: _MicSttSource.flush() drains and drops whatever whisper-stream
    captured while the window was closed (ambient noise / the assistant's own TTS),
    so it can never leak into the next utterance. Only speech written to the log AFTER
    flush -- inside the open window -- is returned by poll_segment."""
    from whisper import CAPTURE_BANNER

    source = _mic_source_with_gate(None)
    open(source._log_path, "wb").close()  # isolate from other mic tests' log lines

    def feed(text: str) -> None:
        with open(source._log_path, "ab") as handle:
            handle.write(text.encode("utf-8"))

    # Window CLOSED: the banner plus ambient/self-echo speech accumulate in the log.
    feed(CAPTURE_BANNER + "\n")
    feed("### Transcription 0 START | t0 = 0 ms | t1 = 3000 ms\n")
    feed("[00:00:00.000 --> 00:00:03.000]   thanks for watching\n")

    # Owner opens the window: flush drops all of that closed-window capture.
    source.flush()
    assert source._queue == []

    # Window OPEN: real speech arrives and poll_segment returns it (0 timeout: the log
    # already holds it, so the first poll finds it without waiting).
    feed("### Transcription 1 START | t0 = 5000 ms | t1 = 8000 ms\n")
    feed("[00:00:00.000 --> 00:00:03.000]   what is the capital of Kenya\n")
    assert source.poll_segment(0.0) == "what is the capital of Kenya"
    # The closed-window self-echo was discarded, never surfaced as a segment.
    assert source.poll_segment(0.0) is None


def test_push_to_talk_end_to_end_over_real_mic_source():
    """push_to_talk (end to end): PushToTalkSttSource driving the real _MicSttSource --
    Enter opens a window, the closed-window junk is dropped, and the utterance written
    while the window is open is returned; the loop returns to the prompt after."""
    from assistant import PUSH_TO_TALK_PROMPT, PushToTalkSttSource
    from whisper import CAPTURE_BANNER

    mic = _mic_source_with_gate(None)
    open(mic._log_path, "wb").close()  # isolate from other mic tests' log lines

    def feed(text: str) -> None:
        with open(mic._log_path, "ab") as handle:
            handle.write(text.encode("utf-8"))

    # Pre-window: banner + self-echo captured while the window is closed.
    feed(CAPTURE_BANNER + "\n")
    feed("### Transcription 0 START | t0 = 0 ms | t1 = 2000 ms\n")
    feed("[00:00:00.000 --> 00:00:02.000]   you're welcome\n")

    prompts: list[str] = []

    # The owner's real speech arrives AFTER the window opens. flush() marks the window
    # open (and drops the closed-window self-echo); wrap it so the real utterance is
    # written to the log only once the window is open -- exactly the real timeline.
    real_flush = mic.flush

    def flush_then_speak():
        real_flush()  # drop the closed-window "you're welcome" self-echo
        feed("### Transcription 1 START | t0 = 4000 ms | t1 = 7000 ms\n")
        feed("[00:00:00.000 --> 00:00:03.000]   what time is it\n")

    mic.flush = flush_then_speak  # type: ignore[method-assign]

    enters = [""]  # one Enter press, then EOF ends the session

    def read_line():
        return enters.pop(0) if enters else None

    src = PushToTalkSttSource(
        mic, read_line=read_line, emit=prompts.append, settle_s=0.0, poll_s=0.0
    )
    assert src.next_utterance() == "what time is it"
    assert prompts.count(PUSH_TO_TALK_PROMPT) == 1  # the window prompt was shown


# ---- M2: service lifecycle wired into the launcher ---------------------- #


def _fake_service_deps(created: dict, ready: bool = True, exits_early: bool = False):
    """Build launcher deps that start model services with fake processes only.

    `ready` drives whether the injected readiness check passes; `exits_early`
    models a process that dies before readiness (a failed start). No real binary
    or subprocess is ever spawned.
    """

    def spec_builder(model_id, ctx=None, gpu=None):
        return ServiceSpec(
            name=f"llama_cpp:{model_id}",
            command=["llama", "--port", "8080"],
            port=None,
            health_path=None,
            ready_timeout_s=1.0,
            stop_timeout_s=0.5,
        )

    def factory(spec):
        seq = [1] if exits_early else [None]
        proc = FakeProcess(poll_sequence=seq)
        mp = ManagedProcess(
            spec,
            launcher=make_fake_launcher(proc),
            readiness=lambda s, h: ready,
            kill_runner=lambda pid: None,
            poll_interval_s=0.01,
        )
        created[spec.name] = (mp, proc)
        return mp

    return {"process_factory": factory, "spec_builder": spec_builder}


def test_no_orphan_process_after_menu_exit(tmp_path):
    """AC5 / Reviewer L-1: starting a model then quitting the menu leaves no
    service running (stop_all runs in the menu's finally block)."""
    created: dict = {}
    choices = iter(["1", "q"])  # start model #1, then quit

    settings = Settings()
    settings.base_dir = tmp_path
    settings.launcher.auto_journal = False

    def config_load(_base=None, env=None):
        return settings, _models(), []

    deps = {
        "config_load": config_load,
        "health_checker": _StubChecker(_pass_report()),
        "output_fn": lambda *_a: None,
        "input_fn": lambda _p: next(choices),
    }
    deps.update(_fake_service_deps(created))

    launcher = Launcher(deps=deps)
    code = launcher.run([])
    assert code == 0
    # A service was started, and after menu exit nothing is left RUNNING.
    assert created, "the model selection should have started a service"
    statuses = list(launcher._service_manager.monitor().values())
    assert ServiceStatus.RUNNING not in statuses
    # The started process received a stop signal (proof stop_all reached it).
    _mp, proc = created["llama_cpp:qwen3-14b"]
    assert proc.signals, "stop_all should have signalled the running service"


def test_service_status_json_reports_idle_when_nothing_started(tmp_path):
    """AC6: --service-status --json runs with nothing started and reports an
    honest empty/idle state, exit 0, no GPU or process needed."""
    lines: list[str] = []
    settings = Settings()
    settings.base_dir = tmp_path

    def config_load(_base=None, env=None):
        return settings, _models(), []

    deps = {
        "config_load": config_load,
        "health_checker": _StubChecker(_pass_report()),
        "output_fn": lines.append,
    }
    code = Launcher(deps=deps).run(["--service-status", "--json"])
    assert code == 0
    data = json.loads("\n".join(lines))
    assert data["running_model"] is None
    assert data["services"] == []


def test_smoke_start_ready_exits_zero_and_cleans_up(tmp_path):
    """AC7 (fake-process form): a model that becomes ready yields outcome 'ready',
    exit 0, and no service left running afterward."""
    created: dict = {}
    lines: list[str] = []
    settings = Settings()
    settings.base_dir = tmp_path

    def config_load(_base=None, env=None):
        return settings, _models(), []

    deps = {
        "config_load": config_load,
        "health_checker": _StubChecker(_pass_report()),
        "output_fn": lines.append,
    }
    deps.update(_fake_service_deps(created, ready=True))

    launcher = Launcher(deps=deps)
    code = launcher.run(["--smoke-start", "qwen3-14b", "--json"])
    assert code == 0
    result = json.loads(lines[-1])
    assert result["outcome"] == "ready"
    assert result["model_id"] == "qwen3-14b"
    # Cleanup: the harness stopped the service before returning.
    statuses = list(launcher._service_manager.monitor().values())
    assert ServiceStatus.RUNNING not in statuses


def test_smoke_start_failure_reports_and_exits_nonzero(tmp_path):
    """AC7 cleanup guarantee: a model that never becomes ready still cleans up,
    reports outcome 'failed', and exits nonzero (honest failure, no orphan)."""
    created: dict = {}
    lines: list[str] = []
    settings = Settings()
    settings.base_dir = tmp_path

    def config_load(_base=None, env=None):
        return settings, _models(), []

    deps = {
        "config_load": config_load,
        "health_checker": _StubChecker(_pass_report()),
        "output_fn": lines.append,
    }
    deps.update(_fake_service_deps(created, ready=False, exits_early=True))

    launcher = Launcher(deps=deps)
    code = launcher.run(["--smoke-start", "qwen3-14b", "--json"])
    assert code == 1
    result = json.loads(lines[-1])
    assert result["outcome"] == "failed"
    assert result["reason"]  # a specific, non-empty reason (not swallowed)
    statuses = list(launcher._service_manager.monitor().values())
    assert ServiceStatus.RUNNING not in statuses


def test_smoke_start_ready_but_unclean_shutdown_reports_failed_honestly(tmp_path):
    """Owner-observed defect 2026-08-22 (found via a real autotune run): a model
    that became ready but did not shut down cleanly used to print
    outcome="ready"/reason="health endpoint confirmed ready" while ALSO exiting
    1 - the exit code already required both ready and clean, but the JSON never
    got told the shutdown half failed. autotune.py's run_smoke_trial (and any
    other caller) has no way to tell a genuine ready success from an unclean
    shutdown if the JSON lies about which one happened. Simulated here with a
    process that ignores the graceful stop signal and a kill_runner that can't
    actually kill it - a real "still running after stop_all" case, not a
    fabricated one.
    """
    created: dict = {}
    lines: list[str] = []
    settings = Settings()
    settings.base_dir = tmp_path

    def config_load(_base=None, env=None):
        return settings, _models(), []

    def spec_builder(model_id, ctx=None, gpu=None):
        return ServiceSpec(
            name=f"llama_cpp:{model_id}",
            command=["llama", "--port", "8080"],
            port=None,
            health_path=None,
            ready_timeout_s=1.0,
            stop_timeout_s=0.1,
        )

    def factory(spec):
        # dies_on_signal=False: ignores the graceful stop. kill_runner is a
        # no-op: the escalation to kill also fails to actually end it. Net
        # effect: still alive after stop_all(), so _confirm_clean() is False
        # even though it genuinely became ready first.
        proc = FakeProcess(poll_sequence=[None], dies_on_signal=False)
        mp = ManagedProcess(
            spec,
            launcher=make_fake_launcher(proc),
            readiness=lambda s, h: True,
            kill_runner=lambda pid: None,
            poll_interval_s=0.01,
        )
        created[spec.name] = (mp, proc)
        return mp

    deps = {
        "config_load": config_load,
        "health_checker": _StubChecker(_pass_report()),
        "output_fn": lines.append,
        "process_factory": factory,
        "spec_builder": spec_builder,
    }

    launcher = Launcher(deps=deps)
    code = launcher.run(["--smoke-start", "qwen3-14b", "--json"])
    result = json.loads(lines[-1])
    # The bug: these three used to disagree with each other.
    assert code == 1
    assert result["outcome"] == "failed"
    assert "clean" in result["reason"] or "running" in result["reason"]
    assert result["reason"] != "health endpoint confirmed ready"


def test_smoke_start_names_the_exception_that_broke_the_shutdown(tmp_path):
    """Regression 2026-08-22: a raising cleanup must be NAMED in the verdict.

    When stop_all() itself raised, the old code assigned the cleanup error to
    result["reason"] only if that field was still empty - and on the
    became-ready-then-failed-to-stop path it never is. So the one fact that
    would have identified the real cause (a SystemError from a consoleless
    CTRL_BREAK; see services.ManagedProcess.stop) was dropped, and the verdict
    read like a VRAM problem. The exception type and message must survive into
    the JSON.
    """
    lines: list[str] = []
    settings = Settings()
    settings.base_dir = tmp_path

    def config_load(_base=None, env=None):
        return settings, _models(), []

    def spec_builder(model_id, ctx=None, gpu=None):
        return ServiceSpec(
            name=f"llama_cpp:{model_id}",
            command=["llama", "--port", "8080"],
            port=None,
            health_path=None,
            ready_timeout_s=1.0,
            stop_timeout_s=0.1,
        )

    def factory(spec):
        proc = FakeProcess(poll_sequence=[None], dies_on_signal=False)

        def refuse_signal(_signal: int) -> None:
            # Exactly what Windows does to a process with no console.
            raise SystemError(
                "<built-in function kill> returned a result with an exception set"
            )

        proc.send_signal = refuse_signal  # type: ignore[method-assign]
        mp = ManagedProcess(
            spec,
            launcher=make_fake_launcher(proc),
            readiness=lambda s, h: True,
            kill_runner=lambda pid: None,
            poll_interval_s=0.01,
        )
        # Make stop() raise on its FIRST call only, standing in for a cleanup
        # that blows up before the manager can report a status at all. One-shot
        # on purpose: the harness also registers stop_all as an atexit backstop,
        # and a permanently-raising stop would throw again at interpreter exit
        # and spray an unrelated traceback over every later test's output.
        real_stop = mp.stop
        raised: list[bool] = []

        def stop_once():
            if not raised:
                raised.append(True)
                raise SystemError("no console")
            return real_stop()

        mp.stop = stop_once  # type: ignore[method-assign]
        return mp

    deps = {
        "config_load": config_load,
        "health_checker": _StubChecker(_pass_report()),
        "output_fn": lines.append,
        "process_factory": factory,
        "spec_builder": spec_builder,
    }

    launcher = Launcher(deps=deps)
    code = launcher.run(["--smoke-start", "qwen3-14b", "--json"])
    result = json.loads(lines[-1])
    assert code == 1
    assert result["outcome"] == "failed"
    assert "SystemError" in result["reason"], result["reason"]
    assert "no console" in result["reason"]


# ---- M3: whisper-stream listen lifecycle wired into the launcher --------- #


def _fake_stream_factory(created: dict):
    """A process factory that starts whisper-stream with a fake process only."""

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


def test_smoke_listen_lifecycle_ok_with_fake_process(tmp_path):
    """AC6 (fake-process form): --smoke-listen starts whisper-stream, waits the
    window, stops it cleanly, reports outcome 'ok', and leaves no orphan. No real
    binary or microphone is touched."""
    created: dict = {}
    lines: list[str] = []
    settings = Settings()
    settings.base_dir = tmp_path
    settings.paths.whisper_stream = "whisper-stream.exe"
    settings.paths.whisper_model = "ggml.bin"

    def config_load(_base=None, env=None):
        return settings, _models(), []

    deps = {
        "config_load": config_load,
        "health_checker": _StubChecker(_pass_report()),
        "output_fn": lines.append,
        "process_factory": _fake_stream_factory(created),
    }
    code = Launcher(deps=deps).run(["--smoke-listen", "--duration", "0.01", "--json"])
    assert code == 0
    result = json.loads(lines[-1])
    assert result["outcome"] == "ok"
    assert result["service"] == "whisper_stream"
    # Cleanup proof: the started process received a stop signal.
    _mp, proc = created["whisper_stream"]
    assert proc.signals, "the listen window should have stopped whisper-stream"


def test_smoke_listen_reports_failure_when_stream_path_unset(tmp_path):
    """Honest guard: with no whisper-stream path configured, --smoke-listen fails
    with a clear reason and exit 1 rather than pretending to capture."""
    lines: list[str] = []
    settings = Settings()  # no whisper_stream / whisper_model paths
    settings.base_dir = tmp_path

    def config_load(_base=None, env=None):
        return settings, _models(), []

    deps = {
        "config_load": config_load,
        "health_checker": _StubChecker(_pass_report()),
        "output_fn": lines.append,
    }
    code = Launcher(deps=deps).run(["--smoke-listen", "--duration", "0.01", "--json"])
    assert code == 1
    result = json.loads(lines[-1])
    assert result["outcome"] == "failed"
    assert result["reason"]


# --------------------------------------------------------------------------- #
# Launch-time thinking selection (owner request 2026-09-02): a model number
# alone starts the model as registered; an optional suffix picks a thinking
# level for THAT LAUNCH ONLY, never writing models.yaml.
# --------------------------------------------------------------------------- #


class _RecordingController:
    """Captures the kwargs _dispatch_model forwards to controller.switch."""

    running_model_id = None

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def switch(self, model_id, *args, **kwargs):
        from services import ServiceStatus

        self.calls.append((model_id, args, kwargs))
        return ServiceStatus.RUNNING

    def snapshot(self):
        return {"services": [{"model_id": "qwen3-14b", "port": 8080, "pid": 1}]}


def _dispatch(reasoning_text: str):
    lines: list[str] = []
    controller = _RecordingController()
    launcher = Launcher(deps={"output_fn": lines.append})
    installed = [m for m in _models().models if m.status == "installed"]
    launcher._dispatch_model(
        controller, installed, 1, reasoning_text=reasoning_text
    )
    return controller, "\n".join(lines)


def test_plain_model_number_passes_no_reasoning_override():
    """The keyword must be OMITTED, not passed as None.

    build_start_spec reads reasoning=None as the deliberate 'force thinking off
    for this scenario', so passing None here would silently disable thinking on
    every ordinary model start.
    """
    controller, _text = _dispatch("")
    assert len(controller.calls) == 1
    assert "reasoning" not in controller.calls[0][2]


def test_model_number_with_a_level_forwards_the_override():
    controller, text = _dispatch("low")
    assert controller.calls[0][2]["reasoning"] == {"effort": "low"}
    assert "thinking for this launch" in text


def test_model_number_with_level_and_budget_forwards_both():
    controller, _text = _dispatch("low/2048")
    assert controller.calls[0][2]["reasoning"] == {"effort": "low", "budget": 2048}


def test_model_number_with_off_turns_thinking_off_for_the_launch():
    controller, _text = _dispatch("off")
    assert controller.calls[0][2]["reasoning"] == {"enabled": False}


def test_a_bad_level_refuses_to_start_rather_than_starting_wrong():
    """A typo must not quietly start the model at a setting the owner did not
    choose - that is the whole reason this is validated instead of passed
    through to server_args."""
    controller, text = _dispatch("low/abc")
    assert controller.calls == []
    assert "not starting" in text


def test_menu_parses_the_thinking_suffix_after_the_model_number(tmp_path):
    """End to end through the menu loop: '1 low' reaches _dispatch_model."""
    lines: list[str] = []
    choices = iter(["1 low", "q"])
    settings = Settings()
    settings.base_dir = tmp_path
    settings.launcher.auto_journal = False
    seen: list[tuple] = []

    def config_load(_base=None, env=None):
        return settings, _models(), []

    launcher = Launcher(
        deps={
            "config_load": config_load,
            "health_checker": _StubChecker(_pass_report()),
            "output_fn": lines.append,
            "input_fn": lambda _p: next(choices),
        }
    )
    launcher._dispatch_model = lambda *a, **k: seen.append((a, k))
    assert launcher.run([]) == 0
    assert len(seen) == 1
    assert seen[0][0][2] == 1  # model number
    assert seen[0][1]["reasoning_text"] == "low"


def test_menu_still_advertises_the_thinking_suffix():
    """An unadvertised suffix is an undiscoverable feature."""
    lines: list[str] = []
    Launcher(deps=_deps(_pass_report(), lines, input_fn=lambda _p: "q")).run([])
    text = "\n".join(lines)
    assert "thinking level" in text and "3 low/2048" in text


# --------------------------------------------------------------------------- #
# Where the switched-to model's memory went (owner report 2026-09-03, "when I
# switch models Open WebUI does not chat"): the smoke verdict carries the
# measured placement, the switch path says it out loud, and neither invents a
# number when the counters cannot be read.
# --------------------------------------------------------------------------- #


def test_smoke_start_verdict_carries_the_measured_gpu_placement(tmp_path, monkeypatch):
    import gpu_ledger

    seen: list = []

    def fake_placement(pid, run=None):
        seen.append(pid)
        return gpu_ledger.GpuPlacement(pid=int(pid), dedicated_mb=14483, shared_mb=782)

    monkeypatch.setattr(gpu_ledger, "placement_of", fake_placement)
    created: dict = {}
    lines: list[str] = []
    settings = Settings()
    settings.base_dir = tmp_path

    def config_load(_base=None, env=None):
        return settings, _models(), []

    deps = {
        "config_load": config_load,
        "health_checker": _StubChecker(_pass_report()),
        "output_fn": lines.append,
    }
    deps.update(_fake_service_deps(created, ready=True))
    code = Launcher(deps=deps).run(["--smoke-start", "qwen3-14b", "--json"])
    assert code == 0
    result = json.loads(lines[-1])
    assert result["gpu_placement"] == {"dedicated_mb": 14483, "shared_mb": 782, "spilled": True}
    assert seen == [result["pid"]]


def test_smoke_placement_is_null_not_zero_when_unmeasurable(monkeypatch):
    import gpu_ledger

    monkeypatch.setattr(gpu_ledger, "placement_of", lambda pid, run=None: None)
    assert Launcher._smoke_placement(4242) is None


def test_describe_placement_finds_the_served_pid_and_is_honest_when_it_cannot(monkeypatch):
    import gpu_ledger

    class _Controller:
        def snapshot(self):
            return {"services": [
                {"name": "whisper", "model_id": None, "status": "running", "port": 8091, "pid": 11},
                {"name": "llama", "model_id": "qwen3-8-27b", "status": "running", "port": 8080, "pid": 4242},
            ]}

    asked: list = []

    def fake_placement(pid, run=None):
        asked.append(pid)
        return gpu_ledger.GpuPlacement(pid=pid, dedicated_mb=13900, shared_mb=158)

    monkeypatch.setattr(gpu_ledger, "placement_of", fake_placement)
    line = Launcher._describe_placement(_Controller(), "qwen3-8-27b")
    assert asked == [4242]
    assert line == "qwen3-8-27b: 13900 MB on the GPU, 158 MB shared."

    monkeypatch.setattr(gpu_ledger, "placement_of", lambda pid, run=None: None)
    assert Launcher._describe_placement(_Controller(), "qwen3-8-27b") == (
        "qwen3-8-27b: GPU placement not measurable here"
    )
    # A model the controller is not serving: nothing to measure, say so.
    assert "not measurable" in Launcher._describe_placement(_Controller(), "other")


def test_router_switch_logs_the_placement_line_not_just_prints_it():
    """Under pythonw (the desktop) print goes nowhere; the line must reach
    launcher.log or the owner never sees why a switched-to model crawls."""
    import inspect

    import launcher as launcher_module

    src = inspect.getsource(launcher_module.Launcher._ensure_router)
    body = src[src.index("def switch_fn"):src.index("ModelRouter(")]
    assert "_describe_placement(controller, model_id)" in body
    assert 'get_logger("launcher")' in body


def test_speech_service_wiring_injects_the_loaded_noise_suppressor():
    """The router must receive the loaded config, not invent its own default."""
    import inspect

    import launcher as launcher_module

    src = inspect.getsource(launcher_module.Launcher._start_speech_services)
    assert "AudioNoiseSuppressor" in src
    assert "settings.speech.noise_suppression" in src
    assert 'wiring["filter_upload"] = suppressor.process' in src
    assert 'wiring["noise_suppression_mode"]' in src
