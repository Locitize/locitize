"""Service lifecycle tests using a fake process launcher (no real subprocesses).

Proves start/readiness/stop/restart/timeout logic and the taskkill escalation
path by injecting a FakeProcess and asserting against it (Architecture 12).
"""

from __future__ import annotations

from urllib.parse import urlparse

from fakes import FakeProcess, make_fake_launcher
from services import (
    ManagedProcess,
    ModelController,
    PortAllocator,
    PortUnavailableError,
    ServiceManager,
    ServiceSpec,
    ServiceStatus,
    SingleServiceController,
    build_readiness_url,
    strip_managed_flags,
)


def _spec(**over) -> ServiceSpec:
    base = dict(
        name="svc",
        command=["fake", "--run"],
        port=None,
        health_path=None,
        ready_timeout_s=1.0,
        stop_timeout_s=0.5,
    )
    base.update(over)
    return ServiceSpec(**base)


def test_start_reaches_running_when_ready():
    """A process that stays alive and passes readiness reaches RUNNING."""
    proc = FakeProcess(poll_sequence=[None])
    mp = ManagedProcess(
        _spec(),
        launcher=make_fake_launcher(proc),
        readiness=lambda spec, handle: True,  # ready immediately
        poll_interval_s=0.01,
    )
    assert mp.start() is ServiceStatus.RUNNING


def test_start_errors_when_process_exits_early():
    """A process that exits before readiness yields STOPPED_ERROR (not RUNNING)."""
    proc = FakeProcess(poll_sequence=[1])  # already exited
    mp = ManagedProcess(
        _spec(),
        launcher=make_fake_launcher(proc),
        readiness=lambda spec, handle: False,
        poll_interval_s=0.01,
    )
    assert mp.start() is ServiceStatus.STOPPED_ERROR


def test_stop_escalates_to_kill_on_wait_timeout():
    """When a process ignores the graceful signal, the manager force-kills it.

    Models a genuine escalation: the child does NOT die on the graceful signal
    (dies_on_signal=False) and the graceful wait times out (wait_raises=True), so
    stop() must taskkill it and then confirm it exited. The trailing 0 in the poll
    sequence is the confirmation the kill landed. (A process that DID exit from the
    graceful signal is intentionally not force-killed -- that avoids taskkilling a
    since-reused PID; see test_stop_sends_graceful_signal_first.)
    """
    proc = FakeProcess(poll_sequence=[None, None, None, 0], dies_on_signal=False)
    proc.wait_raises = True  # force graceful stop to time out
    killed: list[int] = []
    mp = ManagedProcess(
        _spec(),
        launcher=make_fake_launcher(proc),
        kill_runner=lambda pid: killed.append(pid),
        readiness=lambda spec, handle: True,
        poll_interval_s=0.01,
        kill_confirm_timeout_s=0.1,
        kill_max_attempts=2,
    )
    mp.start()
    status = mp.stop()
    # The escalation kill was invoked against the PID we launched, and only that.
    assert killed == [proc.pid]
    assert status is ServiceStatus.STOPPED


def test_stop_sends_graceful_signal_first():
    """Graceful stop sends a signal to the process group before escalating."""
    # Sequence: start readiness poll (None), stop initial poll (None), then the
    # post-wait poll (0 = exited) after the graceful signal.
    proc = FakeProcess(poll_sequence=[None, None, 0])
    mp = ManagedProcess(
        _spec(),
        launcher=make_fake_launcher(proc),
        readiness=lambda spec, handle: True,
        poll_interval_s=0.01,
    )
    mp.start()
    mp.stop()
    assert len(proc.signals) == 1  # exactly one graceful signal delivered


def test_stop_still_kills_when_the_graceful_signal_raises_systemerror():
    """Regression, 2026-08-22: a consoleless GUI must not skip the taskkill.

    LOCITIZE Desktop runs under pythonw.exe, which Windows never gives a console.
    There `os.kill(pid, CTRL_BREAK_EVENT)` fails inside GenerateConsoleCtrlEvent
    with WinError 6 and CPython raises `SystemError`, NOT OSError - so the old
    `except (OSError, ValueError)` here let it escape stop(), stop_all() and the
    entire escalation below it. The llama-server stayed alive and every auto-tune
    trial came back "model became ready but did not shut down cleanly afterward".

    This is the exact shape of that failure: the graceful signal raises
    SystemError, and stop() must still force-kill the child and report STOPPED.
    """
    proc = FakeProcess(poll_sequence=[None, None, None, 0], dies_on_signal=False)

    def refuse_signal(_signal: int) -> None:
        raise SystemError(
            "<built-in function kill> returned a result with an exception set"
        )

    proc.send_signal = refuse_signal  # type: ignore[method-assign]
    killed: list[int] = []
    mp = ManagedProcess(
        _spec(),
        launcher=make_fake_launcher(proc),
        kill_runner=lambda pid: killed.append(pid),
        readiness=lambda spec, handle: True,
        poll_interval_s=0.01,
        kill_confirm_timeout_s=0.1,
        kill_max_attempts=2,
    )
    mp.start()
    status = mp.stop()
    assert killed == [proc.pid], "the escalation kill must still run"
    assert status is ServiceStatus.STOPPED


def test_stop_does_not_wait_out_the_graceful_window_it_could_not_signal():
    """An undeliverable signal escalates at once instead of sleeping first.

    There is nothing in flight to wait for when the signal never left, and
    stop_timeout_s per service is pure dead time on every consoleless shutdown.
    """
    proc = FakeProcess(poll_sequence=[None, None, None, 0], dies_on_signal=False)

    def refuse_signal(_signal: int) -> None:
        raise SystemError("no console")

    proc.send_signal = refuse_signal  # type: ignore[method-assign]
    mp = ManagedProcess(
        _spec(),
        launcher=make_fake_launcher(proc),
        kill_runner=lambda pid: None,
        readiness=lambda spec, handle: True,
        poll_interval_s=0.01,
        kill_confirm_timeout_s=0.1,
        kill_max_attempts=2,
    )
    mp.start()
    mp.stop()
    assert proc.waited is False, "no graceful wait when no signal was delivered"


def test_stop_all_reaches_every_service_when_one_signal_raises_systemerror():
    """One consoleless-signal failure must not abandon the services behind it.

    The 2026-08-21 desktop shutdown trace in locitize-data/logs/errors.log shows
    the SystemError escaping stop_all() mid-loop; anything not yet stopped at
    that point was simply orphaned. stop_all must get through all of them.
    """
    stopped: list[str] = []

    def make(name: str, raises: bool) -> ManagedProcess:
        proc = FakeProcess(poll_sequence=[None, None, None, 0], dies_on_signal=False)
        if raises:
            def refuse_signal(_signal: int) -> None:
                raise SystemError("no console")

            proc.send_signal = refuse_signal  # type: ignore[method-assign]
        return ManagedProcess(
            _spec(name=name),
            launcher=make_fake_launcher(proc),
            kill_runner=lambda _pid, n=name: stopped.append(n),
            readiness=lambda spec, handle: True,
            poll_interval_s=0.01,
            kill_confirm_timeout_s=0.1,
            kill_max_attempts=1,
        )

    manager = ServiceManager()
    for name, raises in (("first", False), ("second", True), ("third", False)):
        manager.register(make(name, raises))
        manager.start(name)
    manager.stop_all()
    assert sorted(stopped) == ["first", "second", "third"]


def test_port_allocator_auto_increments_on_conflict():
    """auto policy skips an occupied port and returns the next free one in range."""
    occupied = {8080}
    alloc = PortAllocator(8080, 8099, "auto", is_free=lambda p: p not in occupied)
    assert alloc.ensure_free(8080) == 8081


def test_port_allocator_strict_raises_on_conflict():
    """strict policy raises rather than silently reassigning."""
    alloc = PortAllocator(8080, 8099, "strict", is_free=lambda p: False)
    try:
        alloc.ensure_free(8080)
        raise AssertionError("expected PortUnavailableError")
    except PortUnavailableError:
        pass


def test_service_manager_stop_all_reverse_order():
    """stop_all stops services in reverse start order (best effort)."""
    order: list[str] = []

    def make(name: str) -> ManagedProcess:
        proc = FakeProcess(poll_sequence=[None, 0])
        mp = ManagedProcess(
            _spec(name=name),
            launcher=make_fake_launcher(proc),
            readiness=lambda spec, handle: True,
            kill_runner=lambda pid: None,
            poll_interval_s=0.01,
        )
        # Wrap stop to record call order.
        original = mp.stop

        def stop_recorded(_orig=original, _name=name):
            order.append(_name)
            return _orig()

        mp.stop = stop_recorded  # type: ignore[method-assign]
        return mp

    manager = ServiceManager()
    a, b = make("a"), make("b")
    manager.register(a)
    manager.register(b)
    manager.start("a")
    manager.start("b")
    manager.stop_all()
    assert order == ["b", "a"]


def _recording_proc(name: str, stopped: list[str]) -> ManagedProcess:
    """Build a fake-launched ManagedProcess that records the order of stop() calls."""
    proc = FakeProcess(poll_sequence=[None, 0])
    mp = ManagedProcess(
        _spec(name=name),
        launcher=make_fake_launcher(proc),
        readiness=lambda spec, handle: True,
        kill_runner=lambda pid: None,
        poll_interval_s=0.01,
    )
    original = mp.stop

    def stop_recorded(_orig=original, _name=name):
        stopped.append(_name)
        return _orig()

    mp.stop = stop_recorded  # type: ignore[method-assign]
    return mp


def test_service_manager_stop_all_covers_registered_but_not_started_service():
    """H-1: stop_all must stop a service that is REGISTERED but whose start() has
    not yet returned (mid-flight) -- present in _services but not yet in
    _start_order.

    ModelController/SingleServiceController register a ManagedProcess BEFORE calling
    the up-to-60s blocking manager.start(). A window close during a model start or
    switch used to leave that mid-flight child untouched, because the old stop_all
    iterated only _start_order, orphaning a multi-GB llama-server. This proves both
    a started service and a registered-but-never-started one are stopped, and that a
    started service still stops first.
    """
    stopped: list[str] = []
    manager = ServiceManager()
    started, mid_flight = (
        _recording_proc("started", stopped),
        _recording_proc("mid-flight", stopped),
    )
    manager.register(started)
    manager.register(mid_flight)
    manager.start("started")  # only this one finished starting
    # "mid-flight" stands in for a service whose blocking start() has not returned:
    # registered, so in _services, but not yet appended to _start_order.
    assert "mid-flight" in manager._services
    assert "mid-flight" not in manager._start_order
    manager.stop_all()
    # Both are stopped, and the cleanly-started one stops before the mid-flight one.
    assert set(stopped) == {"started", "mid-flight"}
    assert stopped.index("started") < stopped.index("mid-flight")


def test_append_log_preserves_previous_session_evidence(tmp_path):
    """D-M4-3: an append_log spec keeps prior content and adds a dated separator.

    Two starts against the same log file must both leave a session banner and the
    first start's bytes must survive the second start (truncation would destroy the
    evidence the D-M4-1 investigation lost).
    """
    log_file = tmp_path / "whisper_server.log"
    log_file.write_bytes(b"previous session trace\n")

    def start_once() -> None:
        proc = FakeProcess(poll_sequence=[None, 0])
        mp = ManagedProcess(
            _spec(name="whisper_server", log_path=str(log_file), append_log=True),
            launcher=make_fake_launcher(proc),
            readiness=lambda spec, handle: True,
            kill_runner=lambda pid: None,
            poll_interval_s=0.01,
        )
        mp.start()
        mp.stop()

    start_once()
    start_once()
    text = log_file.read_text(encoding="ascii")
    # Original evidence survived and two dated banners were written.
    assert "previous session trace" in text
    assert text.count("locitize service start whisper_server") == 2


def test_truncating_log_default_starts_empty(tmp_path):
    """A spec WITHOUT append_log (transcript-capture default) truncates as before."""
    log_file = tmp_path / "whisper_stream.log"
    log_file.write_bytes(b"stale transcript\n")
    proc = FakeProcess(poll_sequence=[None, 0])
    mp = ManagedProcess(
        _spec(name="whisper_stream", log_path=str(log_file)),  # append_log defaults False
        launcher=make_fake_launcher(proc),
        readiness=lambda spec, handle: True,
        kill_runner=lambda pid: None,
        poll_interval_s=0.01,
    )
    mp.start()
    mp.stop()
    assert log_file.read_bytes() == b""  # truncated, no stale transcript surfaced


class _StubbornProcess(FakeProcess):
    """A process that outlives the ready window and ignores signals and taskkill.

    Models the D-M4-1 reality: a 15GB llama-server that never becomes ready within
    the window and stays alive even after a force-kill is issued (it sits in an
    uninterruptible load). poll() therefore always reports alive (None), so the
    manager must ESCALATE and the escalation must never be believed without
    confirmation. Records every kill it was asked to perform.
    """

    def __init__(self, kills: list[int]) -> None:
        # poll_sequence=[None] -> always alive; never dies on the graceful signal;
        # wait() raises so the graceful window is treated as elapsed.
        super().__init__(pid=44920, poll_sequence=[None], dies_on_signal=False)
        self.wait_raises = True
        self._kills = kills

    def record_kill(self, pid: int) -> None:
        self._kills.append(pid)


def test_timeout_orphan_child_is_stopped_on_timeout_and_by_stop_all():
    """D-M4-1: a readiness-timeout must STOP the child (not orphan it), and the

    child must stay registered so a later stop_all() re-reaches and re-kills it.
    A fake process that outlives the ready window and survives kills proves both:
    (1) start() escalates to taskkill on the timeout itself, and (2) stop_all()
    issues further kills against the same still-live child rather than skipping it.
    """
    kills: list[int] = []
    proc = _StubbornProcess(kills)
    mp = ManagedProcess(
        _spec(name="llama_cpp:qwen3-6-27b", ready_timeout_s=0.05, stop_timeout_s=0.02),
        launcher=make_fake_launcher(proc),
        readiness=lambda spec, handle: False,  # never becomes ready
        kill_runner=proc.record_kill,
        poll_interval_s=0.005,
        kill_confirm_timeout_s=0.02,
        kill_max_attempts=2,
    )
    manager = ServiceManager()
    manager.register(mp)

    # (1) Start hits the readiness timeout. The partial child must be force-killed
    # before STOPPED_ERROR is reported -- it must NOT be left running.
    status = manager.start("llama_cpp:qwen3-6-27b")
    assert status is ServiceStatus.STOPPED_ERROR
    assert kills, "readiness timeout must taskkill the still-loading child, not orphan it"
    assert kills == [proc.pid] * len(kills)  # only ever the PID we launched
    kills_after_timeout = len(kills)

    # The service must remain registered and its handle retained, because it could
    # not be confirmed dead -- otherwise nothing could ever finish killing it.
    assert "llama_cpp:qwen3-6-27b" in manager._services
    assert mp.pid == proc.pid

    # (2) A subsequent stop_all() (the atexit / GuiController.shutdown backstop)
    # must re-reach this still-live child and issue further kills, never skip it.
    manager.stop_all()
    assert len(kills) > kills_after_timeout, (
        "stop_all must re-kill a child that survived the timeout stop"
    )


def test_stop_reports_stopped_once_kill_confirms_death():
    """A force-kill that DOES take effect is confirmed and reported STOPPED.

    Complements the orphan test: when taskkill actually reaps the process (poll
    flips to an exit code after the kill), stop() confirms the exit and returns
    STOPPED rather than a false STOPPED_ERROR.
    """
    # Alive through start + graceful window, then exits once the kill lands.
    proc = FakeProcess(poll_sequence=[None, None, None, 0], dies_on_signal=False)
    proc.wait_raises = True

    def kill(pid: int) -> None:
        # Model taskkill actually working: the next poll reports exited.
        proc._sequence = [0]

    mp = ManagedProcess(
        _spec(ready_timeout_s=1.0, stop_timeout_s=0.02),
        launcher=make_fake_launcher(proc),
        readiness=lambda spec, handle: True,
        kill_runner=kill,
        poll_interval_s=0.005,
        kill_confirm_timeout_s=0.1,
        kill_max_attempts=2,
    )
    mp.start()
    assert mp.stop() is ServiceStatus.STOPPED


def test_reassigned_port_propagates_to_argv_and_health_probe():
    """Reviewer M-1: an auto-reassigned port reaches BOTH the launch argv and the
    readiness health probe (previously a silent no-op)."""
    captured: dict = {}

    def capturing_launcher(**kwargs):
        captured["args"] = list(kwargs["args"])
        return FakeProcess(poll_sequence=[None])

    probed_ports: list = []

    def readiness(spec, handle):
        # The probe spec must carry the resolved port; the URL it composes must
        # target that port on loopback.
        probed_ports.append(spec.port)
        captured["probe_url"] = build_readiness_url(spec.port, spec.health_path)
        return True

    # Port 8080 is occupied, so the auto allocator reassigns to 8081.
    alloc = PortAllocator(8080, 8099, "auto", is_free=lambda p: p != 8080)
    spec = ServiceSpec(
        name="llama",
        command=["llama", "--host", "127.0.0.1", "--port", "8080"],
        port=8080,
        health_path="/health",
        ready_timeout_s=1.0,
        stop_timeout_s=0.5,
    )
    mp = ManagedProcess(
        spec,
        launcher=capturing_launcher,
        readiness=readiness,
        port_allocator=alloc,
        poll_interval_s=0.01,
    )
    assert mp.start() is ServiceStatus.RUNNING
    assert mp.resolved_port == 8081
    # The launch argv carries the reassigned port after --port (not the old 8080).
    args = captured["args"]
    assert args[args.index("--port") + 1] == "8081"
    assert "8080" not in args
    # The readiness probe targeted the reassigned port on loopback.
    assert probed_ports[-1] == 8081
    assert captured["probe_url"] == "http://127.0.0.1:8081/health"


def _kokoro_like_spec(**over) -> ServiceSpec:
    base = dict(
        name="kokoro_server",
        command=["py", "kokoro_server.py", "--port", "8092"],
        port=8092,
        health_path="/health",
        reuse_marker="kokoro_server.py",
        ready_timeout_s=1.0,
        stop_timeout_s=0.5,
    )
    base.update(over)
    return ServiceSpec(**base)


def _never_launch(**kwargs):
    raise AssertionError("a second copy must not be spawned")


def test_a_healthy_copy_on_the_port_is_reused_not_duplicated():
    """2026-09-24: five kokoro copies stacked on 8092/8082/8084/8086/8087 because
    each start found 8092 busy and was auto-reassigned. A healthy copy of the
    same service on the requested port is now adopted instead."""
    alloc = PortAllocator(8080, 8099, "auto", is_free=lambda p: p != 8092)
    mp = ManagedProcess(
        _kokoro_like_spec(),
        launcher=_never_launch,
        readiness=lambda spec, handle: spec.port == 8092,
        port_allocator=alloc,
        listener_cmdline=lambda port: r"python.exe x\platform\kokoro_server.py --port 8092",
    )
    assert mp.start() is ServiceStatus.RUNNING
    assert mp.resolved_port == 8092
    assert mp.pid is None
    assert mp.snapshot() is ServiceStatus.RUNNING
    # Not launched here, so stop() must not kill it.
    assert mp.stop() is ServiceStatus.STOPPED


def test_a_foreign_listener_on_the_port_still_gets_reassigned():
    captured: dict = {}

    def launcher(**kwargs):
        captured["args"] = list(kwargs["args"])
        return FakeProcess(poll_sequence=[None])

    alloc = PortAllocator(8080, 8099, "auto", is_free=lambda p: p != 8092)
    mp = ManagedProcess(
        _kokoro_like_spec(),
        launcher=launcher,
        readiness=lambda spec, handle: True,
        port_allocator=alloc,
        listener_cmdline=lambda port: "other-app.exe --serve",
        poll_interval_s=0.01,
    )
    assert mp.start() is ServiceStatus.RUNNING
    assert mp.resolved_port == 8080
    assert captured["args"][captured["args"].index("--port") + 1] == "8080"


def test_an_unhealthy_copy_is_not_adopted():
    alloc = PortAllocator(8080, 8099, "auto", is_free=lambda p: p != 8092)
    launched: list = []

    def launcher(**kwargs):
        launched.append(kwargs["args"])
        return FakeProcess(poll_sequence=[None])

    mp = ManagedProcess(
        _kokoro_like_spec(),
        launcher=launcher,
        # The existing copy on 8092 fails health; the fresh one comes up.
        readiness=lambda spec, handle: handle is not None,
        port_allocator=alloc,
        listener_cmdline=lambda port: "kokoro_server.py --port 8092",
        poll_interval_s=0.01,
    )
    assert mp.start() is ServiceStatus.RUNNING
    assert len(launched) == 1


def test_a_spec_without_reuse_marker_never_adopts():
    alloc = PortAllocator(8080, 8099, "auto", is_free=lambda p: p != 8092)
    launched: list = []

    def launcher(**kwargs):
        launched.append(kwargs["args"])
        return FakeProcess(poll_sequence=[None])

    mp = ManagedProcess(
        _kokoro_like_spec(reuse_marker=None),
        launcher=launcher,
        readiness=lambda spec, handle: True,
        port_allocator=alloc,
        listener_cmdline=lambda port: "kokoro_server.py --port 8092",
        poll_interval_s=0.01,
    )
    assert mp.start() is ServiceStatus.RUNNING
    assert len(launched) == 1


def test_health_path_cannot_move_probe_off_loopback():
    """Security SEC-1: no crafted health_path can steer the readiness probe host
    off 127.0.0.1, because the host is composed as a fixed constant."""
    crafted = [
        "/health",
        "//evil.example.com/health",
        "/@evil.example.com",
        "/..//evil",
        "/health?redirect=http://evil.example.com",
    ]
    for path in crafted:
        url = build_readiness_url(8080, path)
        assert url is not None
        # Whatever the path, the resolved host stays loopback.
        assert urlparse(url).hostname == "127.0.0.1"


def _switch_factory(created: dict):
    """Return a ManagedProcess factory that records each process it builds."""

    def factory(spec: ServiceSpec) -> ManagedProcess:
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


def test_switch_stops_old_model_and_starts_new():
    """AC4: switching stops the old process (confirmed exited) before running the
    new one, and only the new model is RUNNING afterward."""
    created: dict = {}

    def spec_builder(model_id, ctx=None, gpu=None):
        return ServiceSpec(
            name=f"llama_cpp:{model_id}",
            command=["llama", "--port", "8080"],
            port=None,  # no allocator -> resolved port stays None (not the focus)
            health_path=None,
            ready_timeout_s=1.0,
            stop_timeout_s=0.5,
        )

    manager = ServiceManager()
    controller = ModelController(manager, spec_builder, _switch_factory(created))

    assert controller.start("model-a") is ServiceStatus.RUNNING
    assert controller.running_model_id == "model-a"

    assert controller.switch("model-b") is ServiceStatus.RUNNING
    assert controller.running_model_id == "model-b"

    # The old process was gracefully stopped (received the stop signal, exited).
    old_mp, old_proc = created["llama_cpp:model-a"]
    assert old_proc.signals, "old process should have received a stop signal"
    assert old_mp.status_value is ServiceStatus.STOPPED

    # The controller reports exactly one RUNNING service: the new model.
    snap = controller.snapshot()
    running = [s for s in snap["services"] if s["status"] == "RUNNING"]
    assert [s["model_id"] for s in running] == ["model-b"]


def test_a_model_that_died_underneath_us_is_not_reported_running():
    """Found live: an out-of-band Offload GPU killed llama-server, the cached id
    stayed set, the router believed the model was up, forwarded to a closed port
    and answered 502 until some OTHER model was requested. A crash would do the
    same. The process is the truth, not the bookkeeping."""
    created: dict = {}

    def spec_builder(model_id, ctx=None, gpu=None):
        return ServiceSpec(
            name=f"llama_cpp:{model_id}",
            command=["llama", "--port", "8080"],
            port=None,
            health_path=None,
            ready_timeout_s=1.0,
            stop_timeout_s=0.5,
        )

    controller = ModelController(ServiceManager(), spec_builder, _switch_factory(created))
    assert controller.start("model-a") is ServiceStatus.RUNNING
    assert controller.running_model_id == "model-a"

    # The process exits underneath the controller - nothing told it.
    _, proc = created["llama_cpp:model-a"]
    proc._sequence[:] = [1]

    assert controller.running_model_id is None
    assert controller.running_port is None
    assert controller.snapshot()["running_model"] is None

    # Asking for the SAME model must start it again, not answer "already running".
    first_mp, _ = created["llama_cpp:model-a"]
    assert controller.switch("model-a") is ServiceStatus.RUNNING
    assert controller.running_model_id == "model-a"
    assert created["llama_cpp:model-a"][0] is not first_mp, "a fresh process should have been started"


# ---- L-4: platform-managed flag rejection (AC2, keyword 'managed_flag') ---- #


def test_managed_flag_strips_space_form_port_and_host():
    """A --port/--host in server_args (space form) is stripped with its value, so
    the platform-resolved value is the only one that reaches the argv."""
    cleaned = strip_managed_flags(
        "svc",
        ["--port", "9999", "--host", "0.0.0.0", "--threads", "8"],
        log=_NullLog(),
    )
    # Both managed flags and their values are gone; the harmless flag survives.
    assert cleaned == ["--threads", "8"]
    assert "9999" not in cleaned
    assert "0.0.0.0" not in cleaned


def test_managed_flag_strips_equals_form_port():
    """The '--port=9999' inline form is also rejected (single token)."""
    cleaned = strip_managed_flags("svc", ["--port=9999", "--keep", "x"], log=_NullLog())
    assert cleaned == ["--keep", "x"]


def test_managed_flag_build_start_spec_platform_port_always_wins():
    """Reviewer L-4 end to end: a model whose server_args tries to set --port cannot
    override the platform-resolved port -- build_start_spec emits exactly one --port
    carrying the settings port, never the owner's 9999."""
    from config import Model, ModelRegistryData, Settings
    from models import ModelRegistry

    settings = Settings()
    settings.paths.llama_cpp = "llama-server.exe"
    data = ModelRegistryData(
        models=[
            Model(
                id="m",
                name="M",
                description="d",
                location="/locitize-test/m.gguf",
                context_size=8192,
                gpu_layers=-1,
                status="installed",
                server_args=["--port", "9999", "--host", "0.0.0.0"],
            )
        ]
    )
    spec = ModelRegistry(data, settings).build_start_spec("m")
    assert spec.command.count("--port") == 1
    assert spec.command[spec.command.index("--port") + 1] == "8080"
    assert "9999" not in spec.command
    # The loopback host is the only --host, never the owner's 0.0.0.0.
    assert spec.command.count("--host") == 1
    assert spec.command[spec.command.index("--host") + 1] == "127.0.0.1"
    assert "0.0.0.0" not in spec.command


class _NullLog:
    """Swallow warnings so managed-flag tests assert behavior, not log output."""

    def warning(self, *_args, **_kwargs) -> None:
        return None


# ---- Whisper service lifecycle (AC3, keyword 'whisper_lifecycle') ---------- #


def _whisper_settings(occupied_port: int | None = None):
    """Minimal Settings with whisper paths set so build_whisper_server_spec works."""
    from config import Settings

    settings = Settings()
    settings.paths.whisper = "whisper-server.exe"
    settings.paths.whisper_model = "ggml.bin"
    # ports.whisper defaults to 8091 (Data Model); left as-is to prove the policy.
    return settings


def _whisper_factory(created: dict, occupied: set[int] | None = None):
    """Factory building a fake-process whisper ManagedProcess with a real allocator.

    The PortAllocator honors the reserved range/policy exactly as production does,
    so the resolved port reflects settings.ports.whisper and the auto-reassign
    policy -- but no real whisper-server is spawned (fake launcher/readiness)."""
    occupied = occupied or set()

    def factory(spec: ServiceSpec) -> ManagedProcess:
        proc = FakeProcess(poll_sequence=[None])
        alloc = PortAllocator(8080, 8099, "auto", is_free=lambda p: p not in occupied)
        mp = ManagedProcess(
            spec,
            launcher=make_fake_launcher(proc),
            readiness=lambda s, h: True,
            kill_runner=lambda pid: None,
            port_allocator=alloc,
            poll_interval_s=0.01,
        )
        created[spec.name] = (mp, proc)
        return mp

    return factory


def test_whisper_lifecycle_starts_on_reserved_port_and_stops_clean():
    """AC3: the whisper service starts on settings.ports.whisper (8091), reaches
    RUNNING, stops cleanly, and leaves no orphan -- same guarantees as llama.cpp."""
    from whisper import WHISPER_SERVER_NAME, build_whisper_server_spec

    settings = _whisper_settings()
    created: dict = {}
    manager = ServiceManager()
    controller = SingleServiceController(
        manager,
        lambda: build_whisper_server_spec(settings),
        _whisper_factory(created),
    )

    assert controller.start() is ServiceStatus.RUNNING
    # Port policy honored: the reserved whisper port from settings (8091).
    assert controller.resolved_port == 8091
    assert controller.is_running() is True

    assert controller.stop() is ServiceStatus.STOPPED
    assert controller.is_running() is False
    # No orphan: the manager reports the service STOPPED, and the fake process
    # received the graceful stop signal (proof stop() reached it).
    assert all(s is ServiceStatus.STOPPED for s in manager.monitor().values())
    _mp, proc = created[WHISPER_SERVER_NAME]
    assert proc.signals, "whisper process should have received a stop signal"


def test_whisper_lifecycle_reassigns_when_reserved_port_busy():
    """AC3: if the reserved whisper port is occupied, the auto policy reassigns
    within the range (never silently binds the busy port)."""
    from whisper import build_whisper_server_spec

    settings = _whisper_settings()
    created: dict = {}
    manager = ServiceManager()
    # 8080 and 8091 busy -> allocator must land on some other free port in range.
    controller = SingleServiceController(
        manager,
        lambda: build_whisper_server_spec(settings),
        _whisper_factory(created, occupied={8080, 8091}),
    )

    assert controller.start() is ServiceStatus.RUNNING
    assert controller.resolved_port != 8091
    assert 8080 <= controller.resolved_port <= 8099
    controller.stop()
    assert all(s is ServiceStatus.STOPPED for s in manager.monitor().values())


def test_whisper_lifecycle_stop_all_reaches_whisper():
    """AC3 / L-1: a whisper service registered on the shared manager is torn down by
    stop_all (the atexit/finally no-orphan path), just like a model service."""
    from whisper import WHISPER_SERVER_NAME, build_whisper_server_spec

    settings = _whisper_settings()
    created: dict = {}
    manager = ServiceManager()
    controller = SingleServiceController(
        manager,
        lambda: build_whisper_server_spec(settings),
        _whisper_factory(created),
    )
    controller.start()
    manager.stop_all()
    _mp, proc = created[WHISPER_SERVER_NAME]
    assert proc.signals, "stop_all should have signalled the whisper service"
    assert all(s is ServiceStatus.STOPPED for s in manager.monitor().values())


def test_monitor_returns_snapshot():
    """monitor() returns a status per registered service without side effects."""
    proc = FakeProcess(poll_sequence=[None])
    mp = ManagedProcess(
        _spec(name="svc"),
        launcher=make_fake_launcher(proc),
        readiness=lambda spec, handle: True,
        poll_interval_s=0.01,
    )
    manager = ServiceManager()
    manager.register(mp)
    manager.start("svc")
    snap = manager.monitor()
    assert snap["svc"] is ServiceStatus.RUNNING


# --------------------------------------------------------------------------- #
# Stale-server guard (owner-observed 2026-09-03).
#
# A force-killed session left its llama-server alive. The next session started
# a SECOND model beside it - the port allocator correctly moved to 8081, and
# nothing checked the GPU - so two 13GB models shared a 16.3GB card and the new
# one ran with layers spilled to CPU at 0.31 tok/s. It looked hung.
#
# The M8.2 one-model-at-a-time discipline only ever covered models THIS
# controller started; an orphan from a dead session was invisible to it.
# --------------------------------------------------------------------------- #


def _guard_spec_builder(model_id, ctx=None, gpu=None, **kwargs):
    return ServiceSpec(
        name=f"llama_cpp:{model_id}",
        command=["llama", "--port", "8080"],
        port=None,
        health_path=None,
        ready_timeout_s=1.0,
        stop_timeout_s=0.5,
    )


def test_a_stray_server_refuses_the_start_instead_of_stacking_on_it():
    """Starting anyway is the one option that is definitely wrong: it produces
    a model that loads, serves, and is unusably slow, with nothing saying why."""
    import pytest

    created: dict = {}
    controller = ModelController(
        ServiceManager(),
        _guard_spec_builder,
        _switch_factory(created),
        lambda own_pids: ["llama-server.exe pid 4388"],
    )
    with pytest.raises(ValueError) as excinfo:
        controller.start("model-a")
    message = str(excinfo.value)
    assert "already holding the GPU" in message
    assert "pid 4388" in message          # names the stray, so it can be found
    assert "Offload GPU" in message       # and the remedy that already exists
    assert created == {}, "no process may be spawned when the guard refuses"


def test_a_clean_gpu_starts_normally():
    created: dict = {}
    controller = ModelController(
        ServiceManager(), _guard_spec_builder, _switch_factory(created),
        lambda own_pids: [],
    )
    assert controller.start("model-a") is ServiceStatus.RUNNING


def test_our_own_tracked_server_is_not_a_stray():
    """A switch stops the current model first, so a server WE started must
    never trip the guard - otherwise every switch after the first refuses."""
    created: dict = {}
    seen: list = []

    def check(own_pids):
        seen.append(set(own_pids))
        return []

    controller = ModelController(
        ServiceManager(), _guard_spec_builder, _switch_factory(created), check
    )
    controller.start("model-a")
    controller.switch("model-b")
    # The second call was told about the pid from the first start.
    assert len(seen) == 2
    assert seen[1], "the guard must be given the pids this controller owns"


def test_a_guard_that_cannot_see_does_not_block_the_start():
    """A guard that cannot see is not a reason to refuse a start the owner
    asked for - it degrades to the pre-guard behaviour."""

    def boom(own_pids):
        raise RuntimeError("nvidia-smi went away")

    created: dict = {}
    controller = ModelController(
        ServiceManager(), _guard_spec_builder, _switch_factory(created), boom
    )
    assert controller.start("model-a") is ServiceStatus.RUNNING


def test_no_guard_configured_is_the_existing_behaviour():
    created: dict = {}
    controller = ModelController(
        ServiceManager(), _guard_spec_builder, _switch_factory(created)
    )
    assert controller.start("model-a") is ServiceStatus.RUNNING


def test_closing_the_app_stops_services_in_parallel_with_a_short_grace():
    """The close path: every service stops at once, each with the short grace,
    so closing takes about one stop rather than the sum of every full window."""
    import threading
    import time

    from services import ServiceManager

    manager = ServiceManager()
    graces, running_at_once, lock = [], [], threading.Lock()
    active = [0]

    class SlowService:
        def __init__(self, name):
            self.name = name

        def stop(self, grace_s=None):
            with lock:
                active[0] += 1
                running_at_once.append(active[0])
            graces.append(grace_s)
            time.sleep(0.3)
            with lock:
                active[0] -= 1

    for name in ("a", "b", "c"):
        manager._services[name] = SlowService(name)
    started = time.monotonic()
    manager.stop_all(grace_s=0)
    elapsed = time.monotonic() - started
    assert graces == [0, 0, 0]
    assert max(running_at_once) == 3
    assert elapsed < 0.8  # parallel: about one stop, not three
