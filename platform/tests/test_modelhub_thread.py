"""Thread E's non-blocking contract (AC-M14-23).

WHY THIS SUITE EXISTS
---------------------
The ops worker is a SINGLE SERIALIZED consumer of command_q: it pulls one
Command, runs it to completion, then pulls the next. That is deliberate - it is
what makes start/stop/switch need no locking - but it means a multi-gigabyte
download running inside _dispatch would block start, stop, benchmark and
fine-tune for as long as the transfer takes. It would freeze the product, not
just the page.

So a download runs on Thread E ("locitize-model-download"), created by the ops
worker and then left alone. These tests hold a fake download open with an event
and prove the ops worker keeps serving other commands the whole time - which is
the actual defect being guarded against, not a property anyone can eyeball.

Everything here is headless: no network, no Qt, no real model file.
"""

import queue
import threading
import time

import gui_controller
import modelhub


class BlockingDownloader:
    """A fake Downloader whose download() waits until the test releases it."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancelled = False

    def download(self, repo_id, filename, **kwargs):
        self.entered.set()
        cancel = kwargs.get("cancel_event")
        # Wait for either the test's release or a cancel, whichever comes first.
        while not self.release.is_set():
            if cancel is not None and cancel.is_set():
                self.cancelled = True
                return modelhub.DownloadOutcome(
                    False, cancelled=True, error="cancelled by the user"
                )
            time.sleep(0.005)
        return modelhub.DownloadOutcome(
            True, path=None, sha256="e" * 64, verification=modelhub.V_API
        )

    def catalog(self):
        return {"items": [], "reason": ""}


def make_controller(downloader):
    """Build a bare controller with only the fields the hub paths touch.

    Constructed with __new__ rather than the real __init__ on purpose: this
    suite is about the queue and thread behaviour, and pulling in a registry, a
    service manager and a whisper controller would test those instead.
    """
    controller = gui_controller.GuiController.__new__(gui_controller.GuiController)
    controller._settings = type("S", (), {"data_dir": None})()
    controller.command_q = queue.Queue()
    controller.result_q = queue.Queue()
    controller._hub_downloader_factory = lambda: downloader
    controller._hub_downloader = None
    controller._hub_thread = None
    controller._hub_cancel = threading.Event()
    controller._hub_job_id = None
    controller._hub_shutting_down = False
    controller._hub_lock = threading.Lock()
    # Teardown step 0c cancels an in-progress auto-tune before joining the
    # workers, so the shutdown sequence this suite exercises needs the event.
    controller._autotune_cancel = threading.Event()
    controller._ops_thread = None
    controller._monitor_thread = None
    controller._shutdown_started = False
    controller._shutdown_done = False
    controller.shutdown_errors = []
    return controller


def arm_for_shutdown(controller):
    """Give a bare controller the fields the real shutdown() touches.

    Records what teardown actually reached: `calls` names every step that ran, so
    a test can assert the child-process steps (stop_proxy, stop_all) happened even
    when an earlier step blew up.
    """
    calls = []
    controller._shutdown_started = False
    controller._shutdown_done = False
    controller.shutdown_errors = []
    controller._assistant_handle = None
    controller._stop_event = threading.Event()
    controller.stop_proxy = lambda: calls.append("stop_proxy")
    controller._manager = type(
        "M", (), {"stop_all": lambda self: calls.append("stop_all")}
    )()
    return calls


def drain_kinds(result_q, timeout=2.0):
    """Collect every Result kind currently queued, waiting briefly for the first."""
    kinds = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            kinds.append(result_q.get(timeout=0.05).kind)
        except queue.Empty:
            if kinds:
                break
    return kinds


def test_modelhub_thread_put_returns_immediately():
    """The GUI-thread intent is a plain queue put and never blocks."""
    controller = make_controller(BlockingDownloader())
    started = time.perf_counter()
    for _ in range(100):
        controller.request_hub_download({"repo_id": "o/r", "filename": "m.gguf"})
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5, f"100 puts took {elapsed:.3f}s - the intent is blocking"
    assert controller.command_q.qsize() == 100


def test_modelhub_thread_dispatch_returns_while_the_download_runs():
    """The ops worker's handler starts Thread E and returns without waiting."""
    downloader = BlockingDownloader()
    controller = make_controller(downloader)
    started = time.perf_counter()
    controller._dispatch(
        gui_controller.Command("hub_download", {"repo_id": "o/r", "filename": "m.gguf"})
    )
    elapsed = time.perf_counter() - started
    assert downloader.entered.wait(timeout=2.0), "Thread E never started"
    assert elapsed < 0.5, f"_dispatch blocked for {elapsed:.3f}s"
    assert controller._hub_thread.is_alive()
    assert controller._hub_thread.name == "locitize-model-download"
    assert controller._hub_thread.daemon is True
    downloader.release.set()
    controller._hub_thread.join(timeout=3.0)


def test_modelhub_thread_other_commands_run_during_a_download():
    """The freeze this design exists to prevent, asserted directly.

    A real ops worker is started, a download is queued and held open, and an
    unrelated command is then queued. It must complete while the transfer is
    still in flight.
    """
    downloader = BlockingDownloader()
    controller = make_controller(downloader)
    served = []

    # A cheap, unrelated command standing in for start/stop/benchmark. Patching
    # the catalog handler keeps this test about scheduling, not about disk.
    controller._do_hub_catalog = lambda: (
        served.append("catalog"),
        controller.result_q.put(gui_controller.Result("hub_catalog", True, {})),
    )

    ops = threading.Thread(target=controller._ops_loop, name="ops", daemon=True)
    ops.start()
    try:
        controller.request_hub_download({"repo_id": "o/r", "filename": "m.gguf"})
        assert downloader.entered.wait(timeout=2.0), "the download never started"
        # The transfer is now mid-flight and will not finish on its own.
        assert not downloader.release.is_set()
        controller.request_hub_catalog()
        deadline = time.monotonic() + 3.0
        while not served and time.monotonic() < deadline:
            time.sleep(0.01)
        assert served == ["catalog"], (
            "an unrelated command did not run while a download was in flight - "
            "the ops worker is blocked"
        )
        assert downloader.entered.is_set() and not downloader.release.is_set()
    finally:
        downloader.release.set()
        controller.command_q.put(gui_controller.Command("sentinel"))
        ops.join(timeout=3.0)
        if controller._hub_thread is not None:
            controller._hub_thread.join(timeout=3.0)


def test_modelhub_thread_refuses_a_second_concurrent_download():
    """Exactly one download at a time; the second is refused, not queued."""
    downloader = BlockingDownloader()
    controller = make_controller(downloader)
    controller._dispatch(
        gui_controller.Command("hub_download", {"repo_id": "o/r", "filename": "a.gguf"})
    )
    assert downloader.entered.wait(timeout=2.0)
    controller._dispatch(
        gui_controller.Command("hub_download", {"repo_id": "o/r", "filename": "b.gguf"})
    )
    result = controller.result_q.get(timeout=2.0)
    assert result.kind == "hub_download_done"
    assert result.ok is False
    assert result.error == "A download is already running."
    downloader.release.set()
    controller._hub_thread.join(timeout=3.0)


def test_modelhub_thread_cancel_reaches_the_worker():
    """A cancel command sets the event Thread E watches every chunk."""
    downloader = BlockingDownloader()
    controller = make_controller(downloader)
    controller._dispatch(
        gui_controller.Command("hub_download", {"repo_id": "o/r", "filename": "m.gguf"})
    )
    assert downloader.entered.wait(timeout=2.0)
    controller._dispatch(gui_controller.Command("hub_cancel"))
    controller._hub_thread.join(timeout=3.0)
    assert downloader.cancelled is True
    kinds = drain_kinds(controller.result_q)
    assert "hub_download_done" in kinds


def test_modelhub_thread_publishes_progress_onto_the_same_result_queue():
    """Thread E is a fourth PRODUCER on result_q - it adds no second consumer."""

    class ProgressingDownloader:
        def download(self, repo_id, filename, **kwargs):
            callback = kwargs.get("progress")
            for done in (1000, 2000):
                callback(modelhub.Progress("downloading", done, 2000, 500.0, 1.0))
            return modelhub.DownloadOutcome(True, sha256="f" * 64)

    controller = make_controller(ProgressingDownloader())
    controller._dispatch(
        gui_controller.Command("hub_download", {"repo_id": "o/r", "filename": "m.gguf"})
    )
    controller._hub_thread.join(timeout=3.0)
    kinds = drain_kinds(controller.result_q)
    assert kinds.count("hub_download_progress") == 2
    assert kinds[-1] == "hub_download_done"


def test_modelhub_thread_never_raises_across_the_boundary():
    """An exception inside Thread E becomes a failure Result, never a crash."""

    class ExplodingDownloader:
        def download(self, *args, **kwargs):
            raise RuntimeError("disk on fire")

    controller = make_controller(ExplodingDownloader())
    controller._dispatch(
        gui_controller.Command("hub_download", {"repo_id": "o/r", "filename": "m.gguf"})
    )
    controller._hub_thread.join(timeout=3.0)
    result = controller.result_q.get(timeout=2.0)
    assert result.ok is False
    assert "disk on fire" in result.error


def test_modelhub_thread_shutdown_cancels_and_joins():
    """Window close cancels the transfer and joins Thread E with a bounded wait."""
    downloader = BlockingDownloader()
    controller = make_controller(downloader)
    controller._shutdown_done = False
    controller._assistant_handle = None
    controller._stop_event = threading.Event()
    controller.stop_proxy = lambda: None
    controller._manager = type("M", (), {"stop_all": lambda self: None})()
    controller._dispatch(
        gui_controller.Command("hub_download", {"repo_id": "o/r", "filename": "m.gguf"})
    )
    assert downloader.entered.wait(timeout=2.0)
    controller.shutdown()
    assert controller._hub_cancel.is_set()
    assert not controller._hub_thread.is_alive()
    assert downloader.cancelled is True


def test_modelhub_thread_shutdown_refuses_a_download_queued_behind_it():
    """A download still WAITING in command_q at close must never start.

    The race this guards (found in review): shutdown() cancels and joins Thread
    E, then puts the sentinel. The ops worker drains FIFO, so a hub_download
    already sitting in the queue was dispatched AFTER the join - installing a
    fresh, unset cancel event and starting a Thread E that nothing cancels or
    joins, free to keep writing a .partial nobody will clean up.

    Driven synchronously (the ops loop is run by this thread, after shutdown)
    so the ordering is deterministic rather than timing-dependent.
    """
    downloader = BlockingDownloader()
    controller = make_controller(downloader)
    controller._shutdown_done = False
    controller._assistant_handle = None
    controller._stop_event = threading.Event()
    controller.stop_proxy = lambda: None
    controller._manager = type("M", (), {"stop_all": lambda self: None})()

    # Queued but not yet dispatched - exactly the state the race needs.
    controller.request_hub_download({"repo_id": "o/r", "filename": "m.gguf"})
    controller.shutdown()
    # Now let the ops worker drain what shutdown left behind, to the sentinel.
    controller._ops_loop()

    alive = [t.name for t in threading.enumerate() if t.name == "locitize-model-download"]
    assert alive == [], f"a download thread survived shutdown: {alive}"
    assert not downloader.entered.is_set(), "the queued download started anyway"
    assert controller._hub_thread is None
    refusals = []
    while not controller.result_q.empty():
        refusals.append(controller.result_q.get())
    done = next(r for r in refusals if r.kind == "hub_download_done")
    assert done.ok is False
    assert "shutting down" in done.error


def test_modelhub_thread_page_paint_catalog_never_touches_the_network(tmp_path):
    """Opening the Get models page reads the shipped catalog from DISK (rule HF-1).

    Found by the mutation harness: swapping the handler's catalog() read for a
    live search() left the whole suite green, so "the page paint makes no network
    call" was asserted nowhere for the controller path. The proof is mechanical -
    a REAL Downloader whose opener raises the moment anything opens it - rather
    than an assertion about which method was called.
    """

    class ExplodingOpener:
        def open(self, *args, **kwargs):
            raise AssertionError("the catalog page paint made a network call")

    hub = modelhub.Downloader(
        modelhub.HubConfig(),
        opener_factory=lambda hosts: (ExplodingOpener(), None),
        models_dir=tmp_path,
    )
    controller = make_controller(hub)
    controller._dispatch(gui_controller.Command("hub_catalog"))
    result = controller.result_q.get(timeout=2.0)
    assert result.kind == "hub_catalog"
    assert result.ok is True
    assert "items" in result.payload


def test_modelhub_thread_shutdown_survives_an_unstarted_download_thread():
    """HIGH-4: a close landing between Thread(...) and .start() must still tear down.

    A thread object constructed but never started is exactly what _hub_thread held
    for the instant between those two statements in _do_hub_download. join() on
    such a thread raises "RuntimeError: cannot join thread before it is started".
    Before the fix that RuntimeError escaped shutdown() BEFORE stop_proxy() and
    _manager.stop_all(), so the llama.cpp and whisper children were left running -
    and because the idempotence flag was already latched, neither a retry nor the
    atexit backstop could recover. This drives the REAL shutdown() and fails
    without the fix (it errors out at the join).
    """
    controller = make_controller(BlockingDownloader())
    calls = arm_for_shutdown(controller)
    controller._hub_thread = threading.Thread(
        target=lambda: None, name="locitize-model-download", daemon=True
    )  # constructed, deliberately never started

    controller.shutdown()  # must not raise

    assert calls == ["stop_proxy", "stop_all"], (
        f"teardown never reached the child-process steps: {calls} "
        f"errors={controller.shutdown_errors}"
    )
    assert controller._stop_event.is_set()
    assert controller.command_q.get_nowait().kind == "sentinel"
    assert controller._hub_cancel.is_set()
    assert controller._hub_shutting_down is True
    assert controller.shutdown_errors == []
    assert controller._shutdown_done is True


def test_modelhub_thread_shutdown_attempts_every_step_when_earlier_steps_raise():
    """The invariant behind HIGH-4: one broken step never cancels the rest.

    Three separate teardown steps are made to raise. The two steps that reap child
    processes come last, so they are the ones a mid-sequence exception strands;
    they must still run, the failures must be recorded rather than propagated
    (desktop.py's closeEvent cannot handle an exception), and _shutdown_done must
    still end True because teardown WAS attempted end to end.
    """
    controller = make_controller(BlockingDownloader())
    calls = arm_for_shutdown(controller)

    class ExplodingHandle:
        def stop(self):
            raise RuntimeError("assistant wedged")

    class ExplodingHubThread:
        def is_alive(self):
            raise RuntimeError("thread state unreadable")

    class ExplodingOpsThread:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            raise RuntimeError("join refused")

    controller._assistant_handle = ExplodingHandle()
    controller._hub_thread = ExplodingHubThread()
    controller._ops_thread = ExplodingOpsThread()

    controller.shutdown()  # must not raise

    assert calls == ["stop_proxy", "stop_all"], (
        f"an earlier failure aborted teardown: {calls} {controller.shutdown_errors}"
    )
    assert [entry.split(":")[0] for entry in controller.shutdown_errors] == [
        "assistant",
        "download",
        "join",
    ]
    assert controller._stop_event.is_set()
    assert controller._shutdown_done is True


def test_modelhub_thread_download_start_is_atomic_against_a_concurrent_close(monkeypatch):
    """The construct/start window is closed at the source, not only at the join.

    The download thread is held INSIDE start() while a close runs concurrently on
    another thread - the precise interleaving HIGH-4 described. With the fix the
    closer blocks on _hub_lock until the thread is running and published, so it
    always joins a live thread; without it, the closer observed a
    constructed-but-unstarted _hub_thread and raised.
    """
    downloader = BlockingDownloader()
    controller = make_controller(downloader)
    calls = arm_for_shutdown(controller)

    in_start = threading.Event()
    release_start = threading.Event()
    real_thread_cls = threading.Thread

    class GatedThread(real_thread_cls):
        """Pauses inside start() to hold the race window open deterministically."""

        def start(self):
            if self.name == "locitize-model-download":
                in_start.set()
                release_start.wait(5.0)
            super().start()

    monkeypatch.setattr(gui_controller.threading, "Thread", GatedThread)
    starter = real_thread_cls(
        target=controller._do_hub_download,
        args=({"repo_id": "o/r", "filename": "m.gguf"},),
        daemon=True,
    )
    starter.start()
    assert in_start.wait(5.0), "the download thread never reached start()"

    closer_error = []

    def close():
        try:
            controller.shutdown()
        except BaseException as exc:  # noqa: BLE001 - the defect under test
            closer_error.append(repr(exc))

    closer = real_thread_cls(target=close, daemon=True)
    closer.start()
    time.sleep(0.2)  # let the closer reach the lock / the unguarded join
    release_start.set()
    starter.join(timeout=5.0)
    closer.join(timeout=10.0)

    assert closer_error == [], f"shutdown raised during the race: {closer_error}"
    assert calls == ["stop_proxy", "stop_all"], f"teardown was cut short: {calls}"
    assert downloader.cancelled is True
    survivors = [t.name for t in threading.enumerate() if t.name == "locitize-model-download"]
    assert survivors == [], f"a download thread survived shutdown: {survivors}"


def test_modelhub_thread_cancel_message_matches_real_behaviour():
    """The cancel message must not promise a resume that is not implemented.

    Architecture M14.14.3 specifies a resumable .partial plus a sidecar. That is
    NOT built yet - a cancelled transfer deletes its partial file - so the
    message says exactly that, and models_hub.resume_enabled ships false so the
    declared default matches the real behaviour rather than the intended one.
    """
    import config

    downloader = BlockingDownloader()
    controller = make_controller(downloader)
    controller._dispatch(
        gui_controller.Command("hub_download", {"repo_id": "o/r", "filename": "m.gguf"})
    )
    assert downloader.entered.wait(timeout=2.0)
    controller._dispatch(gui_controller.Command("hub_cancel"))
    controller._hub_thread.join(timeout=3.0)
    results = []
    while not controller.result_q.empty():
        results.append(controller.result_q.get())
    done = next(r for r in results if r.kind == "hub_download_done")
    message = done.payload["message"]
    assert "deleted" in message
    assert "resume" not in message.lower()
    assert config.ModelsHubConfig().resume_enabled is False


def test_modelhub_thread_is_the_only_new_thread_name():
    """The threading contract gained exactly one row, named as designed."""
    downloader = BlockingDownloader()
    controller = make_controller(downloader)
    before = {t.name for t in threading.enumerate()}
    controller._dispatch(
        gui_controller.Command("hub_download", {"repo_id": "o/r", "filename": "m.gguf"})
    )
    assert downloader.entered.wait(timeout=2.0)
    new = {t.name for t in threading.enumerate()} - before
    assert new == {"locitize-model-download"}
    downloader.release.set()
    controller._hub_thread.join(timeout=3.0)


def test_modelhub_thread_shutdown_errors_reach_the_application_log(caplog):
    """A failed teardown step is written to the log, not just held in memory.

    SEC-M14-5 / Reviewer MEDIUM-8: shutdown_errors used to die with the process,
    so "LOCITIZE closed but a child survived" left no trace anyone could read.
    """
    import logging

    controller = make_controller(BlockingDownloader())
    arm_for_shutdown(controller)

    class ExplodingHandle:
        def stop(self):
            raise RuntimeError("assistant wedged")

    controller._assistant_handle = ExplodingHandle()
    with caplog.at_level(logging.ERROR, logger="gui_controller"):
        controller.shutdown()

    assert controller.shutdown_errors, "the failure should have been recorded"
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "shutdown step failed" in logged
    assert "assistant wedged" in logged


def test_modelhub_thread_clean_shutdown_logs_nothing(caplog):
    """A teardown with no failures stays silent - no noise in the log at close."""
    import logging

    controller = make_controller(BlockingDownloader())
    arm_for_shutdown(controller)
    with caplog.at_level(logging.ERROR, logger="gui_controller"):
        controller.shutdown()
    assert controller.shutdown_errors == []
    assert [r for r in caplog.records if "shutdown step failed" in r.getMessage()] == []


# =========================================================================== #
# DEFECT-QA-M14-2: a refusal must be attributable to the request that was
# refused, not mistakable for the live job's terminal result.
# =========================================================================== #


def test_modelhub_thread_refusal_names_the_refused_job_not_the_live_one():
    """The "already running" Result carries the REFUSED request's own job id.

    Without this the view could not tell the refusal apart from the running
    download's terminal result, so it applied terminal UI - disabling Cancel and
    hiding the progress bar of a transfer that was still moving bytes.
    """
    import modelhub

    downloader = BlockingDownloader()
    controller = make_controller(downloader)
    controller._dispatch(
        gui_controller.Command("hub_download", {"repo_id": "o/r", "filename": "a.gguf"})
    )
    assert downloader.entered.wait(timeout=2.0)
    live_job = controller._hub_job_id
    controller._dispatch(
        gui_controller.Command("hub_download", {"repo_id": "o/r", "filename": "b.gguf"})
    )
    result = controller.result_q.get(timeout=2.0)
    assert result.ok is False
    assert result.error == "A download is already running."
    assert result.payload["job_id"] == modelhub.hub_job_id("o/r", "b.gguf")
    assert result.payload["job_id"] != live_job
    assert result.payload["refused"] is True
    # The live job is untouched: it is still the controller's current job.
    assert controller._hub_job_id == live_job
    downloader.release.set()
    controller._hub_thread.join(timeout=3.0)
