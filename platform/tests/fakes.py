"""Fake providers and processes for the LOCITIZE test suite.

These fakes are the injected collaborators that let the platform be tested with
no GPU, no external binaries, and no real subprocesses (Architecture section 12).
They live in a shared module so multiple test files reuse them.
"""

from __future__ import annotations

from typing import Any

from health import GpuInfo


class FakeSystemInfoProvider:
    """Fixed host facts so health tests are deterministic."""

    def __init__(
        self,
        version: tuple[int, int, int] = (3, 11, 7),
        in_venv: bool = True,
        ram_total: float = 32000.0,
        ram_available: float = 16000.0,
        disk_free: float = 500000.0,
    ) -> None:
        self._version = version
        self._in_venv = in_venv
        self._ram_total = ram_total
        self._ram_available = ram_available
        self._disk_free = disk_free

    def python_version(self) -> tuple[int, int, int]:
        return self._version

    def in_virtualenv(self) -> bool:
        return self._in_venv

    def ram_total_mb(self) -> float:
        return self._ram_total

    def ram_available_mb(self) -> float:
        return self._ram_available

    def disk_free_mb(self, path: str) -> float:
        return self._disk_free


class FakeGpuInfoProvider:
    """GPU provider whose gpus() is whatever the test sets (None = no GPU)."""

    def __init__(self, gpus: list[GpuInfo] | None) -> None:
        self._gpus = gpus

    def gpus(self) -> list[GpuInfo] | None:
        return self._gpus


class FakeBinaryProbeProvider:
    """Binary provider driven by a set of paths that 'exist' and 'run'."""

    def __init__(self, existing: set[str] | None = None, runnable: set[str] | None = None) -> None:
        self._existing = existing or set()
        self._runnable = runnable or set()

    def exists(self, path: str) -> bool:
        return bool(path) and path in self._existing

    def runnable(self, path: str) -> bool:
        return bool(path) and path in self._runnable


class FakePortProbeProvider:
    """Port provider that reports a fixed set of ports as occupied."""

    def __init__(self, occupied: set[int] | None = None) -> None:
        self._occupied = occupied or set()

    def is_free(self, port: int) -> bool:
        return port not in self._occupied


class FakeProcess:
    """Minimal stand-in for subprocess.Popen for service lifecycle tests.

    `poll_sequence` lets a test script the process lifecycle: each poll() pops the
    next value (None = alive, int = exited with that code); the last value repeats.
    """

    def __init__(
        self,
        pid: int = 4321,
        poll_sequence: list[int | None] | None = None,
        dies_on_signal: bool = True,
    ) -> None:
        self.pid = pid
        self._sequence = list(poll_sequence) if poll_sequence else [None]
        self.signals: list[int] = []
        self.waited = False
        self.wait_raises = False
        # A well-behaved process exits when it receives the graceful stop signal;
        # modelling that makes stop/switch tests deterministic without hand-tuning
        # the poll sequence. Set False to model a process that ignores the signal
        # (so the manager must escalate to kill).
        self.dies_on_signal = dies_on_signal

    def poll(self) -> int | None:
        if len(self._sequence) > 1:
            return self._sequence.pop(0)
        return self._sequence[0]

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        if self.wait_raises:
            # Simulate a graceful-stop timeout so the manager escalates to kill.
            raise TimeoutError("wait timed out")
        return 0

    def send_signal(self, signal: int) -> None:
        self.signals.append(signal)
        if self.dies_on_signal:
            # The graceful signal took effect: every later poll reports exited.
            self._sequence = [0]


class FakeSttSource:
    """Replays a fixed list of utterances, then ends the session (None).

    The M7 STT seam for headless tests: no microphone, no whisper. Each
    next_utterance() pops the next canned line; when the list is exhausted it
    returns None so AssistantLoop.run() ends cleanly.
    """

    def __init__(self, utterances: list[str]) -> None:
        self._pending = list(utterances)
        self.calls = 0

    def next_utterance(self) -> str | None:
        self.calls += 1
        if not self._pending:
            return None
        return self._pending.pop(0)


class FakeStreamingLlm:
    """Streams canned deltas as an LlmClient, recording the messages it received.

    `deltas` is the ordered list of reply-text chunks each turn yields (the same
    canned list is replayed every turn). `token_count` is what count_tokens returns
    (None to force the caller's chars/4 fallback). `stop_after` optionally sets the
    shared interrupt Event after yielding that many deltas, to test the M7.5
    interrupt path deterministically.
    """

    def __init__(
        self,
        deltas: list[str],
        token_count: int | None = None,
        stop_after: int | None = None,
    ) -> None:
        self._deltas = list(deltas)
        self._token_count = token_count
        self._stop_after = stop_after
        self.received: list[list[Any]] = []

    def chat_stream(self, messages: Any, interrupt: Any = None) -> Any:
        # Snapshot the exact messages this turn was asked to send (so a test can
        # assert history trimming / role ordering reached the client).
        self.received.append(list(messages))
        for i, delta in enumerate(self._deltas):
            if interrupt is not None and interrupt.is_set():
                return
            yield delta
            if self._stop_after is not None and (i + 1) >= self._stop_after:
                if interrupt is not None:
                    interrupt.set()

    def chat(self, messages: Any) -> str:
        return "".join(self.chat_stream(messages))

    def count_tokens(self, text: str) -> int | None:
        return self._token_count


class FakeTtsSink:
    """Records every speak(text, voice) call instead of playing audio (M7.8).

    Lets a test assert the per-sentence speak sequence and the chosen voice without
    any audio hardware. drain() clears not-yet-consumed calls the way the real
    KokoroTtsSink drops pending playback on interrupt.
    """

    def __init__(self) -> None:
        self.spoken: list[tuple[str, str]] = []
        self.drained = 0

    def speak(self, text: str, voice: str) -> None:
        self.spoken.append((text, voice))

    def drain(self) -> None:
        self.drained += 1


class FakeHttpResponse:
    """A minimal urlopen-style response over a fixed byte body (context manager).

    Iterating yields the body split into lines (with trailing newlines), matching
    how urllib streams an SSE response line by line, so the llm_client SSE parser is
    tested against realistic framing. read() returns the whole body (for /tokenize).
    """

    def __init__(self, body: bytes) -> None:
        self._body = body
        self.closed = False

    def __iter__(self) -> Any:
        for line in self._body.splitlines(keepends=True):
            yield line

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def make_fake_launcher(process: FakeProcess) -> Any:
    """Return a process-launcher callable that yields the given fake process."""

    def _launch(**kwargs: Any) -> FakeProcess:
        # Record nothing here; tests inspect the returned process directly.
        return process

    return _launch
