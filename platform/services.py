"""Process/service lifecycle management for the LOCITIZE platform.

This module starts, stops, restarts, and monitors the local subprocesses the
platform supervises (llama.cpp server, Whisper, Kokoro, and future services),
with Windows-correct process-tree handling and loopback-only port allocation
(Architecture section 4, Permission Matrix section 3).

Testability (Architecture section 12): ManagedProcess takes an injectable
process launcher (default subprocess.Popen) and an injectable "taskkill" runner,
so the full start/stop/restart/timeout/escalation logic is unit-tested with a
fake process object and NO real processes are spawned.

Security invariants enforced here:
- Ports bind 127.0.0.1 only, within the reserved range 8080-8099. Never 0.0.0.0.
- The manager only ever kills PIDs it launched (never a name pattern or unknown
  PID), and taskkill /F /T is scoped to one PID it created.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from logger import get_logger

# Windows creation flag: put the child in its own process group so we can send
# CTRL_BREAK_EVENT to it and its descendants. Imported defensively because the
# constant only exists in subprocess on Windows.
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

# Owner request 2026-08-21 (black-flash-on-launch root cause, same fix applied
# to health.py's nvidia-smi calls): every managed child here (llama-server,
# whisper-server, kokoro) is a console app; the GUI runs under pythonw.exe,
# which has none of its own. Its stdout/stderr are already redirected to a log
# file above (never an unread PIPE), but redirecting output does NOT stop
# Windows from allocating a new console window for the child - only this flag
# does. Combines with CREATE_NEW_PROCESS_GROUP (the flags are independent bits).
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Alternative to CREATE_NO_WINDOW for a service whose OWN startup spawns
# short-lived console child processes (M17.10). CREATE_NO_WINDOW leaves the
# service with no console, so each console child it spawns makes Windows
# allocate a brand-new console window - a visible black flash (seen with Open
# WebUI's uvicorn startup helpers). Giving the service its own console and
# hiding it (CREATE_NEW_CONSOLE + STARTUPINFO SW_HIDE) means those children
# attach to the hidden console instead of popping their own. Mutually exclusive
# with CREATE_NO_WINDOW per the Win32 docs, so a spec picks exactly one path.
CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
_STARTF_USESHOWWINDOW = getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
_SW_HIDE = 0  # ShowWindow's SW_HIDE


def _hidden_console_startupinfo() -> "subprocess.STARTUPINFO | None":
    """STARTUPINFO that hides the new console window, or None off Windows."""
    if sys.platform != "win32":
        return None
    info = subprocess.STARTUPINFO()
    info.dwFlags |= _STARTF_USESHOWWINDOW
    info.wShowWindow = _SW_HIDE
    return info


# Loopback host is hardcoded so a service can never be exposed to the LAN.
LOOPBACK_HOST = "127.0.0.1"

# The folder every managed child process runs in, relative to the data root.
# It is deliberately NOT the data root itself: a child that writes a stray
# relative file should drop it somewhere obviously disposable rather than into
# the folder the product tells the user to back up.
SERVICE_CWD_NAME = "run"


def resolve_service_cwd(settings: Any) -> str:
    """The working directory every managed service must be launched in.

    Invariant W1 (DEC-M14-9) says LOCITIZE writes nothing beneath the install
    directory at runtime. Until now the ServiceSpec builders all passed
    `cwd=None`, which makes a child inherit LOCITIZE's own working directory - and
    LOCITIZE.bat line 9 does `cd /d "%~dp0"`, i.e. the install directory. Any
    relative path llama-server, whisper, kokoro or Open WebUI wrote therefore
    landed in the install tree, invisible to both halves of the W1 fence
    (review round 6, MEDIUM-1).

    All configured binary and weight paths are absolute by contract
    (settings.default.yaml: "Absolute paths to the external binaries and
    weights"), so moving the working directory cannot change how any argv value
    resolves; a relative one was already resolving against whatever directory
    the launcher happened to be started from, which was never a defined value.

    Returns a string, not a Path, because ServiceSpec.cwd is a string field.
    """
    root = getattr(settings, "data_dir", None)
    if not root:
        # No data root configured at all: the system temp directory is the one
        # place guaranteed to be writable and guaranteed NOT to be the install
        # tree. Falling back to None here would restore the exact defect.
        return tempfile.gettempdir()
    return str(Path(root) / SERVICE_CWD_NAME)


def _ensure_working_dir(cwd: str | None) -> str | None:
    """Create a spec's working directory if needed; degrade rather than crash.

    Popen raises when cwd does not exist, and a service failing to start because
    a scratch folder was missing would be a worse outcome than the stray-file
    risk this directory exists to contain - so an un-creatable directory falls
    back to the temp directory rather than to None (None means "inherit the
    install directory", which is the defect).
    """
    if not cwd:
        return cwd
    try:
        Path(cwd).mkdir(parents=True, exist_ok=True)
        return cwd
    except OSError:
        return tempfile.gettempdir()


class ServiceStatus(Enum):
    """Lifecycle state of a managed service."""

    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    UNHEALTHY = "UNHEALTHY"
    STOPPED_ERROR = "STOPPED_ERROR"


@dataclass
class ServiceSpec:
    """Declarative description of a launchable service (Architecture 4.1).

    `env` carries process-environment values injected at launch (secrets come
    from the process environment, never from yaml). `port`/`health_path` drive
    the readiness check.

    The readiness URL is intentionally NOT stored here as a full string. We store
    only `health_path` (e.g. "/health") and always compose the probe URL from the
    fixed loopback host plus the resolved port at probe time (see
    build_readiness_url). This single-sources the port so a reassigned port always
    reaches both the launch argv and the readiness probe (Reviewer M-1), and it
    makes it impossible for a crafted health_path to move the probe off
    127.0.0.1 (Security SEC-1): the host is a hardcoded constant, never derived
    from owner-supplied text.

    `log_path` optionally routes the child's stdout/stderr to a file. When unset,
    the child's output goes to the null device. Either way the child's pipes are
    drained by the OS, never by an unread parent pipe -- a large model emits tens
    of thousands of log lines while loading, and an unread PIPE would fill its
    buffer and deadlock the child before it ever becomes ready.
    """

    name: str
    command: list[str]
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    port: int | None = None
    health_path: str | None = None
    log_path: str | None = None
    ready_timeout_s: float = 60.0
    stop_timeout_s: float = 10.0
    # D-M4-3: when True the child's log is opened for APPEND with a dated session
    # separator instead of being truncated, so a failed start no longer destroys
    # the previous session's evidence (the D-M4-1 investigation lost the owner's
    # original whisper_server.log failure trace exactly because every start
    # overwrote it). Left False for transcript-capture streams (whisper-stream),
    # whose log is consumed as this-session data and must start empty.
    append_log: bool = False
    # M17.10: launch with a hidden console (CREATE_NEW_CONSOLE + SW_HIDE) instead
    # of CREATE_NO_WINDOW, for a service whose own startup spawns console child
    # processes that would otherwise each flash a new console window. Set only for
    # Open WebUI, whose uvicorn startup does exactly that; llama-server/whisper/
    # kokoro do not, and keep the plain CREATE_NO_WINDOW path.
    hidden_console: bool = False
    # Reuse-instead-of-duplicate (2026-09-24). When set, a start that finds the
    # requested port already held by a HEALTHY listener whose command line
    # contains this marker adopts that listener instead of spawning a second copy
    # on an auto-reassigned port. Owner-observed: five kokoro_server processes
    # (8092, 8082, 8084, 8086, 8087), each holding VRAM, because every launcher /
    # CLI start found 8092 busy and the auto allocator moved the new copy to the
    # next free port in the range - once onto 8080, llama-server's own port.
    # Requires health_path, so a half-dead listener is never adopted. None keeps
    # the old behaviour (llama-server must never adopt: a stray on its port is a
    # different model, which the stray guard handles).
    reuse_marker: str | None = None


# Injectable seams for testing --------------------------------------------- #


@runtime_checkable
class ProcessHandle(Protocol):
    """The subset of subprocess.Popen the manager relies on.

    A fake implementing these members lets tests drive lifecycle logic without
    spawning a real OS process.
    """

    pid: int

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def send_signal(self, signal: int) -> None: ...


# A process launcher takes the same kwargs as subprocess.Popen and returns a
# ProcessHandle. Tests inject a fake launcher.
ProcessLauncher = Callable[..., ProcessHandle]

# A kill runner performs the escalation kill (default wraps taskkill). Injectable
# so tests can assert it was called without touching the OS.
KillRunner = Callable[[int], None]

# A readiness checker answers "is the service ready?" given the spec and handle.
# Injectable so tests avoid real sockets/HTTP.
ReadinessChecker = Callable[[ServiceSpec, ProcessHandle], bool]


def _default_launcher(**kwargs: Any) -> ProcessHandle:
    """Production launcher: subprocess.Popen with the Windows process group flag."""
    return subprocess.Popen(**kwargs)  # type: ignore[return-value]


def _default_kill_runner(pid: int) -> None:
    """Production escalation kill: taskkill /F /T scoped to one PID we launched.

    /T kills the whole tree (llama.cpp/whisper may spawn helpers); /F forces it.
    This only ever receives a PID the manager itself created.
    """
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        capture_output=True,
        check=False,
        creationflags=CREATE_NO_WINDOW,
    )


def build_readiness_url(port: int | None, health_path: str | None) -> str | None:
    """Compose the loopback-only readiness URL from a resolved port and path.

    The host is the hardcoded LOOPBACK_HOST constant, never owner-supplied text,
    so no value of `health_path` can steer the readiness GET off 127.0.0.1
    (Security SEC-1). Returns None when there is no health path or port to probe.
    The port is always the resolved port because this is called with the probe
    spec (see ManagedProcess._probe_spec), which carries the reassigned port
    (Reviewer M-1).
    """
    if not health_path or port is None:
        return None
    return f"http://{LOOPBACK_HOST}:{port}{health_path}"


def _default_listener_cmdline(port: int) -> str:
    """Command line of whatever listens on loopback `port`, or "" if unknown.

    Used only to decide whether a busy port already holds our own service
    (ServiceSpec.reuse_marker). Any failure returns "", which means "not
    provably ours" and falls back to a normal start.
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
                return " ".join(psutil.Process(conn.pid).cmdline() or [])
    except Exception:  # noqa: BLE001 - a probe never blocks a start
        return ""
    return ""


def _default_readiness(spec: ServiceSpec, handle: ProcessHandle) -> bool:
    """Production readiness: HTTP health path, else TCP port, else process alive.

    Uses stdlib urllib for the HTTP probe (no third-party dependency). The URL is
    rebuilt every call from the (resolved) port and the loopback host, so it can
    never target the wrong port or a non-loopback host.
    """
    url = build_readiness_url(spec.port, spec.health_path)
    if url:
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError):
            return False
    if spec.port is not None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect((LOOPBACK_HOST, spec.port))
                return True
            except OSError:
                return False
    # No port/url: consider ready if the process is still alive after settling.
    return handle.poll() is None


class PortAllocator:
    """Finds and reserves a free loopback port within the reserved range.

    Binds 127.0.0.1 only; a service is never bound to a public interface.
    """

    def __init__(
        self,
        range_start: int,
        range_end: int,
        allocation: str = "auto",
        is_free: Callable[[int], bool] | None = None,
    ) -> None:
        self._start = range_start
        self._end = range_end
        self._allocation = allocation
        # Injectable free-check for tests; defaults to a real loopback bind.
        self._is_free = is_free or self._default_is_free

    def _default_is_free(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((LOOPBACK_HOST, port))
                return True
            except OSError:
                return False

    def is_free(self, port: int) -> bool:
        return self._is_free(port)

    def ensure_free(self, port: int) -> int:
        """Return a usable port for the requested one.

        strict policy: the exact port must be free or raise. auto policy:
        increment within the reserved range until a free port is found.
        """
        if self._is_free(port):
            return port
        if self._allocation == "strict":
            raise PortUnavailableError(
                f"port {port} is occupied and allocation policy is 'strict'"
            )
        # auto: scan the reserved range for the next free port.
        for candidate in range(self._start, self._end + 1):
            if candidate == port:
                continue
            if self._is_free(candidate):
                return candidate
        raise PortUnavailableError(
            f"no free port in reserved range {self._start}-{self._end}"
        )


class PortUnavailableError(RuntimeError):
    """Raised when a required loopback port cannot be allocated."""


class ManagedProcess:
    """Concrete Service wrapping one OS process (tree).

    All external effects (spawn, kill, readiness) go through injected callables so
    the lifecycle is testable without real processes.
    """

    def __init__(
        self,
        spec: ServiceSpec,
        launcher: ProcessLauncher | None = None,
        kill_runner: KillRunner | None = None,
        readiness: ReadinessChecker | None = None,
        port_allocator: PortAllocator | None = None,
        poll_interval_s: float = 0.2,
        kill_confirm_timeout_s: float = 3.0,
        kill_max_attempts: int = 3,
        listener_cmdline: Callable[[int], str] | None = None,
    ) -> None:
        self._spec = spec
        self._listener_cmdline = listener_cmdline or _default_listener_cmdline
        # True when start() adopted an already-running healthy copy (reuse_marker)
        # instead of spawning one. There is no handle: this process did not launch
        # it, so stop() leaves it running and snapshot() probes it for liveness.
        self._adopted = False
        self._launcher = launcher or _default_launcher
        self._kill_runner = kill_runner or _default_kill_runner
        self._readiness = readiness or _default_readiness
        self._allocator = port_allocator
        self._poll_interval = poll_interval_s
        # D-M4-1: after issuing taskkill we WAIT this long for the OS to actually
        # reap the child before believing it is gone, and retry the kill up to
        # kill_max_attempts times. A 15GB llama-server mid-load can sit in an
        # uninterruptible GPU/driver call when the first taskkill lands, so a single
        # fire-and-forget kill that is never confirmed can leave the process alive
        # (the orphan the defect reproduced). Tests pass tiny values to stay fast.
        self._kill_confirm_timeout = kill_confirm_timeout_s
        self._kill_max_attempts = max(1, kill_max_attempts)
        self._handle: ProcessHandle | None = None
        self._status = ServiceStatus.STOPPED
        self._resolved_port: int | None = None
        # File object receiving the child's merged stdout/stderr, if log_path set.
        self._log_handle: Any = None
        self._log = get_logger("launcher")

    @property
    def status_value(self) -> ServiceStatus:
        return self._status

    @property
    def resolved_port(self) -> int | None:
        return self._resolved_port

    @property
    def pid(self) -> int | None:
        """OS pid of the launched process, or None if not started."""
        return self._handle.pid if self._handle is not None else None

    @property
    def returncode(self) -> int | None:
        """Exit code if the process has exited, else None (still running)."""
        return self._handle.poll() if self._handle is not None else None

    def start(self) -> ServiceStatus:
        """Start the service: port preflight -> launch -> readiness wait."""
        self._status = ServiceStatus.STARTING
        spec = self._spec
        self._adopted = False

        if self._adopt_existing():
            self._resolved_port = spec.port
            self._adopted = True
            self._status = ServiceStatus.RUNNING
            self._log.info(
                "service %s: reusing the healthy copy already on port %s",
                spec.name,
                spec.port,
            )
            return self._status

        # Port preflight (Architecture 4.2 step 1).
        if spec.port is not None and self._allocator is not None:
            try:
                self._resolved_port = self._allocator.ensure_free(spec.port)
            except PortUnavailableError as exc:
                self._status = ServiceStatus.STOPPED_ERROR
                self._log.error("service %s: %s", spec.name, exc)
                return self._status
        else:
            self._resolved_port = spec.port

        # M-1 fix: rebuild the launch argv so the --port value carries the
        # resolved (possibly reassigned) port. Without this, an auto-reassigned
        # port changed nothing observable -- the child was still told to bind the
        # originally requested port. Single-sourced with readiness via
        # _resolved_port so argv and the health probe can never disagree.
        command = _command_with_port(spec.command, self._resolved_port)

        # Launch (Architecture 4.2 step 2). The child's stdout+stderr are routed
        # to a log file (if the spec names one) or the null device, NEVER to an
        # unread parent PIPE: a large model floods its output while loading and an
        # unread pipe buffer would fill and deadlock the child before readiness.
        stdout_target = self._open_output_target(spec)
        # A child inherits the parent's working directory when cwd is None, and
        # LOCITIZE.bat cd's into the install directory - so every relative file a
        # managed child writes would land in the install tree (invariant W1,
        # review round 6 MEDIUM-1). Every spec now names its own directory; this
        # makes sure it exists before Popen, which fails outright on a missing one.
        working_dir = _ensure_working_dir(spec.cwd)
        # Console strategy (M17.10). Default: CREATE_NO_WINDOW - the service has no
        # console. A service whose own startup spawns console children (Open WebUI)
        # sets hidden_console: give it a hidden console (CREATE_NEW_CONSOLE +
        # SW_HIDE) so those children attach to it instead of each flashing a new
        # window. CREATE_NEW_PROCESS_GROUP is kept in both paths so stop() can
        # still signal the group; if that ever fails the taskkill fallback (which
        # the smoke tests assert leaves no orphan) still stops it.
        if spec.hidden_console:
            creationflags = CREATE_NEW_PROCESS_GROUP | CREATE_NEW_CONSOLE
            startupinfo = _hidden_console_startupinfo()
        else:
            creationflags = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
            startupinfo = None
        try:
            self._handle = self._launcher(
                args=command,
                cwd=working_dir,
                env=self._merged_env(),
                stdout=stdout_target,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                startupinfo=startupinfo,
            )
        except (OSError, ValueError) as exc:
            self._close_log_handle()
            self._status = ServiceStatus.STOPPED_ERROR
            self._log.error("service %s failed to launch: %s", spec.name, exc)
            return self._status

        # Readiness (Architecture 4.2 step 3).
        if self._await_ready():
            self._status = ServiceStatus.RUNNING
        else:
            # Timed out or died: stop the partial process, report the error.
            self.stop()
            self._status = ServiceStatus.STOPPED_ERROR
            self._log.error("service %s did not become ready in time", spec.name)
        return self._status

    def _adopt_existing(self) -> bool:
        """True when the requested port already serves a healthy copy of this service.

        All must hold: the spec opts in (reuse_marker + port + health_path), the
        port is occupied, the listener's command line names the marker, and the
        health probe answers. Anything unprovable falls through to a normal start,
        so a foreign listener still gets the old reassignment behaviour.
        """
        spec = self._spec
        if not spec.reuse_marker or spec.port is None or not spec.health_path:
            return False
        if self._allocator is not None and self._allocator.is_free(spec.port):
            return False
        try:
            described = self._listener_cmdline(spec.port) or ""
        except Exception:  # noqa: BLE001 - unprovable ownership means no adoption
            return False
        marker = spec.reuse_marker.replace("\\", "/").lower()
        if marker not in described.replace("\\", "/").lower():
            return False
        return bool(self._readiness(spec, None))

    def _await_ready(self) -> bool:
        """Poll readiness until ready, process exit, or timeout."""
        deadline = time.monotonic() + self._spec.ready_timeout_s
        assert self._handle is not None
        while time.monotonic() < deadline:
            if self._handle.poll() is not None:
                # Process exited before becoming ready.
                return False
            if self._readiness(self._probe_spec(), self._handle):
                return True
            time.sleep(self._poll_interval)
        return False

    def _probe_spec(self) -> ServiceSpec:
        """Spec view carrying the resolved (possibly reassigned) port for readiness.

        The readiness URL is composed from this port at probe time, so a
        reassigned port propagates to the health probe (Reviewer M-1). The
        health_path is copied verbatim; the loopback host is fixed in
        build_readiness_url, so the path cannot move the probe host (SEC-1).
        """
        if self._resolved_port == self._spec.port:
            return self._spec
        return ServiceSpec(
            name=self._spec.name,
            command=self._spec.command,
            cwd=self._spec.cwd,
            env=self._spec.env,
            port=self._resolved_port,
            health_path=self._spec.health_path,
            log_path=self._spec.log_path,
            ready_timeout_s=self._spec.ready_timeout_s,
            stop_timeout_s=self._spec.stop_timeout_s,
            append_log=self._spec.append_log,
        )

    def _open_output_target(self, spec: ServiceSpec) -> Any:
        """Return the stdout target for the child (a log file or the null device).

        Stores the opened file handle so stop() can close it. Falling back to the
        null device (never an unread PIPE) guarantees the child never blocks on a
        full output-pipe buffer.

        D-M4-3: when spec.append_log is set the file is opened for APPEND ("ab") and
        a dated separator line is written first, so each service start adds to the
        log under a timestamped banner instead of overwriting the previous session's
        evidence. Transcript-capture streams leave append_log False and keep the
        truncating "wb" behaviour (their log is this-session data, tailed from byte
        zero).
        """
        if spec.log_path:
            mode = "ab" if spec.append_log else "wb"
            try:
                self._log_handle = open(spec.log_path, mode)  # noqa: SIM115
                if spec.append_log:
                    self._write_session_separator(spec.name)
                return self._log_handle
            except OSError as exc:
                # A missing logs/ dir or permission issue must not abort the
                # launch; degrade to discarding output rather than failing.
                self._log.warning(
                    "service %s: could not open log %s (%s); discarding output",
                    spec.name,
                    spec.log_path,
                    exc,
                )
        return subprocess.DEVNULL

    def _write_session_separator(self, service_name: str) -> None:
        """Write a dated banner to the append-mode child log (D-M4-3).

        Marks where this start's output begins so an owner can tell one session's
        llama.cpp / whisper output from the last. ASCII only; UTC timestamp so log
        entries are unambiguous across timezones.
        """
        if self._log_handle is None:
            return
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        banner = f"\n===== LOCITIZE service start {service_name} {stamp} =====\n"
        try:
            self._log_handle.write(banner.encode("ascii", errors="replace"))
            self._log_handle.flush()
        except OSError:
            # A separator write failure must never abort a launch; the child output
            # that follows is what matters.
            pass

    def _close_log_handle(self) -> None:
        """Close the child-output log file if one was opened (idempotent)."""
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None

    def stop(self) -> ServiceStatus:
        """Stop the service: graceful CTRL_BREAK, then confirmed taskkill /F /T.

        D-M4-1 no-orphan guarantee: this never reports STOPPED unless the child is
        actually confirmed gone (poll() returns an exit code). If the graceful
        signal does not settle the process it escalates to taskkill and then WAITS
        for the OS to reap it, retrying the kill; only when confirmation fails after
        every attempt does it return STOPPED_ERROR -- and in that case it leaves
        self._handle set and the service registered, so a subsequent stop_all()
        finds it and kills it again. No path abandons a live handle.
        """
        handle = self._handle
        if handle is None:
            # An adopted copy was not launched here, so it is not ours to kill.
            self._adopted = False
            self._status = ServiceStatus.STOPPED
            return self._status
        if handle.poll() is not None:
            # Already dead: release the log file and report STOPPED.
            self._status = ServiceStatus.STOPPED
            self._close_log_handle()
            return self._status

        # Graceful: signal the process group (valid due to CREATE_NEW_PROCESS_GROUP).
        signalled = True
        try:
            handle.send_signal(_ctrl_break_signal())
        except BaseException as exc:  # noqa: BLE001 - see the two paragraphs below
            # Signal delivery can fail if the process is mid-exit, or if THIS
            # process has no console to deliver a CTRL_BREAK_EVENT through.
            #
            # Bug fixed 2026-08-22. The except clause here used to be
            # `(OSError, ValueError)`, which is not what Windows actually
            # raises in the console-less case. LOCITIZE Desktop runs under
            # pythonw.exe - a GUI-subsystem binary that is never given a
            # console - and there `os.kill(pid, CTRL_BREAK_EVENT)` fails inside
            # GenerateConsoleCtrlEvent with WinError 6 ("The handle is
            # invalid") and CPython surfaces it as
            #   SystemError: <built-in function kill> returned a result with an
            #   exception set
            # SystemError is neither OSError nor ValueError, so it escaped this
            # handler, escaped stop(), escaped stop_all() - and took the
            # taskkill escalation below with it. The child llama-server was
            # left running and the caller was told the shutdown was unclean.
            # That is exactly what an auto-tune run saw for every trial (each
            # trial subprocess inherits pythonw.exe via sys.executable), and it
            # is also what orphaned the model on desktop shutdown
            # (locitize-data/logs/errors.log, 2026-08-21 10:11 and 10:16).
            # Catching BaseException here is deliberate: this is a best-effort
            # courtesy signal on a path whose real guarantee is the confirmed
            # taskkill below, and NOTHING it raises may be allowed to skip that.
            signalled = False
            self._log.debug(
                "service %s: graceful stop signal not deliverable (%s: %s); "
                "escalating immediately",
                self._spec.name,
                type(exc).__name__,
                exc,
            )
        if signalled:
            # Only wait out the graceful window when a signal was actually
            # delivered. When delivery failed there is nothing in flight to wait
            # for, and sleeping stop_timeout_s per service would just make every
            # console-less (GUI) shutdown slower for no chance of a better result.
            try:
                handle.wait(timeout=self._spec.stop_timeout_s)
            except Exception:  # noqa: BLE001 - wait() timeout type varies by platform
                # Graceful window elapsed without exit; fall through to escalation.
                pass

        if handle.poll() is not None:
            # The graceful signal took effect.
            self._status = ServiceStatus.STOPPED
            self._close_log_handle()
            return self._status

        # Escalate: force-kill the whole tree by the PID we launched, and CONFIRM
        # the process is actually reaped before believing it. Confirmation is the
        # fix for the reproduced orphan: a single unconfirmed taskkill could return
        # while a still-loading multi-GB child lingered.
        if self._terminate_and_confirm(handle):
            self._status = ServiceStatus.STOPPED
        else:
            # Still alive after every kill attempt: an error state, but the handle
            # is deliberately KEPT (self._handle unchanged, service still
            # registered) so stop_all() re-reaches and re-kills it -- no live
            # process ever escapes the manager's reach (D-M4-1).
            self._status = ServiceStatus.STOPPED_ERROR
            self._log.error(
                "service %s could not be confirmed stopped after %d kill attempts; "
                "left registered for stop_all retry",
                self._spec.name,
                self._kill_max_attempts,
            )
        # Only release the log file once the process is confirmed gone; while it is
        # still (believed) alive the child may keep writing to it.
        if self._status is ServiceStatus.STOPPED:
            self._close_log_handle()
        return self._status

    def _terminate_and_confirm(self, handle: ProcessHandle) -> bool:
        """Force-kill the PID we launched and confirm it exited; True iff dead.

        Retries the kill up to kill_max_attempts, waiting kill_confirm_timeout each
        time for the OS to reap the process. Returns False only if the child is
        still alive after every attempt (the caller then keeps it registered for a
        later retry).
        """
        for _ in range(self._kill_max_attempts):
            self._kill_runner(handle.pid)
            if self._wait_for_exit(handle, self._kill_confirm_timeout):
                return True
        return handle.poll() is not None

    def _wait_for_exit(self, handle: ProcessHandle, timeout_s: float) -> bool:
        """Poll until the process exits or timeout_s elapses; True iff it exited."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if handle.poll() is not None:
                return True
            time.sleep(self._poll_interval)
        return handle.poll() is not None

    def restart(self) -> ServiceStatus:
        """Stop then start with the same spec (Architecture 4.4)."""
        self.stop()
        return self.start()

    def is_healthy(self) -> bool:
        """True if the process is alive and passes the readiness check."""
        if self._handle is None or self._handle.poll() is not None:
            return False
        return self._readiness(self._probe_spec(), self._handle)

    def snapshot(self) -> ServiceStatus:
        """Non-mutating status read for monitor()."""
        if self._handle is None:
            if self._adopted and self._status == ServiceStatus.RUNNING:
                return (
                    ServiceStatus.RUNNING
                    if self._readiness(self._spec, None)
                    else ServiceStatus.STOPPED_ERROR
                )
            return ServiceStatus.STOPPED
        if self._handle.poll() is not None:
            # Process died out from under us.
            return (
                ServiceStatus.STOPPED_ERROR
                if self._status == ServiceStatus.RUNNING
                else ServiceStatus.STOPPED
            )
        return self._status

    def _merged_env(self) -> dict[str, str] | None:
        """Merge the process environment with the spec's env additions.

        Returning None lets Popen inherit the parent environment when the spec
        adds nothing, avoiding a needless full-env copy.
        """
        if not self._spec.env:
            return None
        import os

        merged = dict(os.environ)
        merged.update(self._spec.env)
        return merged


def _ctrl_break_signal() -> int:
    """Return CTRL_BREAK_EVENT on Windows, else SIGTERM as a portable fallback."""
    import signal

    return getattr(signal, "CTRL_BREAK_EVENT", getattr(signal, "SIGTERM", 15))


# Flags the platform resolves itself and therefore forbids in owner-supplied
# server_args. If an owner puts --port/--host in a model's or service's
# server_args, it could append a second, conflicting value to the argv and race
# the platform-resolved value (Reviewer L-4). We strip them (with a WARNING) so
# the platform-resolved value always wins.
_MANAGED_FLAGS = ("--port", "--host")


def strip_managed_flags(
    service_id: str,
    args: list[str],
    log: Any = None,
) -> list[str]:
    """Return a copy of `args` with any platform-managed flag (and its value) removed.

    Reviewer L-4: --port and --host are resolved by the platform (port allocator +
    hardcoded loopback host) and baked into the argv by the spec builders. Allowing
    the same flag through owner-supplied server_args would append a second value to
    the argv; llama.cpp/whisper-server take the last one, silently overriding the
    platform-resolved port/host and defeating the M-1 port-propagation fix. This
    strips the offending flag (both the "--port 9999" space form and the
    "--port=9999" form, and the value token that follows the space form) and logs
    one WARNING per stripped flag naming the offending service, so the
    platform-resolved value is always the only one in the final argv.
    """
    if log is None:
        log = get_logger("launcher")
    out: list[str] = []
    index = 0
    count = len(args)
    while index < count:
        token = args[index]
        # Compare on the flag name only so "--port=9999" is caught alongside
        # the bare "--port" form.
        flag_name = token.split("=", 1)[0]
        if flag_name in _MANAGED_FLAGS:
            log.warning(
                "service %s: ignoring platform-managed flag '%s' in server_args; "
                "the platform-resolved value always wins",
                service_id,
                token,
            )
            # Space-separated form ("--port", "9999"): also drop the value token.
            # The "=" form carries its value inline, so only the one token is dropped.
            if "=" not in token and index + 1 < count and not args[index + 1].startswith("-"):
                index += 2
            else:
                index += 1
            continue
        out.append(token)
        index += 1
    return out


class SingleServiceController:
    """Start/stop/inspect exactly one managed service (not a switchable model).

    ModelController serves the one active llama.cpp *model* (with switch
    semantics). Auxiliary services -- whisper-server (HTTP transcription) and
    whisper-stream (mic capture) -- are single, non-switchable services, so they
    get this simpler controller that reuses the identical ServiceManager start/stop
    machinery and the same stop_all()/atexit no-orphan guarantee M2 proved for
    llama.cpp. All process effects go through the injected process_factory, so the
    whisper lifecycle is unit-tested with a fake process and no real binary is
    spawned in the test suite (Architecture section 12).

    `spec_builder` is a zero-argument callable returning the ServiceSpec (the
    launcher closes over settings to build it), so this controller stays decoupled
    from config/whisper spec construction.
    """

    def __init__(
        self,
        service_manager: ServiceManager,
        spec_builder: Callable[[], ServiceSpec],
        process_factory: ManagedProcessFactory,
    ) -> None:
        self._manager = service_manager
        self._spec_builder = spec_builder
        self._process_factory = process_factory
        self._proc: ManagedProcess | None = None
        self._spec_name: str | None = None

    @property
    def resolved_port(self) -> int | None:
        return self._proc.resolved_port if self._proc is not None else None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def is_running(self) -> bool:
        """True only if the service was started and is currently RUNNING."""
        return (
            self._proc is not None
            and self._proc.snapshot() is ServiceStatus.RUNNING
        )

    def start(self) -> ServiceStatus:
        """Build the spec and start the service via the shared ServiceManager."""
        spec = self._spec_builder()
        self._spec_name = spec.name
        proc = self._process_factory(spec)
        self._manager.register(proc)
        self._proc = proc
        return self._manager.start(spec.name)

    def stop(self) -> ServiceStatus:
        """Stop the service if it was started; STOPPED if nothing to stop."""
        if self._proc is None or self._spec_name is None:
            return ServiceStatus.STOPPED
        return self._manager.stop(self._spec_name)

    def snapshot(self) -> dict[str, Any]:
        """Honest single-service status (empty 'services' list before first start)."""
        if self._proc is None or self._spec_name is None:
            return {"name": None, "status": ServiceStatus.STOPPED.value, "services": []}
        service = {
            "name": self._spec_name,
            "status": self._proc.snapshot().value,
            "port": self._proc.resolved_port,
            "pid": self._proc.pid,
        }
        return {
            "name": self._spec_name,
            "status": service["status"],
            "services": [service],
        }


def _command_with_port(command: list[str], resolved_port: int | None) -> list[str]:
    """Return a copy of `command` with the value after --port set to resolved_port.

    This is the single point where a reassigned port reaches the launch argv
    (Reviewer M-1). If there is no --port flag or no resolved port, the command is
    returned unchanged. Only the token immediately following --port is replaced,
    so no other argument can be affected.
    """
    if resolved_port is None or "--port" not in command:
        return list(command)
    out = list(command)
    idx = out.index("--port")
    if idx + 1 < len(out):
        out[idx + 1] = str(resolved_port)
    return out


class ServiceManager:
    """Registry of named services with lifecycle orchestration.

    stop_all() runs in reverse start order and is safe to register with atexit so
    a launcher crash does not orphan child processes.
    """

    def __init__(self) -> None:
        # Insertion order is start order; reversed for shutdown.
        self._services: dict[str, ManagedProcess] = {}
        self._start_order: list[str] = []
        self._lock = threading.Lock()

    def register(self, service: ManagedProcess) -> None:
        with self._lock:
            self._services[service._spec.name] = service

    def start(self, name: str) -> ServiceStatus:
        service = self._require(name)
        status = service.start()
        with self._lock:
            if name not in self._start_order:
                self._start_order.append(name)
        return status

    def stop(self, name: str) -> ServiceStatus:
        return self._require(name).stop()

    def restart(self, name: str) -> ServiceStatus:
        return self._require(name).restart()

    def stop_all(self) -> None:
        """Stop every REGISTERED service (best effort), reverse start order first.

        H-1 fix: teardown now covers every service in `_services`, not only those
        that finished starting (`_start_order`). A ManagedProcess is registered
        BEFORE its potentially blocking start() returns (ModelController.start and
        SingleServiceController.start both register, then call the up-to-60s
        `manager.start()`). If the window closed while a model start/switch was
        mid-flight -- the child already spawned but readiness not yet returned --
        the service was in `_services` but not yet in `_start_order`, so the old
        loop skipped it and both shutdown() and the atexit backstop orphaned a
        multi-GB llama-server. Covering `_services` closes that window.

        Ordering: cleanly-started services stop first in reverse start order (the
        original clean dependency ordering); any registered-but-unordered service
        (mid-flight or never started) stops after, in reverse registration order.
        ManagedProcess.stop is idempotent and safe on a never-started proc, so
        stopping a registered-but-unstarted service is harmless. Names are
        snapshotted under the lock; stop() itself runs unlocked because it can block
        on process teardown and must not hold the registration lock.
        """
        with self._lock:
            started = list(self._start_order)
            all_names = list(self._services.keys())
        started_set = set(started)
        # Reverse start order for services that finished starting, then any
        # remaining registered service in reverse registration order.
        ordered = list(reversed(started))
        ordered.extend(
            name for name in reversed(all_names) if name not in started_set
        )
        for name in ordered:
            service = self._services.get(name)
            if service is not None:
                service.stop()

    def monitor(self) -> dict[str, ServiceStatus]:
        """Return a status snapshot for every registered service (no side effects)."""
        return {name: svc.snapshot() for name, svc in self._services.items()}

    def _require(self, name: str) -> ManagedProcess:
        service = self._services.get(name)
        if service is None:
            raise KeyError(f"no service registered under '{name}'")
        return service


# A spec builder turns a model id (plus optional ctx/gpu overrides) into a
# ServiceSpec. ModelController is decoupled from models.py through this callable
# so services.py stays at the bottom of the import graph.
SpecBuilder = Callable[..., ServiceSpec]

# A process factory turns a ServiceSpec into a ManagedProcess. Injected so switch
# and stop semantics are unit-tested with fake processes (no real binaries).
ManagedProcessFactory = Callable[[ServiceSpec], ManagedProcess]


def make_process_factory(
    port_allocator: PortAllocator | None = None,
) -> ManagedProcessFactory:
    """Return a factory that builds a real ManagedProcess for a spec.

    The production launcher passes a PortAllocator built from settings; tests pass
    their own factory yielding fake-launcher ManagedProcess instances instead.
    """

    def _factory(spec: ServiceSpec) -> ManagedProcess:
        return ManagedProcess(spec, port_allocator=port_allocator)

    return _factory


class ModelController:
    """Starts, stops, and switches the single active llama.cpp model service.

    LOCITIZE serves one local chat model at a time. Selecting a model starts a
    llama.cpp service for it; selecting a different model performs a clean switch:
    stop the currently running model, confirm the process is gone, then start the
    newly selected one. The controller never leaves two models reported running.

    Every ManagedProcess it starts is registered with the injected ServiceManager,
    so stop_all() -- wired to atexit and the menu's finally block (Reviewer L-1)
    -- tears down anything still alive when the launcher exits or crashes.

    All process effects go through the injected process_factory, so the
    start/switch/stop logic is unit-tested with fake processes and no real binary
    is ever spawned in the test suite (Architecture section 12).
    """

    def __init__(
        self,
        service_manager: ServiceManager,
        spec_builder: SpecBuilder,
        process_factory: ManagedProcessFactory,
        stale_server_check: Callable[[set[int]], list[str]] | None = None,
    ) -> None:
        self._manager = service_manager
        self._spec_builder = spec_builder
        self._process_factory = process_factory
        # Owner-observed 2026-09-03: a force-killed session left its
        # llama-server alive. The next session started a SECOND model beside
        # it - the port allocator correctly moved to 8081, nothing checked the
        # GPU - and two 13GB models on a 16.3GB card left the new one with
        # layers spilled to CPU, answering at 0.31 tok/s. It looked hung.
        #
        # The one-model-at-a-time discipline (M8.2) only ever covered models
        # THIS controller started; an orphan from a dead session was invisible
        # to it. This callable is given the pids we own and returns a
        # description of any LOCITIZE server we do NOT own that is holding the
        # GPU. Injected rather than imported so services.py keeps knowing
        # nothing about GPUs, and so the whole path is testable with no
        # nvidia-smi. None -> no check, which is every existing caller.
        self._stale_server_check = stale_server_check
        self._current_model_id: str | None = None
        self._current_name: str | None = None
        self._current: ManagedProcess | None = None
        # Every service started this session (running or since stopped), for an
        # honest status snapshot after a switch.
        self._started: dict[str, ManagedProcess] = {}

    def _child_alive(self) -> bool:
        """Is the tracked model process still there? A cached id is not evidence.

        The id was only ever cleared by our own stop/switch. A llama-server
        that died underneath us - a crash, an out-of-band Offload GPU - left it
        set, so the router believed the model was up, forwarded to a closed
        port, and answered 502 until some other model was requested. Ask the
        process, not the bookkeeping.
        """
        return self._current is not None and self._current.returncode is None

    @property
    def running_model_id(self) -> str | None:
        """The model id currently RUNNING, or None - None too if its process has exited."""
        return self._current_model_id if self._child_alive() else None

    @property
    def running_port(self) -> int | None:
        """The loopback port the running model service resolved to, or None.

        The assistant (M7) composes its /v1/chat/completions URL from this so it
        talks to exactly the port the ModelController started -- honoring the
        PortAllocator's auto-reassignment (SEC-1 loopback discipline). None when no
        model is running.
        """
        return self._current.resolved_port if self._child_alive() else None

    def start(
        self,
        model_id: str,
        ctx_size: int | None = None,
        gpu_layers: int | None = None,
        **spec_kwargs: Any,
    ) -> ServiceStatus:
        """Build the spec for a model and start its service. Returns the status.

        Extra keyword arguments (M5: server_args / draft_model / spec_config sweep
        overrides) are forwarded verbatim to the spec builder. The terminal and GUI
        callers pass none, so their spec builders (which accept only model_id/ctx/
        gpu) are unaffected; only the benchmark runner supplies them, against the
        registry's own build_start_spec which understands them.
        """
        self._refuse_if_a_stale_server_holds_the_gpu()
        spec = self._spec_builder(model_id, ctx_size, gpu_layers, **spec_kwargs)
        proc = self._process_factory(spec)
        self._manager.register(proc)
        self._started[spec.name] = proc
        status = self._manager.start(spec.name)
        if status is ServiceStatus.RUNNING:
            self._current_model_id = model_id
            self._current_name = spec.name
            self._current = proc
        else:
            # A failed start owns nothing; do not track it as the running model.
            self._current_model_id = None
            self._current_name = None
            self._current = None
        return status

    def _own_pids(self) -> set[int]:
        """Pids of every service THIS controller started and still tracks."""
        return {
            proc.pid
            for proc in self._started.values()
            if getattr(proc, "pid", None)
        }

    def _refuse_if_a_stale_server_holds_the_gpu(self) -> None:
        """Raise ValueError when a LOCITIZE server we do not own has the GPU.

        Refuses rather than reaps. The orphan is a real process holding real
        VRAM, and killing something this controller never started is the
        owner's call - Offload GPU already does exactly that, deliberately, on
        a click. Starting anyway is the one option that is definitely wrong: it
        produces a model that loads, serves, and is unusably slow, with nothing
        on any surface saying why.

        A guard that cannot see is not a reason to refuse a start the owner
        asked for, so any failure inside the check is swallowed and the start
        proceeds - the pre-guard behaviour.

        ValueError because every caller already handles it as the config guard:
        the launcher menu prints it as a remedy line, the GUI turns it into a
        failed-start Result, and the model router returns it as a 503.
        """
        if self._stale_server_check is None:
            return
        try:
            stale = self._stale_server_check(self._own_pids())
        except Exception:  # noqa: BLE001 - see the docstring
            return
        if not stale:
            return
        listed = "; ".join(stale)
        raise ValueError(
            f"another LOCITIZE model server is already holding the GPU "
            f"({listed}). Starting a second one would leave both spilling to "
            f"CPU. Use Offload GPU to clear it, then retry"
        )

    def switch(
        self,
        model_id: str,
        ctx_size: int | None = None,
        gpu_layers: int | None = None,
        **spec_kwargs: Any,
    ) -> ServiceStatus:
        """Switch to a different model: stop old (confirm gone), then start new."""
        reserved = getattr(self, "_session_reserved_model", None)
        if reserved and reserved != model_id:
            raise ValueError("A coding session is using this model. Release it on the Sessions page before switching.")
        # Already serving this exact model: nothing to switch.
        if (
            self._current is not None
            and self._current_model_id == model_id
            and self._current.status_value is ServiceStatus.RUNNING
            and self._child_alive()          # status is bookkeeping; the process is the truth
        ):
            return ServiceStatus.RUNNING
        # Stop the currently running model first and confirm it exited before
        # starting the replacement, so the two never run simultaneously.
        if self._current is not None:
            stop_status = self.stop()
            if stop_status is ServiceStatus.STOPPED_ERROR:
                # The old process could not be confirmed gone; refuse to stack a
                # second model on top of it.
                return ServiceStatus.STOPPED_ERROR
        return self.start(model_id, ctx_size, gpu_layers, **spec_kwargs)

    def stop(self) -> ServiceStatus:
        """Stop the currently running model, if any. Returns the stop status."""
        if self._current is None or self._current_name is None:
            return ServiceStatus.STOPPED
        status = self._manager.stop(self._current_name)
        # Clear current tracking regardless; a STOPPED_ERROR is surfaced upward so
        # the caller can decide (switch refuses to start a new model over it).
        self._current = None
        self._current_model_id = None
        self._current_name = None
        return status

    def snapshot(self) -> dict[str, Any]:
        """Honest status of every service this controller started.

        Safe to call with nothing started (returns an empty service list), which
        is exactly what the non-interactive --service-status path needs.
        """
        services = []
        for name, proc in self._started.items():
            model_id = name.split(":", 1)[1] if ":" in name else name
            services.append(
                {
                    "name": name,
                    "model_id": model_id,
                    "status": proc.snapshot().value,
                    "port": proc.resolved_port,
                    "pid": proc.pid,
                }
            )
        return {"running_model": self.running_model_id, "services": services}
