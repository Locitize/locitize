"""Cancellable, single-instance Open WebUI installer for the desktop.

The install is an explicit user action. It runs outside the controller's ops
worker, shares the launcher's Open WebUI start lock, logs every line, and owns
the pip process tree so closing the desktop cannot leave an installer behind.

Open WebUI remains pinned in setup_env. Its transitive dependencies are not yet
locked; release qualification must retest the pin whenever upstream dependency
resolution changes.
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


@dataclass(frozen=True)
class CommandOutcome:
    returncode: int
    lines: tuple[str, ...] = ()
    cancelled: bool = False


@dataclass(frozen=True)
class InstallResult:
    ok: bool
    state: str
    message: str
    log_path: str


def _friendly_failure(text: str, log_path: Path) -> str:
    low = text.lower()
    if any(token in low for token in (
        "temporary failure in name resolution", "name or service not known",
        "getaddrinfo failed", "no such host", "dns",
    )):
        return "Open WebUI could not be downloaded because this computer is offline or DNS is unavailable."
    if any(token in low for token in (
        "certificate verify failed", "certificate_verify_failed", "tls certificate",
        "ssl: certificate",
    )):
        return "Open WebUI could not be downloaded because the TLS certificate could not be verified."
    if any(token in low for token in (
        "no space left on device", "disk full", "winerror 112",
        "not enough space on the disk",
    )):
        return "Open WebUI could not be installed because the disk is full."
    if "requires-python" in low or "incompatible python" in low:
        return "Open WebUI requires Python 3.11 or 3.12."
    return f"Open WebUI could not be installed. See {log_path}."


class OpenWebUIInstaller:
    """One reusable installer coordinator; at most one worker may run."""

    def __init__(
        self,
        settings: Any,
        start_lock: threading.Lock,
        *,
        runner: Callable[..., CommandOutcome] | None = None,
        verifier: Callable[[Any], tuple[bool, str]] | None = None,
        disk_usage: Callable[[str], Any] = shutil.disk_usage,
        base_python: Path | str | None = None,
        stop_fn: Callable[[], None] | None = None,
    ) -> None:
        self.settings = settings
        self.start_lock = start_lock
        self._runner = runner or self._run_process
        self._verifier = verifier or self._verify
        self._disk_usage = disk_usage
        self._base_python = Path(base_python or self._default_python())
        self._stop_fn = stop_fn
        self._guard = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self.state = "idle"

    @staticmethod
    def _default_python() -> Path:
        current = Path(sys.executable)
        console = current.with_name("python.exe")
        return console if console.is_file() else current

    @property
    def log_path(self) -> Path:
        return Path(self.settings.data_dir) / "logs" / "openwebui-install.log"

    def start(
        self,
        progress: Callable[[dict[str, Any]], None],
        done: Callable[[InstallResult], None],
    ) -> bool:
        with self._guard:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._cancel = threading.Event()
            self.state = "running"
            thread = threading.Thread(
                target=self._work,
                args=(progress, done),
                name="locitize-openwebui-install",
                daemon=True,
            )
            self._thread = thread
            thread.start()
        return True

    def cancel(self) -> None:
        self._cancel.set()
        with self._guard:
            if self.state == "running":
                self.state = "cancelling"
            process = self._process
        if process is not None and process.poll() is None:
            self._terminate_tree(process)

    def shutdown(self) -> None:
        self.cancel()
        with self._guard:
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)

    def _publish(
        self,
        progress: Callable[[dict[str, Any]], None],
        message: str,
        *,
        phase: str,
    ) -> None:
        progress({"state": self.state, "phase": phase, "message": message})
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    def _finish(
        self,
        done: Callable[[InstallResult], None],
        *,
        ok: bool,
        state: str,
        message: str,
    ) -> None:
        with self._guard:
            self.state = state
        done(InstallResult(ok, state, message, str(self.log_path)))

    def _work(self, progress, done) -> None:
        from setup_env import OPENWEBUI_VERSION

        package = f"open-webui=={OPENWEBUI_VERSION}"
        expected = self._expected_endpoints()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\nOpen WebUI install started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        try:
            import egress_log

            egress_log.record_installer_event(
                action="openwebui-install", program="pip", package=package,
                expected_endpoints=expected,
                outcome="started", user_initiated=True,
            )
            result = self._install(progress, package)
        except Exception as exc:  # noqa: BLE001 - worker boundary
            result = InstallResult(
                False, "failed", _friendly_failure(str(exc), self.log_path),
                str(self.log_path),
            )
        try:
            import egress_log

            egress_log.record_installer_event(
                action="openwebui-install", program="pip", package=package,
                expected_endpoints=expected,
                outcome=result.state, user_initiated=True,
            )
        finally:
            self._finish(
                done, ok=result.ok, state=result.state, message=result.message
            )

    def _install(self, progress, package: str) -> InstallResult:
        if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
            return InstallResult(
                False, "failed", "Open WebUI requires Python 3.11 or 3.12.",
                str(self.log_path),
            )
        from setup_plan import REQUIREMENTS
        from webui import webui_venv_python

        python = webui_venv_python(self.settings)
        venv = python.parent.parent
        estimate_mb = int(REQUIREMENTS["webui_pip"].size_mb or 0)
        required = (estimate_mb + 512) * 1024 * 1024
        for location in (self._existing_parent(venv), Path(tempfile.gettempdir())):
            if self._disk_usage(str(location)).free < required:
                return InstallResult(
                    False, "failed",
                    "Open WebUI could not be installed because the disk is full.",
                    str(self.log_path),
                )

        while not self.start_lock.acquire(timeout=0.1):
            if self._cancel.is_set():
                return self._cancelled()
        try:
            if self._cancel.is_set():
                return self._cancelled()
            from webui import ensure_single_openwebui_processes

            if self._stop_fn is not None:
                self._stop_fn()
            ensure_single_openwebui_processes(
                int(self.settings.ports.openwebui), force=True, venv_dir=venv
            )
            if not python.is_file():
                self._publish(progress, "Creating the Open WebUI environment...", phase="venv")
                outcome = self._runner(
                    [str(self._base_python), "-m", "venv", str(venv)],
                    self._cancel, lambda line: self._publish(progress, line, phase="venv"),
                    self._set_process,
                )
                if outcome.cancelled:
                    return self._cancelled()
                if outcome.returncode != 0 or not python.is_file():
                    return self._failed("\n".join(outcome.lines))

            self._publish(
                progress,
                f"Installing Open WebUI ({estimate_mb} MB download estimate)...",
                phase="install",
            )
            outcome = self._runner(
                [str(python), "-m", "pip", "install", "--disable-pip-version-check", package],
                self._cancel, lambda line: self._publish(progress, line, phase="install"),
                self._set_process,
            )
            if outcome.cancelled:
                return self._cancelled()
            if outcome.returncode != 0:
                return self._failed("\n".join(outcome.lines))
            ok, detail = self._verifier(self.settings)
            if self._cancel.is_set():
                return self._cancelled()
            if not ok:
                return self._failed(detail)

            from config import write_openwebui_enabled

            write_openwebui_enabled(self.settings.data_dir, True)
            self.settings.openwebui.enabled = True
            self._publish(progress, "Open WebUI is installed.", phase="complete")
            return InstallResult(
                True, "succeeded", "Open WebUI is installed.", str(self.log_path)
            )
        finally:
            self.start_lock.release()

    @staticmethod
    def _existing_parent(path: Path) -> Path:
        candidate = path
        while not candidate.exists() and candidate.parent != candidate:
            candidate = candidate.parent
        return candidate

    @staticmethod
    def _expected_endpoints() -> list[str]:
        configured = [
            OpenWebUIInstaller._safe_endpoint(value)
            for key in ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL")
            for value in os.environ.get(key, "").split()
            if value.strip()
        ]
        if configured:
            return configured
        return ["https://pypi.org", "https://files.pythonhosted.org"]

    @staticmethod
    def _safe_endpoint(value: str) -> str:
        parsed = urllib.parse.urlparse(value.strip())
        if not parsed.hostname:
            return "configured pip index"
        scheme = parsed.scheme or "https"
        try:
            port = f":{parsed.port}" if parsed.port else ""
        except ValueError:
            port = ""
        return f"{scheme}://{parsed.hostname}{port}"

    def _verify(self, settings) -> tuple[bool, str]:
        from setup_env import OPENWEBUI_VERSION
        from webui import openwebui_package_version, webui_console_script, webui_venv_python

        if not webui_console_script(settings).is_file():
            return False, "The Open WebUI entry point was not created."
        if openwebui_package_version(settings) != OPENWEBUI_VERSION:
            return False, "The installed Open WebUI version did not match the pinned version."
        outcome = self._runner(
            [str(webui_venv_python(settings)), "-m", "pip", "check"],
            self._cancel, lambda _line: None, self._set_process,
        )
        if outcome.cancelled:
            return False, "Installation was cancelled."
        if outcome.returncode != 0:
            return False, "Open WebUI dependency verification failed."
        return True, "verified"

    def _failed(self, detail: str) -> InstallResult:
        return InstallResult(
            False, "failed", _friendly_failure(detail, self.log_path),
            str(self.log_path),
        )

    def _cancelled(self) -> InstallResult:
        return InstallResult(
            False, "cancelled", "Open WebUI installation was cancelled.",
            str(self.log_path),
        )

    def _set_process(self, process) -> None:
        with self._guard:
            self._process = process

    def _run_process(self, argv, cancel, on_line, set_process) -> CommandOutcome:
        try:
            process = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=os.environ.copy(),
                creationflags=_NO_WINDOW | _NEW_GROUP,
            )
        except OSError as exc:
            return CommandOutcome(1, (str(exc),))
        set_process(process)
        lines: list[str] = []
        output: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            assert process.stdout is not None
            for raw in process.stdout:
                output.put(raw.rstrip())
            output.put(None)

        threading.Thread(target=read_output, name="openwebui-install-output", daemon=True).start()
        ended = False
        while not ended:
            if cancel.is_set() and process.poll() is None:
                self._terminate_tree(process)
            try:
                line = output.get(timeout=0.1)
            except queue.Empty:
                if process.poll() is not None:
                    ended = True
                continue
            if line is None:
                ended = True
            else:
                lines.append(line)
                del lines[:-80]
                on_line(line)
        code = process.wait()
        set_process(None)
        return CommandOutcome(code, tuple(lines), cancelled=cancel.is_set())

    @staticmethod
    def _terminate_tree(process: subprocess.Popen[str]) -> None:
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    capture_output=True, timeout=10, check=False,
                    creationflags=_NO_WINDOW,
                )
                return
            except (OSError, subprocess.SubprocessError):
                pass
        try:
            process.kill()
        except OSError:
            pass
