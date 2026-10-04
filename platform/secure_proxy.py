"""Caddy TLS reverse proxy for https://locitize.local (owner request 2026-08-14).

Goal: the Chat button opens https://<hostname>/ with a real padlock instead of
"Not secure" on a bare loopback port. This module makes LOCITIZE own that chain
end-to-end so a fresh machine (or a reboot that lost the proxy) self-heals on
chat launch:

  1. discover caddy.exe (settings override, PATH, winget user package dir) --
     optionally installing it user-scope via winget (no elevation, one-time)
  2. materialize the Caddyfile if missing (loopback-only bind, tls internal,
     reverse_proxy to the Open WebUI port)
  3. start caddy detached if 127.0.0.1:443 is not answering
  4. trust Caddy's local root CA in the CURRENT-USER store via certutil when an
     ssl handshake against the hostname fails (Windows may show the owner one
     consent dialog -- never silent, never the machine store)

Design and safety (Permission Matrix section 7, mirroring proxy.py):
- Everything binds/probes 127.0.0.1 only; the Caddyfile template hardcodes
  `bind 127.0.0.1` so nothing is ever reachable from the LAN.
- NEVER performs elevation. The hosts-file line (hostname -> 127.0.0.1) is the
  single elevated prerequisite; when it is missing, ensure() returns a remedy
  string for the owner instead of touching the hosts file.
- The winget install and certutil trust are current-user scope: reversible with
  `winget uninstall CaddyServer.Caddy` / `certutil -user -delstore Root <serial>`.
- All child processes run with a discrete argv (never a shell string).

The module is import-light and GUI-agnostic: callers (gui_controller) get back
(ok, message) and surface it on the monitor rail / chat status line.
"""

from __future__ import annotations

import socket
import ssl
import subprocess
import sys
from pathlib import Path
from typing import Callable

from config import Settings

# Where the winget user-scope package lands (version-independent glob). Kept as
# a constant so a packaging change is a one-line edit.
_WINGET_PACKAGES = (
    Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
)
_WINGET_ID = "CaddyServer.Caddy"
# Caddy's local-CA root certificate inside its default storage. certutil needs
# the file path; caddy creates it on first `tls internal` issuance.
_CADDY_ROOT_CRT = (
    Path.home() / "AppData/Roaming/Caddy/pki/authorities/local/root.crt"
)

_PROBE_TIMEOUT_S = 0.8
_START_WAIT_S = 6.0


def _port_open(host: str, port: int, timeout: float = _PROBE_TIMEOUT_S) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _hostname_resolves_loopback(hostname: str) -> bool:
    """True when the hosts-file line (or DNS) maps hostname to loopback."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except OSError:
        return False
    return any(info[4][0] in ("127.0.0.1", "::1") for info in infos)


def _tls_trusted(hostname: str) -> bool:
    """Full handshake with the system trust store (Windows CA store on win32)."""
    context = ssl.create_default_context()
    try:
        with socket.create_connection((hostname, 443), timeout=2.0) as sock:
            with context.wrap_socket(sock, server_hostname=hostname):
                return True
    except (OSError, ssl.SSLError):
        return False


def find_caddy(settings: Settings) -> str | None:
    """Resolve caddy.exe: explicit setting, PATH, then the winget package dir."""
    explicit = settings.secure_proxy.caddy_path
    if explicit and Path(explicit).exists():
        return explicit
    import shutil

    on_path = shutil.which("caddy")
    if on_path:
        return on_path
    if _WINGET_PACKAGES.exists():
        for candidate in _WINGET_PACKAGES.glob(f"{_WINGET_ID}_*/caddy.exe"):
            return str(candidate)
    return None


def _install_caddy() -> tuple[bool, str]:
    """User-scope winget install; no elevation, ~15 MB download, one-time."""
    winget = (
        Path.home() / "AppData/Local/Microsoft/WindowsApps/winget.exe"
    )
    exe = str(winget) if winget.exists() else "winget"
    try:
        proc = subprocess.run(
            [
                exe,
                "install",
                "--id",
                _WINGET_ID,
                "--accept-source-agreements",
                "--accept-package-agreements",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            # Same reasoning as the caddy start call below: winget.exe is a
            # console app, the GUI has no console of its own, so without this
            # a new one flashes visibly for the run.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"winget install failed to run: {exc}"
    if proc.returncode != 0:
        tail = (proc.stdout or proc.stderr or "").strip()[-200:]
        return False, f"winget install exited {proc.returncode}: {tail}"
    return True, "caddy installed (winget, user scope)"


def resolve_caddyfile_path(settings: Settings) -> Path:
    """Return the Caddyfile LOCITIZE generates, or the one the user configured.

    <data root>/Caddyfile when secure_proxy.caddyfile is blank, never the
    install directory (DEC-M14-9). The generated file is derived from the user's
    own settings (their hostname, their port), it must survive a reinstall
    alongside the certificate trust it implies, and the install tree is
    read-only at runtime (invariant W1). An explicitly configured path is
    honoured exactly as written.
    """
    if settings.secure_proxy.caddyfile:
        return Path(settings.secure_proxy.caddyfile)
    return Path(settings.data_dir) / "Caddyfile"


def _caddyfile_body(hostname: str, upstream_port: int) -> str:
    return (
        "\n\n"
        "# LOCITIZE secure proxy (managed by secure_proxy.py; regenerated if deleted)\n"
        f"# https://{hostname}/ -> 127.0.0.1:{upstream_port} (Open WebUI)\n"
        "# bind 127.0.0.1 keeps both listeners loopback-only; nothing on the LAN.\n"
        f"{hostname} {{\n"
        "\tbind 127.0.0.1\n"
        "\ttls internal\n"
        f"\treverse_proxy 127.0.0.1:{upstream_port}\n"
        "}\n"
    )


def ensure(
    settings: Settings,
    notify: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """Make https://<hostname>/ live and trusted; return (ok, status message).

    Idempotent and cheap when everything is already up (two socket probes and
    one TLS handshake). Never elevates; the only owner-visible side effects are
    a one-time winget install and at most one Windows certificate consent
    dialog, both announced through notify() first.
    """
    say = notify or (lambda _msg: None)
    cfg = settings.secure_proxy
    if not cfg.enabled:
        return False, "secure proxy disabled in settings"
    if sys.platform != "win32":
        return False, "secure proxy is Windows-only in this build"
    hostname = cfg.hostname

    # The hosts line is the one elevated prerequisite LOCITIZE never writes itself.
    if not _hostname_resolves_loopback(hostname):
        return False, (
            f"'{hostname}' does not resolve to 127.0.0.1; add the hosts line "
            f"once from an elevated prompt: "
            # $env:SystemRoot expands to the Windows directory, so the remedy carries
            # no hardcoded drive letter and works on any install.
            f"Add-Content $env:SystemRoot\\System32\\drivers\\etc\\hosts "
            f'"127.0.0.1 {hostname}"'
        )

    caddy = find_caddy(settings)
    if caddy is None:
        if not cfg.auto_install:
            return False, (
                "caddy.exe not found and secure_proxy.auto_install is false; "
                "install with: winget install CaddyServer.Caddy"
            )
        say("installing caddy (winget, user scope, one-time)...")
        ok, message = _install_caddy()
        if not ok:
            return False, message
        caddy = find_caddy(settings)
        if caddy is None:
            return False, "caddy installed but not found; set secure_proxy.caddy_path"

    caddyfile = resolve_caddyfile_path(settings)
    if not caddyfile.exists():
        caddyfile.parent.mkdir(parents=True, exist_ok=True)
        caddyfile.write_text(
            _caddyfile_body(hostname, settings.ports.openwebui), encoding="utf-8"
        )
        say(f"wrote {caddyfile}")

    if not _port_open("127.0.0.1", 443):
        say("starting caddy...")
        try:
            # `caddy start` self-daemonizes and returns, but the daemon child
            # INHERITS our stdio handles -- capturing pipes here deadlocks the
            # reader for the daemon's lifetime, so everything goes to DEVNULL
            # and readiness is judged by the port probe below instead of output.
            # CREATE_NO_WINDOW keeps the GUI free of console flashes.
            subprocess.run(
                [caddy, "start", "--config", str(caddyfile)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=_START_WAIT_S * 5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"caddy start failed: {exc}"
        import time

        deadline = time.monotonic() + _START_WAIT_S
        while time.monotonic() < deadline:
            if _port_open("127.0.0.1", 443):
                break
            time.sleep(0.25)
        else:
            return False, "caddy started but 127.0.0.1:443 never answered"

    if not _tls_trusted(hostname):
        # First issuance creates root.crt; trust it user-scope. Windows may show
        # one consent dialog here -- announced, never silent.
        if not _CADDY_ROOT_CRT.exists():
            return False, (
                f"caddy root certificate not found at {_CADDY_ROOT_CRT}; "
                "visit the https URL once so caddy issues it, then relaunch chat"
            )
        say("trusting caddy local root CA (you may see one Windows dialog)...")
        try:
            proc = subprocess.run(
                ["certutil", "-user", "-addstore", "Root", str(_CADDY_ROOT_CRT)],
                capture_output=True,
                text=True,
                timeout=120,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"certutil trust failed to run: {exc}"
        if proc.returncode != 0:
            return False, (
                "certutil could not add the root CA (owner may have declined); "
                "https will warn until it is trusted"
            )
        if not _tls_trusted(hostname):
            return False, "root CA added but the TLS handshake still fails"

    return True, f"https://{hostname}/ live and trusted"
