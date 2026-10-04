"""Detect a Tailscale Serve URL that already points at this machine's loopback chat.

LOCITIZE stays bound to 127.0.0.1. The owner can (and on this machine, does)
front Open WebUI with `tailscale serve`, which is a Tailscale-owned HTTPS
listener on the tailnet only. This module reads that configuration from the
local `tailscale` CLI. It never binds a new port, never calls Funnel, and
never contacts the internet.

The returned URL is what a phone on the same tailnet opens. Discovery is a
local subprocess; a missing binary or a stopped tailscaled is None, not an
error the Chat page has to explain as a crash.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
_TIMEOUT_S = 4.0

# Documented OWUI Serve HTTPS port (alt URL; WEBUI_URL uses root :443).
_PREFERRED_OWUI_SERVE_PORT = "4443"
# Explicit WEBUI_URL for machines whose MagicDNS name should survive a boot where
# Tailscale is not up yet (e.g. "https://<host>.<tailnet>.ts.net").
_WEBUI_URL_ENV = "LOCITIZE_WEBUI_URL"
_DEFAULT_OPENWEBUI_PORT = 8096


@dataclass(frozen=True)
class PhoneAccess:
    """One already-configured tailnet URL that proxies a local loopback port."""

    url: str
    proxy_target: str
    hostname: str
    tailnet_only: bool = True


def discover_phone_access(
    runner: Callable[..., Any] | None = None,
    openwebui_port: int | None = None,
) -> PhoneAccess | None:
    """Return the Serve HTTPS URL if Tailscale is already proxying loopback chat.

    ``runner`` is the subprocess.run seam (tests inject a fake). Production uses
    the real CLI with CREATE_NO_WINDOW so a Chat-page paint never flashes a
    console. ``openwebui_port`` prefers the handler that already points at
    Open WebUI; if Serve has only one web handler, that one is used.
    """
    run = runner or _run_tailscale
    raw = run(["serve", "status", "--json"])
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    return _from_serve_status(payload, openwebui_port=openwebui_port)


def magicdns_name(runner: Callable[..., Any] | None = None) -> str | None:
    """Return this node's MagicDNS name (no trailing dot), or None."""
    run = runner or _run_tailscale
    raw = run(["status", "--json"])
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    self = payload.get("Self")
    if not isinstance(self, dict):
        return None
    dns = str(self.get("DNSName") or "").strip().rstrip(".")
    return dns or None


def resolve_webui_serve_url(
    openwebui_port: int | None = None,
    runner: Callable[..., Any] | None = None,
) -> str:
    """Root HTTPS URL Open WebUI should advertise as WEBUI_URL (no port).

    WEBUI_URL is https://<magicdns> (Serve root :443 -> OWUI), not :4443.
    Order: ``LOCITIZE_WEBUI_URL`` if set, then live MagicDNS from
    ``tailscale status``, then the loopback Open WebUI address so a machine
    without Tailscale still gets a working, never-empty WEBUI_URL.
    """
    override = os.environ.get(_WEBUI_URL_ENV, "").strip().rstrip("/")
    if override:
        return override
    magic = magicdns_name(runner=runner)
    if magic:
        return f"https://{magic}"
    return f"http://127.0.0.1:{openwebui_port or _DEFAULT_OPENWEBUI_PORT}"


def _run_tailscale(args: list[str]) -> str:
    """Run `tailscale <args>` and return stdout, or empty on any failure."""
    exe = shutil.which("tailscale")
    if not exe:
        return ""
    try:
        proc = subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
            creationflags=_NO_WINDOW,
            env=os.environ.copy(),
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout or ""


def _serve_https_url(host_key: str) -> tuple[str, str] | None:
    """Return (hostname, https_url) for a Serve Web host key like host:4443."""
    key = str(host_key).strip().rstrip(".")
    if not key:
        return None
    hostname, sep, port = key.partition(":")
    hostname = hostname.strip().rstrip(".")
    if not hostname:
        return None
    if sep and port.isdigit() and port not in ("443", "80"):
        return hostname, f"https://{hostname}:{port}/"
    return hostname, f"https://{hostname}/"


def _from_serve_status(
    payload: dict[str, Any], openwebui_port: int | None = None
) -> PhoneAccess | None:
    """Pick the HTTPS Serve handler that fronts a loopback URL."""
    web = payload.get("Web")
    if not isinstance(web, dict) or not web:
        return None
    candidates: list[PhoneAccess] = []
    for host_key, spec in web.items():
        if not isinstance(spec, dict):
            continue
        handlers = spec.get("Handlers")
        if not isinstance(handlers, dict):
            continue
        root = handlers.get("/") or handlers.get("")
        if not isinstance(root, dict):
            continue
        proxy = str(root.get("Proxy") or "").strip()
        if not proxy:
            continue
        parsed = _serve_https_url(str(host_key))
        if parsed is None:
            continue
        hostname, url = parsed
        candidates.append(
            PhoneAccess(
                url=url,
                proxy_target=proxy,
                hostname=hostname,
                tailnet_only=True,
            )
        )
    if not candidates:
        return None
    if openwebui_port is not None:
        needle = f"127.0.0.1:{int(openwebui_port)}"
        matches = [item for item in candidates if needle in item.proxy_target]
        if matches:
            # Prefer the documented OWUI Serve port when several handlers exist
            # (e.g. live :4443 plus a leftover :9443 both -> 8096).
            for item in matches:
                if f":{_PREFERRED_OWUI_SERVE_PORT}" in item.url:
                    return item
            return matches[0]
    return candidates[0]