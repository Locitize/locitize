"""Refuse HTTP requests that did not come from a program on this machine.

LOCITIZE's services listen on 127.0.0.1 only, but a web page the user visits
can still send requests to 127.0.0.1, and a DNS-rebinding page can make its own
host name resolve there. Every legitimate caller (Open WebUI's server, the
desktop app, coding tools) connects to the loopback address and sends no
browser Origin, so two header checks stop both:

  - Host must name the loopback address (and the service's port): a rebinding
    page keeps its own host name in the Host header;
  - a browser Origin, when present, must itself be a loopback address.
"""

from __future__ import annotations

import urllib.parse

_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]")
_LOOPBACK_ORIGIN_HOSTS = ("127.0.0.1", "localhost", "::1")


def foreign_request_reason(host: str | None, origin: str | None, port: int) -> str | None:
    """Why a request must be refused as not coming from this machine, or None."""
    allowed = {f"{name}:{port}" for name in _LOOPBACK_HOSTS} | set(_LOOPBACK_HOSTS)
    if (host or "").strip().lower() not in allowed:
        return "host is not this machine's loopback address"
    if origin is None:
        return None
    parsed = urllib.parse.urlsplit(origin.strip().lower())
    if parsed.scheme in ("http", "https") and (parsed.hostname or "") in _LOOPBACK_ORIGIN_HOSTS:
        return None
    return "cross-site browser requests are not accepted"
