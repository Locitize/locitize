"""Optional loopback reverse proxy for the LOCITIZE command center (Architecture G4).

Goal: let the owner reach the active model's llama.cpp chat UI at a friendly
hostname (http://locitize.local:8085/) instead of http://127.0.0.1:<port>/. It is a
convenience feature that is OFF by default; the plain Chat button to
127.0.0.1:<port> is always available and needs nothing from this module.

Design and safety (Permission Matrix section 7):
- Binds 127.0.0.1 ONLY (a hardcoded host), never a routable interface.
- Forwards every request to 127.0.0.1:<active model port>, read fresh from a
  port_provider callable on each request, so the proxy FOLLOWS the active model
  across a switch and returns a friendly 503 when no model runs.
- Uses http.client for the upstream hop, which does not follow redirects, and
  streams the response body so llama.cpp's chunked/SSE chat streaming survives.
- NEVER performs elevation. Port 80 is an explicit opt-in (proxy.bind_port_80)
  that still requires the owner to have launched LOCITIZE elevated; a non-privileged
  port-80 bind fails and is surfaced as a remedy pointing at the elevated opt-in
  script - it is never silently downgraded and the hosts file is never edited by
  any code path here.

The request-target logic (build_upstream_target) is split out as a pure function
so it is unit-tested without binding a socket (AC13, keyword `proxy_target`).
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

# The proxy only ever forwards to loopback; the host is a hardcoded constant so no
# config value can steer it off 127.0.0.1 (Permission Matrix section 7).
_UPSTREAM_HOST = "127.0.0.1"

# Requests larger than this are refused rather than buffered unbounded. A local
# chat request body is tiny; this is a simple guard, not a security boundary.
_MAX_BODY_BYTES = 32 * 1024 * 1024


def build_upstream_target(active_port: int | None) -> tuple[str, int] | None:
    """Resolve the upstream (host, port) for the active model, or None if none runs.

    Pure function (no socket): given the active model's resolved loopback port it
    returns ("127.0.0.1", port); given None (no model running) it returns None, and
    the handler answers a friendly 503. This is the target-following heart of the
    proxy and is unit-tested directly.
    """
    if active_port is None:
        return None
    if not isinstance(active_port, int) or isinstance(active_port, bool):
        return None
    if not (1 <= active_port <= 65535):
        return None
    return (_UPSTREAM_HOST, active_port)


def effective_bind_port(proxy_config: Any) -> int:
    """The port the proxy will actually bind.

    port 80 only when the owner opted in via proxy.bind_port_80 (which still needs
    an elevated LOCITIZE); otherwise the non-privileged proxy.port (default 8085).
    """
    if getattr(proxy_config, "bind_port_80", False):
        return 80
    return int(proxy_config.port)


class _ProxyHandler(BaseHTTPRequestHandler):
    """Forwards each request to the active model's loopback port, streaming the body.

    The server instance carries the port_provider; the handler reads it fresh per
    request so a model switch is picked up immediately.
    """

    # Quiet by default: the platform logs elsewhere and a per-request stderr line
    # would spam the console. Errors still surface as HTTP status codes.
    def log_message(self, *args: Any) -> None:  # noqa: D401 - stdlib signature
        return

    def do_GET(self) -> None:
        self._forward("GET")

    def do_POST(self) -> None:
        self._forward("POST")

    def do_OPTIONS(self) -> None:
        self._forward("OPTIONS")

    def do_DELETE(self) -> None:
        self._forward("DELETE")

    def _forward(self, method: str) -> None:
        import http.client

        target = build_upstream_target(self.server.port_provider())  # type: ignore[attr-defined]
        if target is None:
            self._send_no_model()
            return
        host, port = target

        body = self._read_body()
        if body is None:
            self.send_error(413, "request body too large")
            return

        # Copy client headers verbatim except hop-by-hop ones; rewrite Host to the
        # upstream so llama.cpp sees a loopback request.
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in ("host", "connection", "proxy-connection")
        }
        headers["Host"] = f"{host}:{port}"

        upstream = http.client.HTTPConnection(host, port, timeout=300)
        try:
            upstream.request(method, self.path, body=body, headers=headers)
            response = upstream.getresponse()
            # Do NOT follow redirects (SEC-M3-1): http.client returns the 3xx as-is
            # and we relay it to the browser, which decides.
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() in ("connection", "transfer-encoding"):
                    continue
                self.send_header(key, value)
            self.end_headers()
            # Stream the body in chunks so SSE/chunked chat streaming is preserved.
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
        except (OSError, http.client.HTTPException):
            self.send_error(502, "upstream model not reachable")
        finally:
            try:
                upstream.close()
            except OSError:
                pass

    def _read_body(self) -> bytes | None:
        """Read the request body honouring Content-Length; None if it is too large."""
        length = self.headers.get("Content-Length")
        if length is None:
            return b""
        try:
            size = int(length)
        except ValueError:
            return b""
        if size > _MAX_BODY_BYTES:
            return None
        return self.rfile.read(size) if size > 0 else b""

    def _send_no_model(self) -> None:
        """Friendly 503 when no model is running (the proxy has nothing to target)."""
        page = (
            b"<html><body><h1>locitize</h1>"
            b"<p>No model is running. Start one in the locitize command center, "
            b"then reload this page.</p></body></html>"
        )
        self.send_response(503, "no model running")
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)


class ReverseProxy:
    """In-process loopback reverse proxy managed as a daemon thread (G4).

    Not a ManagedProcess (that is for external binaries); this is Python code, so
    the GUI owns it directly and shutdown() tears it down. Binding a privileged
    port (80) without elevation raises PermissionError, which the caller renders as
    a remedy - the proxy never elevates itself.
    """

    def __init__(
        self,
        proxy_config: Any,
        port_provider: Callable[[], int | None],
    ) -> None:
        self._config = proxy_config
        self._port_provider = port_provider
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def bind_port(self) -> int:
        return effective_bind_port(self._config)

    def start(self) -> None:
        """Bind 127.0.0.1:<bind_port> and serve on a daemon thread.

        Raises PermissionError (with a remedy) if a privileged port-80 bind is
        attempted without Administrator rights; the platform never elevates.
        """
        port = self.bind_port
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), _ProxyHandler)
        except PermissionError as exc:
            raise PermissionError(
                f"binding port {port} needs Administrator; run locitize elevated and "
                f"the scripts/enable_locitize_local.ps1 opt-in, or use the default "
                f"loopback port {self._config.port}"
            ) from exc
        # Expose the port_provider to the handler instances.
        server.port_provider = self._port_provider  # type: ignore[attr-defined]
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, name="locitize-proxy", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop serving and join the thread (idempotent)."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
