"""The privacy ledger: a verifiable record of every outbound connection (M17.1).

LOCITIZE's headline promise is that nothing leaves your machine unless you asked
it to. This module records the outbound connections LOCITIZE's own app makes,
so that promise can be checked rather than merely asserted.

What it covers: every request the running app makes through
modelhub.open_checked (the T2 allowlist chokepoint) - model search, model and
component downloads, release checks. What it does NOT cover, because those
requests are made by other programs: pip/winget installs, the setup wizard's
own process, Open WebUI (including its optional web search), the fine-tune
studio, and coding tools launched from LOCITIZE (see SECURITY.md). Each record is host,
UTC timestamp, and the reason the connection happened ("hub-search",
"download", "llama.cpp-release"), appended to a JSONL in the data root. Nothing
about the request body or response is logged - the ledger proves WHEN and TO
WHOM the machine talked, which is the privacy question, not what was said.

The reader-facing payoff (Models page, and `launcher.py --egress`): "This
session: 0 outbound connections" or a dated list. A local-first tool that can
show you its own network history is a claim no cloud product can make.

Pure and dependency-free at module scope; the writer is one append. ASCII only.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

EGRESS_FILE = "egress_log.jsonl"

# The active reason for connections happening right now. A call site sets it
# around a burst of requests (a search, a download) so each recorded hop is
# attributed to the user action that caused it, not to a bare stack trace.
_reason = threading.local()
# The data root to log into, set once at startup. None = logging disabled
# (e.g. a unit test that never configured one), which must never raise.
_root: Path | None = None
_lock = threading.Lock()


def configure(data_root: Path | str | None) -> None:
    """Point the ledger at a data root. Called once at launch; None disables."""
    global _root
    _root = Path(data_root) if data_root is not None else None


def egress_path(data_root: Path | str) -> Path:
    return Path(data_root) / "reports" / EGRESS_FILE


class reason:  # noqa: N801 - used as a context manager, reads like one
    """Attribute every connection opened in this block to `label`.

        with egress_log.reason("hub-search"):
            ... requests ...

    Nests and restores the prior label, so a download inside a search still
    reads as "download" for its own hops and reverts afterwards.
    """

    def __init__(self, label: str) -> None:
        self._label = label
        self._prev: str | None = None

    def __enter__(self) -> "reason":
        self._prev = getattr(_reason, "label", None)
        _reason.label = self._label
        return self

    def __exit__(self, *_exc: Any) -> None:
        _reason.label = self._prev


def current_reason() -> str:
    return getattr(_reason, "label", None) or "unattributed"


def record(url: str, when: float | None = None) -> None:
    """Append one egress record. Never raises - a logging failure must never
    break the download it is recording."""
    if _root is None:
        return
    try:
        host = urllib.parse.urlparse(str(url)).hostname or "?"
        line = json.dumps(
            {
                "ts": time.strftime(
                    "%Y-%m-%dT%H:%M:%S",
                    time.localtime(when if when is not None else time.time()),
                ),
                "host": host,
                "reason": current_reason(),
            },
            ensure_ascii=True,
        )
        target = egress_path(_root)
        with _lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:  # noqa: BLE001 - the ledger is a witness, never a gate
        pass


@dataclass
class EgressSummary:
    total: int
    hosts: dict[str, int]
    recent: list[dict[str, str]]


def summarize(data_root: Path | str, limit: int = 20) -> EgressSummary:
    """Read the ledger back for display. A missing file is zero connections -
    the honest, and best, answer."""
    target = egress_path(data_root)
    rows: list[dict[str, str]] = []
    if target.is_file():
        for raw in target.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(raw))
            except (TypeError, json.JSONDecodeError):
                continue
    hosts: dict[str, int] = {}
    for r in rows:
        hosts[r.get("host", "?")] = hosts.get(r.get("host", "?"), 0) + 1
    return EgressSummary(total=len(rows), hosts=hosts, recent=rows[-limit:])


def render_line(summary: EgressSummary) -> str:
    """One honest sentence for a status chip."""
    if summary.total == 0:
        return (
            "0 outbound connections recorded by locitize's own downloads "
            "(installers, Open WebUI and coding tools are not covered)."
        )
    hosts = ", ".join(sorted(summary.hosts))
    noun = "connection" if summary.total == 1 else "connections"
    return f"{summary.total} outbound {noun} recorded ({hosts})."
