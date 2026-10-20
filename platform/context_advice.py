"""Context-ceiling advice (M17.3): manage measured ceiling warnings.

The hardest-won lesson of this project's tuning work is that the VRAM cliff is
a WALL, not a slope: past the point where the KV cache stops fitting, 2048 more
context tokens can cost ~90% of generation throughput. scripts/
measure_context_ceilings.py records each model's measured ceiling in
reports/ctx_ceilings.json; this module reads that back and answers one
question - "is this model's configured context past the cliff?" - so the UI can
warn instead of letting a user silently choose a 10x-slower config.

A missing or unreadable ceilings file means "no measured ceiling known", which
yields no warning rather than a false alarm. Replacing a model's weights removes
only that model's stale measurement with an atomic best-effort edit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def _ceilings_path(data_root: Path | str) -> Path:
    return Path(data_root) / "reports" / "ctx_ceilings.json"


def load_ceilings(data_root: Path | str) -> dict[str, int]:
    """Map model id -> measured best context, from the ceilings report.

    Only entries with a positive integer `best` are returned; a run that failed
    to measure a model contributes nothing rather than a misleading zero.
    """
    target = _ceilings_path(data_root)
    if not target.is_file():
        return {}
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for model_id, row in raw.items():
        best = row.get("best") if isinstance(row, dict) else None
        if isinstance(best, int) and not isinstance(best, bool) and best > 0:
            out[str(model_id)] = best
    return out


def remove_ceiling(data_root: Path | str, model_id: str) -> bool:
    """Atomically remove one model's stale ceiling, best-effort.

    Missing, unreadable, or malformed reports are left untouched. A write or
    replace failure likewise leaves the original report in place. The return
    value says whether an entry was removed; callers must not make saving a
    model depend on it.
    """
    target = _ceilings_path(data_root)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(raw, dict) or str(model_id) not in raw:
        return False

    updated = dict(raw)
    del updated[str(model_id)]
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(updated, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        temp_path = None
        return True
    except (OSError, TypeError, ValueError):
        return False
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def context_warning(
    model_id: str, context_size: int, ceilings: dict[str, int]
) -> str:
    """Return a one-line warning when `context_size` is past the measured cliff.

    Empty string means "fine, or unknown" - a model with no measured ceiling
    gets no warning, because a guess dressed as a warning is worse than
    silence. The threshold is the measured ceiling itself: at or below it the
    config was proven to hold throughput; above it, the cliff was measured.
    """
    best = ceilings.get(model_id)
    if not best or context_size <= best:
        return ""
    return (
        f"context {context_size} is past this model's measured ceiling of "
        f"{best} on this machine - expect a large slowdown (the KV cache "
        f"spills out of VRAM). Re-run auto-tune, or lower the context."
    )
