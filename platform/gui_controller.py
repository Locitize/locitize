"""Headless controller/marshalling seam for the LOCITIZE Tkinter command center.

This module is the testable heart of the GUI (Architecture "GUI Command Center"
G1/G2/G5). It imports NO tkinter, so the whole test suite runs on a headless
machine and never constructs a Tk root (G7). gui.py is a thin presentation shell
that binds each widget to one intent method here and renders the Result objects
this module marshals back through a thread-safe queue.

What lives here (and why it is not in gui.py):
- The threading contract (G2): one long-lived operations worker (Thread B) that
  serializes every lifecycle mutation, and one monitor thread (Thread C) that
  polls /metrics + /slots while a model runs. gui.py owns only the UI thread and
  a root.after() pump that drains result_q.
- Field validation for the gpu_layers / context_size editors (Panel 3).
- Pure /metrics and /slots parsing that degrades honestly - an absent field is
  None (rendered "-"), never a fabricated 0 (G5).
- The button-state machine (which controls are enabled given the current state).
- Orchestration of the EXISTING ModelController / SingleServiceController /
  ServiceManager - this module adds no new process-lifecycle code (G1). The same
  ServiceManager the terminal path uses is passed in, so the existing
  atexit.register(stop_all) no-orphan guarantee already covers the GUI process.

Security: the monitor's loopback HTTP client uses http.client directly (no
urllib), which does not follow redirects (SEC-M3-1), and only ever connects to
127.0.0.1:<resolved port> (Permission Matrix section 7).
"""

from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# A gibibyte would read low against the disk vendor's decimal GB (a 10.26e9-byte
# .gguf shows 9.6 "GiB" but 10.3 "GB"); the owner reads sizes in the same decimal
# GB the model card and disk report, so the Size column divides by 1e9 (G2 batch,
# owner request 2 - "10.3 GB").
_BYTES_PER_GB = 1_000_000_000

# --------------------------------------------------------------------------- #
# Marshalled value types (Data Model 6.3). None of these are persisted or hold
# secrets; they live only in process memory for the GUI session.
# --------------------------------------------------------------------------- #


@dataclass
class Command:
    """A queued GUI intent drained by the ops worker (Thread B)."""

    kind: str  # start, stop, whisper_toggle, listen, speak, audition, save_edits,
    #            save_identity, remember_chat_ui, start_openwebui, sentinel
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    """A marshalled outcome the root.after() pump applies to widgets (G2)."""

    kind: str
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# M13: the single-sourced refusal shown when the owner tries to edit or rename a
# DISCOVERED fine-tune. A discovered model has no models.yaml block to rewrite, so
# the write path would have nothing to target; the owner registers it first. Worded
# exactly as Architecture M13.6 / UX Spec 3.5 require.
def _today_stamp() -> str:
    """Today's date as YYYY-MM-DD, for the dated notes this module writes.

    models.yaml's existing comments are all dated by hand ("raised 10000->65536
    2026-08-21"); a note the auto-tuner writes should be readable the same way,
    so a future reader can tell when the value was last measured.
    """
    import datetime

    return datetime.date.today().isoformat()


DISCOVERED_EDIT_REFUSAL = (
    "Discovered fine-tune: click Register to add it to models.yaml before editing."
)

@dataclass
class MetricsSample:
    """Parsed llama.cpp /metrics snapshot (G5).

    Every numeric field is Optional: a field absent from the payload stays None
    and the monitor renders it "-", never 0, so an idle or partial scrape can
    never fabricate a throughput number.
    """

    gen_tokens_s: float | None = None
    prompt_tokens_s: float | None = None
    kv_cache_usage_ratio: float | None = None
    kv_cache_tokens: int | None = None
    requests_processing: int | None = None
    # Cumulative token counters (owner request 4 - show COUNTS, not just rates).
    # In llama.cpp these are monotonically-increasing session totals, so over a
    # fresh model launch they read as the running prompt/gen token counts the owner
    # wants ("prompt 412 tok | gen 583 tok").
    prompt_tokens_total: int | None = None
    gen_tokens_total: int | None = None
    # Cumulative SECONDS spent prompting/generating (newer builds, M18.10).
    # These advance only while the server is actually working, so a ratio of
    # deltas between two polls is the true recent speed - measured, not gauged.
    prompt_seconds_total: float | None = None
    gen_seconds_total: float | None = None
    # Peak tokens observed resident in a slot's context. This build exposes neither
    # kv_cache_tokens nor kv_cache_usage_ratio, so n_tokens_max is the honest
    # fallback source for "context used" (verified live against the real
    # llama-server /metrics on 2026-07-18; see Builder Verification).
    n_tokens_max: int | None = None
    # Total context window, sourced from /slots n_ctx when present, else the
    # model's configured context_size. Used as the denominator of the ctx display.
    n_ctx: int | None = None
    metrics_available: bool = True
    slots: list[dict[str, Any]] | None = None
    sampled_at: float = 0.0


@dataclass
class SlotsSample:
    """Optional /slots enrichment (may be absent/disabled in a build).

    n_ctx (the total context window) and tokens_used (best-effort current context
    fill) are pulled out of the first slot when present; a stripped build (like the
    owner's, whose /slots exposes only id/is_processing/n_ctx/speculative) leaves
    tokens_used None and the caller degrades honestly.
    """

    slots: list[dict[str, Any]] = field(default_factory=list)
    available: bool = True
    n_ctx: int | None = None
    tokens_used: int | None = None


@dataclass
class UiState:
    """Snapshot the UI thread renders from; the pump is its sole writer (G3)."""

    running_model_id: str | None = None
    running_port: int | None = None
    in_flight: bool = False
    whisper_running: bool = False
    proxy_running: bool = False
    latest_metrics: MetricsSample | None = None


# --------------------------------------------------------------------------- #
# Exact llama.cpp Prometheus metric names, single-sourced so a build/version
# rename is a one-line change (RG2). A name that is not present maps to None.
# --------------------------------------------------------------------------- #

_METRIC_GEN_TPS = "llamacpp:predicted_tokens_seconds"
_METRIC_PROMPT_TPS = "llamacpp:prompt_tokens_seconds"
_METRIC_KV_RATIO = "llamacpp:kv_cache_usage_ratio"
_METRIC_KV_TOKENS = "llamacpp:kv_cache_tokens"
_METRIC_REQ_PROCESSING = "llamacpp:requests_processing"
# Cumulative token counters + peak-context gauge (owner request 4). These names
# were confirmed present in the owner's real llama-server build; the kv_cache_*
# names above were confirmed ABSENT in that same build, so both sets are read and
# any missing name degrades to None (never a fabricated count).
_METRIC_PROMPT_TOKENS_TOTAL = "llamacpp:prompt_tokens_total"
_METRIC_GEN_TOKENS_TOTAL = "llamacpp:tokens_predicted_total"
_METRIC_N_TOKENS_MAX = "llamacpp:n_tokens_max"
# M18.10 (owner report: tok/s stopped registering after a llama.cpp upgrade):
# newer builds REMOVED the direct rate gauges (predicted_tokens_seconds /
# prompt_tokens_seconds) and instead expose cumulative time counters alongside
# the token counters. Both generations of names are read; whichever is present
# feeds the display. Confirmed live against the freshly installed build.
_METRIC_GEN_SECONDS_TOTAL = "llamacpp:tokens_predicted_seconds_total"
_METRIC_PROMPT_SECONDS_TOTAL = "llamacpp:prompt_seconds_total"


# --------------------------------------------------------------------------- #
# Pure validation (Panel 3 editors, G3). Returned as (ok, value, error) so both
# the GUI and the unit tests exercise identical rules.
# --------------------------------------------------------------------------- #


def validate_gpu_layers(text: str) -> tuple[bool, int | None, str | None]:
    """Accept an integer >= -1 (-1 means all layers on GPU); reject anything else.

    Rejects floats, non-numeric text, and values < -1. Returns (True, value, None)
    on success or (False, None, remedy) on failure so the editor can show an inline
    hint and keep Save disabled without ever writing a bad value.
    """
    raw = (text or "").strip()
    if not _is_plain_int(raw):
        return False, None, "gpu_layers must be a whole number (999 or -1 = all layers)"
    value = int(raw)
    if value < -1:
        return False, None, "gpu_layers must be >= -1 (999 or -1 = all layers)"
    return True, value, None


def validate_context_size(text: str) -> tuple[bool, int | None, str | None]:
    """Accept a positive integer; reject floats, non-numeric text, and values <= 0."""
    raw = (text or "").strip()
    if not _is_plain_int(raw):
        return False, None, "context_size must be a whole number of tokens"
    value = int(raw)
    if value <= 0:
        return False, None, "context_size must be greater than 0"
    return True, value, None


def _is_plain_int(raw: str) -> bool:
    """True only for an optionally-signed run of digits (no float, no whitespace).

    int("1.0") raises, but we also reject values Python's int() would accept yet a
    human would not expect here (e.g. leading/trailing spaces are already stripped
    by the caller). A leading '+'/'-' is allowed; anything else is rejected.
    """
    if not raw:
        return False
    body = raw[1:] if raw[0] in "+-" else raw
    return body.isdigit()


# --------------------------------------------------------------------------- #
# Pure /metrics and /slots parsing (G5). Never fabricates a number.
# --------------------------------------------------------------------------- #


def parse_metrics(text: str) -> MetricsSample:
    """Parse llama.cpp Prometheus /metrics text into a MetricsSample.

    Format: one `name value` per non-comment line (comment lines start with '#').
    Only the exact llamacpp: gauge names above are read; every other line is
    ignored. A name that never appears leaves its field None (honest degradation),
    so the caller renders "-" rather than 0. An empty or None body yields a sample
    flagged metrics_available=False.
    """
    if not text or not text.strip():
        return MetricsSample(metrics_available=False, sampled_at=time.time())

    values: dict[str, float] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # A Prometheus sample line is "name value" (labels not used by llama.cpp
        # for these gauges). Split on the last whitespace run so a metric name is
        # never confused with its value.
        parts = stripped.rsplit(None, 1)
        if len(parts) != 2:
            continue
        name, raw_value = parts
        try:
            values[name] = float(raw_value)
        except ValueError:
            # A non-numeric value (e.g. NaN text or a malformed line) is skipped
            # rather than crashing the monitor.
            continue

    return MetricsSample(
        gen_tokens_s=values.get(_METRIC_GEN_TPS),
        prompt_tokens_s=values.get(_METRIC_PROMPT_TPS),
        kv_cache_usage_ratio=values.get(_METRIC_KV_RATIO),
        kv_cache_tokens=_as_int(values.get(_METRIC_KV_TOKENS)),
        requests_processing=_as_int(values.get(_METRIC_REQ_PROCESSING)),
        prompt_tokens_total=_as_int(values.get(_METRIC_PROMPT_TOKENS_TOTAL)),
        gen_tokens_total=_as_int(values.get(_METRIC_GEN_TOKENS_TOTAL)),
        prompt_seconds_total=values.get(_METRIC_PROMPT_SECONDS_TOTAL),
        gen_seconds_total=values.get(_METRIC_GEN_SECONDS_TOTAL),
        n_tokens_max=_as_int(values.get(_METRIC_N_TOKENS_MAX)),
        metrics_available=True,
        sampled_at=time.time(),
    )


def parse_slots(data: Any) -> SlotsSample:
    """Parse llama.cpp /slots JSON (a list of slot objects) into a SlotsSample.

    /slots is optional enrichment (a build may disable it for privacy), so a
    non-list payload yields an unavailable sample rather than an error.
    """
    if not isinstance(data, list):
        return SlotsSample(slots=[], available=False)
    slots: list[dict[str, Any]] = []
    for entry in data:
        if isinstance(entry, dict):
            slots.append(entry)
    n_ctx = _first_int(slots, ("n_ctx",))
    # "tokens used" varies by build key: n_past is the prompt+generated tokens
    # resident in the slot's context; n_decoded/n_tokens are alternate names some
    # builds use. A stripped build (owner's) exposes none of these -> None.
    tokens_used = _first_int(slots, ("n_past", "n_decoded", "n_tokens"))
    return SlotsSample(slots=slots, available=True, n_ctx=n_ctx, tokens_used=tokens_used)


def _first_int(slots: list[dict[str, Any]], keys: tuple[str, ...]) -> int | None:
    """First integer value found under any of `keys` across the slot objects.

    Returns None when no slot carries any of the keys, so a stripped /slots
    payload degrades honestly rather than fabricating a number.
    """
    for slot in slots:
        for key in keys:
            value = slot.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
    return None


def _as_int(value: float | None) -> int | None:
    """Round a gauge float to int, preserving None (absent field)."""
    return None if value is None else int(round(value))


# --------------------------------------------------------------------------- #
# Button-state machine (Panel 2, G3). Pure function of a UiState so the enable/
# disable rules are unit-tested without a widget.
# --------------------------------------------------------------------------- #


def compute_button_states(ui: UiState) -> dict[str, Any]:
    """Return which controls are enabled/labelled for the given UiState.

    Rules (Architecture G3 Panel 2):
    - A command in flight disables every Start and Stop (prevents a second click
      racing the single ops worker).
    - Idle: every Start enabled, Stop disabled, Chat disabled.
    - Running model M: Start on M shows "Running" (disabled), Start on another
      model shows "Switch" (enabled), Stop enabled, Chat enabled.
    """
    running = ui.running_model_id
    if ui.in_flight:
        return {
            "start_enabled": False,
            "stop_enabled": False,
            "chat_enabled": False,
            "working": True,
            "running_model_id": running,
        }
    return {
        "start_enabled": True,
        "stop_enabled": running is not None,
        "chat_enabled": running is not None,
        "working": False,
        "running_model_id": running,
    }


def start_label_for(model_id: str, ui: UiState) -> str:
    """Label for a model row's Start button given the running state."""
    if ui.running_model_id == model_id:
        return "Running"
    if ui.running_model_id is not None:
        return "Switch"
    return "Start"


# --------------------------------------------------------------------------- #
# Save-button state machine (Panel 3, owner request 1). Pure function so the
# dirty/clean/invalid transitions are unit-tested without a widget. The grey-out
# of the button IS the save confirmation - there is no separate "saved" notice.
# --------------------------------------------------------------------------- #


def compute_save_state(
    gpu_text: str, ctx_text: str, saved_gpu: int, saved_ctx: int
) -> dict[str, Any]:
    """Return {'enabled', 'error'} for the Save button given editor vs saved values.

    Owner's exact contract:
    - clean (both editors parse and equal the saved registry values) -> Save
      DISABLED, no error. That greyed state is the confirmation the values are
      persisted; there is no red "saved; applies on next start" notice.
    - dirty + valid (either editor differs and both parse) -> Save ENABLED.
    - invalid input -> Save stays ENABLED and the validation error is surfaced
      (invalid text is by definition not equal to a valid saved value, so it is a
      dirty state the owner can see and correct). A click in this state is rejected
      by save_model_edits (validated again there) and never writes.
    """
    gpu_ok, gpu_val, gpu_err = validate_gpu_layers(gpu_text)
    ctx_ok, ctx_val, ctx_err = validate_context_size(ctx_text)
    if not gpu_ok or not ctx_ok:
        return {"enabled": True, "error": gpu_err or ctx_err}
    changed = gpu_val != saved_gpu or ctx_val != saved_ctx
    return {"enabled": changed, "error": None}


# --------------------------------------------------------------------------- #
# Identity (id/name) editors (Settings page, owner request: rename/re-id a model).
# Pure validation + a save-state machine mirroring compute_save_state above, so
# the id/name fields follow the identical dirty/clean/invalid contract the
# gpu_layers/context_size editors already use.
# --------------------------------------------------------------------------- #


def validate_model_id(text: str, existing_ids: list[str], current_id: str) -> tuple[bool, str | None, str | None]:
    """Accept a non-empty id not already used by ANOTHER model in the registry.

    Returns (True, value, None) on success or (False, None, remedy) on failure.
    Renaming a model's own id back to itself (current_id) is always valid; that
    is the "unchanged" case, not a collision.
    """
    value = (text or "").strip()
    if not value:
        return False, None, "id must not be empty"
    if value != current_id and value in existing_ids:
        return False, None, f"id '{value}' is already used by another model"
    return True, value, None


def validate_model_name(text: str) -> tuple[bool, str | None, str | None]:
    """Accept a non-empty display name; reject blank/whitespace-only text."""
    value = (text or "").strip()
    if not value:
        return False, None, "name must not be empty"
    return True, value, None


def parse_capabilities_text(text: str) -> list[str]:
    """Turn a comma-separated Capabilities field into a clean tag list.

    Owner request 2026-08-21: free-form (no fixed tag set - "coding", "voice
    cloning", whatever the owner wants to note), lower-cased for consistent
    display/sort, empty entries from stray/doubled commas dropped, order and
    duplicates as typed otherwise preserved. An empty or blank field is always
    valid - it means "just a plain chat model" (format_capabilities renders it
    "Chat"), never an error.
    """
    return [part.strip().lower() for part in (text or "").split(",") if part.strip()]


def find_id_change_blockers(
    model_id: str,
    draft_model_refs: list[tuple[str, str | None]],
    default_model: str | None,
) -> list[str]:
    """Return remedy strings for why model_id's id cannot change right now.

    An id is not just a label: a model's draft_model and settings.yaml's
    launcher.default_model both reference OTHER models by id (2026-08-16 owner
    incident: renaming qwen3-14b's id silently orphaned both, and Config.load()
    only reports the default_model break as a WARNING, easy to miss). Pure
    function of what the caller already loaded, so gui_controller adds no new
    file/registry access here; GuiController.save_model_identity feeds it from
    self._registry / self._settings. Returns [] when nothing blocks the change.
    """
    blockers: list[str] = []
    referencing = [
        other_id for other_id, draft in draft_model_refs if draft == model_id
    ]
    if referencing:
        blockers.append(
            f"id '{model_id}' is used as draft_model by "
            f"{', '.join(referencing)}; update that reference first"
        )
    if default_model == model_id:
        blockers.append(
            f"id '{model_id}' is settings.yaml's launcher.default_model; "
            f"update that first"
        )
    return blockers


def compute_identity_save_state(
    id_text: str,
    name_text: str,
    saved_id: str,
    saved_name: str,
    existing_ids: list[str],
) -> dict[str, Any]:
    """Return {'enabled', 'error'} for the identity Save button (same contract as
    compute_save_state): clean -> disabled/no error, dirty+valid -> enabled,
    invalid -> enabled with the validation error surfaced, never written.
    """
    id_ok, id_val, id_err = validate_model_id(id_text, existing_ids, saved_id)
    name_ok, name_val, name_err = validate_model_name(name_text)
    if not id_ok or not name_ok:
        return {"enabled": True, "error": id_err or name_err}
    changed = id_val != saved_id or name_val != saved_name
    return {"enabled": changed, "error": None}


# --------------------------------------------------------------------------- #
# Size column formatting (Panel 1, owner request 2). Decimal GB (1e9), matching
# how the disk vendor and model card report a .gguf's size.
# --------------------------------------------------------------------------- #


def format_size_gb(num_bytes: int | None) -> str:
    """Render a .gguf byte size as decimal GB with one decimal, or '-' when absent.

    None (location unset or file missing/unstattable) renders '-' so the column is
    honest about an unknown size rather than showing 0. A 10,263,894,400-byte model
    renders "10.3 GB" (owner's reference value).
    """
    if num_bytes is None:
        return "-"
    return f"{num_bytes / _BYTES_PER_GB:.1f} GB"


def format_mtime(mtime: float | None) -> str:
    """Render a file mtime as a local "YYYY-MM-DD HH:MM" string, or '-' when absent.

    Used by the Fine-tune page's Modified column, which is one of the two ways the
    owner tells same-named fine-tune builds apart (size is the other).
    """
    if not mtime:
        return "-"
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
    except (OSError, ValueError, OverflowError):
        return "-"


def total_size_display(rows: list[dict[str, Any]]) -> str:
    """Sum every row's size_bytes as decimal GB (Panel 1 footer, owner request).

    Rows whose size_bytes is None (location unset or unstattable) are skipped in
    the sum rather than counted as 0, and the count of skipped rows is folded into
    the label so the total is honest about what it does and does not cover.
    """
    known = [row["size_bytes"] for row in rows if row.get("size_bytes") is not None]
    missing = len(rows) - len(known)
    total = format_size_gb(sum(known)) if known else "-"
    if missing:
        return f"Total: {total} ({missing} unknown)"
    return f"Total: {total}"


# The fit basis moved to config.py so health.py's VramProbe and the GUI's fit
# columns share one definition (see compute_vram_need_mb's docstring). Imported
# here rather than re-defined, so gui_controller.compute_vram_need_mb stays the
# same callable every existing caller and test already uses.
from config import compute_vram_need_mb  # noqa: E402  (re-export, single source)


def format_gpu_portion_display(vram_need_mb: float, gpu_total_mb: float | None) -> str:
    """Render how much of a model's true VRAM need (compute_vram_need_mb) the
    detected GPU can actually hold (owner fix, 2026-08-16: a single combined
    "VRAM (fits)" cell next to an almost-identical Size column was confusing --
    "show the official model size, how much of it fits on the GPU, and if not
    all, what goes to the CPU" as three separate, literal answers instead).

    vram_need_mb / gpu_total_mb are in the same raw-number convention
    compute_vram_need_mb and VramProbe (health.py) already use without
    reconciling MiB vs decimal-MB (see compute_vram_need_mb's docstring).

    - vram_need_mb falsy (0, no estimate and no file on disk) -> "-".
    - No GPU detected -> "-": "how much fits on the GPU" is unanswerable
      without a detected card, so this never guesses.
    - Otherwise, the smaller of vram_need_mb and gpu_total_mb: the model's own
      need if it fully fits, or the GPU's whole capacity if it does not (the
      GPU can only ever hold up to what it has, never more).
    """
    if not vram_need_mb or gpu_total_mb is None:
        return "-"
    on_gpu_mb = min(vram_need_mb, gpu_total_mb)
    return f"{on_gpu_mb / 1000:.1f} GB"


def format_cpu_portion_display(vram_need_mb: float, gpu_total_mb: float | None) -> str:
    """Render how much of a model's true VRAM need (compute_vram_need_mb) would
    spill to system RAM / CPU offload, the sibling of format_gpu_portion_display.

    - vram_need_mb falsy, or no GPU detected -> "-" (same reasoning as the GPU
      column: nothing to compute without both numbers).
    - Fits entirely (need <= capacity) -> "-", read as "nothing spills."
    - Otherwise exceeds capacity by ANY amount, however small (owner fix,
      2026-08-16, third pass: a 16.3379GB model on his reported 16303 "16GB"
      card overflows by only ~35MB, which the previous "round to 1 decimal,
      then suppress anything under 0.05 GB" rule silently swallowed as "-" --
      "I have just 16GB GPU RAM" [it should have shown a spill]). Rounds the
      spill UP to the nearest 0.1 GB (never down), so a genuine overflow is
      never truncated back to a misleading zero the way plain 1-decimal
      rounding would.
    """
    if not vram_need_mb or gpu_total_mb is None:
        return "-"
    spill_mb = vram_need_mb - gpu_total_mb
    if spill_mb <= 0:
        return "-"
    spill_gb = math.ceil(spill_mb / 100) / 10
    return f"{spill_gb:.1f} GB"


@dataclass
class SystemSpecs:
    """Static host RAM + GPU facts (owner request: show the machine's specs).

    None fields mean "not detected" -- no provider injected, no NVIDIA driver,
    or nvidia-smi errored -- rendered honestly by format_system_specs rather
    than a fabricated 0 (G5). Hardware does not change mid-session, so
    GuiController.system_specs() computes this once and caches it.
    """

    ram_total_mb: float | None = None
    gpu_name: str | None = None
    gpu_vram_total_mb: float | None = None


def format_system_specs(specs: SystemSpecs) -> str:
    """Render SystemSpecs as one line for the always-visible header."""
    ram = f"{specs.ram_total_mb / 1024:.1f} GB RAM" if specs.ram_total_mb else "RAM: -"
    if specs.gpu_name and specs.gpu_vram_total_mb:
        gpu = f"{specs.gpu_name} ({specs.gpu_vram_total_mb / 1024:.1f} GB VRAM)"
    else:
        gpu = "GPU: not detected"
    return f"{ram}  |  {gpu}"


def format_score(score: float | None) -> str:
    """Render measured generation throughput for the model-table benchmark cell.

    None (no matching successful run) renders '-' so the column is honest about an
    unmeasured model rather than showing 0. The unit is always explicit because a
    quality percentage also exists in the detailed benchmark report.
    """
    if score is None:
        return "-"
    return f"{score:.1f} tok/s"


def format_capabilities(capabilities: list[str]) -> str:
    """Render a Model.capabilities list for the Capabilities column (owner request
    2026-08-21).

    Every tag is asserted explicitly in the row (config.py never infers one), so
    this is pure display formatting - title-case, comma-joined. An empty list is
    every plain chat model, which is the common case; it renders "Chat" rather
    than a blank cell so the column always says something useful.
    """
    tags = [str(c).strip() for c in (capabilities or []) if str(c).strip()]
    if not tags:
        return "Chat"
    return ", ".join(tag.title() for tag in tags)


# --------------------------------------------------------------------------- #
# Live-monitor line (Panel 4, owner request 4). Counts + context fill, not just
# rates. Pure function of a MetricsSample so the format is unit-tested with canned
# payloads. Every piece renders "-" (or "idle") when its source is absent (G5).
# --------------------------------------------------------------------------- #


def format_monitor_line(sample: MetricsSample, context_size: int | None = None) -> str:
    """Build the owner's live line: rates + token counts + context fill.

    Target format:
        gen 69.8 tok/s | prompt 22.6 tok/s | prompt 412 tok | gen 583 tok | ctx 995/16384 (6%)

    Rate wording is decided from requests_processing, NOT from the rate gauge
    reading 0 (AC13 / Architecture M5.12). The llama.cpp gauges reflect the LAST
    request and read 0 between requests, so keying "idle" off rate==0 would
    mislabel a genuinely active-but-slow generation (rate momentarily 0 while a
    request is in flight) as idle. Instead: a non-zero rate always shows the
    number; when the rate is 0/absent we look at requests_processing -- if a
    request is in flight (>0) we show "..." (working, not idle), otherwise "idle".
    Counts come from the cumulative /metrics counters; an absent counter renders
    "-". The ctx segment prefers a real token count (kv_cache_tokens, then a
    /slots token count, then n_tokens_max) over a total (n_ctx from /slots, else
    the model's context_size); when no used-count source exists it falls back to
    the kv ratio percentage, else "-".
    """
    gen_rate = _rate_text(sample.gen_tokens_s, sample.requests_processing)
    prompt_rate = _rate_text(sample.prompt_tokens_s, sample.requests_processing)
    prompt_n = "-" if sample.prompt_tokens_total is None else str(sample.prompt_tokens_total)
    gen_n = "-" if sample.gen_tokens_total is None else str(sample.gen_tokens_total)
    ctx = _format_ctx(sample, context_size)
    return (
        f"gen {gen_rate} tok/s | prompt {prompt_rate} tok/s | "
        f"prompt {prompt_n} tok | gen {gen_n} tok | ctx {ctx}"
    )


def _rate_text(rate: float | None, requests_processing: int | None) -> str:
    """Render one throughput gauge: a number, "..." (working), or "idle" (AC13).

    Idle is decided from requests_processing, not from rate == 0: the llama.cpp
    rate gauges read 0 between requests, so only a NON-processing state is truly
    idle. A request in flight with no rate yet reads "..." so an active-but-slow
    generation is never mislabeled idle (Architecture M5.12).
    """
    if rate:  # a real, non-zero measured rate always wins
        return f"{rate:.1f}"
    if requests_processing and requests_processing > 0:
        return "..."  # active request, rate not yet reported; not idle
    return "idle"


def _format_ctx(sample: MetricsSample, context_size: int | None) -> str:
    """Render 'used/total (pct%)', degrading honestly when a source is absent."""
    total = sample.n_ctx or context_size
    # Used-count sources in preference order: a direct KV token count, then a ratio
    # applied to the total, then the peak-tokens gauge this build actually exposes.
    used = sample.kv_cache_tokens
    if used is None and sample.kv_cache_usage_ratio is not None and total:
        used = round(sample.kv_cache_usage_ratio * total)
    if used is None:
        used = sample.n_tokens_max
    if used is None or not total:
        # No usable count/total pair: fall back to a bare percentage if the ratio
        # is present, otherwise be honest that context fill is unknown.
        if sample.kv_cache_usage_ratio is not None:
            return f"{sample.kv_cache_usage_ratio * 100:.0f}%"
        return "-"
    pct = round(used / total * 100)
    return f"{used}/{total} ({pct}%)"


# --------------------------------------------------------------------------- #
# Model-table sorting (Panel 1, owner request 3). Pure function so the ordering
# (incl. numeric size with missing-last, and stable ties) is unit-tested.
# --------------------------------------------------------------------------- #


def sort_model_rows(
    rows: list[dict[str, Any]], column: str, descending: bool
) -> list[dict[str, Any]]:
    """Return a new list of row dicts sorted by `column`.

    - "size" sorts numerically on size_bytes; rows with no size (None) always sort
      LAST regardless of direction, so a missing file never masquerades as smallest
      or largest.
    - "vram" sorts numerically on vram_need_mb; rows with no need (0/None) sort
      LAST regardless of direction, same missing-last rule as "size".
    - "benchmark_tok_s" sorts numerically with unmeasured rows last.
    - "name"/"id"/"status" sort case-insensitively on their string value.
    Python's sort is stable, so rows equal on the key keep their prior order (this
    is what lets the caller preserve the selected row across a resort).
    """
    if column == "size":
        present = [r for r in rows if r.get("size_bytes") is not None]
        missing = [r for r in rows if r.get("size_bytes") is None]
        present.sort(key=lambda r: r["size_bytes"], reverse=descending)
        return present + missing
    if column == "vram":
        present = [r for r in rows if r.get("vram_need_mb")]
        missing = [r for r in rows if not r.get("vram_need_mb")]
        present.sort(key=lambda r: r["vram_need_mb"], reverse=descending)
        return present + missing
    if column == "benchmark_tok_s":
        present = [r for r in rows if r.get("benchmark_tok_s") is not None]
        missing = [r for r in rows if r.get("benchmark_tok_s") is None]
        present.sort(key=lambda r: r["benchmark_tok_s"], reverse=descending)
        return present + missing
    if column == "context":
        present = [r for r in rows if r.get("context_size")]
        missing = [r for r in rows if not r.get("context_size")]
        present.sort(key=lambda r: r["context_size"], reverse=descending)
        return present + missing
    return sorted(rows, key=lambda r: str(r.get(column, "")).lower(), reverse=descending)


# Remedy strings for a non-RUNNING lifecycle outcome, mirroring the honest
# messages the terminal path already prints (G2 error marshalling).
_STATUS_REMEDY = {
    "STOPPED": "did not start (STOPPED); check the model location and logs",
    "STOPPED_ERROR": "did not start cleanly (STOPPED_ERROR); check VRAM/port, see logs",
    "UNHEALTHY": "started but did not become healthy; check logs",
    "STARTING": "still starting; try again in a moment",
}


# --------------------------------------------------------------------------- #
# M10 Talk panel seams (Architecture M10.3). Pure orchestration over injected
# collaborators, imported by the launcher; no tkinter here (G7 headless-testable).
# --------------------------------------------------------------------------- #


# Unique gate tokens. Identity comparison (is) keeps them unambiguous on a plain
# queue.Queue of arbitrary objects.
_TALK_TOKEN = object()
_STOP_SENTINEL = object()


@dataclass
class AssistantEvents:
    """The three marshalling callbacks the assistant session pushes to the pump (M10.2).

    Each callback is wired by GuiController to enqueue a Result on result_q, so every
    assistant turn event reaches the UI thread through the one pump (the sole widget
    writer). The launcher's session builder calls these; it never touches a widget.
    """

    on_state: Callable[[str], None]  # idle/listening/thinking/speaking
    on_user: Callable[[str], None]  # a completed user utterance ("You:")
    on_reply: Callable[[str], None]  # a completed LOCITIZE reply line ("LOCITIZE: ...")


class GuiSttSource:
    """Talk-gate STT source: PushToTalkSttSource driven by GUI clicks (M10.3).

    A thin decorator over the UNCHANGED assistant.PushToTalkSttSource. It does NOT
    reimplement capture; it injects a talk-gate read_line (a queue.Queue the Talk
    button feeds) and adds the state/user-turn marshalling the loop does not emit:

      - read_line() BLOCKS on the gate. A Talk token -> fire on_state("listening")
        and return "" (which PushToTalkSttSource treats as "open ONE capture
        window"). The STOP sentinel -> return None (which ends the loop cleanly).
      - next_utterance() wraps PushToTalkSttSource.next_utterance(): it fires
        on_state("idle") before blocking, and on a captured (non-None) utterance
        fires on_state("thinking") + on_user(utterance) so the "You:" line appears
        before the reply is generated. A no-speech window reloops INSIDE
        PushToTalkSttSource (its no-speech notice routes to the status line) and
        never yields a fabricated turn.

    The idle-floor (fired at the top of every next_utterance) guarantees the button
    always returns to "Talk" even if an on_speaking end signal is ever missed (RM2).

    Pure orchestration over injected seams (a fake mic + a REAL PushToTalkSttSource
    built here), so it is unit-tested headless with no mic/LLM/GPU/Tk.
    """

    def __init__(
        self,
        mic: Any,
        gate: "queue.Queue[Any]",
        on_state: Callable[[str], None] | None = None,
        on_user: Callable[[str], None] | None = None,
        *,
        max_capture_s: float = 15.0,
        settle_s: float = 0.7,
        poll_s: float = 0.3,
        clock: Callable[[], float] | None = None,
        emit: Callable[[str], None] | None = None,
    ) -> None:
        from assistant import PushToTalkSttSource

        self._gate = gate
        self._on_state = on_state or (lambda _s: None)
        self._on_user = on_user or (lambda _u: None)
        # Build the REAL push-to-talk source with our talk-gate read_line; capture is
        # its unchanged flush + settle + max-capture pipeline (no capture code here).
        self._ptt = PushToTalkSttSource(
            mic,
            read_line=self._read_line,
            emit=emit,
            max_capture_s=max_capture_s,
            settle_s=settle_s,
            poll_s=poll_s,
            clock=clock,
        )

    def _read_line(self) -> str | None:
        """Block on the talk gate; a Talk click opens a window, STOP ends the loop."""
        token = self._gate.get()
        if token is _STOP_SENTINEL:
            return None  # end the session
        # Any other token is a Talk click: open ONE capture window.
        self._on_state("listening")
        return ""

    def next_utterance(self) -> str | None:
        """Idle until a Talk click, capture one utterance, mark thinking + emit "You:"."""
        self._on_state("idle")  # idle-floor: button always returns to "Talk"
        utterance = self._ptt.next_utterance()
        if utterance is None:
            return None
        self._on_state("thinking")
        self._on_user(utterance)
        return utterance

    def talk(self) -> None:
        """Put a Talk token on the gate (opens one capture window). Non-blocking."""
        self._gate.put(_TALK_TOKEN)

    def stop(self) -> None:
        """End the session: STOP sentinel unblocks read_line, and stop the PTT source."""
        self._gate.put(_STOP_SENTINEL)
        self._ptt.stop()


# --------------------------------------------------------------------------- #
# The controller: owns the queues and the two background threads (G2).
# --------------------------------------------------------------------------- #


# Sentinel for "this worker has not yet observed the controller's running
# model". Distinct from (None, None), which honestly means "nothing running".
_UNOBSERVED = object()


class GuiController:
    """Marshals GUI intents onto the existing controllers via two worker threads.

    Constructed with the SAME ServiceManager instance the terminal path uses, so
    the platform's atexit.register(stop_all) no-orphan guarantee (Reviewer L-1)
    already covers the GUI process; shutdown() additionally joins the threads and
    calls stop_all() itself for a clean window close.

    Injection points keep the whole thing headless-testable (G7):
    - `fetch` replaces the loopback /metrics|/slots GET (tests feed canned bodies).
    - `listen_fn(seconds, emit)` runs the existing --listen capture and calls
      emit(line) per deduplicated, VAD-gated transcript segment; the launcher
      supplies the real one (which reuses the M3 whisper-stream pipeline unchanged)
      and tests supply a fake, so gui_controller adds no new capture code.
    - `proxy_factory` builds the optional reverse proxy (tests pass a fake).
    """

    def __init__(
        self,
        settings: Any,
        model_registry: Any,
        model_controller: Any,
        whisper_controller: Any,
        service_manager: Any,
        *,
        fetch: Callable[[int, str], str | None] | None = None,
        listen_fn: Callable[[float, Callable[[str], None]], bool] | None = None,
        speak_fn: Callable[[str, str], tuple[bool, str]] | None = None,
        proxy_factory: Callable[..., Any] | None = None,
        openwebui_start_fn: Callable[[], bool] | None = None,
        assistant_start_fn: Callable[[AssistantEvents, str, bool], Any] | None = None,
        describe_fn: Callable[[str, str], tuple[bool, str]] | None = None,
        memory_search_fn: Callable[[str], list[dict[str, Any]]] | None = None,
        benchmark_fn: Callable[[str], tuple[bool, str, float | None]] | None = None,
        benchmark_history_fn: Callable[
            [list[dict[str, Any]]], dict[str, float]
        ]
        | None = None,
        gpu_provider: Any = None,
        sys_provider: Any = None,
        finetune_controller_factory: Callable[[], Any] | None = None,
        hub_downloader_factory: Callable[[], Any] | None = None,
        apply_noise_suppression_fn: Callable[[str], str] | None = None,
        second_eye_fn: Callable[..., int] | None = None,
    ) -> None:
        self._settings = settings
        self._registry = model_registry
        self._controller = model_controller
        self._whisper = whisper_controller
        self._manager = service_manager
        self._fetch = fetch or self._loopback_fetch
        self._listen_fn = listen_fn
        # M9-lite: starts the real Open WebUI service on the shared ServiceManager
        # (so the no-orphan backstop covers it) and returns True when it went ready.
        # The launcher supplies the real one; tests supply a fake or None. Keeping it
        # injected means gui_controller drives the service without gui.py touching
        # any subprocess/HTTP code (presentation discipline, Reviewer G1).
        self._openwebui_start_fn = openwebui_start_fn
        # speak_fn(text, voice) -> (ok, detail) runs the real Kokoro synthesize+play
        # off the UI thread. The launcher supplies the real one (reusing the M6 TTS
        # path on the SHARED ServiceManager, so no orphan); tests supply a fake, so
        # gui_controller adds no HTTP/subprocess/audio code (presentation discipline).
        self._speak_fn = speak_fn
        self._proxy_factory = proxy_factory
        # M10 single-front-door builders (all injected so gui_controller stays free of
        # subprocess/HTTP/model code and headless-testable). assistant_start_fn builds
        # the M7 loop on the SHARED manager and returns an AssistantHandle; the other
        # three reuse the existing --describe / ConversationMemory / --benchmark paths.
        self._assistant_start_fn = assistant_start_fn
        self._describe_fn = describe_fn
        self._memory_search_fn = memory_search_fn
        self._benchmark_fn = benchmark_fn
        # Loads measured generation throughput from the append-only benchmark
        # record. Injected by launcher so this headless seam performs no report I/O
        # in tests and never confuses benchmark_score (quality %) with tok/s.
        self._benchmark_history_fn = benchmark_history_fn
        # health.py's GpuInfoProvider/SystemInfoProvider (owner request: show host
        # RAM/GPU specs and per-model VRAM+spillage). Injected, like every other
        # collaborator here, so gui_controller stays free of subprocess/psutil code
        # itself (G1/G7) -- the launcher supplies the real NvidiaSmiGpuInfoProvider/
        # DefaultSystemInfoProvider, tests supply a fake or leave this None. None
        # degrades to "not detected" (system_specs()), never a fabricated value.
        self._gpu_provider = gpu_provider
        self._sys_provider = sys_provider
        self._system_specs_cache: SystemSpecs | None = None
        # Last Talk-loop state pushed by AssistantEvents.on_state. request_talk
        # barges in only while thinking/speaking; interrupting while idle would
        # leave the Event set and abort the next turn.
        self._assistant_ui_state = "idle"
        self._apply_noise_suppression_fn = apply_noise_suppression_fn
        self._second_eye_fn = second_eye_fn
        self._second_eye_stop = threading.Event()
        self._second_eye_thread: threading.Thread | None = None
        self._second_eye_stop_file: Path | None = None

        self._assistant_handle: Any = None

        self.command_q: "queue.Queue[Command]" = queue.Queue()
        self.result_q: "queue.Queue[Result]" = queue.Queue()

        # Shared monitor state. int/reference assignment is atomic in CPython, so
        # these are read by the monitor and written by the ops worker without a
        # lock; only lifecycle mutations (the controllers) need serialization and
        # those all run on the single ops worker.
        self._active_port: int | None = None
        # Last (model_id, port) this worker published. Starts at _UNOBSERVED, a
        # sentinel no real state equals, so the FIRST monitor cycle always
        # paints the truth - a window opened while a model already runs must
        # not sit blank waiting for a change. (None, None) is a legitimate
        # state ("nothing is running") and therefore cannot be the sentinel.
        self._last_running_observed: Any = _UNOBSERVED
        self._metrics_available = True
        # M18.10: the previous poll's cumulative counters, for delta-derived
        # tok/s on builds without the direct rate gauges. Reset on model start.
        self._prev_metrics_sample: MetricsSample | None = None
        # Per-session cache of each model's .gguf size in bytes (owner request 2).
        # Populated by a single os.stat at list_models time (no polling loop); a
        # None entry records "location unset or file unstattable" so we never
        # re-stat a missing file every refresh.
        self._size_cache: dict[str, int | None] = {}

        self._stop_event = threading.Event()
        self._ops_thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None

        self._proxy: Any = None
        # Two flags, deliberately distinct (see shutdown()):
        #   _shutdown_started - re-entrancy latch, set on entry so a second close
        #                       (or the atexit backstop) never runs teardown twice.
        #   _shutdown_done    - set only AFTER every teardown step has been
        #                       attempted end to end, so it is an honest answer to
        #                       "did teardown actually run?" rather than merely
        #                       "did someone call shutdown?".
        self._shutdown_started = False
        self._shutdown_done = False
        # Failures raised by individual teardown steps, recorded rather than
        # propagated (desktop.py's closeEvent cannot handle an exception).
        self.shutdown_errors: list[str] = []

        # M13 fine-tune studio state. The SingleServiceController is built lazily
        # on the SHARED ServiceManager (so stop_all() reaps the studio on window
        # close exactly like every other managed child) the first time the owner
        # starts it; tests inject a fake factory instead. _finetune_status is the
        # last state this controller published, so a page repaint never has to
        # re-probe, and _finetune_started records that LOCITIZE launched the studio
        # at least once this session (which gates the honest orphan warning).
        self._finetune_controller_factory = finetune_controller_factory
        self._finetune_controller: Any = None
        self._finetune_status = "stopped"
        self._finetune_started = False
        # D-M13-2: a SEPARATE, sticky latch - true once a studio child was
        # actually launched this session, whether that launch ended Running or
        # in an error. It is never cleared by Stop, because the log file it
        # points at outlives the process and is exactly what the owner needs
        # after a stop or a failed start. _finetune_started must stay
        # live-only (the orphan warning depends on it), so this cannot reuse it.
        self._finetune_launched_once = False
        # Last answer of the train.log freshness probe, refreshed only on a
        # background thread (the ops worker and the monitor loop). The probe walks
        # the outputs tree, which can live on a removable or network drive
        # (failure mode F7), so the GUI thread reads this cached bool and never
        # stats anything itself - see finetune_run_active().
        self._finetune_run_active_cache = False

        # M14.14.3 Thread E state. `_hub_downloader_factory` builds the
        # modelhub.Downloader lazily (tests inject a fake); building one makes no
        # network call, but building it lazily keeps import-time work at zero,
        # which is part of how egress rule HF-1 stays true.
        #
        # `_hub_thread` is the AT MOST ONE download worker. It exists because the
        # ops worker is a single serialized consumer: a 9 GB transfer inside
        # _dispatch would block start/stop/benchmark/fine-tune for an hour - it
        # would freeze the product, not just the page. Thread E does the blocking
        # work and publishes onto the same result_q the pump already drains, so
        # this adds a fourth background PRODUCER and no new consumer, no QThread,
        # no signal type, and no event loop.
        self._hub_downloader_factory = hub_downloader_factory
        self._hub_downloader: Any = None
        self._hub_thread: threading.Thread | None = None
        self._hub_cancel = threading.Event()
        self._hub_job_id: str | None = None
        # Latched by shutdown() before it cancels and joins Thread E. Without it,
        # a hub_download still sitting in command_q would be dispatched by the
        # ops worker AFTER the join, start a brand-new Thread E with a fresh
        # (unset) cancel event, and outlive shutdown - writing to a .partial
        # nothing will ever clean up.
        self._hub_shutting_down = False
        # Guards the (_hub_shutting_down, _hub_cancel, _hub_thread) trio, which is
        # written by the ops worker (Thread B) and read by the GUI thread inside
        # shutdown(). Without it those two threads can interleave BETWEEN the
        # construction of Thread E and its start(), and a join() against a thread
        # that was never started raises RuntimeError - which used to abort the
        # rest of teardown and orphan the model + whisper children (HIGH-4).
        self._hub_lock = threading.Lock()
        # ---- auto-tune cancellation (owner request 2026-08-22) -------------- #
        # An auto-tune occupies the single ops worker for MINUTES (several real
        # llama-server loads). Every other cancel in this controller travels as a
        # Command on command_q - which is exactly what cannot work here, because
        # the ops worker is the thing that is busy: a queued cancel would not be
        # read until the run it was meant to interrupt had already finished.
        # So this one is an Event the Qt thread sets DIRECTLY, bypassing the
        # queue, and the ops worker polls inside its trial wait loop. An Event is
        # safe to set from another thread by construction; nothing else about the
        # threading model changes.
        self._autotune_cancel = threading.Event()
        # Read by the UI to decide whether Stop should be offered and what it
        # means. Written only by the ops worker, and only bool-assigned, so a
        # racing read sees one state or the other and never a torn value.
        self._autotune_running = False
        self._autotune_model_id: str | None = None

    # ---- read accessors for the presentation layer ----------------------- #

    def hub_enabled(self) -> bool:
        """True when the Get models section may operate (models_hub.enabled).

        A pure settings read on the GUI thread: it opens no file and makes no
        network call, so it is safe to call while building the page.
        """
        return bool(getattr(getattr(self._settings, "models_hub", None), "enabled", False))

    def refresh_models(self) -> list[dict[str, Any]]:
        """Re-read models.yaml, swap the registry contents in place, and return
        fresh rows (Desktop 'Refresh' button, owner request 2026-08-13).

        Runs the same Config.load path as startup so validation and overrides
        apply identically. On ERROR-level parse issues the registry is left
        untouched and the error is raised for the caller's status chip, so a
        bad edit never yields a half-parsed model list.
        """
        from config import Config

        # Owner request 2026-08-21: Config.load() with no argument re-resolves
        # the PRODUCTION data root from the environment/BASE_DIR from scratch,
        # ignoring wherever self._settings was actually built from - invisible
        # in real usage (they're the same path there) but wrong in principle,
        # and it silently reloaded the wrong registry entirely in a test that
        # pointed self._settings at a tmp_path fixture. Passing the SAME
        # data_dir this controller's settings came from keeps this read-your-
        # own-write instead of read-whatever-the-environment-says-right-now.
        _settings, models, issues = Config.load(self._settings.data_dir)
        errors = [i for i in issues if getattr(i, "severity", "") == "ERROR"]
        if errors:
            raise RuntimeError("; ".join(str(i) for i in errors[:3]))
        self._registry.reload(models)
        return self.list_models()

    def system_specs(self) -> SystemSpecs:
        """Static host RAM + GPU facts (owner request), computed once and cached.

        Hardware does not change during a session, so a Refresh click or a later
        list_models() call never re-shells to nvidia-smi or re-queries psutil.
        None fields mean "not detected"; the injected providers already degrade
        to None on a probe error (health.py's contract), so this trusts that
        rather than adding a second layer of defensive try/except.
        """
        if self._system_specs_cache is not None:
            return self._system_specs_cache
        ram_mb = self._sys_provider.ram_total_mb() if self._sys_provider else None
        gpu_name: str | None = None
        gpu_vram_mb: float | None = None
        if self._gpu_provider is not None:
            gpus = self._gpu_provider.gpus()
            if gpus:
                target = max(gpus, key=lambda g: g.vram_total_mb)
                gpu_name = target.name
                gpu_vram_mb = target.vram_total_mb
        self._system_specs_cache = SystemSpecs(ram_mb, gpu_name, gpu_vram_mb)
        return self._system_specs_cache

    def list_models(self) -> list[dict[str, Any]]:
        """Model rows for Panel 1/3, so gui.py never imports config/models itself.

        Returns id, name, location, status, and the persisted gpu_layers/
        context_size (used to prefill the editors and compute the unsaved marker).
        launchable is false for a future/empty-location model so the row disables
        its Start with an honest reason.
        """
        gpu_total_mb = self.system_specs().gpu_vram_total_mb
        rows: list[dict[str, Any]] = []
        import context_advice
        ceilings = context_advice.load_ceilings(self._settings.data_dir)
        for model in self._registry.all():
            size_bytes = self._model_size_bytes(model.id, model.location)
            vram_need_mb = compute_vram_need_mb(model.vram_estimate_mb, size_bytes)
            rows.append(
                {
                    "id": model.id,
                    "name": model.name,
                    "location": model.location,
                    "status": model.status,
                    "gpu_layers": model.gpu_layers,
                    "context_size": model.context_size,
                    # Human-grouped for the Context column (e.g. 131,072).
                    "context_display": f"{model.context_size:,}" if model.context_size else "-",
                    # M17.3: non-empty when the configured context is past the
                    # measured cliff, so the table can flag a slow config.
                    "ctx_warning": context_advice.context_warning(
                        model.id, model.context_size, ceilings
                    ),
                    "launchable": bool(model.location) and model.status == "installed",
                    # Size column (owner request 2): raw bytes for numeric sorting,
                    # plus the pre-formatted decimal-GB string the table renders.
                    "size_bytes": size_bytes,
                    "size_display": format_size_gb(size_bytes),
                    # On GPU / On CPU columns (owner request 2026-08-16: the official
                    # model size, how much of it fits on the GPU, and if not all, what
                    # goes to the CPU -- three separate answers, not one combined cell).
                    # vram_need_mb (the larger of the owner's tuned estimate and the
                    # on-disk file size, see compute_vram_need_mb) is the true full
                    # footprint both are computed from, and is also the numeric sort key.
                    "vram_estimate_mb": model.vram_estimate_mb,
                    "vram_need_mb": vram_need_mb,
                    "gpu_portion_display": format_gpu_portion_display(vram_need_mb, gpu_total_mb),
                    "cpu_portion_display": format_cpu_portion_display(vram_need_mb, gpu_total_mb),
                    # benchmark_score remains the deterministic quality percentage
                    # persisted by M5. The visible benchmark cell is populated below
                    # from the measured generation-throughput history instead.
                    "benchmark_score": model.benchmark_score,
                    "benchmark_tok_s": None,
                    "score_display": format_score(None),
                    # Capabilities column (owner request 2026-08-21): what the
                    # model is actually good for, so the owner can tell at a
                    # glance which one to switch to.
                    "capabilities": list(model.capabilities),
                    "capabilities_display": format_capabilities(model.capabilities),
                }
            )
        self._attach_benchmark_throughput(rows)
        return rows

    def _attach_benchmark_throughput(self, rows: list[dict[str, Any]]) -> None:
        """Attach honest generation tok/s values supplied by benchmark history."""
        speeds: dict[str, float] = {}
        if self._benchmark_history_fn is not None:
            try:
                speeds = dict(self._benchmark_history_fn(rows) or {})
            except Exception:  # noqa: BLE001 - history display must not break inventory
                speeds = {}
        for row in rows:
            speed = speeds.get(str(row.get("id", "")))
            row["benchmark_tok_s"] = speed
            row["score_display"] = format_score(speed)

    def _model_size_bytes(self, model_id: str, location: str) -> int | None:
        """Stat a model's .gguf once per session, caching the result (or None).

        The stat happens here (in the controller, never in the presentation shell)
        so gui.py stays free of filesystem I/O. A missing/unset location or any
        OSError yields None -> the Size column shows "-" honestly. The value is
        cached so re-sorts and refreshes never re-hit the disk.
        """
        if model_id in self._size_cache:
            return self._size_cache[model_id]
        size: int | None = None
        if location:
            try:
                size = os.stat(location).st_size
            except OSError:
                size = None
        self._size_cache[model_id] = size
        return size

    # ---- lifecycle of the background threads ------------------------------ #

    def start_threads(self) -> None:
        """Start the ops worker and monitor daemon threads (called before mainloop)."""
        self._ops_thread = threading.Thread(
            target=self._ops_loop, name="locitize-gui-ops", daemon=True
        )
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="locitize-gui-monitor", daemon=True
        )
        self._ops_thread.start()
        self._monitor_thread.start()

    # ---- intent methods (called on the UI thread; they only enqueue) ------ #

    def request_start(
        self, model_id: str, reasoning: dict | None = None
    ) -> None:
        """Enqueue a start/switch for model_id (the worker decides which).

        Owner request 2026-09-02: `reasoning` is an optional one-launch thinking
        override (already validated by config.parse_reasoning_choice at the UI
        edge), matching the terminal menu's "3 low" suffix. None means "use the
        row's own reasoning:" - it is dropped from the payload rather than
        forwarded, because build_start_spec reads an explicit None as the
        deliberate "force thinking off for this scenario".
        """
        payload: dict = {"model_id": model_id}
        if reasoning:
            payload["reasoning"] = reasoning
        self.command_q.put(Command("start", payload))

    def request_stop(self) -> None:
        self.command_q.put(Command("stop"))

    def request_free_gpu(self) -> None:
        """Offload GPU (M17.8): stop the running model AND sweep LOCITIZE's own
        orphaned servers (a crashed session, a measurement probe) that the plain
        stop cannot reach. Foreign apps are reported but never touched."""
        self.command_q.put(Command("free_gpu"))

    def request_whisper_toggle(self) -> None:
        self.command_q.put(Command("whisper_toggle"))

    def request_listen(self, seconds: float) -> None:
        self.command_q.put(Command("listen", {"seconds": seconds}))

    def request_speak(self, text: str, voice: str) -> None:
        """Enqueue a Speak-test intent for the ops worker (never runs on UI thread)."""
        self.command_q.put(Command("speak", {"text": text, "voice": voice}))

    def request_audition(self) -> None:
        """Enqueue an Audition-all intent: speak the sample sentence in every voice."""
        self.command_q.put(Command("audition"))

    # ---- M10 Talk panel + single-front-door intents ---------------------- #

    def request_start_assistant(self, voice: str = "", speak: bool = True) -> None:
        """Enqueue starting the assistant session (blocks the ops worker, not the UI).

        Service startup (model + whisper + Kokoro cold start) can take up to the
        per-service ready_timeout_s, so it MUST run on the ops worker (Thread B) with
        the panel showing an honest "starting services..." status (M10.2). The chosen
        Panel 6 voice and the speak toggle are carried in the payload.
        """
        self.command_q.put(
            Command("start_assistant", {"voice": voice, "speak": speak})
        )

    def request_talk(self) -> None:
        """Open ONE capture window (a Talk click). Non-blocking (Thread A -> talk gate).

        If the assistant is thinking or speaking, barge in first: stop generation
        and audio, then queue the capture window so the next turn listens at once.
        Do not interrupt while idle -- that would leave the loop Event set and
        abort the following turn.
        """
        handle = self._assistant_handle
        if handle is None:
            return
        if self._assistant_ui_state in ("thinking", "speaking"):
            handle.interrupt()
        handle.talk()

    def request_interrupt(self) -> None:
        """Barge-in: stop the current reply's generation + audio (M7.5). Non-blocking."""
        handle = self._assistant_handle
        if handle is not None:
            handle.interrupt()

    def request_end_assistant(self) -> None:
        """Enqueue ending the assistant session (handle.stop() joins, so ops worker)."""
        self.command_q.put(Command("end_assistant"))

    def request_describe(self, path: str, prompt: str = "") -> None:
        """Enqueue a vision-describe, or refuse honestly while a Talk session is live.

        Describe switches the running model (to the VL model), so it is mutually
        exclusive with a live Talk session; the guard returns an honest remedy Result
        instead of disrupting the conversation (M10.5).
        """
        if self._assistant_handle is not None:
            self.result_q.put(
                Result(
                    "describe",
                    False,
                    {},
                    error="end the assistant session first (Vision switches the running model)",
                )
            )
            return
        self.command_q.put(Command("describe", {"path": path, "prompt": prompt}))

    def request_memory_search(self, query: str) -> None:
        """Enqueue a read-only memory search (always available, even during a session)."""
        self.command_q.put(Command("memory_search", {"query": query}))

    def request_benchmark(self, model_id: str) -> None:
        """Enqueue a single-model benchmark, or refuse honestly while a session is live.

        Benchmark runs the model exclusively (it drives the ModelController), so it is
        mutually exclusive with a live Talk session (M10.5).
        """
        if self._assistant_handle is not None:
            self.result_q.put(
                Result(
                    "benchmark",
                    False,
                    {"model_id": model_id},
                    error="end the assistant session first (Benchmark runs the model exclusively)",
                )
            )
            return
        self.command_q.put(Command("benchmark", {"model_id": model_id}))

    def _is_discovered(self, model_id: str) -> bool:
        """True when model_id names a discovered fine-tune rather than a yaml row.

        Cheap and string-only: discovered ids are the only ones in the reserved
        "ft:" namespace (config refuses to load a models.yaml row that claims it),
        so no filesystem work is needed to answer this.
        """
        return bool(model_id) and model_id.startswith("ft:")

    def available_voices(self) -> list[str]:
        """The on-disk voice names for the Voice panel dropdown (read accessor).

        Read here (in the controller) so gui.py never imports tts/config or touches
        the filesystem itself; returns [] honestly when no voices are configured.
        """
        from tts import list_voices

        return list_voices(self._settings.paths.kokoro_voices)

    def phone_access(self) -> Any:
        """Tailscale Serve URL for a phone on this tailnet, or None.

        Local CLI only. Missing Tailscale, a stopped daemon, or Serve pointing
        elsewhere is None - the Chat page hides the phone strip rather than
        inventing a URL. Result is cached for the window lifetime; Serve config
        does not change while chatting.
        """
        cached = getattr(self, "_phone_access_cache", _UNOBSERVED)
        if cached is not _UNOBSERVED:
            return cached
        from tailscale_phone import discover_phone_access

        port = int(getattr(getattr(self._settings, "ports", None), "openwebui", 8096) or 8096)
        try:
            found = discover_phone_access(openwebui_port=port)
        except Exception:  # noqa: BLE001 - Chat page must still paint
            found = None
        self._phone_access_cache = found
        return found

    def noise_suppression_mode(self) -> str:
        """Current Open WebUI upload filter preset (off / balanced / strong)."""
        speech = getattr(self._settings, "speech", None)
        mode = str(getattr(speech, "noise_suppression", "balanced") or "balanced")
        return mode.strip().lower() or "balanced"

    def set_noise_suppression(self, mode: str) -> None:
        """Persist the Voice Setup preset and rebind the live router processor."""
        from config import write_speech_noise_suppression

        write_speech_noise_suppression(self._settings.data_dir, mode)
        speech = getattr(self._settings, "speech", None)
        if speech is not None:
            speech.noise_suppression = str(mode).strip().lower()
        if callable(self._apply_noise_suppression_fn):
            self._apply_noise_suppression_fn(mode)
        self.result_q.put(
            Result(
                "noise_suppression",
                True,
                {"mode": self.noise_suppression_mode()},
            )
        )

    def request_second_eye_start(
        self, goal: str, interval_s: float = 1.0
    ) -> None:
        """Start the screen watcher on a dedicated thread (not the ops worker)."""
        if self._assistant_handle is not None:
            self.result_q.put(
                Result(
                    "second_eye",
                    False,
                    {},
                    error="end the assistant session first (Watch uses the vision model)",
                )
            )
            return
        if self._second_eye_fn is None:
            self.result_q.put(
                Result(
                    "second_eye",
                    False,
                    {},
                    error="Watch my screen is not available in this session",
                )
            )
            return
        goal_text = str(goal or "").strip()
        if not goal_text:
            self.result_q.put(
                Result(
                    "second_eye",
                    False,
                    {},
                    error="say what you are doing first (the goal)",
                )
            )
            return
        if self._second_eye_thread is not None and self._second_eye_thread.is_alive():
            self.result_q.put(
                Result(
                    "second_eye",
                    False,
                    {"running": True},
                    error="already watching the screen",
                )
            )
            return
        run_dir = Path(self._settings.data_dir) / "run"
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.result_q.put(
                Result("second_eye", False, {}, error=f"cannot write stop file: {exc}")
            )
            return
        stop_file = run_dir / "second-eye.stop"
        try:
            stop_file.unlink(missing_ok=True)
        except OSError:
            pass
        self._second_eye_stop.clear()
        self._second_eye_stop_file = stop_file
        interval = max(0.2, float(interval_s or 1.0))

        def _run() -> None:
            try:
                self._second_eye_fn(
                    goal_text, interval, str(stop_file)
                )
            except Exception as exc:  # noqa: BLE001 - watcher must not kill the GUI
                self.result_q.put(
                    Result("second_eye", False, {"running": False}, error=str(exc))
                )
                return
            self.result_q.put(
                Result("second_eye", True, {"running": False, "stopped": True})
            )

        self._second_eye_thread = threading.Thread(
            target=_run, name="locitize-second-eye", daemon=True
        )
        self._second_eye_thread.start()
        self.result_q.put(
            Result(
                "second_eye",
                True,
                {"running": True, "goal": goal_text},
            )
        )

    def request_second_eye_stop(self) -> None:
        """Ask the watcher to exit by creating its stop file."""
        stop_file = getattr(self, "_second_eye_stop_file", None)
        if stop_file is None:
            self.result_q.put(
                Result("second_eye", True, {"running": False, "stopped": True})
            )
            return
        try:
            stop_file.write_text("stop\n", encoding="ascii")
        except OSError as exc:
            self.result_q.put(
                Result("second_eye", False, {}, error=f"could not stop watcher: {exc}")
            )
            return
        self._second_eye_stop.set()

    def _push_assistant_state(self, state: str) -> None:
        """Record Talk-loop state, then publish it for the GUI pump."""
        self._assistant_ui_state = str(state or "idle")
        self.result_q.put(
            Result("assistant_state", True, {"state": self._assistant_ui_state})
        )

    def save_model_edits(self, model_id: str, gpu_text: str, ctx_text: str) -> None:
        """Validate the two fields on the UI thread; enqueue the write on success.

        Invalid input never reaches the worker: it pushes an immediate error
        Result the pump renders inline, and Save stays effectively a no-op.
        """
        if self._is_discovered(model_id):
            self.result_q.put(
                Result("save_edits", False, {"model_id": model_id}, error=DISCOVERED_EDIT_REFUSAL)
            )
            return
        gpu_ok, gpu_val, gpu_err = validate_gpu_layers(gpu_text)
        ctx_ok, ctx_val, ctx_err = validate_context_size(ctx_text)
        if not gpu_ok or not ctx_ok:
            self.result_q.put(
                Result(
                    "save_edits",
                    False,
                    {"model_id": model_id},
                    error=gpu_err or ctx_err,
                )
            )
            return
        self.command_q.put(
            Command(
                "save_edits",
                {"model_id": model_id, "gpu_layers": gpu_val, "context_size": ctx_val},
            )
        )

    def save_model_identity(self, model_id: str, id_text: str, name_text: str) -> None:
        """Validate the id/name fields on the UI thread; enqueue the rename on success.

        Invalid input (blank, or an id colliding with another model) never reaches
        the worker: it pushes an immediate error Result the pump renders inline.
        Changing a RUNNING model's id is rejected here too (before the write),
        since running_model_id and every in-flight process-tracking key are still
        keyed on the old id; the owner stops the model first. An id change that
        would orphan another model's draft_model or settings.yaml's launcher.
        default_model is rejected the same way (2026-08-16 incident: this silently
        broke both before the write path was allowed to check for it).
        """
        if self._is_discovered(model_id):
            self.result_q.put(
                Result(
                    "save_identity", False, {"model_id": model_id}, error=DISCOVERED_EDIT_REFUSAL
                )
            )
            return
        existing_ids = [m["id"] for m in self.list_models()]
        id_ok, id_val, id_err = validate_model_id(id_text, existing_ids, model_id)
        name_ok, name_val, name_err = validate_model_name(name_text)
        if not id_ok or not name_ok:
            self.result_q.put(
                Result(
                    "save_identity",
                    False,
                    {"model_id": model_id},
                    error=id_err or name_err,
                )
            )
            return
        if id_val != model_id:
            if self._controller.running_model_id == model_id:
                self.result_q.put(
                    Result(
                        "save_identity",
                        False,
                        {"model_id": model_id},
                        error="stop this model before changing its id",
                    )
                )
                return
            draft_refs = [
                (m.id, m.draft_model) for m in self._registry.all() if m.id != model_id
            ]
            default_model = getattr(
                getattr(self._settings, "launcher", None), "default_model", None
            )
            blockers = find_id_change_blockers(model_id, draft_refs, default_model)
            if blockers:
                self.result_q.put(
                    Result(
                        "save_identity",
                        False,
                        {"model_id": model_id},
                        error="; ".join(blockers),
                    )
                )
                return
        self.command_q.put(
            Command(
                "save_identity",
                {"model_id": model_id, "new_id": id_val, "new_name": name_val},
            )
        )

    def save_model_capabilities(self, model_id: str, capabilities_text: str) -> None:
        """Enqueue a capabilities write. Any text parses (see parse_capabilities_text);
        the only refusal is a discovered row, same guard as edits/identity."""
        if self._is_discovered(model_id):
            self.result_q.put(
                Result(
                    "save_capabilities", False, {"model_id": model_id},
                    error=DISCOVERED_EDIT_REFUSAL,
                )
            )
            return
        self.command_q.put(
            Command(
                "save_capabilities",
                {"model_id": model_id, "capabilities": parse_capabilities_text(capabilities_text)},
            )
        )

    def request_autotune_context(self, model_id: str) -> None:
        """Enqueue an "Auto-tune context" run for ONE model (owner request 2026-08-22).

        Explicitly opt-in per model: this is the only thing that ever starts an
        auto-tune. It is never triggered by selecting or starting a model,
        because each run performs several real llama-server loads costing 20-30
        seconds and a full VRAM allocation apiece - fine when the owner asks for
        it, unacceptable as a side effect of clicking a row.

        Two refusals before anything is queued, mirroring every other
        model-lifecycle write on this controller:
          - a discovered fine-tune has no models.yaml block to write back into;
          - a model that is currently RUNNING would have its own trial starts
            collide with itself on the reserved port, so it is stopped first by
            the owner rather than surprised by this.
        """
        if self._is_discovered(model_id):
            self.result_q.put(
                Result(
                    "autotune", False, {"model_id": model_id},
                    error=DISCOVERED_EDIT_REFUSAL,
                )
            )
            return
        if self._controller.running_model_id is not None:
            self.result_q.put(
                Result(
                    "autotune", False, {"model_id": model_id},
                    error="stop the running model before auto-tuning context",
                )
            )
            return
        # Clear any cancellation left over from a previous run BEFORE queueing,
        # so a Stop pressed during the last tune cannot abort this one.
        self._autotune_cancel.clear()
        self.command_q.put(Command("autotune", {"model_id": model_id}))

    def autotune_in_progress(self) -> bool:
        """True while an auto-tune is occupying the ops worker.

        The UI asks this to decide whether Stop means "stop the running model"
        or "cancel the auto-tune"; those are different actions and must not be
        conflated, because an auto-tune runs precisely when no model is running.
        """
        return self._autotune_running

    def autotune_model_id(self) -> str | None:
        """Which model the in-progress auto-tune is for, or None when idle."""
        return self._autotune_model_id if self._autotune_running else None

    def cancel_autotune(self) -> bool:
        """Ask a running auto-tune to stop. Returns True if there was one.

        Called on the Qt UI thread, DIRECTLY - not via command_q. See the
        _autotune_cancel comment in __init__: the ops worker is mid-tune and
        cannot service a queued command, so a queued cancel would arrive only
        after the thing it was cancelling had ended, which is no cancel at all.

        Setting the event is the entire operation. The ops worker notices it
        inside autotune.run_smoke_trial's wait loop (within a quarter second),
        kills the in-flight trial's whole process tree so the GPU is released at
        once, and unwinds through autotune_model_context's cancellation path,
        which restores the model's previous settings and reports canceled=True.
        """
        if not self._autotune_running:
            return False
        self._autotune_cancel.set()
        return True

    def delete_model(self, model_id: str) -> None:
        """Enqueue a Delete: removes the models.yaml row AND the file(s) it
        points at (owner request 2026-08-21). The UI is responsible for the
        confirmation dialog naming what will be deleted BEFORE calling this -
        this method only validates state, the same two refusals every other
        model-lifecycle write already enforces:
          - a discovered row (no models.yaml block to remove);
          - the model currently RUNNING (its .gguf is an open file handle on
            Windows, so deleting it under the server would fail anyway - this
            just gives the honest reason instead of a raw OS error).
        """
        if self._is_discovered(model_id):
            self.result_q.put(
                Result(
                    "delete_model", False, {"model_id": model_id},
                    error=DISCOVERED_EDIT_REFUSAL,
                )
            )
            return
        if self._controller.running_model_id == model_id:
            self.result_q.put(
                Result(
                    "delete_model", False, {"model_id": model_id},
                    error="stop this model before deleting it",
                )
            )
            return
        self.command_q.put(Command("delete_model", {"model_id": model_id}))

    # ---- M13 fine-tune studio + discovery -------------------------------- #

    def request_finetune_start(self) -> None:
        """Enqueue starting the fine-tune studio (never blocks the UI thread)."""
        self.command_q.put(Command("finetune_start"))

    def request_finetune_stop(self) -> None:
        """Enqueue stopping the fine-tune studio."""
        self.command_q.put(Command("finetune_stop"))

    def request_finetune_open(self) -> None:
        """Enqueue opening the studio's loopback URL in the owner's browser."""
        self.command_q.put(Command("finetune_open"))

    def request_scan_finetunes(self) -> None:
        """Enqueue a rescan of the fine-tune outputs tree."""
        self.command_q.put(Command("finetune_scan"))

    def request_register_discovered(self, key: str) -> None:
        """Enqueue promoting one discovered fine-tune into models.yaml."""
        self.command_q.put(Command("finetune_register", {"key": key}))

    def request_finetune_delete(self, key: str) -> None:
        """Enqueue permanent deletion of ONE discovered run (owner-confirmed).

        The desktop shows the confirmation dialog BEFORE this is called; the
        worker performs the guarded finetune.delete_run and then re-scans so
        the table refreshes without a manual Rescan click (M18.7)."""
        self.command_q.put(Command("finetune_delete", {"key": key}))

    def request_feature_state(self) -> None:
        """Ask the ops worker which install-time features this machine has
        (M18.12, the Settings Features panel). Detection shells out (nvidia-smi,
        venv module probes), so it never runs on the UI thread."""
        self.command_q.put(Command("feature_state"))

    # ---- M14.14 model acquisition intents (non-blocking puts, like the rest) - #

    def request_hub_catalog(self) -> None:
        """Enqueue publishing the shipped catalog. Reads disk only - NO network.

        This is the one hub intent the page fires on paint, and it is safe to do
        so precisely because it cannot reach the network (HF-1).
        """
        self.command_q.put(Command("hub_catalog"))

    def request_hub_search(self, query: str) -> None:
        """Enqueue a HuggingFace search. Only ever called from a Search press."""
        self.command_q.put(Command("hub_search", {"query": query}))

    def request_hub_search_more(self) -> None:
        """Enqueue the next page of the current search. Only from Load more."""
        self.command_q.put(Command("hub_search_more"))

    def request_hub_files(self, repo_id: str) -> None:
        """Enqueue listing one repository's files. Only from a repo selection."""
        self.command_q.put(Command("hub_files", {"repo_id": repo_id}))

    def request_hub_download(self, payload: dict[str, Any]) -> None:
        """Enqueue starting a download. Only ever called from a Download press."""
        self.command_q.put(Command("hub_download", dict(payload)))

    def request_hub_cancel(self) -> None:
        """Ask the live download to stop at its next chunk boundary.

        Sets the cancel event DIRECTLY on the UI thread (perf audit
        2026-08-31), exactly like cancel_autotune and for the same reason:
        while an auto-tune or benchmark holds the single ops worker, a
        QUEUED cancel would sit behind it for minutes - a Cancel button
        that does nothing. The operation is one lock plus one event.set(),
        both safe from any thread; Thread E notices within one chunk."""
        with self._hub_lock:
            cancel = self._hub_cancel
        cancel.set()

    def request_finetune_state(self) -> None:
        """Enqueue a refresh of the Fine-tune page's lifecycle state.

        The state snapshot touches the filesystem (studio_available() probes the
        checkout, and the run-active probe walks the outputs tree), so the view
        asks for it through the queue like every other potentially slow read
        instead of computing it inline at first paint (Architecture M13.7.4).
        """
        self.command_q.put(Command("finetune_state"))

    def finetune_state(self) -> dict[str, Any]:
        """The Fine-tune page's current lifecycle state.

        WORKER THREAD ONLY. This does real filesystem work - studio_available()
        probes the studio checkout and _probe_finetune_run_active() walks the
        outputs tree - either of which can stall on a disconnected drive. It is
        called from _publish_finetune_state() on the ops worker; the view reaches
        it via request_finetune_state(). It never starts, stops, or probes a
        process.
        """
        import finetune as ft

        cfg = getattr(self._settings, "finetune", None)
        port = getattr(cfg, "port", None)
        ok, reason = ft.studio_available(self._settings)
        status = self._finetune_status if ok else "disabled"
        resolved = self._finetune_resolved_port()
        # The URL is advertised ONLY when the studio is genuinely running on a port
        # LOCITIZE itself resolved (D-M13-1): the configured port may be occupied by a
        # foreign process, so publishing it as a live URL would be a lie.
        url = (
            ft.studio_url(self._settings, resolved)
            if ok and status == "running" and resolved is not None
            else None
        )
        return {
            "status": status,
            "url": url,
            # Report where it actually runs when running, else where it will try.
            "port": resolved if (status == "running" and resolved is not None) else port,
            "reason": "" if ok else reason,
            # UX Spec section 2, Panel A: shown once the studio has been started
            # at least once this session - NOT only while it is running.
            "log_path": (
                self._finetune_log_path() if self._finetune_launched_once else ""
            ),
            "run_active": self._probe_finetune_run_active(),
        }

    def _finetune_resolved_port(self) -> int | None:
        """The port the studio's controller actually started on, or None.

        Read defensively and WITHOUT constructing the controller: calling
        _finetune_service() here would spawn lifecycle machinery from a read-only
        state snapshot, and tests inject a fake controller that need not expose the
        property at all. No controller yet means no resolved port, which is the
        fail-closed answer both the URL and the Open button depend on.
        """
        controller = self._finetune_controller
        if controller is None:
            return None
        return getattr(controller, "resolved_port", None)

    def _probe_finetune_run_active(self) -> bool:
        """Walk the outputs tree for a live train.log and cache the answer.

        WORKER THREAD ONLY (ops loop and monitor loop). This is the read-only
        heuristic behind the honest limitation warning: LOCITIZE only claims a run
        is in flight when the studio it started is up AND a run's train.log was
        written to very recently. It never guesses from the studio merely being
        open.
        """
        import finetune as ft

        if not self._finetune_started:
            self._finetune_run_active_cache = False
            return False
        try:
            active = ft.active_run(self._settings) is not None
        except Exception:  # noqa: BLE001 - a status heuristic must never raise
            active = False
        self._finetune_run_active_cache = active
        return active

    def finetune_run_active(self) -> bool:
        """True when a training run looked live at the last background probe.

        GUI-thread safe by construction: it returns the cached answer and does no
        I/O at all, so the window-close warning check can never freeze on a dead
        network or removable path. The monitor loop refreshes the cache on its
        normal cadence while the studio is up, so the value the close path reads
        is at most one monitor interval old.
        """
        return self._finetune_run_active_cache

    def finetune_warning_text(self) -> str:
        """The verbatim limitation notice for a stop/close during a live run."""
        import finetune as ft

        return ft.orphan_warning_text()

    def list_finetunes(self) -> dict[str, Any]:
        """Discovered fine-tune rows for the Fine-tune page, pre-formatted.

        All display formatting (size, modified date, the honest "unknown" quant)
        happens here so the Qt view stays presentation-only and never touches the
        filesystem itself. `reason` is non-empty whenever the list is empty and is
        rendered verbatim in the page's empty state.
        """
        import finetune as ft

        try:
            result = ft.scan_outputs(self._settings)
            marked, _matched = ft.dedup_against_registry(
                result.items, self._registry.all()
            )
        except Exception as exc:  # noqa: BLE001 - a scan miss is a state, not a crash
            return {"items": [], "root": "", "reason": f"Could not scan fine-tunes: {exc}"}
        items = []
        for item in marked:
            if item.already_registered:
                continue
            items.append(
                {
                    "id": item.id,
                    "name": item.name,
                    "run": item.run,
                    "quant": item.quant or "unknown",
                    "size_bytes": item.size_bytes,
                    "size_display": format_size_gb(item.size_bytes),
                    "mtime": item.mtime,
                    "modified_display": format_mtime(item.mtime),
                    "path": item.path,
                    "base_model": item.base_model or "unknown",
                    "meta_display": ft.describe_meta(item),
                    "registered": False,
                }
            )
        return {"items": items, "root": result.root, "reason": result.reason}

    def list_models_merged(self) -> list[dict[str, Any]]:
        """Model rows for the Models page: manual rows first, then discovered ones.

        list_models() deliberately still returns manual rows only (every existing
        caller and test depends on that); this is the additive merged view the
        Models page uses, with a Source column value on every row.
        """
        import finetune as ft

        rows = self.list_models()
        try:
            registered_ft = self._registry.registered_finetune_ids()
        except Exception:  # noqa: BLE001 - discovery must never break the model list
            registered_ft = set()
        for row in rows:
            row["source"] = "registry"
            row["source_display"] = (
                "Registered (fine-tune)" if row["id"] in registered_ft else "Registered"
            )
        try:
            discovered = self._registry.discovered_items()
        except Exception:  # noqa: BLE001 - same guard
            discovered = []
        gpu_total_mb = self.system_specs().gpu_vram_total_mb
        for item in discovered:
            vram_need_mb = compute_vram_need_mb(0, item.size_bytes)
            rows.append(
                {
                    "id": item.id,
                    "name": item.name,
                    "location": item.path,
                    "status": "installed",
                    "gpu_layers": self._settings.finetune.default_gpu_layers,
                    "context_size": self._settings.finetune.default_context_size,
                    "launchable": True,
                    "size_bytes": item.size_bytes,
                    "size_display": format_size_gb(item.size_bytes),
                    "vram_estimate_mb": 0,
                    "vram_need_mb": vram_need_mb,
                    "gpu_portion_display": format_gpu_portion_display(
                        vram_need_mb, gpu_total_mb
                    ),
                    "cpu_portion_display": format_cpu_portion_display(
                        vram_need_mb, gpu_total_mb
                    ),
                    "benchmark_score": None,
                    "benchmark_tok_s": None,
                    "score_display": "-",
                    "source": ft.DISCOVERED_SOURCE,
                    "source_display": "Discovered",
                }
            )
        # list_models() attached registered-row history; run the same pure lookup
        # again after adding discovered rows so their persisted JSONL runs survive a
        # desktop restart too (no models.yaml registration is required for history).
        self._attach_benchmark_throughput(rows)
        return rows

    def _finetune_log_path(self) -> str:
        """Path of the studio's log file (the one the Error state points the owner at)."""
        from logger import resolve_log_dir

        return str(resolve_log_dir(self._settings) / "finetune_studio.log")

    def _finetune_service(self) -> Any:
        """Lazily build the studio's SingleServiceController on the SHARED manager.

        Built on the shared ServiceManager on purpose: that is what makes
        shutdown() -> stop_all() (and the atexit backstop) reap the Streamlit child
        on window close with no new teardown code. Tests inject a fake factory.
        """
        if self._finetune_controller is not None:
            return self._finetune_controller
        if self._finetune_controller_factory is not None:
            self._finetune_controller = self._finetune_controller_factory()
            return self._finetune_controller

        import finetune as ft
        from services import PortAllocator, SingleServiceController, make_process_factory

        settings = self._settings
        allocator = PortAllocator(
            settings.ports.range_start,
            settings.ports.range_end,
            settings.ports.allocation,
        )
        log_path = self._finetune_log_path()
        # The allocator goes to the SPEC BUILDER, not the process factory
        # (DEC-M13-2). Streamlit's --server.port token is invisible to
        # services._command_with_port, so the port must already be resolved in the
        # argv the spec carries. make_process_factory() therefore gets NO
        # allocator: ManagedProcess must not resolve a second time and drift from
        # what the child was actually told to bind.
        self._finetune_controller = SingleServiceController(
            self._manager,
            lambda: ft.build_studio_spec(settings, log_path, allocator),
            make_process_factory(),
        )
        return self._finetune_controller

    def _publish_finetune_state(self, warning: str = "", error: str = "") -> None:
        """Push one finetune_state Result carrying the current lifecycle snapshot."""
        payload = self.finetune_state()
        if warning:
            payload["warning"] = warning
        self.result_q.put(
            Result("finetune_state", not error, payload, error=error or None)
        )

    def _do_finetune_start(self) -> None:
        """Start the studio on the shared manager; publish the honest outcome.

        Runs on the ops worker because the start blocks until readiness (or the
        timeout). A failure is an Error state plus the log path, never a traceback.
        """
        from services import PortUnavailableError, ServiceStatus

        self._finetune_status = "starting"
        self._publish_finetune_state()
        try:
            status = self._finetune_service().start()
        except PortUnavailableError:
            # Every port in the reserved range is taken (or the policy is strict
            # and the configured one is). That is an actionable configuration
            # situation, not a crash, so say exactly what to do about it.
            ports = self._settings.ports
            configured = getattr(getattr(self._settings, "finetune", None), "port", "?")
            self._finetune_status = "error"
            self._publish_finetune_state(
                error=(
                    f"No free loopback port for the fine-tune studio in the "
                    f"reserved range {ports.range_start}-{ports.range_end}. Free "
                    f"port {configured}, or widen ports.range_end in settings.yaml."
                )
            )
            return
        except ValueError as exc:
            # Honest configuration guard (studio dir/app/interpreter missing).
            self._finetune_status = "error"
            self._publish_finetune_state(error=str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - boundary: never raise off-thread
            # The child may already have been spawned before this failure, so its
            # log file can exist and is the owner's best evidence (D-M13-2).
            self._finetune_launched_once = True
            self._finetune_status = "error"
            self._publish_finetune_state(error=f"fine-tune studio start failed: {exc}")
            return
        # Past the guards above a child process was actually launched, so
        # logs/finetune_studio.log now exists for both outcomes below. The two
        # early returns above (no free port, bad configuration) never spawn one
        # and therefore never latch - pointing at a non-existent log would lie.
        self._finetune_launched_once = True
        if status is ServiceStatus.RUNNING:
            self._finetune_started = True
            self._finetune_status = "running"
            self._publish_finetune_state()
        else:
            self._finetune_status = "error"
            self._publish_finetune_state(
                error=f"Did not start - see {self._finetune_log_path()}"
            )

    def _do_finetune_stop(self) -> None:
        """Stop the studio, surfacing the limitation warning if a run was live.

        The run check happens BEFORE the stop, because once the child is gone the
        evidence (a train.log still being written) may stop updating and the owner
        would get a clean-stop claim that is not true.
        """
        # Probes the tree directly rather than reading the cache: this runs on the
        # ops worker, and the stop-time warning must reflect the run's state right
        # now, not the state at the last monitor tick.
        warning = (
            self.finetune_warning_text() if self._probe_finetune_run_active() else ""
        )
        try:
            self._finetune_service().stop()
        except Exception as exc:  # noqa: BLE001 - boundary guard
            self._finetune_status = "error"
            self._publish_finetune_state(error=f"fine-tune studio stop failed: {exc}")
            return
        self._finetune_status = "stopped"
        self._finetune_started = False
        self._publish_finetune_state(warning=warning)

    def _do_finetune_open(self) -> None:
        """Open the studio's loopback URL in the owner's default browser."""
        import finetune as ft

        if self._finetune_status != "running":
            self._publish_finetune_state(
                error="the fine-tune studio is not running yet"
            )
            return
        # Second, independent fail-closed guard (DEC-M13-2 / SEC-M13-2): without a
        # port LOCITIZE itself resolved there is no URL it can honestly claim is the
        # studio, so the browser is never launched at a guess.
        resolved = self._finetune_resolved_port()
        if resolved is None:
            self._publish_finetune_state(
                error="the fine-tune studio is not running yet"
            )
            return
        url = ft.studio_url(self._settings, resolved)
        if getattr(self._settings.finetune, "open_browser", True):
            webbrowser.open(url)
            self._publish_finetune_state()
            return
        # open_browser is off, so the click cannot launch anything. Say what
        # happened and hand over the URL instead of republishing an identical
        # state, which reads as a dead button (finding L-2).
        payload = self.finetune_state()
        payload["message"] = (
            f"finetune.open_browser is off - open {url} yourself, or set "
            f"finetune.open_browser: true in settings.yaml."
        )
        self.result_q.put(Result("finetune_state", True, payload))

    def _do_finetune_scan(self) -> None:
        """Rescan the outputs tree and publish the discovered-model rows."""
        import finetune as ft

        ft.clear_scan_cache()
        payload = self.list_finetunes()
        self.result_q.put(Result("finetune_models", True, payload))

    def _do_finetune_register(self, key: str) -> None:
        """Promote ONE discovered fine-tune into models.yaml (owner-triggered only).

        The only writer discovery can ever reach, and only through this explicit
        command. A duplicate id is refused with the worded message the UI shows
        verbatim; nothing is written in that case and the caller can retry after
        the owner resolves the clash by hand.
        """
        import finetune as ft
        from config import RegistryWriteError, append_model_entry

        rows = self.list_finetunes()["items"]
        row = next((r for r in rows if r["id"] == key), None)
        if row is None:
            self.result_q.put(
                Result(
                    "finetune_register_result",
                    False,
                    {"key": key},
                    error="that fine-tune is no longer in the scan; click Rescan",
                )
            )
            return
        model_id = ft.register_id_for(key)
        try:
            append_model_entry(
                self._settings.data_dir,
                model_id,
                row["name"],
                row["path"],
                description=f"Fine-tuned model registered from the run folder {row['run']}.",
                context_size=self._settings.finetune.default_context_size,
                gpu_layers=self._settings.finetune.default_gpu_layers,
                quantization="" if row["quant"] == "unknown" else row["quant"],
            )
        # DEC-M14-11: two handlers, one message shape. ValueError is a refusal
        # about the ARGUMENTS (a duplicate id); RegistryWriteError is a refusal
        # about the FILE, already worded with the real path and a next step.
        # OSError is not caught: the chokepoint translates every one of them,
        # so catching it here would only let this call site invent its own
        # unreadable diagnostic again.
        except (ValueError, RegistryWriteError) as exc:
            self.result_q.put(
                Result(
                    "finetune_register_result",
                    False,
                    {"key": key, "model_id": model_id},
                    error=f"Could not register: {exc}",
                )
            )
            return
        # Reload so the new manual row is live immediately; the discovered
        # duplicate disappears on the NEXT scan (the UI says exactly that rather
        # than claiming it already vanished).
        try:
            self.refresh_models()
        except Exception:  # noqa: BLE001 - the write succeeded; a reload miss is minor
            pass
        self.result_q.put(
            Result("finetune_register_result", True, {"key": key, "model_id": model_id})
        )

    def _do_feature_state(self) -> None:
        """Detect per-feature install state via the wizard's own detection.

        A feature counts installed when EVERY requirement it lists is satisfied
        - the same pessimistic detect_state the wizard trusts, so Settings and
        the wizard can never disagree about what this machine has.
        """
        try:
            import setup_env
            import setup_plan

            state, _machine = setup_env.detect_state()
            features = [
                {
                    "key": feature.key,
                    "label": feature.label,
                    "installed": all(state.get(req, False) for req in feature.requires),
                    "core": feature.core,
                }
                for feature in setup_plan.FEATURES
            ]
            self.result_q.put(Result("feature_state", True, {"features": features}))
        except Exception as exc:  # noqa: BLE001 - detection must never crash the app
            self.result_q.put(
                Result("feature_state", False, {}, error=f"could not detect features: {exc}")
            )

    def _do_finetune_delete(self, key: str) -> None:
        """Delete one discovered run via the guarded finetune.delete_run (M18.7)."""
        import finetune as ft

        rows = self.list_finetunes()["items"]
        row = next((r for r in rows if r["id"] == key), None)
        if row is None:
            self.result_q.put(
                Result(
                    "finetune_delete_result",
                    False,
                    {"key": key},
                    error="that fine-tune is no longer in the scan; click Rescan",
                )
            )
            return
        ok, message = ft.delete_run(
            self._settings, row["run"], self._registry.all()
        )
        self.result_q.put(
            Result(
                "finetune_delete_result",
                ok,
                {"key": key, "message": message},
                error="" if ok else message,
            )
        )
        # Refresh the table either way, so what the owner sees is the disk truth.
        self._do_finetune_scan()

    def request_chat(self) -> None:
        """Run the pure chat-UI chooser and act on / surface the decision (M9.3).

        Called on the UI thread by the Chat button. Computes the three runtime facts
        the pure resolve_chat_choice needs (a model is running, Open WebUI is
        installed, Open WebUI is already listening), then:

        - NO_MODEL            -> honest error Result (never opens a dead URL)
        - OPEN_LLAMACPP       -> open the built-in UI immediately
        - OPEN_OPENWEBUI      -> open the running Open WebUI immediately
        - DEGRADE_TO_LLAMACPP -> open the built-in UI, carrying the reason
        - ASK                 -> a "chat_ask" Result so gui.py renders the choice dialog
        - OFFER_START_OPENWEBUI -> a "chat_offer_start" Result so gui.py offers Start

        The decision function is pure and shared with the terminal menu, so both
        surfaces make the identical choice. gui_controller does the small file/port
        checks here (it is the controller, not the presentation shell); gui.py stays
        free of subprocess/HTTP/yaml.
        """
        from webui import ChatDecision, resolve_chat_choice, webui_available

        installed = webui_available(self._settings)
        ready = installed and self._openwebui_listening()
        resolution = resolve_chat_choice(
            preferred=self._settings.chat.preferred_ui,
            cli_override=None,
            model_running=self._active_port is not None,
            webui_installed=installed,
            webui_ready=ready,
        )
        decision = resolution.decision
        if decision is ChatDecision.NO_MODEL:
            self.result_q.put(Result("chat", False, {}, error=resolution.reason))
        elif decision in (ChatDecision.OPEN_LLAMACPP, ChatDecision.DEGRADE_TO_LLAMACPP):
            self._open_llamacpp(reason=resolution.reason)
        elif decision is ChatDecision.OPEN_OPENWEBUI:
            # Perf audit 2026-08-31: _open_openwebui can run secure_proxy
            # work (a winget install, caddy start, certutil with a consent
            # dialog, a .local DNS lookup) - seconds to MINUTES. Never on
            # the UI thread; same routing rationale as
            # start_openwebui_and_open. The status line answers the click
            # immediately so the button never feels dead.
            self.result_q.put(Result(
                "chat_status", True, {"message": "opening Open WebUI..."}
            ))
            self.command_q.put(Command("open_openwebui_ready"))
        elif decision is ChatDecision.ASK:
            self.result_q.put(Result("chat_ask", True, {}))
        elif decision is ChatDecision.OFFER_START_OPENWEBUI:
            self.result_q.put(
                Result("chat_offer_start", True, {"reason": resolution.reason})
            )

    def open_chat(self, choice: str | None = None, remember: bool = False) -> None:
        """Open the chosen chat UI, optionally remembering the choice (M9.3/M9.4).

        Called by gui.py after the ASK dialog resolves (choice='llamacpp'|'openwebui',
        remember from the checkbox), and kept callable with no args as the legacy
        built-in-UI open. remember=True persists the choice via the ops worker
        (write_chat_ui is disk I/O, so it never runs on the UI thread). Runs on the UI
        thread otherwise (webbrowser.open is non-blocking). Honest guard: no model ->
        error Result, never a dead URL.
        """
        if self._active_port is None:
            self.result_q.put(
                Result("chat", False, {}, error="no model running - start one first")
            )
            return
        if remember and choice in ("llamacpp", "openwebui"):
            # Persist off the UI thread; the browser opens immediately regardless.
            self.command_q.put(Command("remember_chat_ui", {"choice": choice}))
        if choice == "openwebui":
            # Defect fix 2026-08-14: the chooser could resolve to Open WebUI while
            # the service was not running; opening the URL alone produced a dead
            # tab and no service, while the status chip still said "chat opened".
            # Route the not-listening case through the same ops-worker start the
            # offer dialog uses (it opens the browser on readiness and degrades
            # honestly to the built-in UI on failure).
            if self._openwebui_listening():
                # Perf audit 2026-08-31: routed via the ops worker for the
                # same secure_proxy reason as request_chat above.
                self.result_q.put(Result(
                    "chat_status", True,
                    {"message": "opening Open WebUI..."},
                ))
                self.command_q.put(Command("open_openwebui_ready"))
            else:
                self.result_q.put(
                    Result(
                        "chat_status",
                        True,
                        {"message": "starting Open WebUI; first launch can take time..."},
                    )
                )
                self.command_q.put(Command("start_openwebui"))
        else:
            self._open_llamacpp()

    def detect_harnesses(self) -> dict[str, str | None]:
        """Resolve claude/codex/opencode on PATH for the picker to grey out
        whichever isn't installed. Cheap (shutil.which x3); safe on the UI
        thread, matching how open_chat's own webui_available() check runs."""
        from harness_launch import detect_harnesses as _detect

        return _detect()

    def last_project_dir(self) -> str:
        """The remembered folder the harness picker's browse dialog defaults
        to; "" if no harness has ever been launched from this install."""
        return self._settings.chat_harness.last_project_dir

    def request_launch_harness(
        self, choice: str, project_dir: str, remember: bool = True
    ) -> None:
        """Queue a "Launch in Claude Code/Codex/OpenCode" request (M-harness).

        Runs on the ops worker (Thread B): each launch writes a provider
        config file (Codex's config.toml, OpenCode's project opencode.json)
        and/or starts the Claude bridge's HTTP server, none of which belong
        on the UI thread. remember persists project_dir via write_chat_harness_dir
        the same way open_chat's remember persists the UI choice.
        """
        self.command_q.put(
            Command(
                "launch_harness",
                {"choice": choice, "project_dir": project_dir, "remember": remember},
            )
        )

    def request_session_launch(self, payload: dict) -> None:
        """Serialize model readiness and a local coding launch on the ops worker."""
        self.command_q.put(Command("session_launch", dict(payload)))

    def _do_session_launch(self, payload: dict) -> None:
        import sqlite3
        import harness_launch
        from session_launch import build_local_launch
        from session_store import SessionStore

        model_id = payload.get("model_id", "")
        choice = payload.get("choice", "")
        project = payload.get("project_dir", "")
        try:
            if self._registry.get(model_id) is None:
                raise ValueError("Select an installed model before launching")
            if harness_launch.detect_executable(choice) is None:
                raise ValueError(f"{choice} is not installed. Add it in Settings > Features.")
            # Validate before starting or replacing any model.
            build_local_launch(choice, project, model_id, 8080, payload.get("resume_id", ""))
            if self._controller.running_model_id != model_id or self._active_port is None:
                self._do_start(model_id)
            if self._controller.running_model_id != model_id or self._active_port is None:
                raise ValueError("The selected model did not become ready. See Models for details.")
            registered = self._registry.get(model_id)
            argv, env = build_local_launch(choice, project, model_id, self._active_port,
                                           payload.get("resume_id", ""), registered.context_size)
            harness_launch.spawn_in_terminal(argv, project, env, f"{choice} - LOCITIZE ({model_id})")
            self._controller._session_reserved_model = model_id
            self._local_session_launches = getattr(self, "_local_session_launches", 0) + 1
            sid = payload.get("resume_id")
            result_payload = dict(payload)
            if sid:
                try:
                    SessionStore(self._settings.data_dir).update(choice, sid, model_id=model_id, project=project)
                except (OSError, ValueError, sqlite3.Error) as exc:
                    result_payload["warning"] = f"Terminal opened, but its profile could not be saved: {exc}"
            self.result_q.put(Result("session_launch", True, result_payload))
        except (OSError, ValueError) as exc:
            self.result_q.put(Result("session_launch", False, dict(payload), error=str(exc)))

    def request_install_harness(self, harness: str) -> None:
        """Queue an "Install <harness>" request (owner request 2026-08-21:
        zero-friction onboarding). Runs on the ops worker: the native
        installer / winget / npm subprocess calls do not belong on the UI
        thread. Fired from the first-run onboarding dialog and from the Chat
        picker's own "Install" action next to a harness not found on PATH.
        """
        self.command_q.put(Command("install_harness", {"harness": harness}))

    def start_openwebui_and_open(self, remember: bool = False) -> None:
        """Enqueue starting Open WebUI on the ops worker, then open it (M9.3).

        Used by the OFFER_START_OPENWEBUI dialog's Start button. The start can take up
        to the readiness timeout (first-run DB migration), so it MUST run on the ops
        worker, never the UI thread (G2). On success the worker opens Open WebUI; on
        failure it degrades to the built-in UI with an honest reason.
        """
        if remember:
            self.command_q.put(Command("remember_chat_ui", {"choice": "openwebui"}))
        self.command_q.put(Command("start_openwebui"))

    def _open_llamacpp(self, reason: str = "") -> None:
        """Open the built-in llama.cpp web UI on the active resolved port."""
        port = self._active_port
        if port is None:
            self.result_q.put(
                Result("chat", False, {}, error="no model running - start one first")
            )
            return
        self._open_url(self._chat_url(port), reason=reason)

    def _open_openwebui(self, reason: str = "") -> None:
        """Open the Open WebUI service, preferring the friendly hostname.

        Owner request 2026-08-14: the machine carries a hosts entry
        (locitize.local -> 127.0.0.1) plus a netsh portproxy (80 -> openwebui
        port), so the address bar can read http://locitize.local/ with no port.
        Probe that mapping live each time and fall back to the plain loopback
        port URL if either half has been removed - the button must never open
        a dead URL.
        """
        url = f"http://127.0.0.1:{self._settings.ports.openwebui}/"
        hostname = self._settings.secure_proxy.hostname
        # Self-heal the Caddy TLS chain (install/start/trust) before choosing the
        # URL, so the padlock survives reboots and fresh machines (owner request
        # 2026-08-14). ensure() is two socket probes when already healthy; on
        # failure the message lands on the status rail and we fall back to the
        # plain URLs rather than opening a dead https page.
        try:
            import secure_proxy

            ok, message = secure_proxy.ensure(
                self._settings,
                notify=lambda msg: self.result_q.put(
                    Result("chat_status", True, {"message": msg})
                ),
            )
        except Exception as exc:  # noqa: BLE001 - never let ensure kill chat open
            ok, message = False, f"secure proxy check failed: {exc}"
        if ok:
            url = f"https://{hostname}/"
        else:
            self.result_q.put(Result("chat_status", True, {"message": message}))
            if self._friendly_openwebui_listening(443):
                url = f"https://{hostname}/"
            elif self._friendly_openwebui_listening(80):
                url = f"http://{hostname}/"
        self._open_url(url, reason=reason)

    def _friendly_openwebui_listening(self, port: int) -> bool:
        """True if locitize.local answers on the given port (hosts + proxy intact).

        Hardening (public-readiness audit): verify the name actually resolves to
        loopback BEFORE connecting, mirroring secure_proxy's own guard - a
        hijacked or externally-resolving locitize.local must never draw a
        connection off this machine.
        """
        import socket

        try:
            resolved = {info[4][0] for info in socket.getaddrinfo("locitize.local", port)}
        except OSError:
            return False
        if not resolved or not all(a in ("127.0.0.1", "::1") for a in resolved):
            return False
        try:
            with socket.create_connection(("locitize.local", port), timeout=0.5):
                return True
        except OSError:
            return False

    def _open_url(self, url: str, reason: str = "") -> None:
        """Open a chat URL in the browser (or surface it as text), then push Result."""
        payload: dict[str, Any] = {"url": url}
        if reason:
            payload["reason"] = reason
        if self._settings.gui.chat_open_browser:
            webbrowser.open(url)
            self.result_q.put(Result("chat", True, payload))
        else:
            payload["opened"] = False
            self.result_q.put(Result("chat", True, payload))

    def _openwebui_listening(self) -> bool:
        """True if something already answers on the Open WebUI loopback port."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            try:
                sock.connect(("127.0.0.1", self._settings.ports.openwebui))
                return True
            except OSError:
                return False

    def _chat_url(self, port: int) -> str:
        """Resolve the chat URL: proxy hostname when running, else loopback:port."""
        if self._proxy is not None and self._settings.proxy.enabled:
            host = self._settings.proxy.hostname
            proxy_port = self._settings.proxy.port
            suffix = "" if proxy_port == 80 else f":{proxy_port}"
            return f"http://{host}{suffix}/"
        return f"http://127.0.0.1:{port}/"

    # ---- optional reverse proxy (G4) -------------------------------------- #

    def start_proxy(self) -> Result:
        """Start the loopback reverse proxy if configured; return the outcome.

        Never binds port 80 without the elevated opt-in: the proxy module raises a
        PermissionError which is converted to a remedy Result pointing at the
        elevated script (Permission Matrix section 7). The proxy follows the active
        model via the port_provider below.
        """
        if self._proxy is not None:
            return Result("proxy", True, {"running": True})
        if self._proxy_factory is None:
            from proxy import ReverseProxy  # imported here to keep import cheap

            factory = ReverseProxy
        else:
            factory = self._proxy_factory
        try:
            self._proxy = factory(
                self._settings.proxy,
                port_provider=lambda: self._active_port,
            )
            self._proxy.start()
        except PermissionError as exc:
            self._proxy = None
            result = Result("proxy", False, {}, error=str(exc))
            self.result_q.put(result)
            return result
        except OSError as exc:
            self._proxy = None
            result = Result(
                "proxy", False, {}, error=f"proxy could not bind: {exc}"
            )
            self.result_q.put(result)
            return result
        result = Result("proxy", True, {"running": True})
        self.result_q.put(result)
        return result

    def stop_proxy(self) -> None:
        """Stop the reverse proxy if it is running (idempotent)."""
        if self._proxy is not None:
            try:
                self._proxy.stop()
            finally:
                self._proxy = None

    # ---- shutdown / no-orphan survival (G2, mandatory) -------------------- #

    def shutdown(self) -> None:
        """Attempt EVERY teardown step, in order, exactly once.

        Called from desktop.py's closeEvent (and gui.py's WM_DELETE_WINDOW handler)
        before the window goes away. Idempotent: a second call, or the atexit
        backstop after a hard crash, is a no-op, so stop_all() runs exactly once
        (AC14, Architecture G2).

        HIGH-4 invariant: no single step's failure may abort the steps after it.
        The steps that reap CHILD PROCESSES - stop_proxy() and _manager.stop_all()
        - are necessarily last, so an exception escaping any earlier step left the
        llama.cpp and whisper children running while the idempotence latch was
        already set, making a retry and the atexit backstop no-ops too. That is a
        silent violation of the no-orphan guarantee, so every step now runs inside
        its own guard and a failure is RECORDED (shutdown_errors) rather than
        raised at a caller that cannot handle it. _shutdown_done is set only after
        the whole sequence has been attempted, so it never certifies a teardown
        that did not happen.
        """
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self.shutdown_errors = []

        def step(name: str, action: Callable[[], None]) -> None:
            """Run one teardown step; record its failure, never propagate it.

            A raising step must not skip the steps after it: `proxy` and
            `services` are the ones that reap the llama.cpp/whisper/Kokoro
            children, so propagating here strands real processes. closeEvent
            has nowhere to handle an exception either, so the failure is
            recorded and shutdown carries on to the next step.
            """
            try:
                action()
            except Exception as exc:  # teardown must attempt every step
                self.shutdown_errors.append(f"{name}: {exc}")

        # (0) M10.4: stop any LIVE assistant session first, BEFORE joining the ops
        #     worker, so its dedicated session thread's talk-gate wait is unblocked
        #     (STOP sentinel) and its whisper/Kokoro children are reaped
        #     deterministically ahead of the shared manager's stop_all().
        step("assistant", self._shutdown_assistant)
        step("second_eye", self.request_second_eye_stop)
        # (0b) M14.14.3: stop a live model download the same way the user's Cancel
        #      does, and join it with a bounded wait.
        step("download", self._shutdown_hub_thread)
        # (0c) cancel a running auto-tune (owner request 2026-08-22). Without
        #      this, closing the window during a tune would leave the ops worker
        #      inside a multi-minute trial: the bounded join below would time
        #      out and the trial's llama-server would be orphaned holding VRAM
        #      after the window had gone. Setting the event makes the trial kill
        #      its own process tree within a quarter second, so the join finds a
        #      worker that is genuinely on its way out.
        step("autotune", self._shutdown_autotune)
        # (1) signal the monitor and hand the ops worker a sentinel to break its get().
        step("signal", self._shutdown_signal_workers)
        # (2) join both workers with a bounded wait so a wedged thread cannot hang
        #     the close (the daemon flag guarantees the process still exits).
        step("join", self._shutdown_join_workers)
        # (3) tear down the proxy and the model + whisper services (shared manager
        #     => one call stops both). These are the no-orphan steps: they run even
        #     if everything above failed.
        step("proxy", self.stop_proxy)
        step("services", lambda: self._manager.stop_all())
        # SEC-M14-5 / Reviewer MEDIUM-8: a teardown step that failed used to be
        # recorded in memory and then discarded with the process. Writing it to
        # the application log is what makes "LOCITIZE closed but a child survived"
        # diagnosable after the fact. Engineer-facing only: a user-facing surface
        # needs a UX decision first, so none is invented here.
        self._log_shutdown_errors()
        self._shutdown_done = True

    def _log_shutdown_errors(self) -> None:
        """Write any recorded teardown failures to the application log.

        Runs at the very end of shutdown and must never raise: this is the last
        thing standing between a failed close and process exit, and an exception
        here would skip _shutdown_done and mislabel the teardown.
        """
        if not self.shutdown_errors:
            return
        try:
            from logger import get_logger

            log = get_logger("gui_controller")
            for entry in self.shutdown_errors:
                log.error("shutdown step failed: %s", entry)
        except Exception:  # noqa: BLE001 - logging must never break the close
            pass

    def _shutdown_assistant(self) -> None:
        """Teardown step 0: end a live assistant session and reap its children."""
        handle = self._assistant_handle
        self._assistant_handle = None
        if handle is not None:
            handle.stop()

    def _shutdown_hub_thread(self) -> None:
        """Teardown step 0b: cancel the model download and join Thread E.

        Thread E is a daemon thread INSIDE this process - it launches no child -
        so it can never leave an orphaned process behind; the join is only so a
        partially-written file is closed tidily before the window goes away. A
        wedged transfer must not hold the close open, hence the bounded wait.

        The shutting-down flag is latched under _hub_lock, which makes this read
        atomic against _do_hub_download's construct-then-start pair: a download
        that has not yet taken the lock is refused, and one that already has is
        fully started before we observe it. is_alive() is the second line of
        defence - a constructed-but-unstarted thread reports False, and join()
        on such a thread raises RuntimeError (HIGH-4).
        """
        with self._hub_lock:
            self._hub_shutting_down = True
            self._hub_cancel.set()
            thread = self._hub_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

    def _shutdown_autotune(self) -> None:
        """Teardown step 0c: abort an in-progress auto-tune before joining workers.

        Unconditionally sets the event rather than checking _autotune_running
        first: a tune that starts in the gap between the check and the set would
        otherwise slip through, and setting an event nobody is waiting on costs
        nothing (request_autotune_context clears it before every run).
        """
        self._autotune_cancel.set()

    def _shutdown_signal_workers(self) -> None:
        """Teardown step 1: wake the monitor and break the ops worker's get()."""
        self._stop_event.set()
        self.command_q.put(Command("sentinel"))

    def _shutdown_join_workers(self) -> None:
        """Teardown step 2: bounded join of the ops and monitor workers.

        is_alive() guards the same never-started case as Thread E: start_threads()
        may never have run (a close during startup, or a headless test).
        """
        for thread in (self._ops_thread, self._monitor_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=5.0)

    # ---- the ops worker (Thread B) ---------------------------------------- #

    def _ops_loop(self) -> None:
        """Serialize every lifecycle mutation; convert failures to Result objects.

        Being the single consumer of command_q guarantees only one start/switch/
        stop runs at a time, so the controllers need no locking. No exception ever
        crosses the thread boundary: each is turned into an error Result the pump
        renders (G2).
        """
        while True:
            command = self.command_q.get()
            if command.kind == "sentinel":
                return
            try:
                self._dispatch(command)
            except Exception as exc:  # noqa: BLE001 - boundary: never raise off-thread
                self.result_q.put(
                    Result(command.kind, False, {}, error=f"unexpected error: {exc}")
                )

    def _dispatch(self, command: Command) -> None:
        """Route one command to the matching existing-controller call."""
        if command.kind == "start":
            self._do_start(
                command.payload["model_id"],
                command.payload.get("reasoning"),
            )
        elif command.kind == "stop":
            self._do_stop()
        elif command.kind == "free_gpu":
            self._do_free_gpu()
        elif command.kind == "whisper_toggle":
            self._do_whisper_toggle()
        elif command.kind == "listen":
            self._do_listen(command.payload.get("seconds", 15.0))
        elif command.kind == "speak":
            self._do_speak(command.payload.get("text", ""), command.payload.get("voice", ""))
        elif command.kind == "audition":
            self._do_audition()
        elif command.kind == "save_edits":
            self._do_save_edits(command.payload)
        elif command.kind == "save_identity":
            self._do_save_identity(command.payload)
        elif command.kind == "save_capabilities":
            self._do_save_capabilities(command.payload)
        elif command.kind == "delete_model":
            self._do_delete_model(command.payload)
        elif command.kind == "autotune":
            self._do_autotune_context(command.payload.get("model_id", ""))
        elif command.kind == "remember_chat_ui":
            self._do_remember_chat_ui(command.payload.get("choice", ""))
        elif command.kind == "open_openwebui_ready":
            self._open_openwebui()
        elif command.kind == "start_openwebui":
            self._do_start_openwebui()
        elif command.kind == "launch_harness":
            self._do_launch_harness(
                command.payload.get("choice", ""),
                command.payload.get("project_dir", ""),
                command.payload.get("remember", False),
            )
        elif command.kind == "session_launch":
            self._do_session_launch(command.payload)
        elif command.kind == "release_sessions":
            self._controller._session_reserved_model = None
            self._local_session_launches = 0
            self.result_q.put(Result("session_launch", True, {"released": True}))
        elif command.kind == "install_harness":
            self._do_install_harness(command.payload.get("harness", ""))
        elif command.kind == "start_assistant":
            self._do_start_assistant(
                command.payload.get("voice", ""), command.payload.get("speak", True)
            )
        elif command.kind == "end_assistant":
            self._do_end_assistant()
        elif command.kind == "describe":
            self._do_describe(
                command.payload.get("path", ""), command.payload.get("prompt", "")
            )
        elif command.kind == "memory_search":
            self._do_memory_search(command.payload.get("query", ""))
        elif command.kind == "benchmark":
            self._do_benchmark(command.payload.get("model_id", ""))
        elif command.kind == "finetune_start":
            self._do_finetune_start()
        elif command.kind == "finetune_stop":
            self._do_finetune_stop()
        elif command.kind == "finetune_open":
            self._do_finetune_open()
        elif command.kind == "finetune_scan":
            self._do_finetune_scan()
        elif command.kind == "feature_state":
            self._do_feature_state()
        elif command.kind == "finetune_delete":
            self._do_finetune_delete(command.payload.get("key", ""))
        elif command.kind == "finetune_register":
            self._do_finetune_register(command.payload.get("key", ""))
        elif command.kind == "finetune_state":
            # Pure read: publishes the current snapshot without touching the child.
            self._publish_finetune_state()
        elif command.kind == "hub_catalog":
            self._do_hub_catalog()
        elif command.kind == "hub_search":
            self._do_hub_search(command.payload.get("query", ""))
        elif command.kind == "hub_search_more":
            self._do_hub_search_more()
        elif command.kind == "hub_files":
            self._do_hub_files(command.payload.get("repo_id", ""))
        elif command.kind == "hub_download":
            self._do_hub_download(command.payload)
        elif command.kind == "hub_cancel":
            self._do_hub_cancel()

    # ---- M14.14 model acquisition handlers (ops worker, Thread B) --------- #

    def _hub(self) -> Any:
        """Return the Downloader, building it on first use. Makes no network call."""
        if self._hub_downloader is None:
            if self._hub_downloader_factory is not None:
                self._hub_downloader = self._hub_downloader_factory()
            else:
                import modelhub
                from config import resolve_models_dir

                self._hub_downloader = modelhub.Downloader(
                    modelhub.HubConfig.from_settings(self._settings),
                    models_dir=resolve_models_dir(self._settings),
                    gpu_provider=self._gpu_provider,
                    system_provider=self._sys_provider,
                )
        return self._hub_downloader

    def _do_hub_catalog(self) -> None:
        """Publish the shipped catalog from disk (page paint; no egress)."""
        payload = self._hub().catalog()
        self.result_q.put(Result("hub_catalog", True, payload))

    def _do_hub_search(self, query: str) -> None:
        """Run ONE search and publish it, success or honest failure alike."""
        payload = self._hub().search(query)
        # An offline or rate-limited answer is a legitimate, fully-rendered
        # state, not an error dialog: ok=False carries the reason line the view
        # prints verbatim above a still-browsable catalog.
        self.result_q.put(
            Result("hub_search", bool(payload.get("ok")), payload,
                   error=None if payload.get("ok") else payload.get("reason"))
        )

    def _do_hub_search_more(self) -> None:
        """Fetch and append the next page of the current search (Load more press)."""
        payload = self._hub().search_more()
        self.result_q.put(
            Result("hub_search_more", bool(payload.get("ok")), payload,
                   error=None if payload.get("ok") else payload.get("reason"))
        )

    def _do_hub_files(self, repo_id: str) -> None:
        """List one repository's downloadable files with fit and verification."""
        payload = self._hub().list_files(repo_id)
        self.result_q.put(
            Result("hub_files", bool(payload.get("ok")), payload,
                   error=None if payload.get("ok") else payload.get("reason"))
        )

    def _do_hub_download(self, payload: dict[str, Any]) -> None:
        """Start Thread E and RETURN IMMEDIATELY - the whole point of Thread E.

        This handler must not block: it runs on the single ops worker, so any
        time spent here is time start/stop/benchmark/fine-tune cannot run.
        """
        # Everything that reads or writes the Thread E trio happens under
        # _hub_lock, INCLUDING start(). shutdown() runs on the GUI thread and used
        # to be able to land between "self._hub_thread = Thread(...)" and
        # ".start()", joining a thread that had never started - a RuntimeError that
        # aborted the rest of teardown (HIGH-4). Holding the lock across
        # construction and start makes that window impossible; the thread is also
        # started BEFORE it is published, so self._hub_thread is never a
        # not-yet-running thread. The refusal Result is published after the lock
        # is released, to keep queue work out of the critical section.
        import modelhub

        refusal: str | None = None
        with self._hub_lock:
            if self._hub_shutting_down:
                # The window is closing and Thread E has already been cancelled and
                # joined. Starting a transfer now would create a worker nothing
                # cancels or joins, so this command is refused instead.
                refusal = "LOCITIZE is shutting down; the download was not started."
            elif self._hub_thread is not None and self._hub_thread.is_alive():
                # One download at a time. No queue: parallel multi-gigabyte
                # transfers fight for bandwidth and give the user two slow
                # downloads instead of one fast one.
                refusal = "A download is already running."
            else:
                job_id = modelhub.hub_job_id(
                    payload.get("repo_id", ""), payload.get("filename", "")
                )
                cancel = threading.Event()
                thread = threading.Thread(
                    target=self._hub_download_worker,
                    args=(dict(payload), job_id, cancel),
                    name="locitize-model-download",
                    daemon=True,
                )
                self._hub_job_id = job_id
                self._hub_cancel = cancel
                thread.start()
                self._hub_thread = thread
        if refusal is not None:
            # DEFECT-QA-M14-2: stamp the REFUSED request's own job id. Without
            # it the view could not tell this result apart from the terminal
            # result of the transfer that is still running, so a refusal
            # disabled Cancel and hid the progress bar of a live multi-gigabyte
            # download - leaving the user unable to see or stop it.
            refused = dict(payload)
            refused["job_id"] = modelhub.hub_job_id(
                payload.get("repo_id", ""), payload.get("filename", "")
            )
            refused["refused"] = True
            self.result_q.put(
                Result("hub_download_done", False, refused, error=refusal)
            )

    def _do_hub_cancel(self) -> None:
        """Signal the live download to stop. The .partial file is DELETED.

        There is no resume: modelhub.download_verified unlinks the partial as
        soon as it sees the cancel event, so cancelling discards the bytes
        already transferred (see config.py's RESERVED/NOT IMPLEMENTED note on
        models_hub.resume_enabled). Cancel then re-download starts from zero.

        Read under _hub_lock so this can never signal the event belonging to a
        previous job while a new one is being installed.
        """
        with self._hub_lock:
            cancel = self._hub_cancel
        cancel.set()

    def _hub_download_worker(
        self, payload: dict[str, Any], job_id: str, cancel: threading.Event
    ) -> None:
        """Thread E: the blocking network + disk work, off the ops worker.

        Publishes progress and one terminal outcome onto the SAME result_q the
        QTimer pump already drains, so the single-GUI-thread-writer invariant is
        unchanged. Every exception is converted to a failure Result here: nothing
        may cross this thread boundary.
        """
        def on_progress(progress: Any) -> None:
            self.result_q.put(
                Result(
                    "hub_download_progress",
                    True,
                    {
                        "job_id": job_id,
                        "repo_id": payload.get("repo_id", ""),
                        "filename": payload.get("filename", ""),
                        "phase": progress.phase,
                        "bytes_done": progress.bytes_done,
                        "bytes_total": progress.bytes_total,
                        "rate_bps": progress.rate_bps,
                        "eta_s": progress.eta_s,
                    },
                )
            )

        try:
            outcome = self._hub().download(
                payload.get("repo_id", ""),
                payload.get("filename", ""),
                expected_sha256=payload.get("sha256"),
                verification=payload.get("verification", "none"),
                size_bytes=payload.get("size_bytes"),
                confirm_unverified=bool(payload.get("confirm_unverified")),
                confirmed_exceeds=bool(payload.get("confirmed_exceeds")),
                cancel_event=cancel,
                progress=on_progress,
            )
        except Exception as exc:  # noqa: BLE001 - boundary: never raise off-thread
            self.result_q.put(
                Result("hub_download_done", False, {"job_id": job_id},
                       error=f"unexpected error: {exc}")
            )
            return

        done: dict[str, Any] = {
            "job_id": job_id,
            "repo_id": payload.get("repo_id", ""),
            "filename": payload.get("filename", ""),
            "path": str(outcome.path) if outcome.path else "",
            "sha256": outcome.sha256,
            "verification": outcome.verification,
            "cancelled": outcome.cancelled,
            "registered": False,
            "model_id": None,
        }
        if not outcome.ok:
            if outcome.cancelled:
                # Worded to match what actually happens. Architecture M14.14.3
                # specifies a resumable .partial plus sidecar; that is NOT built
                # yet, so promising a resume here would be a lie the user finds
                # out about the second time they press Download.
                done["message"] = (
                    "Cancelled. The incomplete file was deleted; pressing "
                    "Download starts again from the beginning."
                )
            self.result_q.put(Result("hub_download_done", False, done,
                                     error=outcome.error))
            return
        if payload.get("register"):
            self._hub_register(done, outcome)
        # M22: give the shared model store a name for the finished file.
        # A hardlink when possible (zero bytes), silently skipped when the
        # store is on another volume or missing - never a second copy and
        # never a failure after a download that already succeeded.
        try:
            import setup_env

            if done.get("path") and setup_env.model_store_path().is_dir():
                setup_env.place_into_models_dir(
                    done["path"], setup_env.model_store_path()
                )
        except Exception:  # noqa: BLE001 - cosmetic convenience only
            pass
        self.result_q.put(Result("hub_download_done", True, done))

    def _hub_register(self, done: dict[str, Any], outcome: Any) -> None:
        """Write the finished download into models.yaml through M13's writer.

        A registry failure NEVER deletes the file the user just waited an hour
        for: the miss is reported and the Register button stays available, which
        is the whole reason this is a separate step from the download itself.
        """
        import datetime

        import modelhub
        from config import RegistryWriteError, append_model_entry

        # Start from "not registered" so every exit path below, including the
        # failure path, leaves the caller an unambiguous answer.
        done.setdefault("registered", False)
        model_id = modelhub.registry_id_for(done["filename"])
        quant = modelhub.quant_from_filename(done["filename"])
        when = datetime.date.today().isoformat()
        try:
            append_model_entry(
                self._settings.data_dir,
                model_id,
                f"{Path(done['filename']).stem}"
                + (f" ({quant})" if quant else ""),
                done["path"],
                description=f"Downloaded from HuggingFace repository {done['repo_id']}.",
                context_size=modelhub.DEFAULT_CONTEXT_SIZE,
                gpu_layers=modelhub.DEFAULT_GPU_LAYERS,
                quantization=quant,
                notes=modelhub.registry_notes(
                    done["repo_id"], done["filename"], outcome.verification, when
                ),
                sha256=outcome.sha256,
            )
        except (ValueError, RegistryWriteError) as exc:
            # DEFECT-QA-M14-4 (H11) was originally fixed here, by wrapping
            # whatever the writer raised in a sentence with a next step. Under
            # DEC-M14-11 that wrapper is GONE rather than duplicated: config's
            # one registry chokepoint now guarantees str(exc) is already a
            # finished sentence naming the real path and a next step, and no
            # call site outside the chokepoint may catch an OSError from a
            # registry write and compose its own diagnostic. What is added here
            # is only the part config cannot know - that a downloaded file was
            # kept, where it is, and which button retries the registration.
            done["register_error"] = (
                f"{str(exc).strip()} The model itself was downloaded and kept at "
                f"{done.get('path', '')}, so nothing was lost: press Register on "
                f"the Models page once the cause above is cleared."
            )
            return
        done["registered"] = True
        done["model_id"] = model_id
        try:
            self.refresh_models()
        except Exception:  # noqa: BLE001 - the write succeeded; a reload miss is minor
            pass

    def _do_remember_chat_ui(self, choice: str) -> None:
        """Persist the chat-UI choice via the targeted atomic write_chat_ui (M9.3)."""
        from config import write_chat_ui

        try:
            write_chat_ui(self._settings.data_dir, choice)
            self._settings.chat.preferred_ui = choice  # reflect in the live session
        except (ValueError, OSError) as exc:
            # Non-fatal: the UI was still opened; report the persistence miss only.
            self.result_q.put(
                Result("chat", False, {}, error=f"could not save preference: {exc}")
            )

    def _do_launch_harness(self, choice: str, project_dir: str, remember: bool) -> None:
        """Wire the chosen coding harness at the running model, then spawn it
        in a new terminal rooted at project_dir (owner request 2026-08-21).

        Runs on the ops worker: no model running -> honest error Result, same
        discipline as open_chat/_open_llamacpp. All file writes and the
        Popen/HTTP-server start happen here, never on the UI thread.
        """
        import harness_launch

        if self._active_port is None or self._controller.running_model_id is None:
            self.result_q.put(
                Result(
                    "launch_harness",
                    False,
                    {},
                    error="no model running - start one first",
                )
            )
            return
        if not project_dir:
            self.result_q.put(
                Result("launch_harness", False, {}, error="no project folder chosen")
            )
            return
        if harness_launch.detect_executable(choice) is None:
            self.result_q.put(
                Result(
                    "launch_harness",
                    False,
                    {},
                    error=f"{choice} was not found on PATH - install it first",
                )
            )
            return

        model_id = self._controller.running_model_id
        port = self._active_port
        note = ""

        try:
            from session_launch import build_local_launch
            registered = self._registry.get(model_id)
            argv, env = build_local_launch(
                choice, project_dir, model_id, port,
                context_size=registered.context_size if registered else None,
            )
            note = "Local model routing applies to this terminal only."

            harness_launch.spawn_in_terminal(
                argv, project_dir, env, title=f"{choice} - LOCITIZE ({model_id})"
            )
            self._controller._session_reserved_model = model_id
            self._local_session_launches = getattr(self, "_local_session_launches", 0) + 1
        except (OSError, ValueError) as exc:
            self.result_q.put(
                Result("launch_harness", False, {}, error=f"could not launch {choice}: {exc}")
            )
            return

        if remember and project_dir != self._settings.chat_harness.last_project_dir:
            from config import write_chat_harness_dir

            try:
                write_chat_harness_dir(self._settings.data_dir, project_dir)
                self._settings.chat_harness.last_project_dir = project_dir
            except (ValueError, OSError):
                pass  # non-fatal: the harness was still launched

        self.result_q.put(
            Result(
                "launch_harness",
                True,
                {"choice": choice, "project_dir": project_dir, "note": note},
            )
        )

    def _do_install_harness(self, harness: str) -> None:
        """Install the chosen coding harness CLI (owner request 2026-08-21:
        zero-friction onboarding - "the user should not have to do
        anything"). Runs on the ops worker: the native installer / winget /
        npm subprocess calls do not belong on the UI thread.

        Deliberately does NOT also write that harness's LOCITIZE provider
        config here: Codex's config.toml and OpenCode's project opencode.json
        are already written fresh on every real _do_launch_harness call (a
        model port and, for OpenCode, a project folder are needed to build
        the config, neither of which exists yet at install time), and Claude
        Code needs no config file at all. Installing just gets the binary
        onto PATH; the existing launch path configures it the first time it
        is actually used.
        """
        import harness_launch

        try:
            ok, message = harness_launch.install_harness(harness)
        except ValueError as exc:
            self.result_q.put(Result("install_harness", False, {}, error=str(exc)))
            return
        self.result_q.put(
            Result(
                "install_harness",
                ok,
                {"harness": harness, "message": message},
                error=None if ok else message,
            )
        )

    def _do_start_openwebui(self) -> None:
        """Start Open WebUI on the shared manager; open it, or degrade honestly (M9.3).

        Runs on the ops worker (the start can block for the readiness timeout during
        the first-run DB migration). Uses the injected start callable so this
        controller adds no subprocess code of its own. On failure it degrades to the
        built-in llama.cpp UI so the owner still gets chat (Architecture M9.5).
        """
        if self._openwebui_start_fn is None:
            self._open_llamacpp(
                reason="Open WebUI cannot be started here; opening the built-in UI"
            )
            return
        try:
            ok = self._openwebui_start_fn()
        except Exception as exc:  # noqa: BLE001 - boundary: honest degrade, never raise
            ok = False
            self.result_q.put(
                Result("chat", False, {}, error=f"Open WebUI start failed: {exc}")
            )
        if ok:
            self._open_openwebui()
        else:
            self._open_llamacpp(
                reason="Open WebUI did not become ready; opening the built-in UI"
            )

    def _do_start_assistant(self, voice: str, speak: bool) -> None:
        """Start the assistant session via the injected builder; marshal the outcome.

        Runs on the ops worker (Thread B), so blocking service startup never freezes
        the UI (M10.2). Builds the three marshalling callbacks (each pushes a Result
        onto result_q), calls assistant_start_fn(events, voice, speak) which returns
        an AssistantHandle, records the resolved model port (so the monitor animates
        during the conversation), and pushes assistant_started. A factory error
        degrades to an honest assistant_error Result -- never a raise across the
        boundary, never a fabricated ready state.
        """
        if self._assistant_start_fn is None:
            self.result_q.put(
                Result("assistant_error", False, {}, error="assistant is not available")
            )
            return
        if self._assistant_handle is not None:
            self.result_q.put(
                Result(
                    "assistant_error",
                    False,
                    {},
                    error="an assistant session is already running",
                )
            )
            return
        events = AssistantEvents(
            on_state=lambda s: self._push_assistant_state(s),
            on_user=lambda t: self.result_q.put(
                Result("assistant_user", True, {"text": t})
            ),
            on_reply=lambda line: self.result_q.put(
                Result("assistant_reply", True, {"line": line})
            ),
        )
        try:
            handle = self._assistant_start_fn(events, voice, speak)
        except Exception as exc:  # noqa: BLE001 - boundary: honest error, never a raise
            self._assistant_handle = None
            self.result_q.put(
                Result(
                    "assistant_error",
                    False,
                    {},
                    error=f"could not start assistant: {exc}",
                )
            )
            return
        self._assistant_handle = handle
        port = getattr(handle, "port", None)
        if port is not None:
            # Tie the monitor (Thread C) to the assistant's chat model so Panel 4's
            # tokens/s + context fill animate during the spoken conversation (M10.2).
            self._active_port = port
            self._metrics_available = True
        # Include the existing authoritative lifecycle snapshot so every view can
        # immediately render the assistant-started model and enable Chat without
        # guessing which configured model the launcher selected.
        self.result_q.put(
            Result(
                "assistant_started",
                True,
                {"port": port, **self._running_snapshot_payload()},
            )
        )

    def _do_end_assistant(self) -> None:
        """Stop the live assistant session cleanly and reset the panel (M10.4).

        handle.stop() puts the STOP sentinel (unblocking the session thread), joins it
        bounded, and reaps the assistant's whisper/Kokoro children on the shared
        manager. The chat model stays running (still reachable via Chat / the monitor).
        """
        handle = self._assistant_handle
        self._assistant_handle = None
        self._assistant_ui_state = "idle"
        if handle is not None:
            try:
                handle.stop()
            except Exception as exc:  # noqa: BLE001 - boundary: report, never raise
                self.result_q.put(
                    Result(
                        "assistant_error",
                        False,
                        {},
                        error=f"error ending assistant session: {exc}",
                    )
                )
        self.result_q.put(Result("assistant_ended", True, {}))

    def _do_describe(self, path: str, prompt: str) -> None:
        """Describe an image via the injected describe_fn; marshal the answer/remedy."""
        if self._describe_fn is None:
            self.result_q.put(
                Result("describe", False, {}, error="vision is not available")
            )
            return
        if not path:
            self.result_q.put(
                Result("describe", False, {}, error="pick an image first")
            )
            return
        try:
            ok, answer = self._describe_fn(path, prompt)
        except Exception as exc:  # noqa: BLE001 - boundary guard
            self.result_q.put(
                Result("describe", False, {}, error=f"describe failed: {exc}")
            )
            return
        self.result_q.put(
            Result(
                "describe",
                bool(ok),
                {"answer": answer if ok else ""},
                error=None if ok else answer,
            )
        )

    def _do_memory_search(self, query: str) -> None:
        """Search past conversations read-only via the injected memory_search_fn (M8.3)."""
        if self._memory_search_fn is None:
            self.result_q.put(
                Result("memory_search", False, {}, error="memory is not available")
            )
            return
        try:
            hits = self._memory_search_fn(query)
        except Exception as exc:  # noqa: BLE001 - boundary guard
            self.result_q.put(
                Result("memory_search", False, {}, error=f"memory search failed: {exc}")
            )
            return
        self.result_q.put(
            Result("memory_search", True, {"hits": list(hits), "query": query})
        )

    def _do_benchmark(self, model_id: str) -> None:
        """Run a single-model benchmark via the injected benchmark_fn; refresh the score."""
        if self._benchmark_fn is None:
            self.result_q.put(
                Result("benchmark", False, {}, error="benchmark is not available")
            )
            return
        try:
            ok, detail, score = self._benchmark_fn(model_id)
        except Exception as exc:  # noqa: BLE001 - boundary guard
            self.result_q.put(
                Result(
                    "benchmark",
                    False,
                    {"model_id": model_id},
                    error=f"benchmark failed: {exc}",
                )
            )
            return
        self.result_q.put(
            Result(
                "benchmark",
                bool(ok),
                {
                    "model_id": model_id,
                    "score": score,
                    "score_display": format_score(score),
                    "detail": detail,
                    # Throughput lives in benchmark_results.jsonl for registered
                    # and discovered models alike, so both survive a restart.
                    "score_persisted": True,
                    "note": "",
                },
                error=None if ok else detail,
            )
        )

    def _do_start(self, model_id: str, reasoning: dict | None = None) -> None:
        """Start (or switch to) model_id via the existing ModelController.

        `reasoning`, when set, is forwarded as a spec-builder keyword exactly as
        the benchmark runner forwards its sweep overrides - the controller's
        **spec_kwargs already reach build_start_spec, so no new plumbing exists
        between here and the argv.
        """
        from services import ServiceStatus

        overrides = {"reasoning": reasoning} if reasoning else {}
        try:
            if self._controller.running_model_id is None:
                status = self._controller.start(model_id, **overrides)
            else:
                status = self._controller.switch(model_id, **overrides)
        except ValueError as exc:
            # Config guard (unknown id, missing path/location): honest remedy. A
            # failed start/switch may have left a DIFFERENT model running (a failed
            # switch stops the old one), so report the controller's real state (O-2).
            running = self._controller.running_model_id
            self._active_port = self._resolve_running_port(running) if running else None
            self.result_q.put(
                Result(
                    "start",
                    False,
                    {"model_id": model_id, **self._running_snapshot_payload()},
                    error=str(exc),
                )
            )
            return

        if status is ServiceStatus.RUNNING:
            port = self._resolve_running_port(model_id)
            self._active_port = port
            self._metrics_available = True  # re-probe /metrics on the new model
            self._prev_metrics_sample = None  # counters restart with the server
            self.result_q.put(
                Result(
                    "start",
                    True,
                    {"model_id": model_id, "port": port, **self._running_snapshot_payload()},
                )
            )
        else:
            # Not RUNNING: do not assume the requested model is up. A failed switch
            # leaves nothing (or the old model) running; the snapshot is the truth
            # the model list must reflect instead of a stale RUNNING chip (O-2).
            self._active_port = None
            self.result_q.put(
                Result(
                    "start",
                    False,
                    {"model_id": model_id, **self._running_snapshot_payload()},
                    error=_STATUS_REMEDY.get(status.value, status.value),
                )
            )

    def _do_stop(self) -> None:
        if getattr(self._controller, "_session_reserved_model", None):
            self.result_q.put(Result("stop", False, {}, error="Release the model on Sessions before stopping it."))
            return
        from services import ServiceStatus

        status = self._controller.stop()
        self._active_port = None
        # Reflect the controller's real post-stop state (O-2): a clean stop leaves
        # nothing running; the snapshot confirms it rather than the GUI guessing.
        if status in (ServiceStatus.STOPPED,):
            self.result_q.put(Result("stop", True, self._running_snapshot_payload()))
        else:
            self.result_q.put(
                Result(
                    "stop",
                    False,
                    self._running_snapshot_payload(),
                    error=_STATUS_REMEDY.get(status.value, status.value),
                )
            )

    def _do_free_gpu(self) -> None:
        if getattr(self._controller, "_session_reserved_model", None):
            self.result_q.put(Result("free_gpu", False, {}, error="Release the model on Sessions before freeing GPU memory."))
            return
        """Stop the supervised model, then clear LOCITIZE's own orphaned GPU
        servers, and report the result (M17.8).

        Beyond the plain stop this also terminates llama-servers the supervisor
        lost track of - a crashed session, or a measurement probe - which the
        model-stop path alone cannot reach. Classification and the kill list come
        from gpu_ledger, whose freeable_pids returns ONLY processes whose exe
        lives under LOCITIZE's bin/: a foreign app (LM Studio, a game) is counted
        in the report so the user sees it, but is never a termination candidate.
        """
        import gpu_ledger
        from pathlib import Path

        # 1. Stop the supervised model cleanly, if one is running. A stop error
        #    must not abort the orphan sweep - freeing VRAM is the whole point.
        try:
            self._controller.stop()
        except Exception:  # noqa: BLE001 - honest degrade, never raise off-thread
            pass
        self._active_port = None

        bin_dir = Path(self._settings.data_dir) / "bin"

        def scan() -> list:
            return gpu_ledger.mark_ours(
                gpu_ledger.parse_compute_apps(gpu_ledger.query_compute_apps()),
                bin_dir=bin_dir,
            )

        # 2. Sweep any LOCITIZE-owned processes still on the card (orphans).
        results = gpu_ledger.terminate_pids(gpu_ledger.freeable_pids(scan()))
        freed = sum(1 for ok in results.values() if ok)

        # 3. Re-read the GPU for an honest after-state to report.
        after = scan()
        vram_free_mb: float | None = None
        if self._gpu_provider is not None:
            try:
                gpus = self._gpu_provider.gpus()
            except Exception:  # noqa: BLE001
                gpus = None
            if gpus:
                vram_free_mb = max(g.vram_free_mb for g in gpus)
        others = [p.short_name() for p in after if not p.is_locitize]
        self.result_q.put(
            Result(
                "gpu_free",
                True,
                {
                    "freed": freed,
                    "remaining_ours": sum(1 for p in after if p.is_locitize),
                    "others": others,
                    "vram_free_mb": vram_free_mb,
                    **self._running_snapshot_payload(),
                },
            )
        )

    def _running_snapshot_payload(self) -> dict[str, Any]:
        """Authoritative {running_model_id, running_port} from the controller.

        O-2 fix: the presentation layer must render what is ACTUALLY running, not an
        optimistic assumption after a start/switch/stop. Reading the controller's
        own snapshot keeps the model list and status chip honest even when an op
        failed part-way (e.g. a switch that stopped the old model then failed to
        start the new one). Never raises off-thread: any error degrades to "nothing
        running".
        """
        try:
            snapshot = self._controller.snapshot()
        except Exception:  # noqa: BLE001 - boundary: honest degrade, never raise
            return {"running_model_id": None, "running_port": None}
        for service in snapshot.get("services", []):
            if service.get("status") == "RUNNING":
                return {
                    "running_model_id": service.get("model_id"),
                    "running_port": service.get("port"),
                }
        return {"running_model_id": None, "running_port": None}

    def _do_whisper_toggle(self) -> None:
        from services import ServiceStatus

        try:
            if self._whisper.is_running():
                self._whisper.stop()
                self.result_q.put(Result("whisper", True, {"running": False}))
            else:
                status = self._whisper.start()
                running = status is ServiceStatus.RUNNING
                self.result_q.put(
                    Result(
                        "whisper",
                        running,
                        {"running": running},
                        error=None if running else "whisper-server did not start; see logs",
                    )
                )
        except ValueError as exc:
            # whisper binary/model path not configured yet.
            self.result_q.put(Result("whisper", False, {"running": False}, error=str(exc)))

    def _do_listen(self, seconds: float) -> None:
        """Run the existing --listen capture, streaming each segment to the pump.

        The actual whisper-stream lifecycle and the M3 deduplicated + VAD-gated
        transcript pipeline are supplied by the injected listen_fn (the launcher's,
        which reuses that pipeline unchanged); this worker only marshals each line
        onto result_q. No new speech-to-text code is added here.
        """
        if self._listen_fn is None:
            self.result_q.put(
                Result("listen", False, {}, error="listen is not available")
            )
            return

        def emit(line: str) -> None:
            self.result_q.put(Result("transcript", True, {"line": line}))

        try:
            ok = self._listen_fn(seconds, emit)
        except Exception as exc:  # noqa: BLE001 - boundary guard
            self.result_q.put(Result("listen", False, {}, error=f"listen failed: {exc}"))
            return
        self.result_q.put(
            Result("listen", bool(ok), {}, error=None if ok else "listen did not complete cleanly")
        )

    def _do_speak(self, text: str, voice: str) -> None:
        """Speak one line via the injected speak_fn (real Kokoro synth+play).

        All TTS effects (service start on the shared manager, HTTP synthesize,
        winsound playback) live in the injected speak_fn, so this worker only
        marshals the outcome onto result_q. A missing speak_fn or empty text is an
        honest error Result, never a crash or a dead button.
        """
        if self._speak_fn is None:
            self.result_q.put(
                Result("speak", False, {}, error="text-to-speech is not available")
            )
            return
        if not text.strip():
            self.result_q.put(
                Result("speak", False, {}, error="enter some text to speak")
            )
            return
        chosen = voice or self._settings.tts.voice
        try:
            ok, detail = self._speak_fn(text, chosen)
        except Exception as exc:  # noqa: BLE001 - boundary guard
            self.result_q.put(Result("speak", False, {"voice": chosen}, error=f"speak failed: {exc}"))
            return
        self.result_q.put(
            Result("speak", bool(ok), {"voice": chosen}, error=None if ok else detail)
        )

    def _do_audition(self) -> None:
        """Speak the sample sentence in every on-disk voice, reporting each.

        Reuses the same injected speak_fn per voice (so no orphan: the service is
        started once on the shared manager and reused), emitting a per-voice line
        the pump renders, then a final outcome Result.
        """
        if self._speak_fn is None:
            self.result_q.put(
                Result("audition", False, {}, error="text-to-speech is not available")
            )
            return
        voices = self.available_voices()
        if not voices:
            self.result_q.put(
                Result("audition", False, {}, error="no voices configured")
            )
            return
        sentence = self._settings.tts.sample_sentence
        for voice in voices:
            self.result_q.put(Result("audition_voice", True, {"voice": voice}))
            try:
                ok, detail = self._speak_fn(sentence, voice)
            except Exception as exc:  # noqa: BLE001 - boundary guard
                self.result_q.put(
                    Result("audition", False, {"voice": voice}, error=f"audition failed: {exc}")
                )
                return
            if not ok:
                self.result_q.put(Result("audition", False, {"voice": voice}, error=detail))
                return
        self.result_q.put(Result("audition", True, {"count": len(voices)}))

    def _do_save_edits(self, payload: dict[str, Any]) -> None:
        """Persist gpu_layers/context_size via config.write_model_fields, then reload.

        Applies on next start (G3): after a successful write the in-memory Model is
        updated so the next start_spec picks up the new values, but a running model
        keeps the values it launched with.
        """
        from config import RegistryWriteError, write_model_fields

        model_id = payload["model_id"]
        gpu = payload["gpu_layers"]
        ctx = payload["context_size"]
        try:
            write_model_fields(self._settings.data_dir, model_id, gpu, ctx)
        # DEC-M14-11: RegistryWriteError already carries the real path and a
        # next step, so str(exc) below is a finished sentence rather than the
        # bare "[Errno 2] ..." NEW-QA-M14-9 measured on this very surface. OSError
        # is deliberately NOT caught here: every filesystem failure of a registry
        # write now arrives translated, and a call site that caught OSError would
        # be free to invent its own message again.
        except (ValueError, RegistryWriteError) as exc:
            self.result_q.put(
                Result("save_edits", False, {"model_id": model_id}, error=str(exc))
            )
            return
        # Keep the in-memory registry consistent with the file so "applies on next
        # start" holds within this session without a full reload.
        model = self._registry.get(model_id)
        if model is not None:
            model.gpu_layers = gpu
            model.context_size = ctx
        self.result_q.put(
            Result(
                "save_edits",
                True,
                {"model_id": model_id, "gpu_layers": gpu, "context_size": ctx},
            )
        )

    def _do_save_identity(self, payload: dict[str, Any]) -> None:
        """Persist an id/name rename via config.write_model_identity, then reload.

        A full registry reload (not an in-place field mutation) is required here,
        unlike _do_save_edits: the registry's lookup dict is keyed by id, so a
        changed id needs a fresh dict, not a mutated value under the old key. Runs
        the same Config.load path refresh_models uses, so a bad concurrent edit to
        models.yaml is caught the same way (registry left untouched, error surfaced)
        rather than half-applied.
        """
        from config import Config, RegistryWriteError, write_model_identity

        model_id = payload["model_id"]
        new_id = payload["new_id"]
        new_name = payload["new_name"]
        try:
            write_model_identity(self._settings.data_dir, model_id, new_id, new_name)
        # Same contract as _do_save_edits above (DEC-M14-11).
        except (ValueError, RegistryWriteError) as exc:
            self.result_q.put(
                Result("save_identity", False, {"model_id": model_id}, error=str(exc))
            )
            return
        # Owner request 2026-08-21: Config.load() with no argument re-resolves
        # the PRODUCTION data root from the environment/BASE_DIR from scratch,
        # ignoring wherever self._settings was actually built from - invisible
        # in real usage (they're the same path there) but wrong in principle,
        # and it silently reloaded the wrong registry entirely in a test that
        # pointed self._settings at a tmp_path fixture. Passing the SAME
        # data_dir this controller's settings came from keeps this read-your-
        # own-write instead of read-whatever-the-environment-says-right-now.
        _settings, models, issues = Config.load(self._settings.data_dir)
        errors = [i for i in issues if getattr(i, "severity", "") == "ERROR"]
        if errors:
            self.result_q.put(
                Result(
                    "save_identity",
                    False,
                    {"model_id": model_id},
                    error="; ".join(str(i) for i in errors[:3]),
                )
            )
            return
        self._registry.reload(models)
        self._size_cache.pop(model_id, None)
        self.result_q.put(
            Result(
                "save_identity",
                True,
                {"model_id": model_id, "new_id": new_id, "new_name": new_name},
            )
        )

    def _do_save_capabilities(self, payload: dict[str, Any]) -> None:
        """Persist a capabilities write via config.write_model_capabilities, then
        reload. The id never changes here, but a full Config.load + registry.reload
        (rather than mutating the cached Model in place) keeps this on the exact
        same read-your-own-write guarantee as _do_save_edits/_do_save_identity: the
        in-memory registry always reflects what is actually on disk, not what the
        write call intended."""
        from config import Config, RegistryWriteError, write_model_capabilities

        model_id = payload["model_id"]
        capabilities = payload["capabilities"]
        try:
            write_model_capabilities(self._settings.data_dir, model_id, capabilities)
        except (ValueError, RegistryWriteError) as exc:
            self.result_q.put(
                Result("save_capabilities", False, {"model_id": model_id}, error=str(exc))
            )
            return
        # Owner request 2026-08-21: Config.load() with no argument re-resolves
        # the PRODUCTION data root from the environment/BASE_DIR from scratch,
        # ignoring wherever self._settings was actually built from - invisible
        # in real usage (they're the same path there) but wrong in principle,
        # and it silently reloaded the wrong registry entirely in a test that
        # pointed self._settings at a tmp_path fixture. Passing the SAME
        # data_dir this controller's settings came from keeps this read-your-
        # own-write instead of read-whatever-the-environment-says-right-now.
        _settings, models, issues = Config.load(self._settings.data_dir)
        errors = [i for i in issues if getattr(i, "severity", "") == "ERROR"]
        if errors:
            self.result_q.put(
                Result(
                    "save_capabilities",
                    False,
                    {"model_id": model_id},
                    error="; ".join(str(i) for i in errors[:3]),
                )
            )
            return
        self._registry.reload(models)
        self.result_q.put(
            Result(
                "save_capabilities",
                True,
                {"model_id": model_id, "capabilities": capabilities},
            )
        )

    def _do_autotune_context(self, model_id: str) -> None:
        """Run the context auto-tuner for ONE model on the ops worker (Thread B).

        This is a LONG handler by this controller's standards - several minutes
        of real llama-server starts - which is exactly why it lives here and not
        on the Qt UI thread. Two consequences the design leans on:

        - Progress is published as it happens. Each step puts an
          "autotune_progress" Result on result_q, which the pump renders
          immediately, so the owner sees which context value is being tried
          instead of a window that looks hung. Anything less would be a frozen
          spinner over a three-minute operation.
        - It occupies the single ops worker for its whole duration, so start,
          stop and benchmark queue behind it. That is deliberate: every one of
          those also wants the GPU, and serialising them is what keeps two
          llama-servers from fighting over the same port and VRAM.

        All of the real work (GGUF read, probe search, YaRN maths, rollback)
        lives in autotune.py; this method supplies the three things only the
        controller knows - which model, how to write models.yaml, and how to run
        a trial - and marshals the outcome back as a Result.
        """
        import autotune
        from config import Config, RegistryWriteError, write_model_tuning

        model = self._registry.get(model_id)
        if model is None:
            self.result_q.put(
                Result("autotune", False, {"model_id": model_id},
                       error=f"model '{model_id}' is not in the registry")
            )
            return
        if not model.location:
            self.result_q.put(
                Result("autotune", False, {"model_id": model_id},
                       error="this model has no file location set, so its GGUF "
                             "header cannot be read")
            )
            return

        def emit(line: str) -> None:
            """Publish one progress line to the pump (never blocks, never raises)."""
            self.result_q.put(
                Result("autotune_progress", True, {"model_id": model_id, "line": line})
            )

        def write_tuning(target_id: str, context_size: int, server_args: list[str]) -> None:
            """The one registry write, through config.py's existing chokepoint."""
            write_model_tuning(
                self._settings.data_dir,
                target_id,
                context_size,
                server_args,
                note=(
                    f"context_size set by LOCITIZE auto-tune "
                    f"{_today_stamp()}: real trial starts on this machine"
                ),
            )

        def trial(context_size: int) -> "autotune.Trial":
            """One real start/health-check/clean-shutdown via the existing CLI."""
            # base_dir defaults to autotune.py's own directory, which IS the
            # platform directory holding launcher.py - the same tree this
            # controller was imported from.
            return autotune.run_smoke_trial(
                model_id, context_size, cancel=self._autotune_cancel
            )

        # Publish "an auto-tune owns the ops worker" BEFORE the first trial, so a
        # Stop click during even the first GGUF read is honoured rather than
        # silently dropped, and clear it in the finally so a crash cannot leave
        # the UI showing a cancellable run that no longer exists.
        self._autotune_model_id = model_id
        self._autotune_running = True
        try:
            outcome = autotune.autotune_model_context(
                model_id=model_id,
                location=model.location,
                current_context=int(model.context_size),
                current_server_args=list(model.server_args or []),
                write_tuning=write_tuning,
                trial_fn=trial,
                progress=emit,
                cancel=self._autotune_cancel,
            )
        except (ValueError, RegistryWriteError) as exc:
            # Same contract as every other registry-writing handler
            # (DEC-M14-11): str(exc) is already a finished sentence.
            self.result_q.put(
                Result("autotune", False, {"model_id": model_id}, error=str(exc))
            )
            return
        finally:
            self._autotune_running = False
            self._autotune_model_id = None

        # Re-read models.yaml so the in-memory registry matches what is really on
        # disk, exactly as _do_save_capabilities does. Passing this controller's
        # OWN data_dir keeps it a read-your-own-write rather than a re-resolve of
        # whatever the environment currently points at.
        _settings, models, issues = Config.load(self._settings.data_dir)
        errors = [i for i in issues if getattr(i, "severity", "") == "ERROR"]
        if not errors:
            self._registry.reload(models)

        self.result_q.put(
            Result(
                "autotune",
                outcome.ok,
                {
                    "model_id": model_id,
                    "native_context": outcome.native_context,
                    "architecture": outcome.architecture,
                    "previous_context": outcome.previous_context,
                    "chosen_context": outcome.chosen_context,
                    "first_failure": outcome.first_failure,
                    "yarn_applied": outcome.yarn_applied,
                    "rope_scale": outcome.rope_scale,
                    "server_args": outcome.server_args,
                    "trials": [
                        {"context_size": t.context_size, "ok": t.ok, "reason": t.reason}
                        for t in outcome.trials
                    ],
                    "detail": outcome.detail,
                    "canceled": outcome.canceled,
                },
                # A cancellation is not an error and must not be reported as one:
                # `error` stays None so nothing downstream renders a red failure
                # for something the owner deliberately did. `canceled` in the
                # payload is what the UI keys its own wording off.
                error=(
                    None
                    if (outcome.ok or outcome.canceled)
                    else (outcome.detail or "auto-tune failed")
                ),
            )
        )

    def _do_delete_model(self, payload: dict[str, Any]) -> None:
        """Delete a model's file(s) from disk, then remove its models.yaml row.

        Owner request 2026-08-21. Files first, registry second: if a file
        delete fails (still open, permission denied, whatever), the registry
        row is left exactly as it was - a partially-applied delete (row gone
        but the file still sitting on disk, or the reverse) is worse than an
        honest failure the owner can retry. A file another row's location or
        mmproj also points at is never unlinked - only the row referencing it
        goes; the shared file stays for its remaining owner.
        """
        from pathlib import Path as _Path

        from config import Config, RegistryWriteError, remove_model_entry

        model_id = payload["model_id"]
        model = self._registry.get(model_id)
        if model is None:
            self.result_q.put(
                Result(
                    "delete_model", False, {"model_id": model_id},
                    error=f"model '{model_id}' is not registered",
                )
            )
            return
        # Re-checked here (not just in delete_model()'s UI-thread guard): a
        # Start press could land on the ops worker between the confirm click
        # and this method actually running.
        if self._controller.running_model_id == model_id:
            self.result_q.put(
                Result(
                    "delete_model", False, {"model_id": model_id},
                    error="stop this model before deleting it",
                )
            )
            return

        other_paths = {
            str(_Path(p).resolve())
            for m in self._registry.all()
            if m.id != model_id
            for p in (m.location, m.mmproj)
            if p
        }

        deleted, skipped_shared, missing = [], [], []
        for raw_path in (p for p in (model.location, model.mmproj) if p):
            path = _Path(raw_path)
            if str(path.resolve()) in other_paths:
                skipped_shared.append(raw_path)
                continue
            try:
                path.unlink()
                deleted.append(raw_path)
            except FileNotFoundError:
                missing.append(raw_path)
            except OSError as exc:
                self.result_q.put(
                    Result(
                        "delete_model", False, {"model_id": model_id},
                        error=(
                            f"could not delete {raw_path}: "
                            f"{exc.strerror or exc}. Close any program using "
                            f"it, then try again. The model list entry was "
                            f"left in place."
                        ),
                    )
                )
                return

        try:
            remove_model_entry(self._settings.data_dir, model_id)
        except (ValueError, RegistryWriteError) as exc:
            self.result_q.put(
                Result("delete_model", False, {"model_id": model_id}, error=str(exc))
            )
            return
        # Owner request 2026-08-21: Config.load() with no argument re-resolves
        # the PRODUCTION data root from the environment/BASE_DIR from scratch,
        # ignoring wherever self._settings was actually built from - invisible
        # in real usage (they're the same path there) but wrong in principle,
        # and it silently reloaded the wrong registry entirely in a test that
        # pointed self._settings at a tmp_path fixture. Passing the SAME
        # data_dir this controller's settings came from keeps this read-your-
        # own-write instead of read-whatever-the-environment-says-right-now.
        _settings, models, issues = Config.load(self._settings.data_dir)
        errors = [i for i in issues if getattr(i, "severity", "") == "ERROR"]
        if errors:
            self.result_q.put(
                Result(
                    "delete_model", False, {"model_id": model_id},
                    error="; ".join(str(i) for i in errors[:3]),
                )
            )
            return
        self._registry.reload(models)
        self._size_cache.pop(model_id, None)
        self.result_q.put(
            Result(
                "delete_model", True,
                {
                    "model_id": model_id,
                    "deleted_files": deleted,
                    "skipped_shared_files": skipped_shared,
                    "missing_files": missing,
                },
            )
        )

    def _resolve_running_port(self, model_id: str) -> int | None:
        """Read the resolved loopback port of the running model from the snapshot.

        The monitor never calls the controller; it reads this captured port. Uses
        the controller's own honest snapshot rather than reaching into internals.
        """
        snapshot = self._controller.snapshot()
        for service in snapshot.get("services", []):
            if service.get("model_id") == model_id and service.get("status") == "RUNNING":
                return service.get("port")
        return None

    # ---- the monitor (Thread C) ------------------------------------------- #

    def _monitor_loop(self) -> None:
        """While a model runs, poll /metrics (+ /slots) and push MetricsSamples.

        Idles cheaply when no model runs or monitoring is disabled. Probes /metrics
        once per model: on an unavailable endpoint it flips metrics_available off
        and stops re-hitting it every cycle (G5), pushing one honest degraded
        sample so the pump can hide Panel 4 with a reason.
        """
        cadence = max(0.1, float(self._settings.gui.monitor_interval_s))
        while not self._stop_event.is_set():
            port = self._active_port
            if port is not None and self._settings.gui.monitor_enabled:
                sample = self._collect_sample(port)
                self.result_q.put(Result("metrics", True, {"sample": sample}))
            # M13: keep the run-active cache warm on this existing background
            # cadence. The GUI thread (window close) reads only the cache, so the
            # outputs-tree walk has to happen somewhere off the Qt thread; this
            # loop already exists and already idles cheaply. No Result is pushed -
            # nothing repaints on this signal, it only has to be fresh when the
            # close path asks. Skipped entirely unless LOCITIZE started the studio.
            if self._finetune_started:
                self._probe_finetune_run_active()
            # M15.7: notice a models.yaml rewritten by someone else (a CLI
            # tune, the ceiling tool, another session) while this window is
            # open. The desktop of 2026-08-25..29 ran four days against a
            # registry that grew 12->33 models underneath it and nothing said
            # so. One stat() per cycle; one status line per change, never a
            # reload behind the user's back - refreshing remains their click.
            self._check_registry_mtime()
            # Owner-observed 2026-09-03: the window showed no running model
            # while one WAS running. Every path that repainted the running
            # model went through a Result this worker had produced, so a model
            # started by something else in this process - the model router,
            # switching because a browser picker changed - was invisible here.
            # The controller is the truth; this asks it, on the cadence that
            # already exists, and publishes only when the answer changes.
            self._check_running_model()
            # Bounded wait so shutdown() interrupts the sleep immediately.
            self._stop_event.wait(cadence)

    def _check_running_model(self) -> None:
        """Publish the controller's real running model when it changes. Never raises.

        Also re-points self._active_port, so the metrics probe follows a model
        the GUI did not start - otherwise the monitor keeps polling the old
        port, or idles because it never had one.

        The first observation always publishes (the remembered value starts at
        _UNOBSERVED, which no real state equals). That is deliberate: a window
        opened while a model is already running must paint the truth rather
        than wait for the state to change.
        """
        try:
            model_id = self._controller.running_model_id
            port = self._controller.running_port
        except Exception:  # noqa: BLE001 - a status read is never a crash
            return
        current = (model_id, port)
        if current == self._last_running_observed:
            return
        self._last_running_observed = current
        self._active_port = port
        self.result_q.put(
            Result(
                "running_model",
                True,
                {"running_model_id": model_id, "running_port": port},
            )
        )

    def _check_registry_mtime(self) -> None:
        """Publish one notice when models.yaml changes on disk. Never raises."""
        try:
            from config import MODELS_FILE

            target = Path(self._settings.data_dir) / MODELS_FILE
            mtime = target.stat().st_mtime if target.is_file() else None
        except Exception:  # noqa: BLE001 - a stat failure is not a crash
            return
        previous = getattr(self, "_registry_mtime", None)
        if previous is None:
            self._registry_mtime = mtime
            return
        if mtime is not None and mtime != previous:
            self._registry_mtime = mtime
            self.result_q.put(
                Result(
                    "status_line",
                    True,
                    {
                        "text": (
                            "models.yaml changed on disk - press Refresh on the "
                            "Models page to load the current registry"
                        )
                    },
                )
            )

    def _derive_rates(self, sample: MetricsSample) -> None:
        """Fill gen/prompt tok/s from cumulative counters when the direct rate
        gauges are absent (newer llama.cpp builds, M18.10).

        The seconds counters advance only while the server is actually working,
        so delta(tokens)/delta(seconds) between two polls is the true speed of
        whatever generation happened in between - a measurement, not a gauge.
        With no previous poll (first sample after a start) the session average
        total/seconds is used, which is equally measured. No activity since the
        last poll leaves the previous displayed value untouched by publishing
        None only when nothing was ever measured.
        """
        prev = self._prev_metrics_sample
        self._prev_metrics_sample = sample

        def rate(tokens, seconds, prev_tokens, prev_seconds, current):
            if current is not None:
                return current  # the old direct gauge exists; trust it
            if tokens is None or seconds is None or seconds <= 0:
                return None
            if (
                prev_tokens is not None
                and prev_seconds is not None
                and seconds > prev_seconds
                and tokens >= prev_tokens
            ):
                return (tokens - prev_tokens) / (seconds - prev_seconds)
            return tokens / seconds  # session average (first poll, or restart)

        prev_gen_tokens = prev.gen_tokens_total if prev else None
        prev_gen_seconds = prev.gen_seconds_total if prev else None
        prev_prompt_tokens = prev.prompt_tokens_total if prev else None
        prev_prompt_seconds = prev.prompt_seconds_total if prev else None
        sample.gen_tokens_s = rate(
            sample.gen_tokens_total, sample.gen_seconds_total,
            prev_gen_tokens, prev_gen_seconds, sample.gen_tokens_s,
        )
        sample.prompt_tokens_s = rate(
            sample.prompt_tokens_total, sample.prompt_seconds_total,
            prev_prompt_tokens, prev_prompt_seconds, sample.prompt_tokens_s,
        )

    def _collect_sample(self, port: int) -> MetricsSample:
        """Fetch and parse one monitor sample, degrading honestly on absence."""
        sample: MetricsSample
        if self._metrics_available:
            body = self._fetch(port, "/metrics")
            if body is None:
                # Probe-once degradation: the endpoint is absent for this build;
                # stop retrying it and report unavailable rather than fabricating.
                self._metrics_available = False
                sample = MetricsSample(metrics_available=False, sampled_at=time.time())
            else:
                sample = parse_metrics(body)
                self._derive_rates(sample)
        else:
            sample = MetricsSample(metrics_available=False, sampled_at=time.time())

        # /slots is optional enrichment; a failure here never affects the metrics
        # view. Absent/disabled -> no slots, no error.
        slots_body = self._fetch(port, "/slots")
        if slots_body:
            try:
                slots = parse_slots(json.loads(slots_body))
                if slots.available:
                    sample.slots = slots.slots
                    # /slots is the authoritative context-window source when present;
                    # fill n_ctx and, if this build exposes it, the live token count
                    # (a stripped build leaves both None and the monitor degrades).
                    if slots.n_ctx is not None:
                        sample.n_ctx = slots.n_ctx
                    if sample.kv_cache_tokens is None and slots.tokens_used is not None:
                        sample.kv_cache_tokens = slots.tokens_used
                else:
                    sample.slots = None
            except (ValueError, TypeError):
                sample.slots = None
        return sample

    def _loopback_fetch(self, port: int, path: str) -> str | None:
        """GET http://127.0.0.1:<port><path> with a 1s timeout; None if unavailable.

        Uses http.client directly (not urllib), which does NOT follow redirects
        (SEC-M3-1), and connects only to the hardcoded loopback host (Permission
        Matrix section 7). Any error (refused, timeout, non-200, malformed) returns
        None so the monitor degrades rather than crashing.
        """
        import http.client

        conn = None
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
            conn.request("GET", path)
            response = conn.getresponse()
            if response.status != 200:
                return None
            return response.read().decode("utf-8", errors="replace")
        except (OSError, http.client.HTTPException):
            return None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except OSError:
                    pass
