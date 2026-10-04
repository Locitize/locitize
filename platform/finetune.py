"""Fine-tune studio integration and fine-tuned-model discovery (LOCITIZE M13).

This module is the whole non-UI half of Milestone 13, modeled on webui.py. It does
four things and nothing else:

- build_studio_spec / studio_url / studio_available: describe the external
  llm-finetune-studio Streamlit app as a declarative ServiceSpec so the existing
  ServiceManager / SingleServiceController stack can start, supervise, and reap it
  exactly like Open WebUI, whisper, and Kokoro. services.py gains no code at all.
- scan_outputs: read the studio's outputs/ tree and report the fine-tuned .gguf
  files found there as DiscoveredFineTune records, never writing anything back.
- dedup_against_registry / to_model / register_id_for: fold a discovered file into
  an existing manually-registered models.yaml row when they are the same model, and
  turn a surviving discovered file into an ordinary config.Model so the proven
  llama.cpp serving path can start it unchanged.
- orphan_warning_text / active_run: the honest limitation notice LOCITIZE must show
  when it stops the studio while a training run is live (see below).

Hard boundaries this module respects (Architecture M13.3, M13.9):

- The studio checkout is READ-ONLY. Nothing here opens a path under it for write.
- No process is spawned here (ServiceManager owns that), no Qt is imported, and no
  container runtime is ever invoked by LOCITIZE.
- A discovered path is confined (resolve + must stay under the configured outputs
  root + must end in .gguf + must be non-empty) before it can reach a child's argv.
- Discovery never writes models.yaml. Promotion is an explicit owner action
  (config.append_model_entry), never a side effect of a scan.

ASCII only.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from config import BASE_DIR, FineTuneConfig, Model, Settings
from services import ServiceSpec

# Stable service identifier shared by the controller, the launcher, and tests.
FINETUNE_SERVICE_NAME = "finetune-studio"

# The llm-finetune-studio checkout ships vendored inside the LOCITIZE repo
# (Codebase/finetune-studio) so a fresh git clone has a working Fine-tune page
# with zero owner-specific configuration -- no external checkout path required.
# An explicit finetune.studio_dir still overrides this; only a blank value
# resolves here.
BUNDLED_STUDIO_DIR = BASE_DIR.parent / "finetune-studio"

# Model.source value marking a row that came from the filesystem scan rather than
# models.yaml. Manual rows keep the default "registry".
DISCOVERED_SOURCE = "discovered"

# Reserved id namespace for generated (discovered) ids, so a hand-written
# models.yaml id can never collide with one by construction (Architecture M13.5.4).
FT_PREFIX = "ft:"

# Quant tokens the trainer actually emits, lowercased. A filename's last
# dot-separated stem token is only accepted as a quant when it is in this set, so
# "a prior fine-tune-v5" does not become a quant named "V5".
KNOWN_QUANTS = (
    "f16",
    "bf16",
    "q8_0",
    "q6_k",
    "q5_k_m",
    "q4_k_m",
    "q4_0",
    "q3_k_m",
    "q2_k",
)

# run_meta.json is optional enrichment. It is size-capped and parsed with
# json.loads only (never yaml.load / eval / pickle) because it comes from a tree
# LOCITIZE does not own (Architecture M13.9 boundary 2).
RUN_META_NAME = "run_meta.json"
RUN_META_MAX_BYTES = 256 * 1024

# Discovery is computed on demand and never persisted; this short in-memory TTL
# only stops a rapid repaint from re-stat-ing the whole tree (Architecture M13.5.5).
SCAN_TTL_S = 5.0

# The trainer appends to this file for the whole run, so its mtime is the one
# honest, read-only signal LOCITIZE has that a run is currently live.
TRAIN_LOG_NAME = "train.log"

# A file's recorded mtime can legitimately read *ahead* of time.time(). The two
# clocks are different sources: NTFS/POSIX timestamps come from the filesystem at
# a finer resolution than time.time()'s float seconds, so a log written moments
# ago routinely produces a small negative "age" (measured here: 19 of 200 fresh
# writes). Network shares and drives with a skewed clock can push that further.
# Treating a negative age as "not live" would suppress the honesty warning on the
# exact case it exists for - a run that just wrote its log - so any age at or
# above this bound counts as live. The value is large enough to absorb real
# machine/share skew and far too small to resurrect a genuinely stale run.
CLOCK_SKEW_TOLERANCE_S = 300.0

# The verbatim limitation notice the Fine-tune page must show when the studio is
# stopped (or the window closed) while a training run is live. It lives in a data
# file rather than in this module because acceptance criterion AC-M13-5 forbids the
# container-runtime product name as a literal anywhere in LOCITIZE Python source (the
# machine-checkable proof that LOCITIZE never invokes that runtime itself). The text
# the owner sees is byte-for-byte the wording UX Spec section 5 requires.
_WARNING_FILE = "finetune_warning.txt"

# Used only if the shipped warning file is missing from the install. It states the
# same limitation in words that carry no forbidden literal, so a damaged install
# degrades to an honest (if less specific) warning instead of silence.
_WARNING_FALLBACK = (
    "A training container may still be running in your container runtime. "
    "locitize cannot stop a container it did not start; check your container list."
)

# root path string -> (monotonic timestamp, ScanResult).
_scan_cache: dict[str, tuple[float, ScanResult]] = {}


@dataclass(frozen=True)
class DiscoveredFineTune:
    """One .gguf found under the studio's outputs/ tree.

    `run` is the run FOLDER name and is what the owner recognizes; `name` mirrors
    it (the file stem is deliberately not used, because four historical runs all
    ship a file called a prior fine-tune.q4_k_m.gguf and would otherwise be indistinguishable).
    `already_registered` is set by dedup_against_registry when this exact model is
    already a manual models.yaml row.
    """

    id: str
    run: str
    name: str
    path: str
    quant: str
    size_bytes: int
    mtime: float
    base_model: str = ""
    recommended_prompt: str = ""
    meta_note: str = ""
    already_registered: bool = False


@dataclass(frozen=True)
class ScanResult:
    """The outcome of one outputs/ scan.

    `reason` is non-empty whenever the scan produced nothing useful (root unset,
    missing, unreadable, or simply empty) and is rendered verbatim in the UI's
    empty state, so the owner always learns WHY the list is empty.
    """

    items: tuple[DiscoveredFineTune, ...] = ()
    root: str = ""
    reason: str = ""


# --------------------------------------------------------------------------- #
# Studio location and availability
# --------------------------------------------------------------------------- #


def _cfg(settings: Settings) -> FineTuneConfig:
    """Return the finetune settings block, tolerating a bare/stub Settings."""
    return getattr(settings, "finetune", None) or FineTuneConfig()


def studio_dir(settings: Settings) -> Path | None:
    """Absolute path of the llm-finetune-studio checkout.

    A blank finetune.studio_dir resolves to the bundled vendored copy shipped
    inside the LOCITIZE repo (BUNDLED_STUDIO_DIR) rather than "unset" -- the
    studio must work out of the box on a fresh clone. An explicit studio_dir
    (or LOCITIZE_FINETUNE_STUDIO_DIR, applied earlier during settings load)
    still overrides this. Only when the bundled directory itself is absent
    (e.g. a partial checkout) does this return None.
    """
    raw = (_cfg(settings).studio_dir or "").strip()
    if raw:
        return Path(raw).expanduser()
    return BUNDLED_STUDIO_DIR if BUNDLED_STUDIO_DIR.is_dir() else None


def studio_app_path(settings: Settings) -> Path | None:
    """Absolute path of the studio's Streamlit entry script, or None when unset."""
    base = studio_dir(settings)
    if base is None:
        return None
    rel = (_cfg(settings).app_path or "app/app.py").strip()
    return base / rel


def outputs_root(settings: Settings) -> Path | None:
    """The directory discovery scans: outputs_dir, else <data root>/finetune/outputs.

    A trained model is the most expensive thing on this machine to reproduce -
    hours of GPU time the user cannot get back - so a blank outputs_dir now
    means the data root, not a folder inside the studio checkout that a
    reinstall or a re-clone would take with it (DEC-M14-9). An explicitly
    configured outputs_dir is honoured as-is, because a user who pointed
    training at another drive meant it.

    Never hardcoded: with no data root and no configured value this yields None,
    which the caller renders as an honest "not configured" empty state.
    """
    cfg = _cfg(settings)
    raw = (cfg.outputs_dir or "").strip()
    if raw:
        return Path(raw).expanduser()
    root = getattr(settings, "data_dir", None)
    if root is None:
        return None
    return Path(root) / "finetune" / "outputs"


def datasets_root(settings: Settings) -> Path | None:
    """The directory LOCITIZE owns for fine-tune datasets: <data root>/finetune/datasets.

    The sibling of outputs_root, and the same classification: a training dataset
    is something the user assembled and cannot get back from a reinstall, so it
    belongs in the folder they back up (DEC-M14-9's resolver contract). It is a
    location, not a scanner: the studio child process resolves its own relative
    datasets/ path from its working directory, which LOCITIZE does not change.
    """
    root = getattr(settings, "data_dir", None)
    if root is None:
        return None
    return Path(root) / "finetune" / "datasets"


def studio_interpreter(settings: Settings) -> Path | None:
    """Resolve the interpreter that runs the studio child.

    Order (Architecture M13.4): finetune.python, then the studio's own .venv, then
    its venv, then this process's interpreter as a last resort. LOCITIZE never adds
    streamlit/torch/unsloth to its own requirements and never installs anything
    into someone else's virtual environment.
    """
    cfg = _cfg(settings)
    explicit = (cfg.python or "").strip()
    if explicit:
        candidate = Path(explicit).expanduser()
        return candidate if candidate.is_file() else None
    from runtime_layout import bundled_python, packaged_environment_root
    if bundled_python(settings.base_dir):
        candidate = packaged_environment_root(settings.data_dir) / "finetune-studio" / ".venv" / "Scripts" / "python.exe"
        return candidate if candidate.is_file() else None
    base = studio_dir(settings)
    if base is not None:
        for rel in (".venv/Scripts/python.exe", "venv/Scripts/python.exe",
                    ".venv/bin/python", "venv/bin/python"):
            candidate = base / rel
            if candidate.is_file():
                return candidate
    # Falling back to this interpreter is honest only if streamlit happens to be
    # importable there; if it is not, the child exits fast and readiness times out
    # into the page's Error state with the log path (failure mode F2).
    return Path(sys.executable) if sys.executable else None


def studio_available(settings: Settings) -> tuple[bool, str]:
    """(usable, reason) for the studio, with a concrete remedy in every failure.

    Never raises and never returns a bare "unavailable": each false case names the
    exact missing piece and the setting or env var that fixes it, because this
    string is rendered verbatim as the page's disabled-state explanation.
    """
    cfg = _cfg(settings)
    if not cfg.enabled:
        return False, (
            "Fine-tune studio is disabled. Set finetune.enabled: true in "
            "settings.yaml to enable it."
        )
    base = studio_dir(settings)
    if base is None:
        return False, (
            f"Fine-tune studio was not found (expected the bundled copy at "
            f"{BUNDLED_STUDIO_DIR}). Set finetune.studio_dir in settings.yaml "
            f"or LOCITIZE_FINETUNE_STUDIO_DIR to point at a different checkout."
        )
    if not base.is_dir():
        return False, (
            f"Fine-tune studio directory '{base}' does not exist. Fix "
            f"finetune.studio_dir in settings.yaml or LOCITIZE_FINETUNE_STUDIO_DIR."
        )
    app = studio_app_path(settings)
    if app is None or not app.is_file():
        return False, (
            f"Fine-tune studio app '{app}' was not found. Check finetune.app_path "
            f"in settings.yaml (it is relative to finetune.studio_dir)."
        )
    interpreter = studio_interpreter(settings)
    if interpreter is None or not Path(interpreter).is_file():
        return False, (
            "No usable Python interpreter was found for the fine-tune studio. "
            "Create the studio's own virtual environment or set finetune.python "
            "in settings.yaml to its python.exe."
        )
    return True, ""


class PortResolver(Protocol):
    """Anything that can turn a preferred port into a usable one.

    Typed structurally (not as services.PortAllocator) so this module keeps its
    one-way dependency on services.py and tests can inject a fake resolver
    without touching real sockets (DEC-M13-2 requirement 1).
    """

    def ensure_free(self, port: int) -> int:  # pragma: no cover - interface only
        ...


def studio_url(settings: Settings, port: int | None = None) -> str:
    """The loopback URL the studio serves on. The host is a hardcoded constant.

    `port=None` keeps the configured port (the pre-M13 behaviour). Callers that
    know the port the studio ACTUALLY bound - after port reassignment - pass it
    here, so the URL shown and opened can never point at a port LOCITIZE did not
    start on (DEC-M13-2 requirement 2, defect D-M13-1).

    The host stays the hardcoded 127.0.0.1 literal: it is never settings-derived
    and never caller-derived, so no caller can aim this URL off-machine (SEC-1).
    """
    effective = _cfg(settings).port if port is None else port
    return f"http://127.0.0.1:{effective}/"


def build_studio_spec(
    settings: Settings,
    log_path: str | None = None,
    port_allocator: PortResolver | None = None,
) -> ServiceSpec:
    """Build the studio's ServiceSpec (the build_openwebui_spec analog).

    Every security-relevant value is fixed here rather than derived from settings:
    the bind address is the literal 127.0.0.1 (never 0.0.0.0, never settings-driven),
    telemetry is switched off by flag, and the child environment is empty so no
    LOCITIZE secret or credential can ever reach it.

    Readiness deliberately uses health_path=None, which routes the existing
    _default_readiness helper to its TCP-connect branch: Streamlit has renamed its
    HTTP health endpoint across major versions, so a hardcoded path would make
    readiness silently version-fragile (Architecture M13.4 / risk RC5).

    Raises ValueError carrying studio_available()'s remedy string when the studio
    cannot be launched, so an unusable studio is an honest disabled state rather
    than a traceback.

    Port resolution (DEC-M13-2, defect D-M13-1). Streamlit spells its port flag
    --server.port, which services._command_with_port (matching the literal
    --port) cannot rewrite, so a port reassigned at launch time never reached the
    child. The fix resolves the port HERE, once, and writes that single value into
    both the argv token and ServiceSpec.port so they cannot drift. The studio's
    ManagedProcess is therefore built with no allocator - it must not resolve a
    second time. PortUnavailableError propagates: a spec that silently carries an
    unusable port would be worse than an honest failure.
    """
    ok, reason = studio_available(settings)
    if not ok:
        raise ValueError(reason)

    cfg = _cfg(settings)
    # One local, used twice below. Never re-derive either value from settings.
    port = cfg.port if port_allocator is None else port_allocator.ensure_free(cfg.port)
    interpreter = studio_interpreter(settings)
    app = studio_app_path(settings)
    base = studio_dir(settings)

    command = [
        str(interpreter),
        "-m",
        "streamlit",
        "run",
        str(app),
        # Loopback constant, never settings-derived (Security boundary 1).
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        # LOCITIZE decides when a browser opens, so the child must not open one itself.
        "--server.headless",
        "true",
        # No usage telemetry leaves this machine.
        "--browser.gatherUsageStats",
        "false",
    ]

    return ServiceSpec(
        name=FINETUNE_SERVICE_NAME,
        command=command,
        # Streamlit resolves the app's relative paths (datasets/, outputs/) from
        # its working directory, so it must be the studio checkout.
        cwd=str(base),
        # No secrets, no injected credentials, ever. The ONLY thing passed is
        # the data-root PATH (M17.5), so the studio's "deploy to LOCITIZE" step
        # writes to the running install's real models.yaml instead of a guessed
        # default - a path is not a secret, and without it a portable data root
        # (locitize-data beside the install) is invisible to the child.
        env={"LOCITIZE_DATA_DIR": str(settings.data_dir)},
        # Same local as the --server.port token above: argv, readiness probe, and
        # the status payload all read one value that cannot disagree.
        port=port,
        health_path=None,
        log_path=log_path,
        ready_timeout_s=float(cfg.ready_timeout_s),
        stop_timeout_s=settings.services.stop_timeout_s,
        # Append so a failed start does not destroy the previous trace.
        append_log=True,
    )


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def delete_run(
    settings: Settings, run_name: str, manual_models: list[Model] | None = None
) -> tuple[bool, str]:
    """Permanently delete one discovered run folder from outputs. Guarded.

    A trained run is hours of GPU time, so this refuses loudly instead of
    guessing (M18.7, owner request: the Fine-tune page had no delete at all):

    - the target must be DIRECTLY under the configured outputs root
      (path confinement - a crafted run name cannot escape it);
    - a run whose files back a REGISTERED model row is refused with the model
      id and the remedy (delete the model from the Models page first), because
      removing it here would silently break a row in models.yaml.

    Returns (ok, message); the message is shown to the owner verbatim. Deleting
    a run that is already gone reports ok (the goal state holds).
    """
    import shutil  # noqa: PLC0415 - only needed for this one destructive path

    root = outputs_root(settings)
    if root is None:
        return False, "no outputs root is configured; nothing to delete"
    # Plain directory containment (path_is_confined is the SERVE guard - it
    # additionally demands a non-empty .gguf file, which a run FOLDER is not).
    # Resolved on both sides so a crafted run name ("..", a symlink) cannot
    # reach outside the outputs tree, and the root itself is never a target.
    try:
        root_resolved = root.resolve()
        candidate = (root / str(run_name)).expanduser().resolve()
    except OSError:
        return False, f"could not resolve the run path for {run_name!r}"
    if candidate == root_resolved or root_resolved not in candidate.parents:
        return False, f"refusing to delete outside the outputs root: {run_name!r}"
    if not candidate.exists():
        clear_scan_cache()
        return True, f"run '{run_name}' is already gone"
    for model in manual_models or []:
        location = getattr(model, "location", "") or ""
        if not location:
            continue
        try:
            resolved_location = Path(location).expanduser().resolve()
        except OSError:
            continue
        if resolved_location == candidate or candidate in resolved_location.parents:
            return False, (
                f"run '{run_name}' backs the registered model "
                f"'{model.id}'; delete that model from the Models page first"
            )
    try:
        shutil.rmtree(candidate)
    except OSError as exc:
        return False, f"could not delete '{run_name}': {exc}"
    clear_scan_cache()
    return True, f"deleted run '{run_name}' and its files"


def clear_scan_cache() -> None:
    """Drop the TTL cache (used by tests and by an explicit owner Rescan)."""
    _scan_cache.clear()


def parse_quant(filename: str) -> str:
    """Uppercased quant token parsed from a .gguf FILE name, or "" if unknown.

    The token is the last dot-separated piece of the stem, accepted only when it
    is a quant the trainer actually emits. Case is normalized because models.yaml
    carries Q4_K_M while the trainer writes q4_k_m, and the two must compare equal.
    """
    stem = Path(filename).stem
    token = stem.rsplit(".", 1)[-1].lower() if "." in stem else ""
    return token.upper() if token in KNOWN_QUANTS else ""


def _read_run_meta(run_dir: Path) -> tuple[str, str, str]:
    """Return (base_model, default_system, note) from an optional run_meta.json.

    run_meta.json exists in zero real runs today, so this is strictly optional
    enrichment: absent, oversized, unreadable, malformed, or not a JSON object all
    yield empty strings. A model whose metadata cannot be read is still a servable
    model, and metadata absence must never make a real .gguf invisible.
    """
    meta_path = run_dir / RUN_META_NAME
    try:
        if not meta_path.is_file():
            return "", "", ""
        if meta_path.stat().st_size > RUN_META_MAX_BYTES:
            return "", "", ""
        raw = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return "", "", ""
    if not isinstance(raw, dict):
        return "", "", ""
    base_model = str(raw.get("base_model", "") or "")
    default_system = str(raw.get("default_system", "") or "")
    epochs = raw.get("epochs")
    lr = raw.get("learning_rate")
    parts = []
    if epochs is not None:
        parts.append(f"epochs={epochs}")
    if lr is not None:
        parts.append(f"lr={lr}")
    return base_model, default_system, ", ".join(parts)


def path_is_confined(candidate: Path, root: Path) -> bool:
    """True when `candidate` really lives under `root` and is a usable .gguf.

    This is the guard that keeps an arbitrary filesystem path off llama-server's
    argv (Architecture M13.9 boundary 2). Both sides are resolved first, so a
    symlink pointing outside the outputs tree fails the containment check rather
    than smuggling its target through.
    """
    try:
        resolved = candidate.resolve()
        root_resolved = root.resolve()
    except OSError:
        return False
    if resolved.suffix.lower() != ".gguf":
        return False
    if resolved != root_resolved and root_resolved not in resolved.parents:
        return False
    try:
        stat = resolved.stat()
    except OSError:
        return False
    # A zero-byte file is an export still in progress, not a servable model.
    return stat.st_size > 0


def scan_outputs(
    settings: Settings,
    *,
    use_cache: bool = True,
    now: float | None = None,
) -> ScanResult:
    """Scan the outputs root for fine-tuned .gguf files. Never raises, never writes.

    Shape of the scan, each choice traceable to the real tree (Architecture M13.5.2):

    - Iterate one level of run folders, then glob "*.gguf" inside each. The
      filename is NEVER constructed from the folder name: four of the six real
      GGUFs are named after a different run and would be missed.
    - A run folder with no .gguf is skipped entirely; folder existence is not a
      model (three real run-* folders hold only a train.log).
    - The glob is non-recursive, so merged_model/, lora_adapter/, and checkpoints/
      intermediates are not surfaced as models.
    - Several .gguf files in one run folder yield several discovered models (the
      owner may export both q4_k_m and f16).
    - Every candidate must pass path_is_confined, which drops escapes and 0-byte
      exports.

    An unset, missing, or unreadable root yields an empty result plus a reason
    string; it never blocks the Models page and never degrades manual-registry
    behavior.
    """
    root = outputs_root(settings)
    if root is None:
        # studio_dir is deliberately NOT named here any more: since DEC-M14-9 a
        # blank outputs_dir resolves to the data root, not to <studio_dir>/outputs,
        # so pointing a user at studio_dir would send them to a key that no longer
        # affects discovery at all (review round 6, MEDIUM-6).
        return ScanResult(
            (),
            "",
            "Fine-tune outputs folder is not configured. Set "
            "finetune.outputs_dir in settings.yaml to the folder your training "
            "runs are written to.",
        )
    if not _cfg(settings).discovery_enabled:
        return ScanResult((), str(root), "Fine-tune discovery is switched off "
                          "(finetune.discovery_enabled: false).")

    key = str(root)
    clock = time.monotonic() if now is None else now
    if use_cache:
        cached = _scan_cache.get(key)
        if cached is not None and (clock - cached[0]) < SCAN_TTL_S:
            return cached[1]

    result = _scan_uncached(settings, root)
    _scan_cache[key] = (clock, result)
    return result


def _scan_uncached(settings: Settings, root: Path) -> ScanResult:
    """The real filesystem walk behind scan_outputs' TTL cache."""
    if not root.is_dir():
        return ScanResult((), str(root), f"Could not read `{root}`: not a directory.")
    try:
        run_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError as exc:
        return ScanResult((), str(root), f"Could not read `{root}`: {exc}.")

    items: list[DiscoveredFineTune] = []
    for run_dir in run_dirs:
        try:
            ggufs = sorted(run_dir.glob("*.gguf"))
        except OSError:
            continue
        usable = [g for g in ggufs if path_is_confined(g, root)]
        if not usable:
            continue
        base_model, default_system, note = _read_run_meta(run_dir)
        for gguf in usable:
            resolved = gguf.resolve()
            stat = resolved.stat()
            # One .gguf per run keeps the plain id; several need the file stem to
            # stay distinct within the run.
            item_id = FT_PREFIX + run_dir.name
            if len(usable) > 1:
                item_id = f"{item_id}:{resolved.stem}"
            items.append(
                DiscoveredFineTune(
                    id=item_id,
                    run=run_dir.name,
                    name=run_dir.name,
                    path=str(resolved),
                    quant=parse_quant(resolved.name),
                    size_bytes=stat.st_size,
                    mtime=stat.st_mtime,
                    base_model=base_model,
                    recommended_prompt=default_system,
                    meta_note=note,
                )
            )

    # The empty state names BOTH the folder that was actually scanned and the key
    # that moves it. Since DEC-M14-9 a blank outputs_dir means the data root, so a
    # user whose runs live in their studio checkout - the pre-M14 default - sees
    # an empty list for a reason they cannot guess from an empty list alone
    # (review round 6, MEDIUM-6).
    reason = (
        ""
        if items
        else (
            f"No fine-tuned models found in `{root}`. If your training runs are "
            f"somewhere else, set finetune.outputs_dir in settings.yaml to that "
            f"folder."
        )
    )
    return ScanResult(tuple(items), str(root), reason)


def active_run(settings: Settings, *, now: float | None = None) -> str | None:
    """Name of a run folder whose train.log was written to very recently, or None.

    This is the only read-only signal LOCITIZE has that a training run is live: the
    trainer appends to train.log for the whole run, so a log touched within
    finetune.active_run_window_s means work is in flight right now. It is a
    heuristic, and the warning it gates says "may still be running" precisely
    because LOCITIZE cannot see inside the studio's own child processes.
    """
    root = outputs_root(settings)
    if root is None or not root.is_dir():
        return None
    window = float(_cfg(settings).active_run_window_s)
    clock = time.time() if now is None else now
    try:
        run_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return None
    for run_dir in run_dirs:
        log_path = run_dir / TRAIN_LOG_NAME
        try:
            if not log_path.is_file():
                continue
            age = clock - log_path.stat().st_mtime
        except OSError:
            continue
        # Lower bound is negative on purpose: see CLOCK_SKEW_TOLERANCE_S. A log
        # whose mtime is ahead of the wall clock is the freshest possible run,
        # never a stale one, so it must read as live rather than be discarded.
        if -CLOCK_SKEW_TOLERANCE_S <= age <= window:
            return run_dir.name
    return None


def orphan_warning_text() -> str:
    """The verbatim limitation notice shown when a live run may outlive the studio.

    Read from the shipped data file (see _WARNING_FILE above for why the wording
    cannot live in this module). A damaged install falls back to a wording that
    states the same limitation rather than saying nothing.
    """
    path = Path(__file__).resolve().parent / _WARNING_FILE
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return _WARNING_FALLBACK
    return text or _WARNING_FALLBACK


# --------------------------------------------------------------------------- #
# Dedup against the manual registry, and promotion
# --------------------------------------------------------------------------- #


def _fingerprint(path_str: str) -> tuple[str, int] | None:
    """(lowercased filename, size in bytes) for a manual row's location, or None.

    Deliberately no hashing: these are multi-gigabyte files, and this runs on every
    Models-page paint. Name plus exact byte size is cheap, deterministic, and
    sufficient to recognize the copy of a fine-tune the user deployed into
    their own models directory with different case in the quant token.
    """
    if not path_str:
        return None
    try:
        candidate = Path(path_str).expanduser()
        stat = candidate.stat()
    except OSError:
        return None
    return candidate.name.lower(), stat.st_size


def dedup_against_registry(
    items: tuple[DiscoveredFineTune, ...] | list[DiscoveredFineTune],
    manual_models: list[Model],
) -> tuple[list[DiscoveredFineTune], set[str]]:
    """Fold discovered files that are already manual rows into those rows.

    Two tiers, in order (Architecture M13.5.4):
      1. Path identity: the manual row's resolved location IS the discovered file.
      2. Provenance fingerprint: same filename (case-insensitively) and the exact
         same byte size, which catches the real case of a fine-tune copied out of
         outputs/ into the user's own models directory before registration.

    Returns (marked_items, matched_manual_ids). A matched item is returned with
    already_registered=True so callers can drop it from the discovered list while
    annotating the manual row as a registered fine-tune. If the manual row's file
    is missing or unreadable, tier 2 is skipped and no match is asserted: visible
    duplication is better than a silently hidden model.
    """
    manual_paths: dict[str, str] = {}
    manual_prints: dict[tuple[str, int], str] = {}
    for model in manual_models or []:
        location = getattr(model, "location", "") or ""
        if not location:
            continue
        try:
            manual_paths.setdefault(str(Path(location).expanduser().resolve()), model.id)
        except OSError:
            pass
        print_key = _fingerprint(location)
        if print_key is not None:
            manual_prints.setdefault(print_key, model.id)

    ordered = list(items or ())
    matched: set[str] = set()
    claimed_by_item: dict[int, str] = {}

    # Pass 1 - exact path identity. This is proof, so it always wins and is never
    # limited by the one-row-one-model rule below.
    for index, item in enumerate(ordered):
        manual_id = manual_paths.get(item.path)
        if manual_id is not None:
            claimed_by_item[index] = manual_id
            matched.add(manual_id)

    # Pass 2 - the (filename, size) fingerprint, which is strong evidence but not
    # proof. A manual row describes exactly ONE model, so once a row has been
    # claimed it cannot absorb further look-alikes: four historical a prior fine-tune runs
    # ship byte-identical copies of the same file, and folding all four into the
    # one manual row would hide three real, distinct builds from the owner.
    for index, item in enumerate(ordered):
        if index in claimed_by_item:
            continue
        manual_id = manual_prints.get((Path(item.path).name.lower(), item.size_bytes))
        if manual_id is None or manual_id in matched:
            continue
        claimed_by_item[index] = manual_id
        matched.add(manual_id)

    marked = [
        replace(item, already_registered=True) if index in claimed_by_item else item
        for index, item in enumerate(ordered)
    ]
    return marked, matched


def to_model(item: DiscoveredFineTune, settings: Settings) -> Model:
    """Turn a discovered file into an ordinary config.Model.

    The result is a normal registry row with a location, so models.build_start_spec
    serves it through the identical, already-proven llama.cpp path; only `source`
    and `source_run` mark where it came from. Defaults come from the finetune
    settings block, not from a models.yaml row (there is none yet).
    """
    cfg = _cfg(settings)
    quant_note = item.quant or "unknown quant"
    return Model(
        id=item.id,
        name=item.run,
        description=f"Fine-tuned model discovered in {item.run} ({quant_note}).",
        location=item.path,
        context_size=cfg.default_context_size,
        gpu_layers=cfg.default_gpu_layers,
        recommended_prompt=item.recommended_prompt,
        benchmark_score=None,
        notes=item.meta_note,
        status="installed",
        quantization=item.quant,
        vram_estimate_mb=0,
        server_args=[],
        source=DISCOVERED_SOURCE,
        source_run=item.run,
    )


def register_id_for(discovered_id: str) -> str:
    """The models.yaml id a discovered model is promoted under.

    The ft: prefix is stripped and any character outside [A-Za-z0-9_-] becomes a
    hyphen, so ft:a prior fine-tune-v5 registers as a prior fine-tune-v5 and a run folder with
    spaces or dots still yields a clean, quotable yaml id.
    """
    raw = discovered_id[len(FT_PREFIX):] if discovered_id.startswith(FT_PREFIX) else discovered_id
    cleaned = "".join(c if (c.isalnum() and c.isascii()) or c in "_-" else "-" for c in raw)
    return cleaned.strip("-") or "finetune"


def resolve_serve_path(item_path: str, settings: Settings) -> str:
    """Re-check a discovered path immediately before it is served. Raises on failure.

    The scan already confined every path, but a scan result can be seconds old and
    the file can be deleted or replaced in between, so the guard runs again at the
    moment the path would reach a child process's argv rather than trusting a
    cached decision.
    """
    root = outputs_root(settings)
    if root is None:
        raise ValueError(
            "fine-tune outputs folder is not configured; cannot serve a discovered "
            "model (set finetune.studio_dir or finetune.outputs_dir)"
        )
    if not path_is_confined(Path(item_path), root):
        raise ValueError(
            f"'{item_path}' is not a readable .gguf inside the configured "
            f"fine-tune outputs folder; refusing to serve it"
        )
    return str(Path(item_path).resolve())


def describe_meta(item: DiscoveredFineTune) -> str:
    """One short metadata line for the UI: honest 'none' when there is nothing."""
    parts = [p for p in (item.base_model, item.meta_note) if p]
    return ", ".join(parts) if parts else "metadata: none"
