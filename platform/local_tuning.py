"""Narrow local tuning adapter; reuses LOCITIZE's registry and owned supervisor.

Cancellation is cooperative: an in-flight bounded load/completion finishes its
cleanup before a terminal cancellation is reported. No unrelated process stops.
"""
from __future__ import annotations

import copy
import json
import math
import re
import threading
import time
import uuid
import webbrowser
import platform as host_platform
import psutil
from pathlib import Path

import codex_picker
import config
import gguf_meta
from benchmark import BenchmarkRunner, RunLock, parse_completion_timings
from health import DefaultSystemInfoProvider, NvidiaSmiGpuInfoProvider
from launcher import Launcher
from logger import resolve_log_dir
from services import ServiceStatus
from router import ModelRouter

TERMINAL = {"completed", "cancelled", "failed", "recovery_failed", "target_not_met"}


def cpu_name():
    """Use the inexpensive Windows registry for missing or architecture-only names."""
    name = host_platform.processor()
    generic = name.strip().lower() if isinstance(name, str) else ""
    descriptor = re.fullmatch(
        r"(?:intel64|amd64|x86) family [0-9]+ model [0-9]+ stepping [0-9]+(?:, [a-z0-9_ -]{1,64})?",
        generic)
    unavailable = not name or generic in {"amd64", "x86_64", "x86", "i386", "aarch64", "arm64"} or bool(descriptor)
    if unavailable and host_platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        except (ImportError, OSError):
            name = None
    if not isinstance(name, str):
        return None
    name = " ".join("".join(c if 32 <= ord(c) < 127 else " " for c in name[:512]).split())
    return name[:256] or None


def effective_context_from_log(log, offset):
    """Read observed slot capacity only from this startup's appended output."""
    if log is None or not log.exists():
        return None
    with log.open("rb") as handle:
        handle.seek(offset)
        data = handle.read().decode("utf-8", errors="replace")
    values = re.findall(r"(?:n_ctx_per_seq|n_ctx_slot|n_ctx_seq)\s*=\s*(\d+)", data)
    return int(values[-1]) if values else None


def candidates(model, native):
    """Bound the search to five trials; partial offload is actually measured."""
    if (not isinstance(native, int) or isinstance(native, bool) or native < 1
            or not isinstance(model.context_size, int)
            or isinstance(model.context_size, bool) or model.context_size < 1):
        raise ValueError("context capacities must be positive integers")
    base = min(model.context_size, native)
    # Always compare a smaller window, even when the saved window already
    # equals/exceeds the native limit. These are bounded trial choices, not
    # computed memory ceilings or claims that a larger context will run well.
    lower = min(8192, max(1, base // 2))
    upper = min(native, max(16384, base * 2))
    if base > 16384 or upper == base:
        # At a native cap or a large saved window, half-sized
        # context prioritizes useful retained history over a sparse upward
        # jump. Real high-context comparisons exposed the missing middle row.
        upper = max(1, base // 2)
    # Layer counts are probe requests, never claims about hardware capacity.
    layers = [model.gpu_layers, 32, 16]
    rows = [(base, layers[0]), (base, layers[1]), (base, layers[2]),
            (lower, layers[0]), (upper, layers[0])]
    return list(dict.fromkeys(rows))


def choose_trial(trials, objective):
    """Rank only measured, usable candidates; balanced retains 85% of best speed."""
    valid = [r for r in trials if r["tokens_per_second"] is not None]
    if not valid:
        raise ValueError("no usable measured configuration")
    best = max(valid, key=lambda r: r["tokens_per_second"])
    if objective == "speed":
        return best
    eligible = [r for r in valid if r["tokens_per_second"] >= best["tokens_per_second"] * .85]
    return max(eligible, key=lambda r: (r["context_effective"] or 0, r["tokens_per_second"]))


MIN_CONTEXT = 8192
MIN_TPS = 20.0
PREFERRED_TPS = 30.0
EXPLORATORY_BUDGET = 16
FINAL_BUDGET = 2


def eligible_trial(row):
    """Eligibility uses observed slot capacity and measured completion speed."""
    context, speed = row.get("context_effective"), row.get("tokens_per_second")
    return (type(context) is int and context >= MIN_CONTEXT and
            isinstance(speed, (int, float)) and math.isfinite(speed) and speed >= MIN_TPS)


def adaptive_choice(rows, objective="balanced"):
    eligible = [row for row in rows if eligible_trial(row)]
    if not eligible:
        return None
    best = max(row["tokens_per_second"] for row in eligible)
    competitive = [row for row in eligible if row["tokens_per_second"] >= best * .85]
    if objective == "speed":
        return max(eligible, key=lambda row: row["tokens_per_second"])
    return max(competitive, key=lambda row: (row["context_effective"], row["tokens_per_second"]))


def adaptive_trials(model, native, block_count, probe, cancelled, emit):
    """Measured coarse/neighbor offload search, then geometric context expansion."""
    rows, seen = [], {}
    maximum_layers = block_count + 1 if type(block_count) is int and 1 <= block_count <= 4096 else None

    def attempt(context, layers, phase):
        key = (context, layers)
        if key in seen:
            return seen[key]
        if cancelled() or len(rows) >= EXPLORATORY_BUDGET:
            return None
        emit(phase, context, layers, rows)
        started = time.monotonic()
        row = probe(context, layers)
        row["phase"] = phase
        row["duration_seconds"] = round(time.monotonic() - started, 3)
        rows.append(row)
        seen[key] = row
        emit(phase, context, layers, rows)
        return row

    def search_center(row):
        requested = row["gpu_layers_requested"]
        actual = row.get("gpu_layers_effective")
        if type(actual) is int and 0 <= actual <= (maximum_layers or 4097):
            return actual
        if maximum_layers is not None and (requested < 0 or requested > maximum_layers):
            return maximum_layers
        return requested

    if native < MIN_CONTEXT:
        return rows
    attempt(min(model.context_size, native), model.gpu_layers, "baseline")
    coarse = ([maximum_layers, round(maximum_layers * 2 / 3), round(maximum_layers / 3), 0]
              if maximum_layers else [min(999, max(0, model.gpu_layers)), 32, 16, 0])
    for layers in dict.fromkeys(coarse):
        attempt(MIN_CONTEXT, layers, "offload_coarse")
    useful = [row for row in rows if row.get("context_effective") == MIN_CONTEXT
              and row.get("tokens_per_second")]
    if not useful or cancelled():
        return rows
    strongest = max(useful, key=lambda row: row["tokens_per_second"])
    center = search_center(strongest)
    for layers in [center - 1, center + 1, center - 2, center + 2]:
        if layers >= 0 and (maximum_layers is None or layers <= maximum_layers):
            attempt(MIN_CONTEXT, layers, "offload_neighbors")
    useful = [row for row in rows if row.get("context_effective") == MIN_CONTEXT
              and row.get("tokens_per_second")]
    center = search_center(max(useful, key=lambda row: row["tokens_per_second"]))
    previous = MIN_CONTEXT
    context = MIN_CONTEXT * 2
    failed_context = None
    while previous < native and len(rows) < EXPLORATORY_BUDGET and not cancelled():
        context = min(context, native)
        trial = attempt(context, center, "context_growth")
        best = max((row["tokens_per_second"] for row in rows if eligible_trial(row)), default=MIN_TPS)
        if not trial or not eligible_trial(trial) or trial["tokens_per_second"] < best * .85:
            # A larger KV window may shift the best offload region. Test a
            # conservative lower request before calling this a context cliff.
            alternate = max(0, center - max(1, center // 3))
            other = attempt(context, alternate, "context_offload_recheck")
            if not other or not eligible_trial(other) or other["tokens_per_second"] < best * .85:
                failed_context = context
                break
            center = alternate
        previous = context
        context *= 2
    if failed_context and failed_context - previous >= 2048:
        middle = ((failed_context + previous) // 2 // 1024) * 1024
        attempt(middle, center, "context_boundary")
    chosen = adaptive_choice(rows)
    if chosen and not cancelled():
        center = search_center(chosen)
        for neighbor in [center - 1, center + 1]:
            if neighbor >= 0 and (maximum_layers is None or neighbor <= maximum_layers):
                attempt(chosen["context_requested"], neighbor, "context_offload_neighbors")
    return rows


class LocalTuning:
    """One job/resource owner per helper, plus the existing cross-process run lock."""

    def __init__(self):
        self._mutex = threading.RLock()
        self._jobs = {}
        self._cancel = {}
        self._thread = None
        self._chat = None
        self._chat_lock = None
        self._router = None
        self._recovery_lock = None
        self._recovery = None

    def _runtime(self):
        """Reload configuration for every job so saved changes are honored."""
        settings, models, _ = config.Config.load()
        log = resolve_log_dir(settings) / "local_tuning_llama.log"
        registry, manager, controller = Launcher()._build_controller(settings, models, log_path=log)
        runner = BenchmarkRunner(controller, registry, settings,
                                 gpu_provider=NvidiaSmiGpuInfoProvider(),
                                 sys_provider=DefaultSystemInfoProvider(), out=lambda _: None)
        return settings, registry, manager, controller, runner, log

    def assess_computer(self):
        """Return real provider readings; unsupported GPU readings stay unknown."""
        system = DefaultSystemInfoProvider()
        gpus = NvidiaSmiGpuInfoProvider().gpus()
        try:
            cpu = {"name": cpu_name(),
                   "logical_cores": psutil.cpu_count(logical=True),
                   "physical_cores": psutil.cpu_count(logical=False)}
        except (OSError, RuntimeError, AttributeError):
            cpu = {"name": None, "logical_cores": None, "physical_cores": None}
        return {"cpu": cpu, "ram_total_mb": system.ram_total_mb(),
                "ram_available_mb": system.ram_available_mb(),
                "gpus": None if gpus is None else
                [{"name": g.name, "vram_total_mb": g.vram_total_mb,
                  "vram_free_mb": g.vram_free_mb} for g in gpus],
                "gpu_source": "nvidia-smi", "gpu_readings_available": bool(gpus)}

    def list_local_models(self):
        """Expose registered IDs and capabilities without private model paths."""
        _, registry, _, _, _, _ = self._runtime()
        rows = []
        for model in registry.all():
            try:
                native = gguf_meta.read_native_context_length(model.location)
            except (ValueError, OSError):
                native = None
            rows.append({"model_id": model.id, "name": model.name,
                         "launchable": model in registry.launchable(),
                         "context_native": native, "context_saved": model.context_size,
                         "gpu_layers_saved": model.gpu_layers,
                         # Configured, not verified: vision needs a real image check.
                         "vision_projector_configured": bool(model.mmproj)})
        return {"models": rows}

    def _model(self, registry, model_id):
        """Allow only an exact registered launchable ID, never caller paths."""
        if not isinstance(model_id, str) or len(model_id) > 128:
            raise ValueError("invalid model ID")
        model = registry.get(model_id)
        if model is None or model not in registry.launchable():
            raise ValueError("unknown or unavailable model ID")
        if model.backend not in ("", "llama-cpp"):
            raise ValueError("unsupported backend; tuning requires llama-cpp")
        return model

    def start_tuning(self, model_id, objective="balanced"):
        """Validate and acquire resources before returning an asynchronous job."""
        if objective not in ("balanced", "speed"):
            raise ValueError("objective must be balanced or speed")
        with self._mutex:
            if self._recovery_lock or self._chat or (self._thread and self._thread.is_alive()):
                raise ValueError("busy: stop the helper's model or wait for tuning")
            runtime = self._runtime()
            model = self._model(runtime[1], model_id)
            try:
                native = gguf_meta.read_native_context_length(model.location)
            except gguf_meta.GgufError as exc:
                raise ValueError("Native GGUF context metadata unavailable; tuning refused") from exc
            if type(native) is not int or native < 1:
                raise ValueError("Native GGUF context metadata unavailable; tuning refused")
            quantization = getattr(model, "quantization", None)
            if not isinstance(quantization, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", quantization):
                quantization = None
            metadata = {"native_context": native, "architecture": None, "block_count": None,
                        "file_bytes": None, "quantization": quantization}
            if objective == "balanced":
                try:
                    header = gguf_meta.read_gguf_header(model.location, include_layer_count=True)
                    metadata["architecture"] = re.sub(r"[^A-Za-z0-9_-]", "", header.architecture)[:64] or None
                    metadata["block_count"] = header.block_count
                    metadata["file_bytes"] = Path(model.location).stat().st_size
                except (gguf_meta.GgufError, OSError):
                    pass
            lock_path = resolve_log_dir(runtime[0]) / "benchmark.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            run_lock = RunLock(lock_path)
            run_lock.__enter__()
            try:
                runtime[4].assert_can_run()
                job_id = uuid.uuid4().hex
                # Limit retained history; the protocol exposes no private snapshots.
                if len(self._jobs) >= 20:
                    oldest = next(iter(self._jobs))
                    self._jobs.pop(oldest)
                    self._cancel.pop(oldest)
                self._jobs[job_id] = {"job_id": job_id, "model_id": model_id,
                    "objective": objective, "state": "queued", "phase": "queued", "trials": [],
                    "trials_completed": 0, "remaining_exploratory_budget": EXPLORATORY_BUDGET if objective == "balanced" else None,
                    "remaining_verification_budget": FINAL_BUDGET if objective == "balanced" else 1,
                    "selected": None, "error": None, "cleanup_confirmed": False,
                    "cancel_requested": False, "started_at": time.time(),
                    "candidate_count": EXPLORATORY_BUDGET if objective == "balanced" else len(candidates(model, native)),
                    "model_metadata": metadata,
                    "hardware_at_start": self.assess_computer() if objective == "balanced" else None,
                    "targets": {"minimum_context": MIN_CONTEXT, "minimum_tokens_per_second": MIN_TPS,
                                "preferred_tokens_per_second": PREFERRED_TPS} if objective == "balanced" else None,
                    "exploratory_budget": EXPLORATORY_BUDGET if objective == "balanced" else None,
                    "verification_budget": FINAL_BUDGET if objective == "balanced" else 1}
                self._cancel[job_id] = threading.Event()
                self._thread = threading.Thread(target=self._run,
                    args=(job_id, runtime, model, native, run_lock), daemon=False)
                self._thread.start()
            except BaseException:
                run_lock.__exit__()
                raise
            return self.tuning_status(job_id)

    def _update(self, job_id, **fields):
        with self._mutex:
            self._jobs[job_id].update(fields)

    def _probe(self, runtime, model_id, ctx, layers):
        """Capture this trial's logs only; report effective values when observable."""
        _, _, manager, _, runner, log = runtime
        offset = log.stat().st_size if log.exists() else 0
        speed = runner.probe_context_throughput(model_id, ctx, gpu_layers=layers)
        if not Launcher()._confirm_clean(manager):
            raise RuntimeError("owned process cleanup failed")
        data = ""
        if log.exists():
            with log.open("rb") as handle:
                handle.seek(offset)
                data = handle.read().decode("utf-8", errors="replace")
        # llama.cpp names this per-request capacity differently across builds.
        # n_ctx_slot is the server's loaded slot capacity, unlike total n_ctx.
        effective = effective_context_from_log(log, offset)
        offload = re.findall(r"offloaded\s+(\d+)/\d+\s+layers", data)
        if speed is not None and (not math.isfinite(speed) or speed <= 0):
            speed = None
        return {"context_requested": ctx,
                "context_effective": effective,
                "gpu_layers_requested": layers,
                "gpu_layers_effective": int(offload[-1]) if offload else None,
                "tokens_per_second": speed,
                "completion": "passed" if speed else "failed",
                "hardware": self.assess_computer(),
                "provenance": "llama.cpp completion timings; warm-up then one timed completion"}

    def _run(self, job_id, runtime, model, native, run_lock):
        """Apply once, verify, and compare-and-restore on cancellation or failure."""
        settings, _, manager, _, _, _ = runtime
        before = (model.gpu_layers, model.context_size)
        applied = None
        state, error = "failed", None
        try:
            self._update(job_id, state="running")
            if self._jobs[job_id]["objective"] == "balanced":
                state, applied = self._run_adaptive(job_id, runtime, model, native, before)
                return
            for index, (ctx, layers) in enumerate(candidates(model, native)):
                if self._cancel[job_id].is_set():
                    state = "cancelled"
                    break
                self._update(job_id, candidate_index=index)
                trial = self._probe(runtime, model.id, ctx, layers)
                with self._mutex:
                    self._jobs[job_id]["trials"].append(trial)
                if index == 0 and not trial["tokens_per_second"]:
                    raise ValueError("baseline completion failed; settings unchanged")
            else:
                if self._cancel[job_id].is_set():
                    state = "cancelled"
                else:
                    chosen = choose_trial(self.tuning_status(job_id)["trials"],
                                          self._jobs[job_id]["objective"])
                    # Unknown effective context cannot support a verified save.
                    if chosen["context_effective"] is None:
                        raise ValueError("effective context unavailable; settings unchanged")
                    selected_fields = (chosen["gpu_layers_requested"], chosen["context_requested"])
                    config.write_model_fields(settings.data_dir, model.id, *selected_fields,
                                              expected_fields=before)
                    applied = selected_fields
                    self._update(job_id, state="verifying", selected=chosen)
                    verification = self._probe(runtime, model.id, applied[1], applied[0])
                    self._update(job_id, verification=verification)
                    if not verification["tokens_per_second"] or verification["context_effective"] != chosen["context_effective"]:
                        raise ValueError("final completion verification failed")
                    state = "cancelled" if self._cancel[job_id].is_set() else "completed"
        except Exception:
            error = "Tuning failed; inspect the local helper log."
        finally:
            try:
                manager.stop_all()
                if not Launcher()._confirm_clean(manager):
                    raise RuntimeError("cleanup not confirmed")
                if applied is not None and self._cancel[job_id].is_set():
                    state = "cancelled"
                if applied is not None and state != "completed":
                    config.write_model_fields(settings.data_dir, model.id, *before,
                                              expected_fields=applied)
                    self._update(job_id, settings_applied=False)
                self._update(job_id, cleanup_confirmed=True)
            except Exception:
                state = "recovery_failed"
                error = "Cleanup or settings restoration failed; inspect locally before retrying."
            finally:
                if state == "recovery_failed":
                    # Keep the resource reserved when recovery is uncertain.
                    # Another job must never co-load over an orphaned process.
                    self._recovery_lock = run_lock
                    self._recovery = {"runtime": runtime, "job_id": job_id,
                                      "before": before, "applied": applied,
                                      "model_id": model.id}
                else:
                    run_lock.__exit__()
                self._update(job_id, state=state, error=error, finished_at=time.time())

    def _run_adaptive(self, job_id, runtime, model, native, before):
        """Verify twice before one compare-and-write; failure never applies settings."""
        cancel = self._cancel[job_id].is_set
        def progress(phase, context, layers, rows):
            best = adaptive_choice(rows)
            durations = [row["duration_seconds"] for row in rows]
            mean = sum(durations) / len(durations) if durations else None
            remaining = EXPLORATORY_BUDGET - len(rows)
            self._update(job_id, phase=phase, current_context_requested=context,
                         current_gpu_layers_requested=layers, trials=copy.deepcopy(rows),
                         trials_completed=len(rows), remaining_exploratory_budget=remaining,
                         remaining_verification_budget=FINAL_BUDGET,
                         best_observed=best, estimated_remaining_seconds=None if mean is None else
                         round(mean * (remaining + FINAL_BUDGET), 1),
                         estimate_note="Rough projection from completed trials; future loads may be slower or faster")
        metadata = self._jobs[job_id].get("model_metadata") or {}
        rows = adaptive_trials(model, native, metadata.get("block_count"),
            lambda context, layers: self._probe(runtime, model.id, context, layers), cancel, progress)
        if cancel():
            return "cancelled", None
        chosen = adaptive_choice(rows)
        observed = max((row for row in rows if row.get("tokens_per_second")),
                       key=lambda row: row["tokens_per_second"], default=None)
        self._update(job_id, best_observed_result=observed, selected=chosen, settings_applied=False)
        if chosen is None:
            self._update(job_id, outcome="target-not-met", reason="No measured result meets context8192 and20tokens/sec",
                         preferred_target_met=False)
            return "target_not_met", None
        best_speed = max(row["tokens_per_second"] for row in rows if eligible_trial(row))
        verifications = []
        for index in range(FINAL_BUDGET):
            if cancel():
                return "cancelled", None
            self._update(job_id, state="verifying", phase="final_verification",
                         remaining_verification_budget=FINAL_BUDGET - index)
            verification = self._probe(runtime, model.id, chosen["context_requested"], chosen["gpu_layers_requested"])
            verifications.append(verification)
            self._update(job_id, verification=verification, verifications=copy.deepcopy(verifications),
                         remaining_verification_budget=FINAL_BUDGET - index - 1)
        if cancel():
            return "cancelled", None
        speeds = [row.get("tokens_per_second") for row in verifications]
        valid = all(eligible_trial(row) and row["context_effective"] == chosen["context_effective"]
                    and row["tokens_per_second"] >= best_speed * .85 for row in verifications)
        if not valid:
            self._update(job_id, outcome="target-not-met", reason="Repeated verification diverged or missed measured targets",
                         preferred_target_met=False)
            return "target_not_met", None
        fields = (chosen["gpu_layers_requested"], chosen["context_requested"])
        config.write_model_fields(runtime[0].data_dir, model.id, *fields, expected_fields=before)
        self._update(job_id, outcome="verified", settings_applied=True,
                     preferred_target_met=min(speeds) >= PREFERRED_TPS,
                     verification_variation={"minimum_tokens_per_second": min(speeds),
                         "maximum_tokens_per_second": max(speeds), "relative_range": (max(speeds) - min(speeds)) / max(speeds)})
        return "completed", fields

    def tuning_status(self, job_id):
        """Copy job data under lock so clients never observe a mutable list."""
        with self._mutex:
            if not isinstance(job_id, str) or job_id not in self._jobs:
                raise ValueError("unknown job ID")
            return copy.deepcopy(self._jobs[job_id])

    def cancel_tuning(self, job_id):
        """Request cancellation; terminal status waits for in-flight probe cleanup."""
        with self._mutex:
            status = self.tuning_status(job_id)
            if status["state"] not in TERMINAL:
                self._cancel[job_id].set()
                self._update(job_id, cancel_requested=True)
            return self.tuning_status(job_id)

    def start_local_chat(self, model_id):
        """Launch only a registered local server and its verified loopback UI."""
        result = self.start_local_model(model_id)
        result["browser_open_requested"] = bool(webbrowser.open(result["url"]))
        return result

    def start_local_model(self, model_id):
        """Verify one real completion and retain the owned server for local clients."""
        with self._mutex:
            if self._recovery_lock or self._chat or (self._thread and self._thread.is_alive()):
                raise ValueError("busy: helper resources already in use")
            runtime = self._runtime()
            self._model(runtime[1], model_id)
            lock_path = resolve_log_dir(runtime[0]) / "benchmark.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = RunLock(lock_path)
            lock.__enter__()
            router = None
            try:
                runtime[4].assert_can_run()
                log = runtime[5]
                offset = log.stat().st_size if log is not None and log.exists() else 0
                status = runtime[3].start(model_id)
                if status is not ServiceStatus.RUNNING:
                    raise ValueError("local model did not become ready")
                port = runtime[3].running_port
                if not isinstance(port, int) or not 1 <= port <= 65535:
                    raise ValueError("invalid local chat port")
                url = f"http://127.0.0.1:{port}/"
                completion = runtime[4]._post_completion(url + "completion", "Write a short greeting.")
                timings = parse_completion_timings(completion or {})
                if timings is None or timings.predicted_per_second <= 0:
                    raise ValueError("local model completion verification failed")
                selected = self._model(runtime[1], model_id)
                picker = self._picker_models(runtime)
                if model_id in [model.id for model in picker]:
                    # Codex /model under the locitize profile: only models the
                    # user put in the picker, one at a time, owned by this helper.
                    advertised, picker_active = picker, True
                    switch = lambda target: self._switch_owned(runtime, picker, target)
                else:
                    # Restrict discovery to the selected model and forbid switching:
                    # local Codex requests cannot silently load another registry row.
                    advertised, picker_active = [selected], False
                    switch = lambda _: (False, "helper model switching is disabled")
                router = ModelRouter(
                    port=runtime[0].ports.router,
                    registry_fn=lambda: list(advertised),
                    running_model_id_fn=lambda: runtime[3].running_model_id,
                    port_provider=lambda: runtime[3].running_port,
                    switch_fn=switch,
                )
                router.start()
                self._router = router
                self._chat, self._chat_lock = runtime, lock
                return {"state": "running", "model_id": model_id,
                        "url": url, "api_base_url": url + "v1",
                        "codex_api_base_url": router.base_url(),
                        "router_port": runtime[0].ports.router,
                        "completion_verified": True,
                        "context_effective": effective_context_from_log(log, offset),
                        "tokens_per_second": timings.predicted_per_second,
                        "codex_picker_models": [model.id for model in advertised] if picker_active else []}
            except BaseException:
                self._router = router
                self._chat, self._chat_lock = runtime, lock
                if self._cleanup_owned(runtime):
                    lock.__exit__()
                    self._chat, self._chat_lock = None, None
                raise

    def _picker_path(self, settings):
        return Path(settings.data_dir) / "codex_picker.json"

    def _picker_models(self, runtime):
        """Picker IDs saved by enable_codex_picker that are still launchable."""
        try:
            saved = json.loads(self._picker_path(runtime[0]).read_text(encoding="utf-8"))
            ids = saved.get("models", [])
        except (OSError, ValueError, AttributeError):
            return []
        models = []
        for model_id in ids[:codex_picker.MAX_MODELS] if isinstance(ids, list) else []:
            try:
                models.append(self._model(runtime[1], model_id))
            except ValueError:
                continue
        return models

    def _switch_owned(self, runtime, picker, target):
        """Router switch: stop the served model, start a picker model, verify it."""
        if target not in [model.id for model in picker]:
            return False, "not in the Codex picker list"
        if not self._mutex.acquire(timeout=5):
            return False, "helper busy"
        try:
            if self._chat is not runtime:
                return False, "helper model is stopped"
            if runtime[3].switch(target) is not ServiceStatus.RUNNING:
                return False, "model did not become ready"
            port = runtime[3].running_port
            completion = runtime[4]._post_completion(
                f"http://127.0.0.1:{port}/completion", "Write a short greeting.")
            timings = parse_completion_timings(completion or {})
            if timings is None or timings.predicted_per_second <= 0:
                return False, "completion verification failed"
            return True, "switched"
        finally:
            self._mutex.release()

    def enable_codex_picker(self, model_ids, default_model_id=None):
        """List tuned local models in `codex --profile locitize`; config.toml untouched."""
        if (not isinstance(model_ids, list) or not 1 <= len(model_ids) <= codex_picker.MAX_MODELS
                or len(set(model_ids)) != len(model_ids)):
            raise ValueError("invalid Codex picker model list")
        default_model_id = default_model_id or model_ids[0]
        if default_model_id not in model_ids:
            raise ValueError("invalid Codex picker model list")
        with self._mutex:
            settings, registry = self._runtime()[:2]
            selected = [self._model(registry, model_id) for model_id in model_ids]
            if any((model.context_size or 0) < codex_picker.MIN_CONTEXT for model in selected):
                raise ValueError("tune the model first: Codex needs a saved context of at least 32768")
            instructions = codex_picker.capture_base_instructions()
            result = codex_picker.enable(selected, default_model_id, settings.ports.router, instructions)
            codex_picker._atomic_write(self._picker_path(settings), json.dumps(
                {"models": model_ids, "default_model_id": default_model_id}) + "\n")
            result["next_step"] = ("Start the default model (start_local_model or server.py --serve), "
                                   "then run codex --profile locitize and choose models with /model.")
            return result

    def disable_codex_picker(self):
        """Remove the helper's picker files; a running model keeps serving."""
        with self._mutex:
            settings, _, _ = config.Config.load()
            self._picker_path(settings).unlink(missing_ok=True)
            return codex_picker.disable()

    def picker_default_model(self):
        settings, _, _ = config.Config.load()
        try:
            saved = json.loads(self._picker_path(settings).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        default = saved.get("default_model_id") if isinstance(saved, dict) else None
        return default if isinstance(default, str) else None

    def _cleanup_owned(self, runtime):
        """Attempt router and model cleanup independently; preserve failed owners."""
        clean = True
        if self._router is not None:
            try:
                self._router.stop()
                self._router = None
            except Exception:
                clean = False
        try:
            runtime[2].stop_all()
            clean = Launcher()._confirm_clean(runtime[2]) and clean
        except Exception:
            clean = False
        return clean

    def _retry_recovery(self):
        """Retry retained tuning ownership and outstanding compare-and-restore."""
        recovery = self._recovery
        if recovery is None:
            return self._recovery_lock is None
        if not self._cleanup_owned(recovery["runtime"]):
            return False
        try:
            if recovery["applied"] is not None:
                config.write_model_fields(recovery["runtime"][0].data_dir,
                    recovery["model_id"], *recovery["before"],
                    expected_fields=recovery["applied"])
            self._recovery_lock.__exit__()
        except Exception:
            return False
        self._update(recovery["job_id"], state="failed", cleanup_confirmed=True,
                     recovery_completed=True, settings_applied=False,
                     error="Tuning failed; owned recovery completed.")
        self._recovery_lock, self._recovery = None, None
        return True

    def stop_local_model(self):
        """Stop only the processes owned by this helper's chat controller."""
        with self._mutex:
            if not self._retry_recovery():
                return {"state": "recovery_failed", "cleanup_confirmed": False}
            if self._chat is None:
                return {"state": "stopped", "cleanup_confirmed": True}
            if not self._cleanup_owned(self._chat):
                return {"state": "recovery_failed", "cleanup_confirmed": False}
            self._chat_lock.__exit__()
            self._chat, self._chat_lock = None, None
            return {"state": "stopped", "cleanup_confirmed": True}

    def close(self):
        """Orderly transport shutdown waits for cooperative tuning cleanup."""
        with self._mutex:
            for event in self._cancel.values():
                event.set()
        if self._thread:
            self._thread.join()
        return self.stop_local_model()
