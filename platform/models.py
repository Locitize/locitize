"""Model registry for the LOCITIZE platform.

Wraps the parsed models.yaml (config.ModelRegistryData) with the queries the
launcher and health system need, and turns "start model X" into a declarative
ServiceSpec for the service manager to launch (Architecture section 2, Data Model
section 1).

This module holds no mutable global state; it is constructed with the parsed
registry and a Settings object and exposes read-only queries plus a pure
spec-builder. It never launches processes itself (that is services.py's job).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import backends
import finetune
import gpu_ledger
from config import Model, ModelRegistryData, Settings
from logger import get_logger
from services import ServiceSpec, resolve_service_cwd, strip_managed_flags

# Reserved id namespace for discovered fine-tunes (single-sourced in finetune.py).
_FT_PREFIX = finetune.FT_PREFIX

# Sentinel distinguishing "caller did not override" from "caller passed None to
# force the feature off" for the optional spec-decoding overrides on
# build_start_spec (used by the benchmark sweep, M5.4/M5.6).
_UNSET = object()

# Single-sourced speculative-decoding flag spellings (M5.6). CONFIRMED against the
# installed binary's own help output:
#   llama-server.exe --help  (build 10037 (56d6e9dde), 2026-07-19)
# which lists, under "----- speculative params -----":
#   --spec-draft-model, -md, --model-draft FNAME   draft model for spec decoding
#   --spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,
#                ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache
#   --spec-draft-n-max N                           tokens to draft (default 3)
#   --spec-draft-n-min N                           minimum draft tokens
#   --spec-draft-ngl, -ngld, --n-gpu-layers-draft N   draft layers in VRAM
#   --spec-ngram-simple-size-n / -size-m / -min-hits  ngram-simple knobs
#   --spec-ngram-mod-n-min / -n-max / -n-match        ngram-mod knobs
# The flags are kept here as constants (never in models.yaml) so a future build
# rename is a one-line code edit, not a per-entry rewrite (Data Model 7.5). Note
# the plan's guessed spellings (--draft-model/--draft-max) do NOT exist in this
# build; these are the real ones.
_SPEC_DRAFT_MODEL_FLAG = "--spec-draft-model"
# M8.1 (vision): the multimodal-projector flag. CONFIRMED against the installed
# binary's own help output on 2026-07-19:
#   -mm,   --mmproj FILE   path to a multimodal projector file. (env: LLAMA_ARG_MMPROJ)
# Single-sourced here (never in models.yaml) so a future build rename is a one-line
# edit. The long form is used for readability in the emitted argv.
_MMPROJ_FLAG = "--mmproj"
# Owner request 2026-08-14: chat UIs (Open WebUI, llama-ui) display the API's
# model id, which defaults to the gguf FILE PATH -- ugly in the model picker.
# `--alias` overrides the API-reported name. CONFIRMED against the installed
# binary's own help output:
#   -a, --alias STRING   set model name aliases, comma-separated (to be used by API)
# Single-sourced here (never in models.yaml); the value is the row's human
# `name` field, so the picker shows "Qwen3.6 27B ThinkingCap Abliterated".
_ALIAS_FLAG = "--alias"
_SPEC_FLAG_BY_KEY = {
    "spec_type": "--spec-type",
    "draft_max": "--spec-draft-n-max",
    "draft_min": "--spec-draft-n-min",
    "draft_gpu_layers": "--spec-draft-ngl",
    "ngram_simple_n": "--spec-ngram-simple-size-n",
    "ngram_simple_m": "--spec-ngram-simple-size-m",
    "ngram_simple_min_hits": "--spec-ngram-simple-min-hits",
    "ngram_mod_n_min": "--spec-ngram-mod-n-min",
    "ngram_mod_n_max": "--spec-ngram-mod-n-max",
    "ngram_mod_match": "--spec-ngram-mod-n-match",
}

# Owner request 2026-09-02: reasoning/thinking control. Single-sourced here for
# the same reason as the speculative flags above, and CONFIRMED the same way -
# against the installed binary's own help output:
#   llama-server.exe --help   (build b10701-cc231cb0d, run 2026-09-02)
# which lists:
#   -rea,  --reasoning [on|off|auto]  Use reasoning/thinking in the chat
#                                     (default: 'auto' (detect from template))
#   --reasoning-effort LEVEL          reasoning effort level given to the chat
#                                     template: 'default' to keep the template
#                                     default, or a level such as 'minimal',
#                                     'low', 'medium', 'high' or 'xhigh'
#   --reasoning-budget N              token budget for thinking: -1 for
#                                     unrestricted, 0 for immediate end,
#                                     N>0 for token budget (default: -1)
# Note the level list in that help text is llama-server's vocabulary, NOT the
# model's: the accepted levels are whatever the row's chat template implements
# (measured 2026-09-02: Qwen3.8-27B-UD-IQ4_XS accepts only xhigh/medium/low and
# aliases 'high' up to 'xhigh'). config.parse_reasoning therefore validates the
# SHAPE of the value and this table only spells the flag.
_REASONING_ENABLED_FLAG = "--reasoning"
_REASONING_FLAG_BY_KEY = {
    "effort": "--reasoning-effort",
    "budget": "--reasoning-budget",
}


class ModelRegistry:
    """Read model queries plus llama.cpp start-spec construction."""

    def __init__(
        self,
        data: ModelRegistryData,
        settings: Settings,
        fit_budget_fn: Any = None,
    ) -> None:
        self._data = data
        self._settings = settings
        # The per-launch fit-margin reading (gpu_ledger.fit_budget); injectable
        # so specs build without a GPU or a binary in tests.
        self._fit_budget_fn = fit_budget_fn if fit_budget_fn is not None else gpu_ledger.fit_budget
        # Index by id for O(1) lookup; ids are validated unique at load time.
        self._by_id = {m.id: m for m in data.models}

    @property
    def version(self) -> int:
        return self._data.version

    def all(self) -> list[Model]:
        """Every declared model, in file order."""
        return list(self._data.models)

    def reload(self, data: ModelRegistryData) -> None:
        """Swap in freshly loaded registry data (Desktop Refresh, owner request
        2026-08-13). Keeps the object identity stable so existing holders
        (controllers, panels) see the new rows without rewiring."""
        self._data = data
        self._by_id = {m.id: m for m in data.models}

    def installed(self) -> list[Model]:
        """Models with status 'installed' (shown and launchable in the menu)."""
        return [m for m in self._data.models if m.status == "installed"]

    def launchable(self) -> list[Model]:
        """Installed models whose file location is set (a precondition to launch).

        A future model or one with an empty location is listed but not launchable;
        this keeps the launcher from trying to start a model with no file.
        """
        return [m for m in self.installed() if m.location]

    def get(self, model_id: str) -> Model | None:
        """Look up a model by id, or None if unknown.

        Manual rows are looked up first and always win. Only an id in the reserved
        "ft:" namespace (which config refuses to load from models.yaml, so it can
        never be a manual id) falls through to the discovered set, which is what
        lets a discovered fine-tune be served through the unchanged
        build_start_spec path without a Register step first (M13.6).
        """
        found = self._by_id.get(model_id)
        if found is not None:
            return found
        if model_id.startswith(_FT_PREFIX):
            for model in self.discovered():
                if model.id == model_id:
                    return model
        return None

    # ---- M13 discovery (additive; all() / installed() / launchable() unchanged) --

    def discovered(self) -> list[Model]:
        """Fine-tuned models found on disk that are NOT already manual rows.

        Recomputed from the filesystem on each call (behind finetune's 5-second TTL
        cache), never persisted, and never written back into models.yaml. Any
        discovery failure degrades to an empty list so a missing or unreadable
        outputs folder can never break the model list.
        """
        return [finetune.to_model(item, self._settings) for item in self.discovered_items()]

    def discovered_items(self) -> list[Any]:
        """The surviving DiscoveredFineTune records (dedup applied), for the UI.

        The UI needs the raw records as well as the Models, because they carry the
        run name, quant, size, and mtime columns the Fine-tune page renders.
        """
        result = finetune.scan_outputs(self._settings)
        marked, _matched = finetune.dedup_against_registry(result.items, self._data.models)
        return [item for item in marked if not item.already_registered]

    def registered_finetune_ids(self) -> set[str]:
        """Manual row ids that a discovered file folded into (the 'fine-tune' badge)."""
        result = finetune.scan_outputs(self._settings)
        _marked, matched = finetune.dedup_against_registry(result.items, self._data.models)
        return matched

    def all_merged(self) -> list[Model]:
        """Manual rows first, then the discovered survivors (deterministic order).

        This is the only merged view; all()/installed()/launchable() deliberately
        keep returning manual rows only, so every existing caller, test, and health
        probe is bit-for-bit unaffected by discovery (Architecture M13.6).
        """
        return self.all() + self.discovered()

    def build_start_spec(
        self,
        model_id: str,
        ctx_size: int | None = None,
        gpu_layers: int | None = None,
        *,
        server_args: list[str] | None = None,
        draft_model: Any = _UNSET,
        spec_config: Any = _UNSET,
        reasoning: Any = _UNSET,
    ) -> ServiceSpec:
        """Build the llama.cpp server ServiceSpec for a model.

        The command is data-driven from settings (binary path, port, health path)
        and the model row (location, context size, gpu layers, extra args), so no
        machine-specific path or flag is hardcoded (Architecture open risk R1).
        `ctx_size`/`gpu_layers`, when given, override the model row's values (used
        by the --smoke-start harness to fit a constrained VRAM budget). Raises
        ValueError for an unknown model or one that is not launchable.

        M5 sweep overrides (keyword-only, all optional; existing positional callers
        are unaffected): `server_args` replaces the model's extra flags for one
        benchmark scenario; `draft_model`/`spec_config` override the model's
        speculative-decoding fields (pass None to force speculation OFF for a
        scenario, omit to use the model's own values). This is how the benchmark
        SweepPlan enumerates {baseline}x{draft on/off}x{ngram on/off} without
        mutating the registry (M5.4/M5.6).

        The port baked into the argv here is the *requested* port from settings.
        If the allocator reassigns it at launch time, ManagedProcess rewrites the
        --port value to the resolved port (Reviewer M-1) -- this method does not
        need to know the final port. The health endpoint is passed as a bare
        health_path, not a full URL, so the readiness probe host is fixed at
        127.0.0.1 by ServiceSpec/build_readiness_url (Security SEC-1).
        """
        model = self.get(model_id)
        if model is None:
            raise ValueError(f"unknown model id '{model_id}'")
        if model.status != "installed":
            raise ValueError(f"model '{model_id}' is not installed (status={model.status})")
        if not model.location:
            raise ValueError(
                f"model '{model_id}' has no location set; "
                f"set it in models.yaml or LOCITIZE_MODEL_*"
            )
        # M13 path confinement (Architecture M13.9 boundary 2). A DISCOVERED model's
        # location came from a filesystem scan rather than the owner's models.yaml,
        # so it is re-checked here - immediately before it becomes llama-server's
        # -m value - against the configured outputs root. This guard therefore
        # precedes every spec build that uses a discovered path, and a manual row
        # is completely unaffected.
        if getattr(model, "source", "registry") == finetune.DISCOVERED_SOURCE:
            model = replace(
                model, location=finetune.resolve_serve_path(model.location, self._settings)
            )

        s = self._settings
        # M18.1: engine resolution happens once, before anything engine-specific.
        engine = backends.get_backend(getattr(model, "backend", None))
        binary = engine.binary_path(s)
        if not binary:
            raise ValueError(
                f"no server binary configured for backend '{engine.name}'; for "
                f"llama.cpp set paths.llama_cpp in settings.yaml or "
                f"LOCITIZE_LLAMACPP_PATH"
            )
        port = s.ports.llama_cpp
        resolved_ctx = ctx_size if ctx_size is not None else model.context_size
        resolved_gpu = gpu_layers if gpu_layers is not None else model.gpu_layers
        # M18.1 (backend seam): the ENGINE-SPECIFIC argv core - binary + model +
        # loopback host/port + context/offload + engine extras (alias, metrics,
        # mmproj) - comes from the model's inference backend (llama.cpp unless
        # the row's `backend:` says otherwise). Everything after this line is
        # engine-independent orchestration and stays here: the managed-flag
        # strip, owner server_args, speculation flags, timeouts, ServiceSpec.
        # Owner rule 2026-09-03 ("do not affect my tok/s"): when the row asks
        # for every layer that fits, measure the fit margin this card needs
        # right now rather than trusting the engine's fixed default - see
        # gpu_ledger.fit_budget for why the engine's own free-memory reading
        # cannot be trusted on Windows. Two readings, ~0.4s, taken here so they
        # describe the card an instant before the load.
        fit_target = None
        if backends.offload_value(resolved_gpu) == backends.FIT_LAYERS:
            budget = self._fit_budget_fn(binary)
            if budget is not None:
                fit_target = budget.target_mib
                get_logger("launcher").info("%s: %s", model.id, budget.describe())
            else:
                get_logger("launcher").info(
                    "%s: fit margin not measurable here; llama-server keeps its default",
                    model.id,
                )
        command = engine.core_command(
            binary, model, s, port, resolved_ctx, resolved_gpu, fit_target=fit_target
        )
        # L-4 (managed-flag rejection): --host/--port are resolved by the platform
        # and already baked into the argv above. Strip them (with a WARNING) from
        # owner-supplied server_args before appending, so a stray --port in
        # models.yaml can never append a second value and override the
        # platform-resolved port (which would silently defeat the M-1 fix).
        # SEC-2 (trust boundary): the remaining server_args flow verbatim from
        # models.yaml (or a sweep override) into the argv. This is owner-authored,
        # single-user, local config passed as a discrete argument list (never a
        # shell string), so no metacharacter or command-chaining injection is
        # possible -- the worst case is the owner's own llama.cpp behaving as the
        # owner's own flags direct.
        resolved_server_args = (
            server_args if server_args is not None else model.server_args
        )
        command.extend(strip_managed_flags(f"llama_cpp:{model.id}", resolved_server_args))

        # M5.6: append speculative-decoding flags, data-driven from the model's
        # draft_model/spec_config (or a sweep override). These are platform-injected
        # additive flags with no --host/--port, so they bypass strip_managed_flags
        # untouched (L-4). Off by default (no fields set -> no flags appended).
        resolved_draft = model.draft_model if draft_model is _UNSET else draft_model
        resolved_spec = model.spec_config if spec_config is _UNSET else spec_config
        command.extend(self._build_spec_args(resolved_draft, resolved_spec))

        # Owner request 2026-09-02: reasoning flags, appended by the same rules as
        # the speculation flags above - platform-injected, additive, no
        # --host/--port, so strip_managed_flags does not apply. Off by default
        # (no `reasoning:` on the row -> no flags -> llama-server keeps its own
        # 'auto' detection and the template's own default effort).
        resolved_reasoning = model.reasoning if reasoning is _UNSET else reasoning
        command.extend(self._build_reasoning_args(resolved_reasoning))

        # D-M4-2: a large model on a cold load can legitimately exceed the global
        # readiness timeout (the 27B needs >60s cold), so an optional per-model
        # ready_timeout_s in models.yaml overrides the global default. None falls
        # back to services.ready_timeout_s so smaller models are unaffected.
        ready_timeout = (
            float(model.ready_timeout_s)
            if model.ready_timeout_s is not None
            else s.services.ready_timeout_s
        )
        return ServiceSpec(
            name=f"llama_cpp:{model.id}",
            command=command,
            # Never None - see resolve_service_cwd. llama-server's -m value is an
            # absolute path by contract, so the working directory cannot change
            # which weights are loaded; it only decides where a stray relative
            # file would land, and "the install tree" is the wrong answer (W1).
            cwd=resolve_service_cwd(s),
            env={},
            port=port,
            health_path=engine.health_path(s),
            ready_timeout_s=ready_timeout,
            # A model server holds nothing worth a graceful shutdown, and
            # llama-server ignores the polite stop on Windows anyway, so the
            # full grace period was pure waiting on every model swap (measured
            # ~11s to drop a model). One second, then the forced stop.
            stop_timeout_s=min(1.0, s.services.stop_timeout_s),
            # D-M4-3: keep the llama.cpp child log across starts so a failed start's
            # trace is not overwritten by the next attempt (this is the evidence the
            # D-M4-1 investigation needed).
            append_log=True,
        )

    def _build_spec_args(
        self, draft_model: str | None, spec_config: dict[str, Any] | None
    ) -> list[str]:
        """Turn draft_model/spec_config into the confirmed server flags (M5.6).

        draft_model may be the id of another registry model (resolved to its
        on-disk location) or a direct path; an id that resolves to an empty
        location raises so the benchmark records the scenario as skipped-with-remedy
        rather than launching a broken draft. spec_config keys are mapped to flags
        via the single-sourced _SPEC_FLAG_BY_KEY table (confirmed vs --help); an
        unknown key is ignored rather than guessed. Returns an empty list when no
        speculation is configured (the common case), so a normal model start argv
        is byte-identical to the pre-M5 behaviour.
        """
        args: list[str] = []
        if draft_model:
            draft_path = self._resolve_draft_path(draft_model)
            args.extend([_SPEC_DRAFT_MODEL_FLAG, draft_path])
        if spec_config:
            # Deterministic flag order (spec_type first, then the rest in table
            # order) so the argv is stable and unit-testable.
            for key, flag in _SPEC_FLAG_BY_KEY.items():
                if key in spec_config and spec_config[key] is not None:
                    args.extend([flag, str(spec_config[key])])
        return args

    def _build_reasoning_args(
        self, reasoning: dict[str, Any] | None
    ) -> list[str]:
        """Turn a row's `reasoning` mapping into the confirmed server flags.

        Mirrors _build_spec_args exactly: deterministic flag order (enabled,
        then the table order) so the argv is stable and unit-testable, and an
        empty/None mapping yields [] so an ordinary model start argv is
        byte-identical to the pre-change behaviour.

        `enabled` is rendered as the on/off words the binary documents, never as
        Python's True/False - `--reasoning True` is not a value llama-server
        accepts. Values are NOT vocabulary-checked here; config.parse_reasoning
        has already checked the shape, and the model's chat template is the
        authority on which effort levels exist (see _REASONING_FLAG_BY_KEY).
        """
        if not reasoning:
            return []
        args: list[str] = []
        if "enabled" in reasoning and reasoning["enabled"] is not None:
            args.extend(
                [_REASONING_ENABLED_FLAG, "on" if reasoning["enabled"] else "off"]
            )
        for key, flag in _REASONING_FLAG_BY_KEY.items():
            if key in reasoning and reasoning[key] is not None:
                args.extend([flag, str(reasoning[key])])
        return args

    def _resolve_draft_path(self, draft_model: str) -> str:
        """Resolve a draft_model reference to a filesystem path.

        If it matches a registry id, use that model's location (raising when the
        location is empty, since a draft with no file cannot load); otherwise treat
        it as a direct path. No URL/token is ever accepted here (Data Model 7.5).
        """
        referenced = self.get(draft_model)
        if referenced is not None:
            if not referenced.location:
                raise ValueError(
                    f"draft model '{draft_model}' has no location set; "
                    f"set it in models.yaml before using speculative decoding"
                )
            return referenced.location
        return draft_model
