"""Inference-backend seam (M18.1): the engine behind a model is pluggable.

LOCITIZE serves every model through llama.cpp today, and the flag spellings for
that engine (--model, --ctx-size, --n-gpu-layers, --alias, --metrics, --mmproj)
were inlined in models.build_start_spec. This module lifts exactly those
ENGINE-SPECIFIC facts behind a small interface, so a second engine - a
TensorRT-LLM server, a NIM container, a vLLM process - plugs in by implementing
the same three answers and registering under a name:

    1. Which binary serves this model on this machine (binary_path).
    2. The engine's own argv core for "serve this file at this port with this
       context and offload" (core_command).
    3. The engine's readiness endpoint (health_path).

Everything NOT engine-specific stays where it was, in ModelRegistry
.build_start_spec: validation, path confinement, owner server_args and the
managed-flag strip, speculative-decoding flags, timeouts, and ServiceSpec
assembly. A backend cannot bypass those rules - it only fills in the engine
vocabulary, which is the part that varies between engines.

A model row selects its engine with an optional `backend:` field in models.yaml
(absent -> "llama-cpp", so every existing registry is unchanged). An unknown
name refuses with the list of registered engines rather than guessing.

Pure: no subprocess, no I/O. ASCII only.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

DEFAULT_BACKEND = "llama-cpp"

# The offload request that means "everything": the registry's 999 (more layers
# than any model has; config.default_gpu_layers) and llama.cpp's own -1. Both
# reach llama-server as -1, which the engine reads as "fit": at load it measures
# the card's FREE memory, projects the model's real footprint at the requested
# context, and keeps whole layers on the CPU when the whole model would not fit
# (--fit, a late-2025 llama.cpp addition, on by default; confirmed in the
# installed build 10701's --help. The setup wizard installs the current
# release; on a build old enough to lack --fit, -1 means the engine's own
# default instead, so keep the binary current).
#
# Why this is not the same as 999. On Windows a CUDA allocation that exceeds
# the card does not fail: WDDM pages it through system RAM, so the model LOADS,
# /health says ok, and every token runs over PCIe. Measured 2026-09-03 on the
# reference card (16.3 GB), Qwen3.8-27B IQ4_XS at its 49152 context beside the
# GPU voice engine: --n-gpu-layers 999 put 782 MB into shared memory and gave
# 5.2 tok/s generation, 43 tok/s prompt processing (a 2842-token chat history
# took 67s before the first word - the owner's "Open WebUI does not chat").
# --n-gpu-layers -1 kept 59 of 66 layers on the card (158 MB shared) and gave
# 24.3 tok/s and 1204 tok/s. A model that fits whole still gets every layer
# (gpt-oss-20b: 25/25 either way). An explicit count below the sentinel is the
# owner's hand tuning and is passed through untouched.
#
# The margin. fit's own default leaves 1024 MiB unused, and on a model that
# only just fits that is layers it did not need to move: the 27B above at
# 24.3 tok/s against 32.6 with the whole model on the card, gemma-4-26b 92
# against 108. The owner's rule is "do not affect my tok/s". But no fixed
# margin is right on Windows, because the free-memory figure fit works from
# does not see other processes (gpu_ledger explains the readings): 256 MiB
# gave the 27B 32.6 tok/s in a sweep and 6.0 tok/s an hour later from the
# desktop with three more applications on the card. So the margin is
# measured per launch - gpu_ledger.fit_budget compares what the engine will
# see with what the card really has - and arrives here as `fit_target`,
# already computed. None (nothing measurable) sends no --fit-target and the
# engine keeps its own default. It is sent only with the fit request itself,
# never with an explicit layer count, and before owner server_args, so a
# row's own --fit-target or --fit off wins (llama.cpp takes the last value).
ALL_LAYERS = 999
FIT_LAYERS = -1


def offload_value(gpu_layers: int) -> int:
    """The --n-gpu-layers value for a row's gpu_layers: the fit request for
    "everything", the owner's number otherwise."""
    if gpu_layers < 0 or gpu_layers >= ALL_LAYERS:
        return FIT_LAYERS
    return gpu_layers


@runtime_checkable
class InferenceBackend(Protocol):
    """The three engine-specific answers a serving engine must provide."""

    name: str

    def binary_path(self, settings: Any) -> str:
        """Absolute path to the engine's server binary on this machine, or ''
        when not configured (the caller raises with a remedy)."""
        ...

    def core_command(
        self,
        binary: str,
        model: Any,
        settings: Any,
        port: int,
        ctx_size: int,
        gpu_layers: int,
        fit_target: int | None = None,
    ) -> list[str]:
        """The engine's argv core: binary + model + host/port + context/offload,
        plus engine-required extras (alias, metrics, projector). Loopback host is
        part of the contract: a backend must bind 127.0.0.1, never the LAN."""
        ...

    def health_path(self, settings: Any) -> str | None:
        """The engine's readiness endpoint (e.g. '/health'), or None for a bare
        TCP-connect readiness check."""
        ...


class LlamaCppBackend:
    """The built-in engine: llama.cpp's llama-server.

    Flag spellings confirmed against the installed binary --help (build 10037):
    --model/--host/--port/--ctx-size/--n-gpu-layers/--alias/--metrics and
    `-mm, --mmproj FILE`. These constants moved here from models.py so that
    adding another engine never means editing the registry's orchestration.
    """

    name = DEFAULT_BACKEND

    _MMPROJ_FLAG = "--mmproj"
    _ALIAS_FLAG = "--alias"

    def binary_path(self, settings: Any) -> str:
        return str(settings.paths.llama_cpp or "")

    def core_command(
        self,
        binary: str,
        model: Any,
        settings: Any,
        port: int,
        ctx_size: int,
        gpu_layers: int,
        fit_target: int | None = None,
    ) -> list[str]:
        command = [
            binary,
            "--model",
            model.location,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(ctx_size),
            "--n-gpu-layers",
            str(offload_value(gpu_layers)),
        ]
        if offload_value(gpu_layers) == FIT_LAYERS and fit_target is not None:
            command.extend(["--fit-target", str(int(fit_target))])
        # Report the human registry name through the API instead of the gguf
        # path, so chat model pickers stay readable (owner request 2026-08-14).
        # Injected before owner server_args so an explicit --alias there wins.
        if model.name:
            command.extend([self._ALIAS_FLAG, model.name])
        # Expose Prometheus /metrics for the live monitor (Architecture G5),
        # gated by settings so a build that rejects the flag can turn it off.
        if settings.services.llama_cpp_metrics:
            command.append("--metrics")
        # A ceiling on reply length for requests that set none (Open WebUI
        # sends no max_tokens). Without it a small model that falls into a
        # loop generates forever and holds the only slot, so every later
        # message waits (seen 2026-10-04: a 0.6B model at 22,000 tokens and
        # counting). Owner server_args come later, so an explicit
        # --n-predict there still wins.
        command.extend(["--n-predict", str(DEFAULT_MAX_REPLY_TOKENS)])
        # M8.1 (vision): a row carrying an mmproj path is served multimodally.
        if model.mmproj:
            command.extend([self._MMPROJ_FLAG, model.mmproj])
        return command

    def health_path(self, settings: Any) -> str | None:
        return settings.services.llama_cpp_health_path or None


# Long enough for a reasoning model's thinking plus a full answer.
DEFAULT_MAX_REPLY_TOKENS = 16384


_REGISTRY: dict[str, InferenceBackend] = {
    DEFAULT_BACKEND: LlamaCppBackend(),
}


def register(backend: InferenceBackend) -> None:
    """Add an engine. A future TensorRT-LLM/NIM/vLLM backend calls this once."""
    _REGISTRY[backend.name] = backend


def get_backend(name: str | None) -> InferenceBackend:
    """Resolve a backend by name; absent/empty means the built-in llama.cpp.

    An unknown name raises with the registered list - refusing loudly beats
    silently serving a model with the wrong engine's flags.
    """
    key = (name or DEFAULT_BACKEND).strip().lower()
    backend = _REGISTRY.get(key)
    if backend is None:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"unknown inference backend '{key}' (registered: {known}); "
            f"fix this model's `backend:` in models.yaml"
        )
    return backend


def registered_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
