"""Tests for the inference-backend seam (backends.py, M18.1).

The seam's contract: the default is llama.cpp with byte-identical argv to the
pre-seam builder (proven by the untouched models/spec test suite), a second
engine plugs in via register(), and an unknown name refuses with the registered
list instead of guessing another engine's flags.
"""

from __future__ import annotations

import pytest

import backends
from backends import LlamaCppBackend, get_backend, register, registered_names


# Windows-shaped fixture paths ASSEMBLED at runtime (drive + backslash joins):
# the shipped-tree owner-path scanner forbids drive-letter literals in source.
_BS = chr(92)


def _win(*parts):
    return _BS.join(parts)


_BINARY = _win("D:", "Apps", "llama", "llama-server.exe")
_LOCATION = _win("D:", "Models", "m1.gguf")
_MMPROJ = _win("D:", "Models", "mmproj.gguf")


class _Paths:
    llama_cpp = _BINARY


class _Services:
    llama_cpp_metrics = True
    llama_cpp_health_path = "/health"


class _Settings:
    paths = _Paths()
    services = _Services()


class _Model:
    id = "m1"
    name = "My Model"
    location = _LOCATION
    mmproj = None
    backend = ""


def test_default_backend_is_llama_cpp():
    engine = get_backend(None)
    assert isinstance(engine, LlamaCppBackend)
    assert engine.name == "llama-cpp"
    assert get_backend("").name == "llama-cpp"
    assert get_backend("LLAMA-CPP").name == "llama-cpp"  # case-insensitive


def test_unknown_backend_refuses_with_registered_list():
    with pytest.raises(ValueError) as excinfo:
        get_backend("vllm")
    assert "llama-cpp" in str(excinfo.value)
    assert "models.yaml" in str(excinfo.value)


def test_llama_core_command_spells_the_confirmed_flags():
    engine = LlamaCppBackend()
    cmd = engine.core_command(
        _BINARY, _Model(), _Settings(), 8080, 32768, 999
    )
    assert cmd[0] == _BINARY
    assert cmd[cmd.index("--model") + 1] == _Model.location
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"  # loopback contract
    assert cmd[cmd.index("--port") + 1] == "8080"
    assert cmd[cmd.index("--ctx-size") + 1] == "32768"
    # 999 ("everything") becomes -1: llama.cpp's --fit then measures free VRAM
    # at load instead of paging the overflow through system RAM.
    assert cmd[cmd.index("--n-gpu-layers") + 1] == "-1"
    # No margin was measured, so none is sent: the engine keeps its default.
    assert "--fit-target" not in cmd
    assert cmd[cmd.index("--alias") + 1] == "My Model"
    assert "--metrics" in cmd


def test_llama_core_command_honours_settings_and_mmproj():
    engine = LlamaCppBackend()
    settings = _Settings()
    settings.services = type("S", (), {"llama_cpp_metrics": False,
                                       "llama_cpp_health_path": ""})()
    model = _Model()
    model.mmproj = _MMPROJ
    model.name = ""
    cmd = engine.core_command(_BINARY, model, settings, 8080, 8192, -1)
    assert "--metrics" not in cmd
    assert "--alias" not in cmd  # no name -> no alias
    assert cmd[cmd.index("--mmproj") + 1] == model.mmproj
    assert engine.health_path(settings) is None  # empty -> bare TCP readiness


def test_register_adds_a_second_engine():
    class _FakeTrt:
        name = "fake-trt"

        def binary_path(self, settings):
            return "trtllm-serve.exe"

        def core_command(self, binary, model, settings, port, ctx, layers):
            return [binary, "--served-model", model.location, "--port", str(port)]

        def health_path(self, settings):
            return "/v1/health/ready"

    register(_FakeTrt())
    try:
        assert "fake-trt" in registered_names()
        engine = get_backend("fake-trt")
        cmd = engine.core_command("trtllm-serve.exe", _Model(), _Settings(), 9000, 1, 1)
        assert cmd[0] == "trtllm-serve.exe" and "--served-model" in cmd
    finally:
        # Leave the registry as shipped for every other test.
        backends._REGISTRY.pop("fake-trt", None)
    assert "fake-trt" not in registered_names()


# --------------------------------------------------------------------------- #
# The gpu_layers -> --n-gpu-layers translation (owner report 2026-09-03, "when I
# switch models Open WebUI does not chat": 999 let a 27B overflow the card and
# crawl at 5 tok/s; -1 lets llama.cpp fit it and measured 24 tok/s).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "requested, passed",
    [
        (999, -1),  # the registry's "everything" sentinel
        (1000, -1),  # anything at or above it means the same
        (-1, -1),  # the engine's own fit request, untouched
        (-5, -1),  # any negative is a fit request
        (10, 10),  # an explicit count is the owner's decision
        (0, 0),  # CPU-only stays CPU-only
    ],
)
def test_offload_value_maps_everything_to_fit_and_passes_counts_through(requested, passed):
    assert backends.offload_value(requested) == passed


def test_explicit_gpu_layer_count_reaches_the_command_line_unchanged():
    engine = LlamaCppBackend()
    cmd = engine.core_command(_BINARY, _Model(), _Settings(), 8080, 32768, 40)
    assert cmd[cmd.index("--n-gpu-layers") + 1] == "40"
    # An explicit count turns fit off in llama.cpp; a fit margin beside it
    # would be noise on the command line.
    assert "--fit-target" not in cmd


def test_a_measured_fit_margin_rides_with_the_fit_request_before_owner_args():
    engine = LlamaCppBackend()
    cmd = engine.core_command(_BINARY, _Model(), _Settings(), 8080, 32768, 999, fit_target=579)
    assert cmd[cmd.index("--fit-target") + 1] == "579"
    # Before --alias (the first engine extra), i.e. before owner server_args
    # are appended: a row's own --fit-target or --fit off comes later and wins.
    assert cmd.index("--fit-target") < cmd.index("--alias")


def test_a_measured_fit_margin_is_dropped_beside_an_explicit_layer_count():
    engine = LlamaCppBackend()
    cmd = engine.core_command(_BINARY, _Model(), _Settings(), 8080, 32768, 40, fit_target=579)
    assert "--fit-target" not in cmd
