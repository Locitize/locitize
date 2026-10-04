"""Vision client + mmproj plumbing tests (keywords 'vision_client', 'vision_mmproj').

Headless and deterministic (Architecture M8.5, AC8): no GPU, no real server, no
real image inference. A fake opener captures the exact /v1/chat/completions request
so the multimodal framing (base64 data: URL + text prompt) is asserted against the
wire shape; a fake ModelController proves describe() requests the model switch; and
build_start_spec is checked to append --mmproj only when the model row carries one.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from config import Model, ModelRegistryData, Settings
from fakes import FakeHttpResponse
from models import ModelRegistry
from services import ServiceStatus
from vision import (
    QwenVisionModel,
    VisionClient,
    VisionError,
    build_vision_messages,
    encode_image_data_uri,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "vision_red_square.png"


class _CapturingOpener:
    """A urlopen-compatible callable that records the request and returns a body."""

    def __init__(self, body: bytes) -> None:
        self._body = body
        self.request = None
        self.timeout = None

    def __call__(self, request, timeout=None):
        self.request = request
        self.timeout = timeout
        return FakeHttpResponse(self._body)


def _answer_body(text: str) -> bytes:
    """A minimal non-streaming chat-completions JSON reply carrying `text`."""
    return json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": text}}]}
    ).encode("utf-8")


# --------------------------------------------------------------------------- #
# vision_client: request framing + response parsing + honest failure
# --------------------------------------------------------------------------- #


def test_vision_client_builds_multimodal_payload():
    """describe() posts a content array with a base64 data: URL and the prompt."""
    opener = _CapturingOpener(_answer_body("A red square on white."))
    client = VisionClient(8080, opener=opener)

    answer = client.describe(_FIXTURE, "What color is the shape?")

    assert answer == "A red square on white."
    # The captured request is the real urllib Request; inspect its JSON body.
    body = json.loads(opener.request.data.decode("utf-8"))
    assert opener.request.full_url == "http://127.0.0.1:8080/v1/chat/completions"
    content = body["messages"][0]["content"]
    kinds = {part["type"] for part in content}
    assert kinds == {"text", "image_url"}
    text_part = next(p for p in content if p["type"] == "text")
    image_part = next(p for p in content if p["type"] == "image_url")
    assert text_part["text"] == "What color is the shape?"
    # The image is carried as a base64 data: URL (never a file path or a URL).
    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    expected = base64.b64encode(_FIXTURE.read_bytes()).decode("ascii")
    assert url.endswith(expected)


def test_vision_client_missing_image_is_honest_error():
    """A missing image raises VisionError before any request is sent."""
    opener = _CapturingOpener(_answer_body("unused"))
    client = VisionClient(8080, opener=opener)
    with pytest.raises(VisionError):
        client.describe("does-not-exist.png", "hi")
    assert opener.request is None  # never posted


def test_vision_client_malformed_response_raises_not_fabricates():
    """A response with no answer content raises VisionError (never a fake answer)."""
    opener = _CapturingOpener(b'{"choices": []}')
    client = VisionClient(8080, opener=opener)
    with pytest.raises(VisionError):
        client.describe(_FIXTURE, "hi")


def test_vision_client_non_json_response_raises():
    opener = _CapturingOpener(b"<html>not json</html>")
    client = VisionClient(8080, opener=opener)
    with pytest.raises(VisionError):
        client.describe(_FIXTURE, "hi")


def test_vision_encode_data_uri_and_messages_are_pure():
    """The pure helpers build the exact data: URL and message shape."""
    uri = encode_image_data_uri(_FIXTURE)
    assert uri.startswith("data:image/png;base64,")
    messages = build_vision_messages("describe", uri)
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][1]["image_url"]["url"] == uri


# --------------------------------------------------------------------------- #
# vision_client: QwenVisionModel drives a model switch via the controller
# --------------------------------------------------------------------------- #


class _FakeController:
    """Records switch requests; reports a running model + port like the real one."""

    def __init__(self, running_model_id=None, port=8080, switch_status=None) -> None:
        self.running_model_id = running_model_id
        self.running_port = port
        self._switch_status = switch_status or ServiceStatus.RUNNING
        self.switched_to: list[str] = []

    def switch(self, model_id: str):
        self.switched_to.append(model_id)
        self.running_model_id = model_id
        return self._switch_status


def test_vision_client_requests_model_switch():
    """describe() switches to the vision model when a different (or no) model runs."""
    controller = _FakeController(running_model_id=None, port=8091)
    opener = _CapturingOpener(_answer_body("It is red."))
    model = QwenVisionModel(
        controller, client_factory=lambda port: VisionClient(port, opener=opener)
    )

    answer = model.describe(str(_FIXTURE), "color?")

    assert answer == "It is red."
    assert controller.switched_to == ["qwen2-5-vl"]
    # The client was built on the controller-resolved port (SEC-1 loopback).
    assert opener.request.full_url.startswith("http://127.0.0.1:8091/")


def test_vision_client_no_switch_when_already_running():
    """No redundant switch when the vision model is already the running one."""
    controller = _FakeController(running_model_id="qwen2-5-vl", port=8080)
    opener = _CapturingOpener(_answer_body("red"))
    model = QwenVisionModel(
        controller, client_factory=lambda port: VisionClient(port, opener=opener)
    )
    model.describe(str(_FIXTURE), "color?")
    assert controller.switched_to == []


def test_vision_client_switch_failure_is_honest_error():
    """A model that will not reach RUNNING raises VisionError, never a fake answer."""
    controller = _FakeController(
        running_model_id=None, switch_status=ServiceStatus.STOPPED_ERROR
    )
    model = QwenVisionModel(controller)
    with pytest.raises(VisionError):
        model.describe(str(_FIXTURE), "color?")


# --------------------------------------------------------------------------- #
# vision_mmproj: build_start_spec appends --mmproj only when the row carries one
# --------------------------------------------------------------------------- #


def _settings_with_llama() -> Settings:
    settings = Settings()
    settings.paths.llama_cpp = "/locitize-test/llama.cpp/llama-server.exe"
    return settings


def _vision_registry(mmproj: str | None) -> ModelRegistry:
    model = Model(
        id="qwen2-5-vl",
        name="Qwen2.5 VL",
        description="vision",
        location="/locitize-test/models/Qwen2.5-VL.gguf",
        context_size=16384,
        gpu_layers=-1,
        mmproj=mmproj,
    )
    return ModelRegistry(ModelRegistryData(models=[model]), _settings_with_llama())


def test_vision_mmproj_flag_appended_when_row_has_mmproj():
    """A model row with an mmproj path yields `--mmproj <path>` in the argv."""
    registry = _vision_registry("/locitize-test/models/mmproj-BF16.gguf")
    spec = registry.build_start_spec("qwen2-5-vl")
    argv = spec.command
    assert "--mmproj" in argv
    assert argv[argv.index("--mmproj") + 1] == "/locitize-test/models/mmproj-BF16.gguf"


def test_vision_mmproj_flag_absent_without_mmproj():
    """A model row with no mmproj is a text-only start (no --mmproj flag)."""
    registry = _vision_registry(None)
    spec = registry.build_start_spec("qwen2-5-vl")
    assert "--mmproj" not in spec.command


# --------------------------------------------------------------------------- #
# resolve_vision_model_id - which registry row answers --describe.
#
# Defect found 2026-09-02: vision.VISION_MODEL_ID ("qwen2-5-vl") was the ONLY
# candidate, and a scan-imported registry names its rows after the FILE
# ("qwen2-5-vl-7b-instruct-q4_k_m"). On the maintainer's machine that meant
# --describe could not find a vision model while SIX were registered.
# --------------------------------------------------------------------------- #

from vision import VISION_MODEL_ID, resolve_vision_model_id  # noqa: E402


def _capability_registry(*rows) -> ModelRegistry:
    """A registry of (id, capabilities) pairs, all installed and launchable."""
    return ModelRegistry(
        ModelRegistryData(
            version=1,
            models=[
                Model(
                    id=model_id,
                    name=model_id,
                    description="d",
                    location=f"/locitize-test/models/{model_id}.gguf",
                    context_size=8192,
                    gpu_layers=-1,
                    status="installed",
                    capabilities=list(capabilities),
                )
                for model_id, capabilities in rows
            ],
        ),
        Settings(),
    )


def test_resolution_falls_back_to_the_vision_capability():
    """The exact reproduction of the defect: no row is named VISION_MODEL_ID."""
    registry = _capability_registry(
        ("qwen3-8-27b-ud-iq4_xs", ["tools", "reasoning"]),
        ("qwen2-5-vl-7b-instruct-q4_k_m", ["vision"]),
    )
    assert registry.get(VISION_MODEL_ID) is None  # the old lookup found nothing
    resolved, alternatives = resolve_vision_model_id(registry)
    assert resolved == "qwen2-5-vl-7b-instruct-q4_k_m"
    assert alternatives == []


def test_resolution_still_prefers_a_row_named_after_the_default_id():
    """A registry that DOES name its row this way behaves exactly as before."""
    registry = _capability_registry(
        ("kimi-vl-a3b-instruct-q6_k", ["vision"]),
        (VISION_MODEL_ID, ["vision"]),
    )
    resolved, alternatives = resolve_vision_model_id(registry)
    assert resolved == VISION_MODEL_ID
    assert alternatives == ["kimi-vl-a3b-instruct-q6_k"]


def test_an_explicit_preference_wins_over_everything():
    registry = _capability_registry(
        (VISION_MODEL_ID, ["vision"]),
        ("kimi-vl-a3b-thinking-2506-q4_k_m", ["vision"]),
    )
    resolved, _alts = resolve_vision_model_id(
        registry, "kimi-vl-a3b-thinking-2506-q4_k_m"
    )
    assert resolved == "kimi-vl-a3b-thinking-2506-q4_k_m"


def test_an_unresolvable_preference_does_not_silently_fall_back():
    """Answering with a different model than the owner named is not a remedy."""
    registry = _capability_registry(("kimi-vl-a3b-instruct-q6_k", ["vision"]))
    resolved, alternatives = resolve_vision_model_id(registry, "no-such-model")
    assert resolved == ""
    assert alternatives == ["kimi-vl-a3b-instruct-q6_k"]


def test_suggestions_are_models_that_can_see_not_every_row():
    """Offering a text-only model as the alternative to a vision model is not a
    remedy - it would answer about the image without looking at it."""
    registry = _capability_registry(
        ("gpt-oss-20b-f16", ["tools", "reasoning"]),
        ("qwen3-14b-q5_0", ["tools"]),
        ("kimi-vl-a3b-instruct-q6_k", ["vision"]),
    )
    _resolved, alternatives = resolve_vision_model_id(registry, "no-such-model")
    assert alternatives == ["kimi-vl-a3b-instruct-q6_k"]


def test_resolution_is_registry_order_and_reports_the_rest():
    """File order, not a quality ranking: nothing here can honestly rank vision
    (the benchmark suite scores quality/reasoning/coding and never sees an
    image), so the pick is stable and the alternatives are always disclosed."""
    registry = _capability_registry(
        ("gemma-4-e2b-it-ud-q4_k_xl", ["tools", "vision"]),
        ("kimi-vl-a3b-thinking-2506-q4_k_m", ["vision"]),
    )
    resolved, alternatives = resolve_vision_model_id(registry)
    assert resolved == "gemma-4-e2b-it-ud-q4_k_xl"
    assert alternatives == ["kimi-vl-a3b-thinking-2506-q4_k_m"]


def test_no_vision_capable_row_resolves_to_nothing():
    """Never guess at a text-only model, which would ignore the image entirely."""
    registry = _capability_registry(("gpt-oss-20b-f16", ["tools", "reasoning"]))
    assert resolve_vision_model_id(registry) == ("", [])
