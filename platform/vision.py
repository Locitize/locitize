"""Single-image vision Q&A client for LOCITIZE (Architecture M8.1, promoted from stub).

This module drives the real qwen2-5-vl model served by llama.cpp with its on-disk
multimodal projector (`--mmproj`, Data Model 9.2) to answer a question about ONE
local image. It is single-image Q&A only -- NOT video, NOT multi-frame (that is
explicitly out of scope and recorded as future in the CLI help and here).

How it fits the architecture:
- `QwenVisionModel.describe(image_path, prompt)` ensures the vision model is RUNNING
  via the EXISTING services.ModelController (a one-model-at-a-time model switch, so
  vision and a separate chat model are never co-resident -- M8.2 VRAM discipline).
  This module never spawns a process itself.
- `VisionClient` then POSTs to the already-running server's `/v1/chat/completions`
  a messages payload whose content array carries the image as a base64 `data:` URL
  plus the text prompt (the llama.cpp mtmd multimodal request shape), and returns
  the model's real answer. All HTTP is loopback-only (127.0.0.1 + resolved port,
  SEC-1) using the standard library -- no new HTTP dependency.

Honest failure only: a missing image, a failed model switch, a transport error, or
a malformed/empty response raises VisionError with a remedy. The client NEVER
fabricates a description.

Test seams (headless, keyword 'vision_client'): VisionClient takes an injectable
`opener` (a urlopen-compatible callable) so request framing and response parsing
are unit-tested with no real server; QwenVisionModel takes an injected controller
so the model-switch request is asserted against a fake. The `--mmproj` flag plumbing
itself is proven in models.build_start_spec (keyword 'vision_mmproj').
"""

from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

# The DEFAULT registry id of the vision model (Data Model section 1.2). Kept as
# the first thing resolve_vision_model_id looks for, so a registry that names its
# vision row this way behaves exactly as it did before resolution existed.
#
# Defect found 2026-09-02: this constant was the ONLY way --describe chose a
# model, and a scan-imported registry names its rows after the file, not after
# this constant. On the owner's machine registry.get("qwen2-5-vl") returned None
# while registry.get("qwen2-5-vl-7b-instruct-q4_k_m") returned the row - so
# --describe could not find a vision model on a machine holding SIX of them.
# A hardcoded id is a claim about someone else's registry; the capability field
# is a fact about this one, which is why resolution now falls back to it.
VISION_MODEL_ID = "qwen2-5-vl"

# The capability a row must declare to be usable for single-image Q&A. Written
# by launcher's --detect-capabilities pass, which only adds it when a projector
# is actually paired (see launcher.py: "a projector on the row IS the proof of
# vision"), so this is evidence rather than an assertion.
VISION_CAPABILITY = "vision"


def resolve_vision_model_id(
    registry: Any, preferred: str = ""
) -> tuple[str, list[str]]:
    """Choose which registry model answers --describe. Returns (id, alternatives).

    Resolution order, most specific first:
      1. `preferred` - the --model flag or settings.vision.model, when it names a
         launchable row. An explicit choice is never second-guessed.
      2. VISION_MODEL_ID, when a row is literally named that.
      3. The launchable rows declaring VISION_CAPABILITY, in registry file order.

    Rung 3 is deliberately file order rather than "best" by some score: this
    module has no honest basis for ranking vision quality (the benchmark suite
    scores quality/reasoning/coding and never looks at an image), and inventing
    a ranking would be a fabricated claim. Stable and explainable beats clever.
    The remaining candidates come back as `alternatives` so the caller can show
    the owner what else was available and how to pin one, rather than silently
    picking for them.

    Returns ("", []) when nothing qualifies, which the caller reports with a
    remedy - never a guess at a text-only model that would ignore the image.
    """
    rows = list(registry.launchable()) if hasattr(registry, "launchable") else []
    by_id = {row.id: row for row in rows}
    capable = [
        row.id
        for row in rows
        if VISION_CAPABILITY in (getattr(row, "capabilities", None) or [])
    ]

    wanted = (preferred or "").strip()
    if wanted:
        if wanted in by_id:
            return wanted, []
        # An explicit choice that does not resolve is an error the caller must
        # report, NOT something to silently fall back from - falling back would
        # answer with a different model than the owner asked for. The suggestions
        # are the models that can SEE, not every launchable row: offering a
        # text-only model as an alternative to a vision model is not a remedy.
        return "", capable

    if VISION_MODEL_ID in by_id:
        return VISION_MODEL_ID, [c for c in capable if c != VISION_MODEL_ID]

    if not capable:
        return "", []
    return capable[0], capable[1:]

# MIME type per image extension, for the data: URL. A small explicit map (not a
# guess) covering the formats a programmatically-drawn test image or an owner photo
# uses; an unknown extension falls back to a generic image type the server tolerates.
_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class VisionError(RuntimeError):
    """Honest vision failure carrying an owner-facing remedy.

    Raised on a missing image, a failed model start/switch, a transport error, or an
    empty/malformed server response. Surfaced to the owner; never converted into a
    fabricated description.
    """

    def __init__(self, message: str, remedy: str = "") -> None:
        super().__init__(message)
        self.remedy = remedy


class VisionModel(ABC):
    """Interface for single-image Q&A (the M1 stub's contract, now real)."""

    @abstractmethod
    def describe(self, image_path: str, prompt: str = "") -> str:
        """Return the model's answer/description for the given image."""


def encode_image_data_uri(image_path: Path | str) -> str:
    """Read a local image file and return a base64 `data:` URL for the request.

    The image is base64-encoded only transiently here for the request body; it is
    never stored (Data Model 9.4 VisionQuery note). Raises VisionError if the file
    is missing so the caller surfaces an honest "image not found" rather than posting
    an empty payload.
    """
    path = Path(image_path)
    if not path.is_file():
        raise VisionError(
            f"image not found: {path}",
            remedy="pass an existing image file path to --describe",
        )
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_vision_messages(prompt: str, data_uri: str) -> list[dict[str, Any]]:
    """Build the /v1/chat/completions `messages` list for one image + a question.

    The content is an ARRAY of parts (the llama.cpp/OpenAI multimodal shape): a text
    part with the question and an image_url part carrying the base64 data: URL. Kept
    as a pure function so the exact wire shape is unit-tested directly (keyword
    'vision_client') against the format confirmed with the running server at Build
    time.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
    ]


class VisionClient:
    """HTTP client for a single non-streaming multimodal chat completion (M8.1).

    Talks only to 127.0.0.1:<resolved-port> (SEC-1). `opener` is the single test
    seam -- an injectable urlopen-compatible callable -- so framing and parsing are
    exercised with no real server; when None the stdlib urllib.request.urlopen is
    used. This client spawns nothing; the server is owned by the ModelController.
    """

    def __init__(
        self,
        port: int,
        host: str = "127.0.0.1",
        timeout_s: float = 180.0,
        opener: Any = None,
    ) -> None:
        self._port = port
        self._host = host
        self._timeout_s = timeout_s
        self._opener = opener

    @property
    def base_url(self) -> str:
        """The loopback base URL this client posts to."""
        return f"http://{self._host}:{self._port}"

    def describe(self, image_path: Path | str, prompt: str) -> str:
        """POST the image + prompt and return the model's answer text.

        Non-streaming (temperature left to the server default): a single request, a
        single JSON reply. Raises VisionError on a transport error or a response that
        carries no answer content, so a broken call is never silently a blank answer.
        """
        import urllib.error
        import urllib.request

        data_uri = encode_image_data_uri(image_path)
        payload = {
            "messages": build_vision_messages(prompt, data_uri),
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        call = self._opener if self._opener is not None else urllib.request.urlopen
        try:
            response = call(request, timeout=self._timeout_s)
        except (urllib.error.URLError, OSError) as exc:
            raise VisionError(
                f"vision server unreachable on {self._host}:{self._port}: {exc}",
                remedy="ensure the vision model started (check logs/) and retry",
            ) from exc
        try:
            raw = response.read().decode("utf-8")
        finally:
            _close(response)
        return _parse_answer(raw)


def _parse_answer(raw: str) -> str:
    """Extract choices[0].message.content from a chat-completions JSON reply.

    Returns the trimmed answer text. Raises VisionError for malformed JSON, a missing
    choices/message/content, or empty content -- a broken response is an honest error,
    never a fabricated description.
    """
    try:
        obj = json.loads(raw)
    except ValueError as exc:
        raise VisionError(
            "vision server returned a non-JSON response",
            remedy="check logs/ for a model-load or mmproj error and retry",
        ) from exc
    choices = obj.get("choices") if isinstance(obj, dict) else None
    if not isinstance(choices, list) or not choices:
        raise VisionError(
            "vision server response had no choices",
            remedy="check logs/ for a model-load or mmproj error and retry",
        )
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    text = content.strip() if isinstance(content, str) else ""
    if not text:
        raise VisionError(
            "vision server returned no answer content",
            remedy="check logs/ for a model-load or mmproj error and retry",
        )
    return text


def _close(response: Any) -> None:
    """Best-effort close of an HTTP response object (never raises)."""
    close: Callable[[], None] | None = getattr(response, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - closing must never mask the real result
            pass


class QwenVisionModel(VisionModel):
    """Real qwen2-5-vl vision client over the existing ModelController (M8.1).

    Construction is dependency injection: it receives the ModelController that owns
    llama.cpp lifecycle plus a client_factory (port -> VisionClient) so tests can
    substitute both. describe() ensures the vision model is RUNNING (a model switch
    if a different model is up -- honoring the one-model-at-a-time invariant and the
    mmproj VRAM discipline, M8.2), then delegates the HTTP call to a VisionClient on
    the controller-resolved loopback port. It spawns no process of its own.
    """

    def __init__(
        self,
        controller: Any,
        default_prompt: str = "Describe this image accurately.",
        model_id: str = VISION_MODEL_ID,
        client_factory: Callable[[int], VisionClient] | None = None,
    ) -> None:
        self._controller = controller
        self._default_prompt = default_prompt
        self._model_id = model_id
        self._client_factory = client_factory or (lambda port: VisionClient(port))

    def describe(self, image_path: str, prompt: str = "") -> str:
        """Ensure the vision model is running, then answer about the image.

        A blank prompt falls back to the configured default question. Raises
        VisionError if the image is missing (checked up front so no needless model
        switch happens) or if the model cannot be brought to RUNNING.
        """
        # ServiceStatus is imported lazily to keep this module import-light and to
        # avoid a hard import cycle with services at module load.
        from services import ServiceStatus

        path = Path(image_path)
        if not path.is_file():
            raise VisionError(
                f"image not found: {path}",
                remedy="pass an existing image file path to --describe",
            )

        # Switch to the vision model unless it is already the running one. switch()
        # stops any current model first (one-model-at-a-time), then starts qwen2-5-vl
        # WITH --mmproj (its models.yaml row carries the projector path, M8.1).
        if self._controller.running_model_id != self._model_id:
            try:
                status = self._controller.switch(self._model_id)
            except ValueError as exc:
                raise VisionError(
                    f"could not start vision model '{self._model_id}': {exc}",
                    remedy="set its location/mmproj in models.yaml and free VRAM",
                ) from exc
            if status is not ServiceStatus.RUNNING:
                raise VisionError(
                    f"vision model '{self._model_id}' did not start "
                    f"({getattr(status, 'value', status)})",
                    remedy="check logs/ for an mmproj/VRAM error, free VRAM, retry",
                )

        port = self._controller.running_port
        if not port:
            raise VisionError(
                "vision model reported no serving port",
                remedy="restart the vision model and retry",
            )
        client = self._client_factory(port)
        return client.describe(path, prompt.strip() or self._default_prompt)
