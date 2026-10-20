"""Pure model-advisor logic for the native fine-tune page.

Suggestions are fetched only after the owner presses the advisor button. The
network request uses modelhub's allowlisted, egress-logged opener, while every
external collaborator remains injectable for offline tests.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any, Callable

import modelhub

HF_API = "https://huggingface.co/api/models"
ALLOWED_HOSTS = ("huggingface.co",)
FETCH_TIMEOUT_S = 20
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
QLORA_GB_PER_B = 0.55
QLORA_OVERHEAD_GB = 3.5


def detect_vram_gb(gpu_provider: Any = None, system_specs: Any = None) -> float | None:
    """Largest GPU's total VRAM in GB through the platform hardware helper.

    ``system_specs`` lets the desktop pass its already-cached hardware reading.
    Other callers may inject a GpuInfoProvider; the production fallback reuses
    health.NvidiaSmiGpuInfoProvider rather than invoking nvidia-smi here.
    """
    if system_specs is not None:
        total_mb = getattr(system_specs, "gpu_vram_total_mb", None)
        return float(total_mb) / 1024.0 if total_mb else None
    if gpu_provider is None:
        from health import NvidiaSmiGpuInfoProvider

        gpu_provider = NvidiaSmiGpuInfoProvider()
    try:
        gpus = gpu_provider.gpus()
    except Exception:  # noqa: BLE001 - missing hardware is an ordinary state
        return None
    if not gpus:
        return None
    total_mb = max(float(getattr(gpu, "vram_total_mb", 0) or 0) for gpu in gpus)
    return total_mb / 1024.0 if total_mb > 0 else None


def params_b_from_id(model_id: str) -> float | None:
    """Parse the final B/M parameter-size tag from a Hugging Face model id."""
    matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*([bBmM])(?![a-zA-Z0-9])", model_id or ""
    )
    if not matches:
        return None
    value, unit = matches[-1]
    number = float(value)
    return number / 1000.0 if unit.lower() == "m" else number


def qlora_fit_gb(params_b: float) -> float:
    """Conservative 4-bit QLoRA estimate (checkpointing, seq 1024, batch 1)."""
    return float(params_b) * QLORA_GB_PER_B + QLORA_OVERHEAD_GB


def fits(params_b: float | None, vram_gb: float | None) -> bool:
    return bool(
        params_b is not None
        and vram_gb
        and qlora_fit_gb(params_b) <= float(vram_gb)
    )


def fetch_candidates(
    query: str = "instruct",
    limit: int = 60,
    opener: Any = None,
    timeout: float = FETCH_TIMEOUT_S,
) -> list[dict[str, Any]]:
    """Fetch popular trainable text-generation models from huggingface.co."""
    url = HF_API + "?" + urllib.parse.urlencode(
        {
            "search": query,
            "pipeline_tag": "text-generation",
            "library": "transformers",
            "sort": "downloads",
            "direction": "-1",
            "limit": str(max(1, int(limit))),
        }
    )
    request = urllib.request.Request(url, headers={"User-Agent": "locitize-desktop"})
    handler = None
    if opener is None:
        opener, handler = modelhub.make_opener(ALLOWED_HOSTS)
    with modelhub.open_checked(
        opener, request, ALLOWED_HOSTS, timeout=timeout, handler=handler
    ) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("huggingface.co returned more model data than expected")
    listed = json.loads(raw.decode("utf-8"))
    out: list[dict[str, Any]] = []
    for item in listed if isinstance(listed, list) else []:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("modelId") or item.get("id") or "").strip()
        if not model_id or item.get("gated"):
            continue
        lowered = model_id.lower()
        if any(
            tag in lowered
            for tag in ("awq", "gptq", "gguf", "bnb", "4bit", "8bit", "fp8")
        ):
            continue
        params = params_b_from_id(model_id)
        if params is None:
            continue
        out.append(
            {
                "id": model_id,
                "downloads": int(item.get("downloads") or 0),
                "params_b": params,
            }
        )
    return out


def suggest(
    vram_gb: float | None,
    query: str = "instruct",
    limit: int = 8,
    opener: Any = None,
    fetch_fn: Callable[..., list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return biggest-fitting candidates and a plain explanatory status."""
    if not vram_gb:
        return [], "no NVIDIA GPU reading; type a model id by hand"
    fetch = fetch_fn or fetch_candidates
    try:
        import egress_log

        with egress_log.reason("finetune-model-advisor"):
            candidates = fetch(query=query, opener=opener)
    except Exception as exc:  # noqa: BLE001 - offline leaves manual entry usable
        return (
            [],
            f"could not reach huggingface.co ({exc}); type a model id by hand",
        )
    fitting = [candidate for candidate in candidates if fits(candidate["params_b"], vram_gb)]
    fitting.sort(key=lambda candidate: (-candidate["params_b"], -candidate["downloads"]))
    top = [dict(candidate) for candidate in fitting[: max(0, int(limit))]]
    for candidate in top:
        candidate["est_gb"] = round(qlora_fit_gb(candidate["params_b"]), 1)
    note = (
        f"estimated for 4-bit QLoRA on {vram_gb:.0f} GB VRAM; "
        f"the run itself is the real test"
    )
    return top, note
