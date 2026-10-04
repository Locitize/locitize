"""Suggest fine-tunable base models that fit THIS machine (M18.6).

The trainer QLoRA-fine-tunes Hugging Face transformers models (4-bit base +
LoRA adapters), so the question a user actually has is "which base models can
MY GPU train?". This module answers it the LOCITIZE way: read the real VRAM
from nvidia-smi, ask huggingface.co for popular text-generation models (a
user-initiated request - the studio never phones home on its own), parse each
model's parameter count, and keep only what the card can hold during training.

The fit rule is a deliberately conservative estimate for 4-bit QLoRA with
gradient checkpointing (seq 1024, batch 1):

    needed_gb ~= params_B * 0.55 (4-bit weights) + 3.5 (LoRA, optimizer,
                 activations, CUDA context)

An estimate is allowed HERE (unlike serving contexts, which LOCITIZE measures)
because the training run itself is the measurement: a too-big pick fails fast
at load, and this filter's whole job is making that unlikely up front. The
report says it is an estimate.

No model names ship in this file - suggestions come live from the hub, sorted
by downloads, so the list is the community's, not ours.
"""

import json
import re
import subprocess
import urllib.parse
import urllib.request

HF_API = "https://huggingface.co/api/models"
QLORA_GB_PER_B = 0.55
QLORA_OVERHEAD_GB = 3.5
_NO_WINDOW = 0x08000000


def detect_vram_gb():
    """Total VRAM of the largest GPU in GB via nvidia-smi, or None."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    values = []
    for line in proc.stdout.strip().splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return max(values) / 1024.0 if values else None


def params_b_from_id(model_id):
    """Parse a parameter count in billions out of a model id, or None.

    Handles the naming the hub actually uses: '7B', '1.5B', '0.5B', '270M',
    'x7b' (case-insensitive). Multi-match takes the LAST (e.g. 'Llama-3.1-8B'
    must read 8, not 3.1) unless an early match is clearly the size tag.
    """
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*([bBmM])(?![a-zA-Z0-9])", model_id or "")
    if not matches:
        return None
    value, unit = matches[-1]
    number = float(value)
    return number / 1000.0 if unit.lower() == "m" else number


def qlora_fit_gb(params_b):
    """Estimated VRAM a QLoRA run needs for a params_b-billion base model."""
    return params_b * QLORA_GB_PER_B + QLORA_OVERHEAD_GB


def fits(params_b, vram_gb):
    return params_b is not None and vram_gb and qlora_fit_gb(params_b) <= vram_gb


def fetch_candidates(query="instruct", limit=60, opener=None):
    """Popular text-generation models from the hub (user-initiated request).

    Returns [{'id', 'downloads', 'params_b'}] for entries whose size is
    parseable from the id; anything unparseable is dropped rather than guessed.
    """
    call = opener or urllib.request.urlopen
    url = (
        HF_API
        + "?"
        + urllib.parse.urlencode(
            {
                "search": query,
                "pipeline_tag": "text-generation",
                "library": "transformers",
                "sort": "downloads",
                "direction": "-1",
                "limit": str(limit),
            }
        )
    )
    request = urllib.request.Request(url, headers={"User-Agent": "locitize-studio"})
    with call(request, timeout=20) as resp:
        listed = json.loads(resp.read())
    out = []
    for item in listed if isinstance(listed, list) else []:
        model_id = str(item.get("modelId") or item.get("id") or "")
        if not model_id or item.get("gated"):
            continue
        # A pre-quantized artifact (AWQ/GPTQ/GGUF/bnb) is a serving format, not
        # a QLoRA-trainable base - the trainer quantizes the fp base itself.
        lowered = model_id.lower()
        if any(tag in lowered for tag in ("awq", "gptq", "gguf", "bnb", "4bit", "8bit", "fp8")):
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


def suggest(vram_gb, query="instruct", limit=8, opener=None):
    """The fitting candidates, biggest-that-fits first, then by downloads.

    Returns (suggestions, note). Suggestions carry an `est_gb` field so the UI
    can show the estimate next to each pick. Errors (offline, API change) come
    back as a note string, never an exception - the studio stays usable with a
    hand-typed model id.
    """
    if not vram_gb:
        return [], "no NVIDIA GPU reading - type a model id by hand"
    try:
        candidates = fetch_candidates(query=query, opener=opener)
    except Exception as exc:  # noqa: BLE001 - offline is a state, not a crash
        return [], f"could not reach huggingface.co ({exc}); type a model id by hand"
    fitting = [c for c in candidates if fits(c["params_b"], vram_gb)]
    fitting.sort(key=lambda c: (-c["params_b"], -c["downloads"]))
    top = fitting[:limit]
    for c in top:
        c["est_gb"] = round(qlora_fit_gb(c["params_b"]), 1)
    note = (
        f"estimated for 4-bit QLoRA on {vram_gb:.0f} GB VRAM; "
        f"the run itself is the real test"
    )
    return top, note
