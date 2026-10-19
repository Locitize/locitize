"""Can this PC fine-tune right now? Plain-language answers instead of tracebacks.

No torch import (importing it is slow and would itself fail on a broken install):
the GPU is read from nvidia-smi and the packages are checked with find_spec.
"""

import importlib.util
import re
import subprocess
from dataclasses import dataclass, field

REQUIRED_MODULES = ("torch", "transformers", "peft", "trl", "accelerate",
                    "bitsandbytes", "datasets")
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class Preflight:
    ok: bool
    problems: list = field(default_factory=list)  # plain-language, one per issue
    gpu_name: str = ""
    free_mb: int = 0
    total_mb: int = 0
    need_mb: int = 0

    def message(self):
        return " ".join(self.problems)


def estimate_vram_mb(model_name, max_seq_length=2048, batch_size=2):
    """Rough 4-bit LoRA training footprint: ~0.7 GB per billion parameters for the
    weights, plus activations that grow with batch and context, plus CUDA overhead.
    Deliberately a little pessimistic - running out mid-run wastes more time than
    a warning does."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*b(?![a-z])", model_name.lower())
    billions = float(match.group(1)) if match else 3.0
    weights = billions * 700
    activations = 250 + (batch_size * max_seq_length) * 0.35 * min(max(billions, 0.5), 8) / 2
    return int(weights + activations + 600)


def query_gpu(runner=None):
    """(name, free_mb, total_mb) of the first NVIDIA GPU, or None."""
    run = runner or subprocess.run
    try:
        proc = run(
            ["nvidia-smi", "--query-gpu=name,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
    if len(parts) < 3:
        return None
    try:
        return parts[0], int(float(parts[1])), int(float(parts[2]))
    except ValueError:
        return None


def missing_modules(finder=None):
    find = finder or importlib.util.find_spec
    out = []
    for name in REQUIRED_MODULES:
        try:
            if find(name) is None:
                out.append(name)
        except (ImportError, ValueError):
            out.append(name)
    return out


def check(model_name, max_seq_length=2048, batch_size=2, gpu=None, finder=None,
          quantize_path=None):
    """Run every check and return a Preflight. `gpu` is injectable for tests
    (pass the tuple query_gpu returns, or False to simulate no GPU)."""
    problems = []
    info = query_gpu() if gpu is None else (gpu or None)
    need = estimate_vram_mb(model_name, max_seq_length, batch_size)
    name, free, total = info if info else ("", 0, 0)

    if info is None:
        problems.append(
            "No NVIDIA graphics card was found. Fine-tuning needs an NVIDIA GPU "
            "(an RTX card with 8 GB or more is recommended). You can still chat "
            "with models; training is not available on this PC."
        )
    missing = missing_modules(finder)
    if missing:
        problems.append(
            "The fine-tuning tools are not installed (missing: "
            + ", ".join(missing)
            + "). Run locitize setup again with 'Fine-tune models' ticked."
        )
    if info is not None and free < need:
        problems.append(
            f"Not enough free graphics memory: this run needs about "
            f"{need / 1024:.1f} GB and only {free / 1024:.1f} GB of the "
            f"{total / 1024:.1f} GB on {name} is free. Stop the running model "
            "(or other GPU programs), pick a smaller base model, or lower the "
            "batch size, then try again."
        )
    return Preflight(ok=not problems, problems=problems, gpu_name=name,
                     free_mb=free, total_mb=total, need_mb=need)


def friendly_failure(log_text):
    """Turn the tail of a failed training log into one plain sentence, or ''."""
    low = (log_text or "").lower()
    if "out of memory" in low or "cuda error: out of memory" in low:
        return ("The GPU ran out of memory. Close other GPU programs, choose a "
                "smaller base model, or lower the batch size and try again.")
    if "no module named" in low:
        return ("A fine-tuning package is missing from the training environment. "
                "Run locitize setup again with 'Fine-tune models' ticked.")
    if "no kernel image is available" in low or "sm_120" in low and "not compatible" in low:
        return ("The installed PyTorch does not support this graphics card. "
                "Run locitize setup again to install the matching build.")
    if "401" in low or "gated repo" in low or "cannot access gated" in low:
        return ("That base model needs a Hugging Face login. Pick a different base model.")
    if "connection" in low and ("huggingface" in low or "hf.co" in low):
        return ("The base model could not be downloaded. Check your internet "
                "connection and try again.")
    return ""


def main(argv=None):
    """Command line for the desktop app, which runs in a different environment:
    `python preflight.py --model <id> [--batch-size N]` prints the check as JSON,
    and `python preflight.py --explain-log <train.log>` prints friendly_failure()."""
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--explain-log", default="")
    args = parser.parse_args(argv)
    if args.explain_log:
        try:
            with open(args.explain_log, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            text = ""
        print(json.dumps({"reason": friendly_failure(text)}))
        return 0
    result = check(args.model, max_seq_length=args.max_seq_length, batch_size=args.batch_size)
    print(json.dumps({
        "ok": result.ok, "problems": result.problems, "gpu_name": result.gpu_name,
        "free_mb": result.free_mb, "total_mb": result.total_mb, "need_mb": result.need_mb,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
