"""Run the fine-tune trainer from the desktop Fine-tune page.

The trainer scripts (finetune-studio/trainer) need CUDA PyTorch and the QLoRA
stack, which live in the fine-tune environment setup builds - never in the
desktop's own venv. So every call here runs one of those scripts with
finetune.studio_interpreter() as a child process: the GPU check, building a
dataset from documents or a word table, distillation, training, and the
before/after comparison. Nothing here imports torch.

Long work (training, distillation) is started with Popen and returns at once;
the page polls the run's log file. Short work (the GPU check, building a dataset,
a comparison) blocks, so the page calls it from a background thread.

Saving a finished model goes through config.append_model_entry - the one
models.yaml writer - never string surgery.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import finetune
import finetune_dataset as ftd

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Built-in base models, smallest first, with the plain-language note the page
# shows under the picker. Any Hugging Face id can be typed in instead.
MODEL_CHOICES = (
    ("unsloth/Qwen2.5-0.5B-Instruct",
     "Tiny and very fast. Good for trying the whole flow in a few minutes."),
    ("unsloth/Qwen2.5-1.5B-Instruct",
     "Small and fast. The best first real choice: teaches a tone, a format or new "
     "vocabulary."),
    ("unsloth/Phi-3.5-mini-instruct",
     "Middle ground (3.8B). Good reasoning for its size."),
    ("unsloth/Qwen2.5-7B-Instruct",
     "Much smarter, slower to train. Pick it when answers need reasoning or general "
     "knowledge."),
    ("unsloth/Meta-Llama-3.1-8B-Instruct",
     "Strong all-rounder, similar to Qwen 7B. Good English writing."),
)
QUANTS = ("q4_k_m", "q8_0", "f16")
QUANT_NOTES = {
    "q4_k_m": "smallest and fastest (recommended)",
    "q8_0": "closer to the original, about twice the size",
    "f16": "full quality, largest file",
}
# Environment variables never handed to a trainer child: credentials for other
# services have no business in a training process. Hugging Face tokens are the
# exception - the trainer downloads base models from there.
_SECRET_NAME = re.compile(r"(TOKEN|SECRET|PASSWORD|API_?KEY|CREDENTIAL)", re.IGNORECASE)
_HF_TOKENS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

TRAIN_LOG = "train.log"
RUN_META = "run_meta.json"


class FineTuneUnavailable(RuntimeError):
    """Fine-tuning cannot run on this install; the message says what to do."""


@dataclass
class Job:
    """One running trainer child (training or distillation)."""

    kind: str
    name: str
    log_path: Path
    process: Any
    out_path: Path | None = None
    meta: dict = field(default_factory=dict)

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def exit_code(self) -> int | None:
        return None if self.process is None else self.process.poll()


# --------------------------------------------------------------------------- #
# Where things are
# --------------------------------------------------------------------------- #


def environment(settings) -> tuple[Path, Path]:
    """(python, trainer_dir) for the fine-tune environment, or FineTuneUnavailable."""
    if settings is None:
        raise FineTuneUnavailable("Fine-tuning is not set up in this window.")
    ok, reason = finetune.studio_available(settings)
    if not ok:
        raise FineTuneUnavailable(reason)
    return Path(finetune.studio_interpreter(settings)), finetune.trainer_dir(settings)


def datasets_dir(settings) -> Path:
    root = finetune.datasets_root(settings)
    if root is None:
        # The studio checkout is read-only to LOCITIZE, so there is no fallback.
        raise FineTuneUnavailable("No data folder is configured for fine-tune datasets.")
    root.mkdir(parents=True, exist_ok=True)
    return root


def outputs_dir(settings) -> Path:
    root = finetune.outputs_root(settings)
    if root is None:
        raise FineTuneUnavailable("No folder is configured for fine-tune results.")
    root.mkdir(parents=True, exist_ok=True)
    return root


def child_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in _HF_TOKENS or not _SECRET_NAME.search(k)}
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(settings, script: str, args: list[str], timeout: float) -> subprocess.CompletedProcess:
    python, trainer = environment(settings)
    return subprocess.run(
        [str(python), str(trainer / script), *[str(a) for a in args]],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, cwd=str(trainer), env=child_env(), creationflags=_NO_WINDOW,
    )


def _last_json_line(text: str) -> dict:
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return {}


# --------------------------------------------------------------------------- #
# Short, blocking calls (run them off the GUI thread)
# --------------------------------------------------------------------------- #


def preflight(settings, model: str, batch_size: int = 2) -> dict:
    """The trainer's GPU and package check: {ok, problems, gpu_name, free_mb, total_mb, need_mb}."""
    try:
        proc = _run(settings, "preflight.py", ["--model", model, "--batch-size", batch_size], 60)
    except FineTuneUnavailable as exc:
        return {"ok": False, "problems": [str(exc)]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "problems": [f"The fine-tune environment did not run: {exc}"]}
    data = _last_json_line(proc.stdout)
    if not data:
        return {"ok": False, "problems": [
            "The fine-tune environment did not answer. Reinstall fine-tuning under "
            "Settings > Features."]}
    return data


def explain_failure(settings, log_path: Path) -> str:
    """One plain sentence for why a run failed, or ''."""
    try:
        proc = _run(settings, "preflight.py", ["--explain-log", str(log_path)], 30)
    except (FineTuneUnavailable, OSError, subprocess.SubprocessError):
        return ""
    return str(_last_json_line(proc.stdout).get("reason", ""))


def _build(settings, args: list[str], out: Path) -> tuple[Path, int, list[str]]:
    proc = _run(settings, "build_dataset.py", args, 3600)
    summary = _last_json_line(proc.stdout)
    if proc.returncode != 0 or not summary.get("rows"):
        notes = summary.get("notes") or [proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else
                                         "no rows could be built"]
        raise RuntimeError("; ".join(str(n) for n in notes))
    return out, int(summary["rows"]), [str(n) for n in summary.get("notes", [])]


def build_from_documents(settings, files: list[str], qa_pairs: int, name: str):
    """Chunk documents into rows (plus Q&A pairs from the running model). Returns (path, rows, notes)."""
    out = datasets_dir(settings) / f"{ftd.safe_name(name, 'documents')}-source.jsonl"
    return _build(settings, ["documents", "--out", out, "--qa-pairs", qa_pairs, *files], out)


def build_from_word_table(settings, table: str, identity: str, templates: int, oversample: int, name: str):
    """Rows from a reviewed word table plus optional persona rows. Returns (path, rows, notes)."""
    out = datasets_dir(settings) / f"{ftd.safe_name(name, 'word-table')}-source.jsonl"
    args = ["wordtable", "--out", out, "--table", table, "--templates", templates, "--oversample", oversample]
    if identity:
        args += ["--identity", identity]
    return _build(settings, args, out)


def compare(settings, base_model: str, merged_dir: str, system: str, prompt: str) -> tuple[str, str]:
    """(original reply, fine-tuned reply) for one message."""
    replies = []
    for model_path in (base_model, merged_dir):
        proc = _run(settings, "infer.py", ["--model-path", model_path, "--system", system, "--prompt", prompt], 900)
        reply = ""
        for line in proc.stdout.splitlines():
            if line.startswith("[RESULT]"):
                try:
                    reply = json.loads(line[len("[RESULT]"):]).get("reply", "")
                except ValueError:
                    pass
        replies.append(reply or "(no reply - " + (proc.stderr.strip().splitlines() or ["unknown error"])[-1] + ")")
    return replies[0], replies[1]


# --------------------------------------------------------------------------- #
# Long-running children
# --------------------------------------------------------------------------- #


def _spawn(settings, script: str, args: list[str], log_path: Path):
    python, trainer = environment(settings)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    try:
        return subprocess.Popen(
            [str(python), str(trainer / script), *[str(a) for a in args]],
            stdout=log_file, stderr=subprocess.STDOUT, cwd=str(trainer),
            env=child_env(), creationflags=_NO_WINDOW,
        )
    finally:
        log_file.close()  # the child holds its own handle


def run_name_for(label: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{ftd.safe_name(label, 'run')}-{stamp}"


def start_training(settings, *, model: str, dataset_path: str, label: str, steps: int,
                   learning_rate: float, lora_r: int, batch_size: int, grad_accum: int,
                   quant: str, default_system: str = "", dataset_rows: int = 0) -> Job:
    out_root = outputs_dir(settings)
    run_name = run_name_for(label)
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "base_model": model, "default_system": default_system, "dataset_path": str(dataset_path),
        "dataset_rows": dataset_rows, "max_steps": int(steps), "learning_rate": learning_rate,
        "lora_r": lora_r, "batch_size": batch_size, "grad_accum": grad_accum, "gguf_quant": quant,
    }
    (run_dir / RUN_META).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    process = _spawn(settings, "train.py", [
        "--dataset", dataset_path, "--model", model, "--output-dir", out_root, "--run-name", run_name,
        "--max-steps", int(steps), "--learning-rate", learning_rate, "--lora-r", lora_r,
        "--batch-size", batch_size, "--grad-accum", grad_accum, "--gguf-quant", quant,
    ], run_dir / TRAIN_LOG)
    return Job("train", run_name, run_dir / TRAIN_LOG, process, meta=meta)


def start_distill(settings, *, teacher: str, system: str, prompts: list[str], max_new_tokens: int,
                  temperature: float, label: str) -> Job:
    name = run_name_for(label or "distill")
    folder = outputs_dir(settings) / "_distill" / name
    folder.mkdir(parents=True, exist_ok=True)
    prompts_path = datasets_dir(settings) / f"{name}-prompts.txt"
    prompts_path.write_text("\n".join(p for p in prompts if p.strip()) + "\n", encoding="utf-8")
    out = folder / "distilled_dataset.jsonl"
    process = _spawn(settings, "distill.py", [
        "--teacher-model", teacher, "--system", system, "--prompts-file", prompts_path,
        "--output", out, "--max-new-tokens", max_new_tokens, "--temperature", temperature,
    ], folder / "distill.log")
    return Job("distill", name, folder / "distill.log", process, out_path=out)


def stop(job: Job) -> None:
    """Stop a child and anything it started (the GGUF converter runs as a grandchild)."""
    if job is None or not job.running():
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(job.process.pid)],
                       capture_output=True, creationflags=_NO_WINDOW, check=False)
    else:
        job.process.terminate()


# --------------------------------------------------------------------------- #
# Reading progress
# --------------------------------------------------------------------------- #

_STAGES = (
    ("DONE", "Done"),
    ("Converting merged model to GGUF", "Making the model file (GGUF)"),
    ("Merging", "Merging the changes into the model"),
    ("Saving LoRA adapter", "Saving"),
    ("Training complete", "Saving"),
    ("Starting training", "Training"),
    ("Attaching LoRA", "Preparing to train"),
    ("Formatting dataset", "Preparing your examples"),
    ("Loading base model", "Downloading / loading the base model"),
    ("Loading dataset", "Reading your dataset"),
    ("Dataset loaded", "Reading your dataset"),
)


def read_new(path: Path, offset: int) -> tuple[str, int]:
    """Text appended to `path` since `offset`, and the new offset."""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return "", offset
    return data.decode("utf-8", errors="replace"), offset + len(data)


def progress(log_text: str) -> dict:
    """Plain-language state of a training or distillation log.

    {stage, current, total, losses, done, failed}. tqdm redraws with carriage
    returns, so the step counter is the LAST `n/total [` in the text.
    """
    statuses = re.findall(r"\[STATUS\]\s*(.+)", log_text)
    stage = "Starting up"
    if statuses:
        last = statuses[-1]
        stage = next((label for key, label in _STAGES if last.startswith(key)), last.strip()[:60])
    steps = re.findall(r"(\d+)/(\d+)\s*\[", log_text) or re.findall(r"\[DISTILL\]\s*(\d+)/(\d+)", log_text)
    current, total = (int(steps[-1][0]), int(steps[-1][1])) if steps else (0, 0)
    losses = [float(v) for v in re.findall(r"'loss':\s*'?([0-9.]+)", log_text)]
    done = bool(statuses) and statuses[-1].startswith("DONE")
    failed = "Traceback (most recent call last)" in log_text and not done
    if done:
        stage = "Done"
    elif failed:
        stage = "Stopped with an error"
    return {"stage": stage, "current": current, "total": total, "losses": losses,
            "done": done, "failed": failed}


# --------------------------------------------------------------------------- #
# Finished runs
# --------------------------------------------------------------------------- #


def run_files(run_dir: Path) -> dict:
    """What a finished run left behind: {gguf: [names], merged: path|None, lora: path|None}."""
    run_dir = Path(run_dir)
    ggufs = sorted(p.name for p in run_dir.glob("*.gguf")) if run_dir.is_dir() else []
    merged = run_dir / "merged_model"
    lora = run_dir / "lora_adapter"
    return {"gguf": ggufs, "merged": merged if merged.is_dir() else None,
            "lora": lora if lora.is_dir() else None}


def folder_size(path: Path) -> int:
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


def read_meta(run_dir: Path) -> dict:
    try:
        data = json.loads((Path(run_dir) / RUN_META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def quant_of(gguf_name: str) -> str:
    parts = gguf_name.rsplit(".", 2)
    return parts[-2].upper() if len(parts) == 3 else ""


def place_file(src: Path, folder: Path, filename: str) -> Path:
    """Put src in folder as filename: a hard link on the same drive, else a copy."""
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / filename
    if dest.exists() and os.path.samefile(dest, src):
        return dest
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)
    return dest


def save_model(settings, *, run_dir: Path, gguf_name: str, folder: str, name: str, register: bool,
               model_id: str, context_size: int, prompt: str) -> tuple[Path, str]:
    """Save a finished run's GGUF to `folder` and (optionally) add it to the model list.

    Returns (saved path, message). Raises ValueError with a plain message when the
    folder or the model list refuses - the page shows it as is.
    """
    from config import RegistryWriteError, append_model_entry

    target = Path(os.path.expandvars(os.path.expanduser(folder.strip())))
    if not target.is_absolute():
        raise ValueError("Use a full folder path, for example one that starts with a drive letter.")
    if not name.strip():
        raise ValueError("Give the model a name.")
    src = Path(run_dir) / gguf_name
    if not src.is_file():
        raise ValueError(f"The model file {gguf_name} is no longer in the run folder.")
    quant = quant_of(gguf_name)
    filename = f"{ftd.safe_name(name, 'model')}.{quant}.gguf" if quant else f"{ftd.safe_name(name, 'model')}.gguf"
    dest = place_file(src, target, filename)
    if not register:
        return dest, f"Saved {dest}"
    meta = read_meta(run_dir)
    provenance = "Fine-tuned in LoCiTiZe"
    if meta.get("base_model"):
        provenance += f" from {meta['base_model']}"
    provenance += f", run {Path(run_dir).name}."
    cfg = getattr(settings, "finetune", None)
    try:
        append_model_entry(
            settings.data_dir, model_id.strip(), name.strip(), str(dest),
            description=provenance, context_size=int(context_size),
            gpu_layers=getattr(cfg, "default_gpu_layers", 999),
            recommended_prompt=prompt.strip(), quantization=quant, notes=provenance,
        )
    except (ValueError, RegistryWriteError) as exc:
        raise ValueError(f"Saved {dest}, but it was not added to your model list: {exc}") from None
    return dest, f"Saved {dest} and added '{name.strip()}' to your model list."
