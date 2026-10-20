"""Run the fine-tune trainer from the desktop Fine-tune page.

The trainer scripts (finetune-studio/trainer) need CUDA PyTorch and the QLoRA
stack, which live in the fine-tune environment setup builds - never in the
desktop's own venv. So every call here runs one of those scripts with
finetune.studio_interpreter() as a child process: the GPU check, building a
dataset from documents or a word table, distillation, training, and the
before/after comparison. Nothing here imports torch.

Long work (training, distillation, model downloads) is started with Popen and returns at once;
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
import urllib.parse
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
HISTORY_LOG_READ_BYTES = 2 * 1024 * 1024
HISTORY_LOG_TAIL_BYTES = 16 * 1024
HISTORY_META_BYTES = 256 * 1024
DOWNLOAD_STATUS_TIMEOUT = 60


class FineTuneUnavailable(RuntimeError):
    """Fine-tuning cannot run on this install; the message says what to do."""


@dataclass
class Job:
    """One running trainer child (training, distillation, or download)."""

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


def _run(
    settings, script: str, args: list[str], timeout: float,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    python, trainer = environment(settings)
    env = child_env()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [str(python), str(trainer / script), *[str(a) for a in args]],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, cwd=str(trainer), env=env, creationflags=_NO_WINDOW,
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


def _download_status_result(model: str, proc: subprocess.CompletedProcess) -> dict:
    fallback = {
        "model": model, "cached": False, "total_bytes": None,
        "missing_bytes": None, "gated": False, "error": None,
        "cache_state": "missing",
    }
    data = _last_json_line(proc.stdout)
    if not isinstance(data, dict) or not data:
        detail = (proc.stderr or "").strip().splitlines()
        message = detail[-1] if detail else "The fine-tune environment did not answer."
        return {**fallback, "error": message}
    result = {**fallback}
    for key in result:
        if key in data:
            result[key] = data[key]
    result["model"] = str(result.get("model") or model)
    result["cached"] = bool(result.get("cached"))
    result["gated"] = bool(result.get("gated"))
    state = str(data.get("cache_state") or "").lower()
    result["cache_state"] = state if state in ("cached", "partial", "missing") else (
        "cached" if result["cached"] else "missing"
    )
    for key in ("total_bytes", "missing_bytes"):
        value = result.get(key)
        result[key] = value if isinstance(value, int) and value >= 0 else None
    result["error"] = str(result["error"]) if result.get("error") else None
    return result


def _download_status_fallback(model: str, error: str) -> dict:
    return {
        "model": model, "cached": False, "total_bytes": None,
        "missing_bytes": None, "gated": False, "error": error,
        "cache_state": "missing",
    }


def local_download_status(settings, model: str) -> dict:
    """Inspect only the local HF cache in an explicitly offline child."""
    try:
        proc = _run(
            settings, "download_model.py", ["--local-status", model],
            DOWNLOAD_STATUS_TIMEOUT, env_overrides={"HF_HUB_OFFLINE": "1"},
        )
    except FineTuneUnavailable as exc:
        return _download_status_fallback(model, str(exc))
    except subprocess.TimeoutExpired:
        return _download_status_fallback(model, "The local model cache check timed out.")
    except (OSError, subprocess.SubprocessError) as exc:
        return _download_status_fallback(model, f"The fine-tune environment did not run: {exc}")
    return _download_status_result(model, proc)


def download_status(settings, model: str) -> dict:
    """User-initiated remote status lookup, including the remaining byte count."""
    try:
        import egress_log

        target = "https://huggingface.co/api/models/" + urllib.parse.quote(model.strip("/"), safe="/")
        with egress_log.reason("finetune-model-size"):
            egress_log.record(target)
            proc = _run(settings, "download_model.py", ["--status", model], DOWNLOAD_STATUS_TIMEOUT)
    except FineTuneUnavailable as exc:
        return _download_status_fallback(model, str(exc))
    except subprocess.TimeoutExpired:
        return _download_status_fallback(
            model, "The model download check timed out. Check your internet connection."
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _download_status_fallback(model, f"The fine-tune environment did not run: {exc}")
    return _download_status_result(model, proc)


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


def start_download(settings, model: str) -> Job:
    """Start a base-model download into the default Hugging Face cache."""
    name = run_name_for(model.rsplit("/", 1)[-1] or "model-download")
    folder = outputs_dir(settings) / "_downloads" / name
    log_path = folder / "download.log"
    process = _spawn(settings, "download_model.py", ["--download", model], log_path)
    return Job("download", name, log_path, process, meta={"model": model})


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


def read_new(path: Path, offset: int, max_bytes: int | None = None) -> tuple[str, int]:
    """Text appended since ``offset`` and the new offset, optionally tail-bounded."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            start = min(offset, end)
            if max_bytes is not None:
                start = max(start, end - int(max_bytes))
            f.seek(start)
            data = f.read() if max_bytes is None else f.read(int(max_bytes))
    except OSError:
        return "", offset
    return data.decode("utf-8", errors="replace"), start + len(data)


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


def download_progress(log_text: str) -> dict:
    """Parse download_model.py byte counters and terminal status from a bounded log."""
    counters = re.findall(r"(?m)^DOWNLOAD\s+(\d+)/(\d+)\s*$", log_text or "")
    current, total = (int(counters[-1][0]), int(counters[-1][1])) if counters else (0, 0)
    done = bool(re.search(r"(?m)^DOWNLOAD DONE\s*$", log_text or ""))
    errors = re.findall(r"(?m)^DOWNLOAD ERROR:\s*(.+)$", log_text or "")
    failed = bool(errors or "Traceback (most recent call last)" in (log_text or "")) and not done
    return {
        "current": min(current, total) if total else current,
        "total": total,
        "fraction": (min(current, total) / total) if total else None,
        "done": done,
        "failed": failed,
        "error": errors[-1].strip() if errors else "",
    }


# --------------------------------------------------------------------------- #
# Finished runs
# --------------------------------------------------------------------------- #


def _read_tail(path: Path, limit: int) -> str:
    """Read at most the last ``limit`` bytes of a text file, replacement-safe."""
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - int(limit)))
            return handle.read(int(limit)).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _history_meta(path: Path) -> tuple[dict, str]:
    try:
        with Path(path).open("rb") as handle:
            raw = handle.read(HISTORY_META_BYTES + 1)
    except FileNotFoundError:
        return {}, ""
    except OSError as exc:
        return {}, f"metadata could not be read: {exc}"
    if len(raw) > HISTORY_META_BYTES:
        return {}, "metadata is too large to display"
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError):
        return {}, "metadata is not valid JSON"
    if not isinstance(data, dict):
        return {}, "metadata is not a JSON object"
    return data, ""


def scan_training_runs(settings, live_job: Job | None = None, limit: int = 50) -> list[dict]:
    """Read training-run history without changing GGUF discovery semantics.

    A filesystem timestamp is never treated as proof that a run is active.
    ``running`` is returned only for the live training ``Job`` owned by this
    process. Logs and metadata are bounded so a damaged run cannot consume
    unbounded memory on the background worker.
    """
    root = finetune.outputs_root(settings) if settings is not None else None
    if root is None or not Path(root).is_dir():
        return []
    try:
        folders = [p for p in Path(root).iterdir()
                   if p.is_dir() and p.name not in ("_distill", "_downloads")]
    except OSError:
        return []

    live_dir = None
    if live_job is not None and live_job.kind == "train":
        try:
            live_dir = live_job.log_path.parent.resolve()
        except OSError:
            live_dir = live_job.log_path.parent

    runs = []
    for folder in folders:
        log_path = folder / TRAIN_LOG
        meta_path = folder / RUN_META
        log_text = _read_tail(log_path, HISTORY_LOG_READ_BYTES)
        state = progress(log_text)
        meta, meta_error = _history_meta(meta_path)
        try:
            resolved = folder.resolve()
        except OSError:
            resolved = folder

        owns_run = live_dir is not None and resolved == live_dir
        if owns_run and live_job.running():
            status = "running"
        elif state["done"]:
            status = "completed"
        elif state["failed"] or (owns_run and live_job.exit_code() not in (None, 0)):
            status = "failed"
        else:
            status = "incomplete"

        mtimes = []
        for candidate in (folder, log_path, meta_path):
            try:
                mtimes.append(candidate.stat().st_mtime)
            except OSError:
                pass
        mtime = max(mtimes, default=0.0)
        runs.append({
            "name": folder.name,
            "path": str(folder),
            "status": status,
            "mtime": mtime,
            "modified_display": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M") if mtime else "unknown",
            "meta": meta,
            "meta_error": meta_error,
            "base_model": str(meta.get("base_model") or ""),
            "dataset_path": str(meta.get("dataset_path") or ""),
            "log_path": str(log_path) if log_path.is_file() else "",
            "log_tail": log_text[-HISTORY_LOG_TAIL_BYTES:],
            "losses": list(state["losses"]),
        })
    runs.sort(key=lambda row: (row["mtime"], row["name"]), reverse=True)
    return runs[:max(0, int(limit))]


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


def _versioned_path(folder: Path, filename: str, number: int) -> Path:
    """Return filename, then filename's ``-2``/``-3`` variants before its tags."""
    if number <= 1:
        return folder / filename
    stem, dot, suffix = filename.partition(".")
    tail = f"{dot}{suffix}" if dot else ""
    return folder / f"{stem}-{number}{tail}"


def planned_destination(src: Path, folder: Path, filename: str) -> Path:
    """First non-conflicting destination, without creating the folder or file."""
    for number in range(1, 100_000):
        dest = _versioned_path(folder, filename, number)
        if dest.exists():
            try:
                if os.path.samefile(dest, src):
                    return dest
            except OSError:
                pass
            continue
        return dest
    raise OSError(f"Could not choose a free filename in {folder}.")


def _copy_exclusive(src: Path, dest: Path) -> None:
    """Copy without the overwrite semantics of shutil.copy2."""
    created = False
    try:
        with src.open("rb") as reader, dest.open("xb") as writer:
            created = True
            shutil.copyfileobj(reader, writer)
        shutil.copystat(src, dest)
    except Exception:
        if created:
            try:
                dest.unlink()
            except OSError:
                pass
        raise


def _place_file(src: Path, folder: Path, filename: str) -> tuple[Path, bool]:
    """Place without replacing any different file; return (path, created_here)."""
    if Path(filename).name != filename or not filename:
        raise ValueError("The saved model filename must be one plain file name.")
    folder.mkdir(parents=True, exist_ok=True)
    for number in range(1, 100_000):
        dest = _versioned_path(folder, filename, number)
        if dest.exists():
            try:
                if os.path.samefile(dest, src):
                    return dest, False
            except OSError:
                pass
            continue
        try:
            os.link(src, dest)
            return dest, True
        except FileExistsError:
            continue
        except OSError:
            try:
                _copy_exclusive(src, dest)
                return dest, True
            except FileExistsError:
                continue
    raise OSError(f"Could not choose a free filename in {folder}.")


def place_file(src: Path, folder: Path, filename: str) -> Path:
    """Hard-link src, or copy it, without deleting or overwriting another file."""
    return _place_file(Path(src), Path(folder), filename)[0]


def registry_model_entry(settings, model_id: str) -> dict[str, Any] | None:
    """Read one raw registry row for preflight/confirmation; never write it."""
    import yaml

    path = Path(settings.data_dir) / "models.yaml"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"Could not read your model list at {path}: {exc}") from None
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not read your model list at {path}: {exc}") from None
    rows = parsed.get("models", []) if isinstance(parsed, dict) else parsed
    for row in rows or []:
        if isinstance(row, dict) and str(row.get("id", "")).strip() == model_id:
            return dict(row)
    return None


def provenance_text(run_dir: Path) -> str:
    """Default description/notes text for a saved fine-tune."""
    meta = read_meta(run_dir)
    provenance = "Fine-tuned in LoCiTiZe"
    if meta.get("base_model"):
        provenance += f" from {meta['base_model']}"
    return provenance + f", run {Path(run_dir).name}."


def save_model(settings, *, run_dir: Path, gguf_name: str, folder: str, name: str, register: bool,
               model_id: str, context_size: int, prompt: str, description: str = "",
               notes: str = "", gpu_layers: int | None = None,
               replace: bool = False) -> tuple[Path, str]:
    """Save a finished run's GGUF to `folder` and (optionally) add it to the model list.

    Returns (saved path, message). Raises ValueError with a plain message when the
    folder or the model list refuses - the page shows it as is.
    """
    from config import RegistryWriteError, append_model_entry, replace_model_entry

    target = Path(os.path.expandvars(os.path.expanduser(folder.strip())))
    if not target.is_absolute():
        raise ValueError("Use a full folder path, for example one that starts with a drive letter.")
    clean_name = name.strip()
    clean_id = model_id.strip()
    if not clean_name:
        raise ValueError("Give the model a name.")
    if register and not clean_id:
        raise ValueError("Give the model an id.")
    if replace and not register:
        raise ValueError("Tick 'Add it to my model list' before replacing an entry.")
    if not isinstance(context_size, int) or isinstance(context_size, bool) or context_size <= 0:
        raise ValueError("Context size must be a positive number.")
    cfg = getattr(settings, "finetune", None)
    resolved_gpu_layers = (
        getattr(cfg, "default_gpu_layers", 999) if gpu_layers is None else gpu_layers
    )
    if (
        not isinstance(resolved_gpu_layers, int)
        or isinstance(resolved_gpu_layers, bool)
        or resolved_gpu_layers < -1
    ):
        raise ValueError("GPU layers must be an integer of -1 or greater.")

    existing = registry_model_entry(settings, clean_id) if register else None
    if existing is not None and not replace:
        raise ValueError(
            f"Model id '{clean_id}' already exists. Choose a new id or tick "
            f"'Replace existing model'."
        )
    if replace and existing is None:
        raise ValueError(
            f"Model id '{clean_id}' does not exist. Untick 'Replace existing model' "
            f"to add it as a new entry."
        )
    old_location = str(existing.get("location", "")).strip() if existing else ""
    src = Path(run_dir) / gguf_name
    if not src.is_file():
        raise ValueError(f"The model file {gguf_name} is no longer in the run folder.")
    quant = quant_of(gguf_name)
    filename = f"{ftd.safe_name(clean_name, 'model')}.{quant}.gguf" if quant else f"{ftd.safe_name(clean_name, 'model')}.gguf"
    dest, created = _place_file(src, target, filename)
    if not register:
        return dest, f"Saved {dest}"
    provenance = provenance_text(run_dir)
    final_description = description.strip() or provenance
    final_notes = notes.strip() or provenance
    fields = {
        "description": final_description,
        "context_size": int(context_size),
        "gpu_layers": resolved_gpu_layers,
        "recommended_prompt": prompt.strip(),
        "quantization": quant,
        "notes": final_notes,
    }
    try:
        if replace:
            replace_model_entry(
                settings.data_dir,
                clean_id,
                name=clean_name,
                location=str(dest),
                **fields,
            )
        else:
            append_model_entry(
                settings.data_dir,
                clean_id,
                clean_name,
                str(dest),
                **fields,
            )
    except (ValueError, RegistryWriteError) as exc:
        removed = False
        if created:
            try:
                dest.unlink()
                removed = True
            except OSError:
                pass
        file_result = (
            "The new saved file was removed; no existing model file was touched."
            if removed
            else f"The file at {dest} was kept because locitize could not prove it was safe to remove."
        )
        raise ValueError(
            f"The model was not added to your model list: {exc} {file_result}"
        ) from None
    location_changed = replace and os.path.normcase(os.path.normpath(old_location)) != (
        os.path.normcase(os.path.normpath(str(dest)))
    )
    if location_changed:
        # The registry write has committed. Ceiling cleanup is a sidecar cache
        # invalidation and must never turn a successful model save into a failure.
        try:
            from context_advice import remove_ceiling

            remove_ceiling(settings.data_dir, clean_id)
        except Exception:
            pass
    action = "replaced" if replace else "added"
    return dest, f"Saved {dest} and {action} '{clean_name}' in your model list."
