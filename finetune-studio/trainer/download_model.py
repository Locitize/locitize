"""Inspect and download the Hugging Face files needed by native training.

This script runs inside the fine-tune environment. It intentionally uses the
default Hugging Face cache so transformers.from_pretrained() reuses the same
files later.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from pathlib import Path, PurePosixPath


_JSON_FILES = {
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
_TOKENIZER_FILES = {"merges.txt", "tokenizer.model", "vocab.txt"}
_BIN_WEIGHT = re.compile(r"^(?:pytorch_model|model)(?:-\d+-of-\d+)?\.bin$", re.IGNORECASE)
_SHARDED_WEIGHT = re.compile(
    r"^(?P<prefix>.+)-(?P<part>\d+)-of-(?P<total>\d+)\.(?P<suffix>safetensors|bin)$",
    re.IGNORECASE,
)
_INDEX_LIMIT = 16 * 1024 * 1024


def _name(sibling) -> str:
    return str(getattr(sibling, "rfilename", "") or "").replace("\\", "/")


def required_files(siblings) -> list:
    """Return the smallest snapshot subset used by AutoModel/AutoTokenizer."""
    rows = [row for row in siblings if _name(row)]
    has_safetensors = any(_name(row).lower().endswith(".safetensors") for row in rows)
    selected = []
    for row in rows:
        path = _name(row)
        pure_path = PurePosixPath(path)
        if pure_path.parent != PurePosixPath("."):
            continue
        base = pure_path.name.lower()
        keep = (
            base in _JSON_FILES
            or (base.startswith("chat_template") and base.endswith(".json"))
            or base in _TOKENIZER_FILES
            or (base.startswith(("tokenizer", "sentencepiece", "spiece")) and base.endswith(".model"))
            or base.endswith(".tiktoken")
            or base.endswith(".safetensors")
            or base.endswith(".safetensors.index.json")
        )
        if not has_safetensors:
            keep = keep or bool(_BIN_WEIGHT.match(base)) or base == "pytorch_model.bin.index.json"
        if keep:
            selected.append(row)
    return selected


def _plain_error(exc: Exception) -> tuple[str, bool]:
    text = str(exc).strip() or exc.__class__.__name__
    low = text.lower()
    status = getattr(getattr(exc, "response", None), "status_code", None)
    gated = (
        exc.__class__.__name__ == "GatedRepoError"
        or status in (401, 403)
        or "gated" in low
        or "unauthorized" in low
        or "forbidden" in low
    )
    if gated:
        return (
            "This model is gated or unauthorized. Accept its license on Hugging Face "
            "and set HF_TOKEN, then try again.",
            True,
        )
    if status == 404 or "repository not found" in low:
        return ("The Hugging Face model was not found. Check the model id.", False)
    return (f"Could not reach or inspect Hugging Face: {text}", False)


def _inspect(model: str, api=None, cache_lookup=None):
    if api is None or cache_lookup is None:
        from huggingface_hub import HfApi, try_to_load_from_cache

        api = api or HfApi()
        cache_lookup = cache_lookup or try_to_load_from_cache
    info = api.model_info(model, files_metadata=True)
    files = required_files(getattr(info, "siblings", ()) or ())
    if not files:
        raise RuntimeError("No training model files were listed for this repository.")
    revision = str(getattr(info, "sha", "") or "main")
    missing = []
    for row in files:
        cached = cache_lookup(model, _name(row), revision=revision)
        if not isinstance(cached, (str, os.PathLike)) or not os.path.isfile(cached):
            missing.append(row)
    return info, files, missing, revision


def _sum_sizes(files) -> int | None:
    sizes = [getattr(row, "size", None) for row in files]
    if any(not isinstance(size, int) or size < 0 for size in sizes):
        return None
    return sum(sizes)


def _cached_names(revision) -> tuple[Path, set[str]]:
    """Return existing files for one cache revision without touching the network."""
    snapshot = Path(getattr(revision, "snapshot_path", "") or "")
    names = set()
    for cached_file in getattr(revision, "files", ()) or ():
        path = Path(getattr(cached_file, "file_path", "") or "")
        if not path.is_file():
            continue
        try:
            name = path.relative_to(snapshot).as_posix()
        except (OSError, ValueError):
            name = str(getattr(cached_file, "file_name", "") or "").replace("\\", "/")
        if name:
            names.add(name)
    return snapshot, names


def _indexed_weights_present(snapshot: Path, names: set[str], index_name: str) -> bool:
    """An index is complete only when every locally listed shard also exists."""
    path = snapshot / Path(index_name)
    try:
        if path.stat().st_size > _INDEX_LIMIT:
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    shards = set()
    for value in weight_map.values():
        shard = str(value or "").replace("\\", "/")
        pure = PurePosixPath(shard)
        if not shard or pure.is_absolute() or ".." in pure.parts:
            return False
        shards.add(pure.as_posix())
    return bool(shards) and shards.issubset(names)


def _weight_files_present(snapshot: Path, names: set[str]) -> bool:
    indexes = sorted(
        name for name in names
        if PurePosixPath(name).parent == PurePosixPath(".")
        and name.lower().endswith((".safetensors.index.json", ".bin.index.json"))
    )
    if indexes:
        return any(_indexed_weights_present(snapshot, names, name) for name in indexes)

    weights = {
        name for name in names
        if PurePosixPath(name).parent == PurePosixPath(".")
        and (name.lower().endswith(".safetensors") or bool(_BIN_WEIGHT.match(name)))
    }
    if not weights:
        return False
    for name in weights:
        match = _SHARDED_WEIGHT.match(name)
        if not match:
            return True
        width = len(match.group("part"))
        total = int(match.group("total"))
        expected = {
            f"{match.group('prefix')}-{part:0{width}d}-of-{total:0{len(match.group('total'))}d}."
            f"{match.group('suffix')}"
            for part in range(1, total + 1)
        }
        if expected.issubset(weights):
            return True
    return False


def _tokenizer_files_present(names: set[str]) -> bool:
    lower = {name.lower() for name in names if PurePosixPath(name).parent == PurePosixPath(".")}
    if {"vocab.json", "merges.txt"}.issubset(lower):
        return True
    return any(
        name in {"tokenizer.json", "tokenizer.model", "vocab.txt"}
        or name.endswith(".tiktoken")
        or (name.startswith(("tokenizer", "sentencepiece", "spiece")) and name.endswith(".model"))
        for name in lower
    )


def _local_revision_state(revision) -> str:
    snapshot, names = _cached_names(revision)
    if not names:
        return "missing"
    complete = (
        "config.json" in {name.lower() for name in names}
        and _tokenizer_files_present(names)
        and _weight_files_present(snapshot, names)
    )
    return "cached" if complete else "partial"


def local_model_status(model: str, cache_info=None) -> dict:
    """Inspect completed files in the local HF cache. This function never egresses."""
    result = {
        "model": model,
        "cached": False,
        "cache_state": "missing",
        "total_bytes": None,
        "missing_bytes": None,
        "gated": False,
        "error": None,
        "cache_dir": "",
    }
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        result["cache_dir"] = str(HF_HUB_CACHE)
        if cache_info is None:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir()
        revisions = []
        for repo in getattr(cache_info, "repos", ()) or ():
            if str(getattr(repo, "repo_id", "") or "").casefold() != model.casefold():
                continue
            if str(getattr(repo, "repo_type", "model") or "model") != "model":
                continue
            revisions.extend(getattr(repo, "revisions", ()) or ())
        reachable = [
            revision for revision in revisions
            if "main" in (getattr(revision, "refs", ()) or ())
        ]
        reachable_states = [_local_revision_state(revision) for revision in reachable]
        all_states = [_local_revision_state(revision) for revision in revisions]
        if "cached" in reachable_states:
            result.update(cached=True, cache_state="cached")
        elif any(state in ("cached", "partial") for state in all_states):
            result["cache_state"] = "partial"
    except Exception as exc:  # noqa: BLE001 - one-line local status contract
        result["error"] = f"Could not inspect the local Hugging Face cache: {exc}"
    return result


def model_status(model: str, api=None, cache_lookup=None) -> dict:
    result = {
        "model": model,
        "cached": False,
        "total_bytes": None,
        "missing_bytes": None,
        "gated": False,
        "error": None,
        "cache_dir": "",
    }
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        result["cache_dir"] = str(HF_HUB_CACHE)
        info, files, missing, _revision = _inspect(model, api=api, cache_lookup=cache_lookup)
        result.update(
            cached=not missing,
            total_bytes=_sum_sizes(files),
            missing_bytes=_sum_sizes(missing),
            gated=bool(getattr(info, "gated", False)),
        )
    except Exception as exc:  # noqa: BLE001 - converted to the one-line CLI contract
        result["error"], result["gated"] = _plain_error(exc)
    return result


class _ByteProgress:
    def __init__(self, total: int | None):
        self.total = int(total or 0)
        self.done = 0
        self.lock = threading.Lock()
        self.tqdm_class = self._make_tqdm_class()

    def _emit(self):
        print(f"DOWNLOAD {min(self.done, self.total)}/{self.total}", flush=True)

    def add(self, amount):
        if not self.total or not amount:
            return
        with self.lock:
            self.done += max(0, int(amount))
            self._emit()

    def _make_tqdm_class(self):
        from huggingface_hub.utils import tqdm as hf_tqdm

        tracker = self

        class DownloadTqdm(hf_tqdm):
            def __init__(self, *args, **kwargs):
                name = kwargs.get("name")
                desc = str(kwargs.get("desc", "") or "")
                is_reconstruct = (
                    name == "huggingface_hub.snapshot_download"
                    or desc.startswith("Reconstructing")
                )
                is_transfer = (
                    name == "huggingface_hub.snapshot_download.transfer"
                    or desc.startswith("Downloading bytes")
                )
                self._download_bytes = kwargs.get("unit") == "B" and (
                    is_reconstruct or (name is None and not is_transfer)
                )
                initial = kwargs.get("initial", 0) if self._download_bytes else 0
                super().__init__(*args, **kwargs)
                if initial:
                    tracker.add(initial)

            def update(self, n=1):
                if self._download_bytes:
                    tracker.add(n)
                return super().update(n)

        return DownloadTqdm


def download_model(model: str, api=None, cache_lookup=None, snapshot=None) -> None:
    if snapshot is None:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download
    info, files, missing, revision = _inspect(model, api=api, cache_lookup=cache_lookup)
    if not missing:
        total = _sum_sizes(files) or 0
        print(f"DOWNLOAD {total}/{total}", flush=True)
        print("DOWNLOAD DONE", flush=True)
        return

    missing_bytes = _sum_sizes(missing)
    tracker = _ByteProgress(missing_bytes)
    tracker._emit()
    snapshot(
        repo_id=model,
        revision=revision,
        allow_patterns=[_name(row) for row in files],
        max_workers=1,
        tqdm_class=tracker.tqdm_class,
    )
    if tracker.total:
        tracker.done = tracker.total
        tracker._emit()
    print("DOWNLOAD DONE", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--local-status", metavar="MODEL")
    action.add_argument("--status", metavar="MODEL")
    action.add_argument("--download", metavar="MODEL")
    args = parser.parse_args(argv)
    if args.local_status:
        # Defense in depth: the desktop also sets this on the child process.
        os.environ["HF_HUB_OFFLINE"] = "1"
        print(json.dumps(local_model_status(args.local_status), separators=(",", ":")), flush=True)
        return 0
    if args.status:
        print(json.dumps(model_status(args.status), separators=(",", ":")), flush=True)
        return 0
    try:
        download_model(args.download)
        return 0
    except Exception as exc:  # noqa: BLE001 - child log must remain plain and actionable
        message, _gated = _plain_error(exc)
        print(f"DOWNLOAD ERROR: {message}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
