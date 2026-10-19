"""Bounded HF selections and verified local jobs for the lightweight MCP helper.

Repository metadata is untrusted data. Only server-issued selections can reach
the checked downloader; immutable revisions and all digests bind their files.
"""
from __future__ import annotations

import copy
import hashlib
import re
import threading
import time
import uuid
from pathlib import Path

import config
import egress_log
import gguf_meta
import health
import modelhub


class DiscoveryError(ValueError):
    """A fixed, path-free error safe to return through the plugin protocol."""


class ModelDiscovery:
    """Own session-only candidates/jobs; completed file shards support safe retry."""

    def __init__(self, root=None, downloader=None, hardware=None):
        self.root = Path(root) if root is not None else config.resolve_data_dir()
        egress_log.configure(self.root)
        self.system = health.DefaultSystemInfoProvider()
        self.hub = downloader or modelhub.Downloader(modelhub.HubConfig(search_limit=10),
            models_dir=self.root / "models", system_provider=self.system)
        self.hardware = hardware
        self._lock = threading.RLock()
        self._candidates = {}
        self._jobs = {}
        self._cancel = {}
        self._thread = None
        self._cursor = None
        self._pages = 0
        self._query = None
        self._closed = False

    def _metadata(self, repo, revision=None):
        """Inspect a validated repo and immutable revision through checked HTTP."""
        repo = modelhub.validate_repo_id(repo)
        base = modelhub.DEFAULT_API_BASE.rstrip("/")
        suffix = "/revision/" + revision if revision else ""
        egress_log._reason.label = "plugin-hf-inspect"
        metadata, _ = self._json(base + "/api/models/" + repo + suffix)
        if not isinstance(metadata, dict):
            raise DiscoveryError("Hugging Face metadata unavailable")
        commit = metadata.get("sha")
        if not isinstance(commit, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", commit):
            raise DiscoveryError("Immutable repository revision unavailable")
        if revision and commit != revision:
            raise DiscoveryError("Selected repository revision changed")
        rows, cursor = self._json(base + "/api/models/" + repo +
            "/tree/" + commit + "?recursive=true")
        if cursor or not isinstance(rows, list) or len(rows) > 2000:
            raise DiscoveryError("Repository file listing exceeds bounded inspection; choose a smaller repository")
        files = modelhub.parse_tree_paths(rows)
        return metadata, commit, files

    def _json(self, url):
        """Only fixed error labels cross the boundary; remote exception text stays private."""
        try:
            return self.hub._get_json_with_link(url)
        except modelhub.HubError as exc:
            states = {"offline": "Hugging Face is offline or unreachable",
                "rate_limited": "Hugging Face rate limit reached; try later",
                "gated": "Gated repositories are unsupported; obtain permitted access separately"}
            raise DiscoveryError(states.get(exc.kind, "Hugging Face metadata unavailable")) from exc

    def search_models(self, query, limit=10, cursor=""):
        """Search only; no weight transfer, registry edits or executable code."""
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 120:
            raise DiscoveryError("Search query must contain 1 to 120 characters")
        if type(limit) is not int or not 1 <= limit <= 10:
            raise DiscoveryError("Search limit must be 1 to 10")
        with self._lock:
            if cursor:
                if cursor != self._cursor or query.strip() != self._query:
                    raise DiscoveryError("Unknown search cursor")
                if self._pages >= 5:
                    raise DiscoveryError("Search page limit reached; start a new search")
                response = self.hub.search_more()
                self._pages += 1
            else:
                response = self.hub.search(query.strip())
                self._query = query.strip()
                self._pages = 1
            if not response.get("ok"):
                return {"items": [], "state": response.get("kind", "unavailable"),
                        "authentication": "Public repositories only; gated access is unsupported"}
            items = response["items"][-10:] if cursor else response["items"][:limit]
            self.hub._last_search = self.hub._last_search[-50:]
            self._cursor = uuid.uuid4().hex if response.get("has_more") else None
            return {"items": [{"repo_id": row["repo_id"],
                "license": row.get("license_tag") or "unknown",
                "source_url": "https://huggingface.co/" + row["repo_id"],
                "downloads": row.get("downloads", 0)} for row in items[:limit]],
                "cursor": self._cursor, "weight_bytes_downloaded": 0}

    def list_model_files(self, repo_id):
        """Issue bounded exact file-set selections; unknown capabilities stay unknown."""
        repo_id = modelhub.validate_repo_id(repo_id)
        with self._lock:
            metadata, revision, files = self._metadata(repo_id)
            card = metadata.get("cardData") if isinstance(metadata.get("cardData"), dict) else {}
            license_tag = modelhub.sanitize_license_tag(card.get("license")) or "unknown"
            projectors = [f for f in files if "mmproj" in f["filename"].lower()]
            candidates = []
            for row in files:
                # Importance matrices are llama.cpp quantization inputs, not
                # runnable model weights. Some publishers suffix them .gguf,
                # so extension-only filtering would offer a false tiny model.
                if row in projectors or "imatrix" in Path(row["filename"]).name.lower():
                    continue
                try:
                    names = modelhub.shard_set_for([f["filename"] for f in files], row["filename"])
                except ValueError:
                    continue
                if names and names[0] != row["filename"]:
                    continue
                names = names or [row["filename"]]
                selected = [next(f for f in files if f["filename"] == n) for n in names]
                # Typical HF metadata does not bind an exact projector/model
                # pair. Keep these downloads weights-only rather than inventing
                # a pairing from file names or repository prose.
                projector = None
                selection = uuid.uuid4().hex
                candidate = {"selection_id": selection, "repo_id": repo_id,
                    "revision": revision, "family": self._family(metadata), "files": copy.deepcopy(selected),
                    "model_files": names, "projector": projector["filename"] if projector else None,
                    "quantization": row["quant"], "total_bytes": sum(max(0, f["size_bytes"]) for f in selected),
                    "verified_metadata": all(f["sha256"] and f["size_bytes"] > 0 for f in selected),
                    "license": license_tag, "gated": bool(metadata.get("gated")),
                    "source_url": "https://huggingface.co/" + repo_id + "/tree/" + revision,
                    "capabilities": {"vision": "reported" if metadata.get("pipeline_tag") == "image-text-to-text" else "unknown",
                        "vision_evidence": "repository pipeline metadata; not locally tested",
                        "projector_pairing": "unsupported: requires separate verified setup",
                        "tools": "unknown", "coding_quality": "unknown",
                        "download_scope": "Model weights only; vision needs separate verified projector setup"}, "created_at": time.time()}
                if len(self._candidates) >= 100:
                    self._candidates.pop(next(iter(self._candidates)))
                self._candidates[selection] = candidate
                candidates.append(self._public(candidate))
                if len(candidates) == 25:
                    break
            return {"repo_id": repo_id, "candidates": candidates,
                    "unpaired_projectors": [{"filename": f["filename"], "size_bytes": f["size_bytes"]} for f in projectors[:10]],
                    "weight_bytes_downloaded": 0, "authentication": "Public ungated files only"}

    @staticmethod
    def _public(candidate):
        view = {key: copy.deepcopy(value) for key, value in candidate.items()
                if key not in {"created_at", "revision", "model_files"}}
        view["files"] = [{"filename": row["filename"], "size_bytes": row["size_bytes"]}
                         for row in candidate["files"]]
        return view

    @staticmethod
    def _family(metadata):
        """Reported architecture groups alternatives; names never imply capability."""
        config_data = metadata.get("config")
        name = config_data.get("model_type") if isinstance(config_data, dict) else None
        if isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            return {"name": name.lower(), "status": "reported", "evidence": "HF config.model_type"}
        return {"name": None, "status": "unknown", "evidence": "No reported model_type"}

    def _selection(self, selection):
        if not isinstance(selection, str) or selection not in self._candidates:
            raise DiscoveryError("Unknown model selection; inspect repository files again")
        row = self._candidates[selection]
        if time.time() - row["created_at"] > 3600:
            raise DiscoveryError("Model selection expired; inspect repository files again")
        return copy.deepcopy(row)

    def recommend_model_candidates(self, selection_ids, use_case="general", prefer_vision=False):
        """Rank estimated current memory/disk fit, never speed or model quality."""
        if (not isinstance(selection_ids, list) or not 1 <= len(selection_ids) <= 25 or
                use_case not in ("general", "coding") or type(prefer_vision) is not bool):
            raise DiscoveryError("Use 1 to 25 selections and general or coding workload")
        with self._lock:
            rows = [self._selection(item) for item in selection_ids]
        hardware = self.hardware() if self.hardware else {}
        available_ram = hardware.get("ram_available_mb")
        vram = max((g.get("vram_free_mb") or 0 for g in (hardware.get("gpus") or [])), default=0)
        try:
            free = self.system.disk_free_mb(str(self.root))
        except OSError:
            free = None
        ranked = []
        for row in rows:
            mb = row["total_bytes"] / 2**20
            # Conservative screening only: unknown KV/cache/context needs are
            # disclosed, and measured tuning remains a separate explicit action.
            allowance = max(512, mb * .2)
            known_size = all(f["size_bytes"] > 0 for f in row["files"])
            ram_fit = None if available_ram is None or not known_size else available_ram >= mb + allowance
            disk_fit = None if free is None or not known_size else free >= mb + 2048
            gpu_fit = vram >= mb + allowance if hardware.get("gpu_readings_available") and known_size else None
            row["fit_estimate"] = {"file_mb": round(mb, 1), "runtime_allowance_mb": round(allowance, 1),
                "available_ram_mb": available_ram, "available_vram_mb": vram if hardware.get("gpu_readings_available") else None,
                "disk_free_mb": free, "ram_fit": ram_fit, "disk_fit": disk_fit,
                "gpu_fit": gpu_fit, "partial_offload_possible": gpu_fit is False and ram_fit is True,
                "limitations": "Estimate only; KV cache, context, projector and other workloads need headroom. Speed/quality/tool/vision support is unmeasured."}
            priority = (disk_fit is not False, ram_fit is not False, row["verified_metadata"],
                        prefer_vision and row["capabilities"]["vision"] == "reported", gpu_fit is True,
                        row["quantization"] in ("Q4_K_M", "Q5_K_M", "Q8_0"))
            ranked.append((priority, self._public(row)))
        ranked.sort(key=lambda item: item[0], reverse=True)
        # First represent each viable reported family, then fill quantization
        # alternatives. Unknown architectures stay unknown, not guessed names.
        counts, shortlist, families = {}, [], set()
        for _, row in ranked:
            family = row["family"]["name"]
            fit = row["fit_estimate"]
            if (family and family not in families and fit["ram_fit"] is True
                    and fit["disk_fit"] is True and row["verified_metadata"]):
                families.add(family)
                counts[row["repo_id"]] = 1
                shortlist.append(row)
                if len(shortlist) == 5:
                    break
        for _, row in ranked:
            if len(shortlist) == 5:
                break
            repo = row["repo_id"]
            if counts.get(repo, 0) >= 2 or any(r["selection_id"] == row["selection_id"] for r in shortlist):
                continue
            counts[repo] = counts.get(repo, 0) + 1
            shortlist.append(row)
        return {"use_case": use_case, "shortlist": shortlist, "weight_bytes_downloaded": 0,
                "family_diversity": {"reported_viable_families": len(families),
                    "limitation": None if len(families) >= 3 else
                    "Fewer than three viable reported families in inspected selections; unknown families are not inferred"},
                "ranking_basis": "Estimated available hardware fit and quantization tradeoffs; not globally best or measured quality"}

    def _fresh(self, candidate):
        """Reinspect pinned files; metadata changes never silently retarget a job."""
        metadata, _, files = self._metadata(candidate["repo_id"], candidate["revision"])
        if metadata.get("gated"):
            raise DiscoveryError("Gated repositories are unsupported; obtain permitted access separately")
        index = {row["filename"]: row for row in files}
        for old in candidate["files"]:
            fresh = index.get(old["filename"])
            if fresh is None or (fresh["sha256"], fresh["size_bytes"]) != (old["sha256"], old["size_bytes"]):
                raise DiscoveryError("Selected file metadata changed; inspect and select again")

    def start_model_download(self, selection_id):
        """Start only an explicitly selected, digest-bound background download."""
        with self._lock:
            if self._closed or (self._thread and self._thread.is_alive()):
                raise DiscoveryError("Download worker busy or closed")
            if len(self._jobs) >= 20:
                raise DiscoveryError("Download job limit reached; reconnect after completion")
            row = self._selection(selection_id)
            if not row["verified_metadata"]:
                raise DiscoveryError("Publisher digest or file size unavailable; unverified downloads are refused")
            if row["gated"]:
                raise DiscoveryError("Gated repositories are unsupported; obtain permitted access separately")
            job = uuid.uuid4().hex
            self._jobs[job] = {"job_id": job, "state": "queued", "bytes_written": 0,
                "files_done": 0, "files_total": len(row["files"]), "total_bytes": row["total_bytes"],
                "selection": row, "error": None, "cleanup_confirmed": False,
                "resume_semantics": "Restart interrupted file; rehash and reuse complete verified files. Session-only job IDs."}
            self._cancel[job] = threading.Event()
            self._start(job)
            return self.download_status(job)

    def _start(self, job):
        self._thread = threading.Thread(target=self._run, args=(job,), daemon=False)
        self._thread.start()

    def _destination(self, job, filename):
        """Job directories isolate repositories with identical file names."""
        folder = self.root / "models" / ("hf-" + job)
        return modelhub.resolve_shard_destination(folder, filename)

    def _verify(self, path, row):
        """Rehash whole files, including reused shards; size alone proves nothing."""
        if not path.is_file() or path.stat().st_size != row["size_bytes"]:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                return False
            handle.seek(0)
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == row["sha256"]

    def _update(self, job, **fields):
        with self._lock:
            self._jobs[job].update(fields)

    def _run(self, job):
        """Use the existing checked streaming verifier for every exact pinned file."""
        row = self._jobs[job]["selection"]
        cancel = self._cancel[job]
        done_bytes = 0
        try:
            self._fresh(row)
            if cancel.is_set():
                self._update(job, state="cancelled", cleanup_confirmed=True)
                return
            folder = self.root / "models" / ("hf-" + job)
            folder.mkdir(parents=True, exist_ok=True)
            disk = self.hub.check_disk(row["total_bytes"], folder)
            if not disk["ok"] or disk["free_mb"] is None:
                raise DiscoveryError("Insufficient or unknown free disk space for complete file set")
            self._update(job, state="downloading")
            for index, file in enumerate(row["files"]):
                if cancel.is_set():
                    self._update(job, state="cancelled", cleanup_confirmed=True)
                    return
                path = self._destination(job, file["filename"])
                if path.exists():
                    if not self._verify(path, file):
                        raise DiscoveryError("Completed download file failed integrity; retained file was not overwritten")
                else:
                    url = modelhub.build_resolve_path_url(modelhub.DEFAULT_API_BASE,
                        row["repo_id"], file["filename"]).replace("/resolve/main/", "/resolve/" + row["revision"] + "/")
                    with egress_log.reason("plugin-hf-download"):
                        outcome = modelhub.download_verified(url, path, file["sha256"],
                            expected_size=file["size_bytes"], max_bytes=file["size_bytes"],
                            require_gguf_magic=True, cancel_event=cancel,
                            progress=lambda progress: self._update(job, bytes_written=done_bytes + progress.bytes_done))
                    if outcome.cancelled:
                        self._update(job, state="cancelled", cleanup_confirmed=True)
                        return
                    if not outcome.ok:
                        raise DiscoveryError("Checked model download failed; incomplete file was not registered")
                    if not self._verify(path, file):
                        raise DiscoveryError("Completed download file failed integrity; retained file was not overwritten")
                done_bytes += file["size_bytes"]
                self._update(job, files_done=index + 1, bytes_written=done_bytes)
            self._update(job, state="cancelled" if cancel.is_set() else "verified", cleanup_confirmed=True)
        except DiscoveryError as exc:
            self._update(job, state="failed", error=str(exc), cleanup_confirmed=True)
        except Exception:
            self._update(job, state="failed", error="Download unavailable; no model registered", cleanup_confirmed=True)

    def download_status(self, job_id):
        with self._lock:
            if not isinstance(job_id, str) or job_id not in self._jobs:
                raise DiscoveryError("Unknown download job")
            return {key: copy.deepcopy(value) for key, value in self._jobs[job_id].items() if key != "selection"}

    def cancel_download(self, job_id):
        """Request cancellation; terminal status confirms current-file cleanup."""
        with self._lock:
            self.download_status(job_id)
            self._cancel[job_id].set()
            self._update(job_id, cancel_requested=True)
            return self.download_status(job_id)

    def resume_download(self, job_id):
        """Retry bound metadata and safely reuse full files; no byte-range promise."""
        with self._lock:
            status = self.download_status(job_id)
            if self._closed or (self._thread and self._thread.is_alive()) or status["state"] not in ("cancelled", "failed"):
                raise DiscoveryError("Only an inactive failed or cancelled download can resume")
            self._cancel[job_id].clear()
            self._update(job_id, state="queued", error=None, cleanup_confirmed=False, cancel_requested=False)
            self._start(job_id)
            return self.download_status(job_id)

    def register_download(self, job_id):
        """Only a reverified complete model set reaches the atomic registry writer."""
        with self._lock:
            status = self.download_status(job_id)
            if status["state"] == "registered":
                return status
            if status["state"] != "verified":
                raise DiscoveryError("Only complete verified downloads can be registered")
            row = self._jobs[job_id]["selection"]
            for file in row["files"]:
                if not self._verify(self._destination(job_id, file["filename"]), file):
                    raise DiscoveryError("Completed download file failed integrity; retained file was not overwritten")
            first = self._destination(job_id, row["model_files"][0])
            native = gguf_meta.read_native_context_length(first)
            if type(native) is not int or native <= 0:
                raise DiscoveryError("Native GGUF context unavailable; verified weights were not registered")
            fingerprint = hashlib.sha256((row["repo_id"] + row["revision"] +
                "|".join(row["model_files"])).encode("ascii")).hexdigest()[:12]
            identity = modelhub.registry_id_for(Path(row["model_files"][0]).name)[:90] + "-hf-" + fingerprint
            _, registry, _ = config.Config.load(self.root)
            existing = next((model for model in registry.models if model.id == identity), None)
            if existing:
                original_root = Path(existing.location).parents[len(Path(row["model_files"][0]).parts) - 1]
                if existing.sha256 != row["files"][0]["sha256"] or any(
                        not self._verify(modelhub.resolve_shard_destination(original_root, file["filename"]), file)
                        for file in row["files"]):
                    raise DiscoveryError("Existing registered model differs; no registry entry was overwritten")
                self._update(job_id, state="registered", model_id=identity, already_registered=True)
                return self.download_status(job_id)
            config.append_model_entry(self.root, identity, identity, str(first),
                context_size=min(native, 8192), gpu_layers=0, quantization=row["quantization"],
                sha256=row["files"][0]["sha256"],
                mmproj=str(self._destination(job_id, row["projector"])) if row["projector"] else "",
                notes="Verified HF commit " + row["revision"] + "; tuning not yet measured")
            self._update(job_id, state="registered", model_id=identity)
            return self.download_status(job_id)

    def close(self):
        """Cancel and drain the owned bounded network worker on disconnect."""
        with self._lock:
            self._closed = True
            for event in self._cancel.values():
                event.set()
            worker = self._thread
        if worker and worker is not threading.current_thread():
            worker.join()
