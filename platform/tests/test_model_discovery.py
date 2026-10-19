"""Offline acceptance of bounded discovery, pinned downloads and registration."""
import copy
import hashlib
import io
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

import config
import modelhub
from model_discovery import DiscoveryError, ModelDiscovery


REVISION = "a" * 40
BODY = b"GGUF" + b"verified-model-bytes" * 20


def file_row(name="test-Q4_K_M.gguf", body=BODY):
    return {"type": "file", "path": name, "size": len(body),
            "lfs": {"oid": hashlib.sha256(body).hexdigest(), "size": len(body)}}


class Hub:
    """Network fixture separates metadata reads from actual weight transfers."""
    config = modelhub.HubConfig()

    def __init__(self):
        self.files = [file_row()]
        self.metadata = {"sha": REVISION, "cardData": {"license": "mit"}}
        self.urls = []
        self._last_search = []
        self.disk = {"ok": True, "free_mb": 10000}

    def _get_json_with_link(self, url):
        self.urls.append(url)
        return copy.deepcopy(self.files if "/tree/" in url else self.metadata), None

    def search(self, query):
        self._last_search = [{"repo_id": "owner/repo", "license_tag": "mit"}]
        return {"ok": True, "items": self._last_search, "has_more": True}

    def search_more(self):
        self._last_search += [{"repo_id": "owner/repo"}] * 10
        return {"ok": True, "items": self._last_search, "has_more": True}

    def check_disk(self, *_):
        return self.disk


@pytest.fixture
def discovery(tmp_path, monkeypatch):
    monkeypatch.setattr("egress_log._root", None)
    config.ensure_user_config(tmp_path)
    hub = Hub()
    owner = ModelDiscovery(tmp_path, hub, hardware=lambda: {
        "ram_available_mb": 4000, "gpu_readings_available": True,
        "gpus": [{"vram_free_mb": 2000}, {"vram_free_mb": 1000}]})
    monkeypatch.setattr(owner.system, "disk_free_mb", lambda _: 10000)
    yield owner
    owner.close()


def select(owner):
    return owner.list_model_files("owner/repo")["candidates"][0]["selection_id"]


def finish(owner, job):
    owner._thread.join(timeout=5)
    assert not owner._thread.is_alive()
    return owner.download_status(job["job_id"])


def transfer(monkeypatch, body=BODY, entered=None, release=None):
    """Exercise the production streaming verifier with a bounded fake response."""
    original = modelhub.download_verified
    seen = []

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(body))}

    class Opener:
        def open(self, target, timeout):
            seen.append(target)
            if entered:
                entered.set()
                assert release.wait(5)
            return Response(body)

    def download(*args, **kwargs):
        return original(*args, **kwargs, opener=Opener())

    monkeypatch.setattr(modelhub, "download_verified", download)
    return seen


def test_search_inspect_recommend_never_transfers_or_registers(discovery, monkeypatch):
    monkeypatch.setattr(modelhub, "download_verified", lambda *_a, **_k: pytest.fail("hidden transfer"))
    before = (discovery.root / "models.yaml").read_bytes()
    assert discovery.search_models("coding GGUF")["weight_bytes_downloaded"] == 0
    selected = select(discovery)
    result = discovery.recommend_model_candidates([selected], "coding", True)
    assert result["weight_bytes_downloaded"] == 0
    row = result["shortlist"][0]
    assert row["fit_estimate"]["available_vram_mb"] == 2000  # no cross-device pooling promise
    assert row["capabilities"]["vision"] == "unknown"
    assert row["capabilities"]["tools"] == "unknown"
    assert (discovery.root / "models.yaml").read_bytes() == before
    assert not (discovery.root / "models").exists()


def test_cursor_is_opaque_query_bound_and_page_limited(discovery):
    result = discovery.search_models("coding")
    with pytest.raises(DiscoveryError):
        discovery.search_models("other", cursor=result["cursor"])
    for _ in range(4):
        result = discovery.search_models("coding", cursor=result["cursor"])
    with pytest.raises(DiscoveryError, match="page limit"):
        discovery.search_models("coding", cursor=result["cursor"])
    assert len(discovery.hub._last_search) <= 50


@pytest.mark.parametrize("defect", ["digest", "size", "gated"])
def test_unverified_or_gated_selection_never_starts(discovery, defect):
    if defect == "digest":
        discovery.hub.files[0]["lfs"]["oid"] = None
    elif defect == "size":
        discovery.hub.files[0]["lfs"]["size"] = 0
    else:
        discovery.hub.metadata["gated"] = "auto"
    with pytest.raises(DiscoveryError):
        discovery.start_model_download(select(discovery))
    assert discovery._thread is None


@pytest.mark.parametrize("changed", ["revision", "digest", "size"])
def test_remote_change_fails_without_weight_transfer(discovery, monkeypatch, changed):
    selection = select(discovery)
    if changed == "revision":
        discovery.hub.metadata["sha"] = "b" * 40
    else:
        discovery.hub.files[0]["lfs"]["oid" if changed == "digest" else "size"] = "b" * 64 if changed == "digest" else 9
    monkeypatch.setattr(modelhub, "download_verified", lambda *_a, **_k: pytest.fail("changed transfer"))
    result = finish(discovery, discovery.start_model_download(selection))
    assert result["state"] == "failed" and result["cleanup_confirmed"]


def test_real_streaming_verifier_progress_pin_register_and_duplicate(discovery, monkeypatch):
    seen = transfer(monkeypatch)
    monkeypatch.setattr("model_discovery.gguf_meta.read_native_context_length", lambda _: 32768)
    job = finish(discovery, discovery.start_model_download(select(discovery)))
    assert job["state"] == "verified" and job["bytes_written"] == len(BODY)
    assert "/resolve/" + REVISION + "/" in str(seen[0])
    registered = discovery.register_download(job["job_id"])
    assert registered["state"] == "registered"
    assert discovery.register_download(job["job_id"]) == registered
    second = finish(discovery, discovery.start_model_download(select(discovery)))
    assert discovery.register_download(second["job_id"])["already_registered"]
    _, registry, _ = config.Config.load(discovery.root)
    assert len(registry.models) == 1
    assert registry.models[0].gpu_layers == 0
    assert registry.models[0].context_size == 8192


@pytest.mark.parametrize("defect", ["corrupt", "disk", "unknown_disk"])
def test_failed_transfer_never_registers(discovery, monkeypatch, defect):
    transfer(monkeypatch, b"GGUF" + b"x" * (len(BODY) - 4) if defect == "corrupt" else BODY)
    if defect == "disk":
        discovery.hub.disk["ok"] = False
    if defect == "unknown_disk":
        discovery.hub.disk["free_mb"] = None
    result = finish(discovery, discovery.start_model_download(select(discovery)))
    assert result["state"] == "failed"
    with pytest.raises(DiscoveryError, match="verified"):
        discovery.register_download(result["job_id"])
    assert config.Config.load(discovery.root)[1].models == []


def test_cancel_drains_and_resume_restarts_interrupted_file(discovery, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    seen = transfer(monkeypatch, entered=entered, release=release)
    job = discovery.start_model_download(select(discovery))
    assert entered.wait(5)
    discovery.cancel_download(job["job_id"])
    release.set()
    cancelled = finish(discovery, job)
    assert cancelled["state"] == "cancelled" and cancelled["cleanup_confirmed"]
    assert list(discovery.root.rglob("*.partial")) == []
    result = finish(discovery, discovery.resume_download(job["job_id"]))
    assert result["state"] == "verified" and len(seen) == 2
    assert "Restart interrupted file" in result["resume_semantics"]


def test_complete_shard_corruption_not_skipped_on_resume(discovery, monkeypatch):
    discovery.hub.files = [file_row("model-00001-of-00002.gguf"), file_row("model-00002-of-00002.gguf")]
    transfer(monkeypatch)
    job = finish(discovery, discovery.start_model_download(select(discovery)))
    candidate = discovery._jobs[job["job_id"]]["selection"]
    first = discovery._destination(job["job_id"], candidate["model_files"][0])
    first.write_bytes(b"GGUF" + b"x" * (len(BODY) - 4))
    discovery._update(job["job_id"], state="failed")
    result = finish(discovery, discovery.resume_download(job["job_id"]))
    assert result["state"] == "failed" and "integrity" in result["error"]
    assert config.Config.load(discovery.root)[1].models == []


def test_incomplete_shards_and_ambiguous_projector_not_offered_as_model(discovery):
    discovery.hub.files = [file_row("model-00001-of-00002.gguf"), file_row("mmproj-one.gguf"), file_row("mmproj-two.gguf")]
    assert discovery.list_model_files("owner/repo")["candidates"] == []
    discovery.hub.files.append(file_row())
    row = discovery.list_model_files("owner/repo")["candidates"][0]
    assert row["projector"] is None
    assert "unsupported" in row["capabilities"]["projector_pairing"]


def test_importance_matrix_gguf_is_not_offered_as_a_model(discovery):
    discovery.hub.files.insert(0, file_row("test.imatrix.gguf"))
    candidates = discovery.list_model_files("owner/repo")["candidates"]
    assert candidates
    assert all("imatrix" not in file["filename"].lower()
               for candidate in candidates for file in candidate["files"])


def test_nonstandard_card_pairing_is_not_trusted(discovery):
    discovery.hub.files.append(file_row("mmproj.gguf"))
    discovery.hub.metadata["cardData"].update(locitize_model_files=["test-Q4_K_M.gguf"], locitize_projector="mmproj.gguf")
    row = discovery.list_model_files("owner/repo")["candidates"][0]
    assert row["total_bytes"] == len(BODY)
    assert row["projector"] is None
    assert "separate verified" in row["capabilities"]["download_scope"]


def test_low_memory_disk_and_unknown_hardware_remain_estimates(discovery, monkeypatch):
    selection = select(discovery)
    discovery.hardware = lambda: {"ram_available_mb": 0, "gpus": None, "gpu_readings_available": False}
    monkeypatch.setattr(discovery.system, "disk_free_mb", lambda _: 0)
    fit = discovery.recommend_model_candidates([selection])["shortlist"][0]["fit_estimate"]
    assert fit["ram_fit"] is False and fit["disk_fit"] is False
    assert fit["gpu_fit"] is None and fit["available_vram_mb"] is None


def test_metadata_byte_cap_uses_checked_opener():
    class Response(io.BytesIO):
        headers = {}
    opener = SimpleNamespace(open=lambda *_a, **_k: Response(b" " * (4 * 1024 * 1024 + 1)))
    hub = modelhub.Downloader(opener_factory=lambda _: opener)
    with pytest.raises(modelhub.HubError, match="4 MiB"):
        hub._get_json_with_link("https://huggingface.co/api/models/owner/repo")


@pytest.mark.parametrize("repo", ["../evil", "owner/repo/extra", "https://evil.invalid/file"])
def test_unsafe_repo_rejected_before_network(discovery, repo):
    with pytest.raises(ValueError):
        discovery.list_model_files(repo)
    assert discovery.hub.urls == []


@pytest.mark.parametrize("native", [None, 0, -1, True, "8192"])
def test_unknown_or_invalid_native_context_never_registers(discovery, monkeypatch, native):
    transfer(monkeypatch)
    monkeypatch.setattr("model_discovery.gguf_meta.read_native_context_length", lambda _: native)
    job = finish(discovery, discovery.start_model_download(select(discovery)))
    with pytest.raises(DiscoveryError, match="Native GGUF"):
        discovery.register_download(job["job_id"])
    assert config.Config.load(discovery.root)[1].models == []


def test_dedup_rechecks_all_existing_registered_shards(discovery, monkeypatch):
    discovery.hub.files = [file_row("model-00001-of-00002.gguf"), file_row("model-00002-of-00002.gguf")]
    transfer(monkeypatch)
    monkeypatch.setattr("model_discovery.gguf_meta.read_native_context_length", lambda _: 8192)
    first = finish(discovery, discovery.start_model_download(select(discovery)))
    discovery.register_download(first["job_id"])
    bound = discovery._jobs[first["job_id"]]["selection"]
    discovery._destination(first["job_id"], bound["model_files"][1]).unlink()
    second = finish(discovery, discovery.start_model_download(select(discovery)))
    with pytest.raises(DiscoveryError, match="Existing registered"):
        discovery.register_download(second["job_id"])
    assert len(config.Config.load(discovery.root)[1].models) == 1


def test_plugin_ledger_records_hosts_and_search_inspect_download_reasons(discovery, monkeypatch):
    import json
    import egress_log
    response_bodies = [json.dumps([{"id": "owner/repo"}]).encode(),
        json.dumps(discovery.hub.metadata).encode(), json.dumps(discovery.hub.files).encode()]
    class Response(io.BytesIO):
        headers = {}
    opener = SimpleNamespace(open=lambda *_a, **_k: Response(response_bodies.pop(0)))
    discovery.hub = modelhub.Downloader(modelhub.HubConfig(search_limit=10),
        opener_factory=lambda _: opener, system_provider=discovery.system)
    discovery.search_models("GGUF")
    selection = select(discovery)
    # The same pinned metadata will be refreshed before the actual transfer.
    response_bodies += [json.dumps({"sha": REVISION}).encode(), json.dumps([file_row()]).encode()]
    transfer(monkeypatch)
    assert finish(discovery, discovery.start_model_download(selection))["state"] == "verified"
    ledger = egress_log.egress_path(discovery.root).read_text()
    records = [json.loads(line) for line in ledger.splitlines()]
    assert {record["reason"] for record in records} >= {"hub-search", "plugin-hf-inspect", "plugin-hf-download"}
    assert all(record["host"] == "huggingface.co" for record in records)
    assert "owner/repo" not in ledger and "GGUF" not in ledger


@pytest.mark.parametrize("kind,expected", [("offline", "offline"), ("rate_limited", "rate limit"), ("gated", "Gated")])
def test_remote_errors_are_distinct_and_do_not_echo_secrets(discovery, monkeypatch, kind, expected):
    def fail(_):
        raise modelhub.HubError(kind, "private path token secret")
    monkeypatch.setattr(discovery.hub, "_get_json_with_link", fail)
    with pytest.raises(DiscoveryError, match=expected) as error:
        discovery.list_model_files("owner/repo")
    assert "secret" not in str(error.value)


def test_missing_file_size_does_not_claim_fit(discovery):
    discovery.hub.files[0]["lfs"]["size"] = 0
    row = discovery.recommend_model_candidates([select(discovery)])["shortlist"][0]
    assert row["fit_estimate"]["ram_fit"] is None
    assert row["fit_estimate"]["disk_fit"] is None
    assert row["fit_estimate"]["gpu_fit"] is None


def test_active_job_rejects_concurrent_start_and_early_registration(discovery, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    transfer(monkeypatch, entered=entered, release=release)
    selection = select(discovery)
    job = discovery.start_model_download(selection)
    assert entered.wait(5)
    try:
        with pytest.raises(DiscoveryError, match="busy"):
            discovery.start_model_download(selection)
        with pytest.raises(DiscoveryError, match="verified"):
            discovery.register_download(job["job_id"])
    finally:
        release.set()
    assert finish(discovery, job)["state"] == "verified"


def test_cancel_request_does_not_wait_for_blocked_metadata(discovery, monkeypatch):
    import time
    selection = select(discovery)
    entered, release = threading.Event(), threading.Event()
    original = discovery._fresh
    def blocked(candidate):
        entered.set()
        assert release.wait(5)
        original(candidate)
    monkeypatch.setattr(discovery, "_fresh", blocked)
    monkeypatch.setattr(modelhub, "download_verified", lambda *_a, **_k: pytest.fail("cancelled transfer"))
    job = discovery.start_model_download(selection)
    assert entered.wait(5)
    try:
        started = time.monotonic()
        assert discovery.cancel_download(job["job_id"])["cancel_requested"]
        assert time.monotonic() - started < .5
    finally:
        release.set()
    assert finish(discovery, job)["state"] == "cancelled"
    assert not (discovery.root / "models").exists()


def test_compact_view_keeps_private_integrity_binding(discovery):
    row = discovery.list_model_files("owner/repo")["candidates"][0]
    assert row["files"] == [{"filename": "test-Q4_K_M.gguf", "size_bytes": len(BODY)}]
    assert "revision" not in row
    private = discovery._selection(row["selection_id"])
    assert private["revision"] == REVISION
    assert private["files"][0]["sha256"] == hashlib.sha256(BODY).hexdigest()
    assert row["family"]["status"] == "unknown"


def test_viable_family_diversity_over_duplicate_repos(discovery):
    ids = []
    for index, family in enumerate(["qwen", "qwen", "gemma", "mistral"]):
        discovery.hub.metadata["config"] = {"model_type": family}
        ids.append(discovery.list_model_files("owner/repo" + str(index))["candidates"][0]["selection_id"])
    result = discovery.recommend_model_candidates(ids, "coding")
    assert [row["family"]["name"] for row in result["shortlist"][:3]] == ["qwen", "gemma", "mistral"]
    assert result["family_diversity"]["limitation"] is None
    discovery.hub.metadata["config"] = {"model_type": "unsafe\n" + "x" * 100}
    row = discovery.list_model_files("owner/other")["candidates"][0]
    assert row["family"]["status"] == "unknown"
    result = discovery.recommend_model_candidates([row["selection_id"]])
    assert result["family_diversity"]["limitation"]


def test_unknown_memory_does_not_claim_viable_family_diversity(discovery):
    ids = []
    for index, family in enumerate(["one", "two", "three"]):
        discovery.hub.metadata["config"] = {"model_type": family}
        ids.append(discovery.list_model_files("owner/model" + str(index))["candidates"][0]["selection_id"])
    discovery.hardware = lambda: {}
    result = discovery.recommend_model_candidates(ids)
    assert result["family_diversity"]["reported_viable_families"] == 0
    assert result["family_diversity"]["limitation"]
    assert all(row["fit_estimate"]["ram_fit"] is None for row in result["shortlist"])
