"""Fine-tune studio + discovery tests (M13, AC-M13-1/2/3).

Everything here is headless: no Streamlit, no GPU, no container runtime, no
display. The discovery fixtures reproduce the SHAPES actually observed in the
owner's real outputs/ tree (Architecture M13.0), which is the point - a scanner
built against an idealized tree would miss four of the six real models.

Keyword map for the acceptance criteria:
  finetune_scan  -> the scanner tests (AC-M13-2)
  finetune_dedup -> the registry dedup tests (AC-M13-3)
  finetune_port  -> studio port reassignment (defect D-M13-1 / DEC-M13-2)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import finetune
import pytest
from config import FineTuneConfig, Model, Settings, append_model_entry
from services import PortUnavailableError

# One .gguf's worth of bytes; the exact value does not matter, only that files
# with the same size compare equal and files with different sizes do not.
_PAYLOAD = b"GGUF" + b"0" * 1024


def _settings(tmp_path: Path, **overrides) -> Settings:
    """A Settings whose fine-tune block points at a fixture tree."""
    cfg = FineTuneConfig(
        enabled=True,
        studio_dir=str(tmp_path / "studio"),
        **overrides,
    )
    return Settings(finetune=cfg, base_dir=tmp_path)


def _write(path: Path, payload: bytes = _PAYLOAD) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


@pytest.fixture()
def outputs_tree(tmp_path: Path) -> Path:
    """Reproduce every real outputs/ shape the scanner must survive.

    - conforming        : <run>.<quant>.gguf, the documented naming
    - mismatched        : a file named after a DIFFERENT run (4 of 6 real files)
    - two-gguf          : one run exporting two quants
    - log-only          : a run folder with train.log and no model at all
    - zero-byte         : an export still in progress
    - meta-good/-bad    : a valid and a malformed run_meta.json
    """
    root = tmp_path / "finetune" / "outputs"
    _write(root / "DemoGPT-v5" / "DemoGPT-v5.q4_k_m.gguf")
    _write(root / "DemoGPT-v4-vocab-regression" / "DemoGPT.q4_k_m.gguf")
    _write(root / "two-quants" / "two-quants.q4_k_m.gguf")
    _write(root / "two-quants" / "two-quants.f16.gguf", _PAYLOAD + b"xx")
    (root / "run-20260815-231314").mkdir(parents=True, exist_ok=True)
    (root / "run-20260815-231314" / "train.log").write_text("step 1\n", encoding="utf-8")
    _write(root / "still-exporting" / "still-exporting.q4_k_m.gguf", b"")
    _write(root / "meta-good" / "meta-good.q8_0.gguf")
    (root / "meta-good" / "run_meta.json").write_text(
        json.dumps({"base_model": "Qwen2.5-1.5B", "default_system": "Be helpful.",
                    "epochs": 2}),
        encoding="utf-8",
    )
    _write(root / "meta-bad" / "meta-bad.q4_k_m.gguf")
    (root / "meta-bad" / "run_meta.json").write_text("{not json", encoding="utf-8")
    # Intermediate artifacts one level deeper must NOT be surfaced as models.
    _write(root / "DemoGPT-v5" / "merged_model" / "leftover.gguf")
    return root


@pytest.fixture(autouse=True)
def _clear_cache():
    """Discovery is TTL-cached in memory; each test starts from a cold cache."""
    finetune.clear_scan_cache()
    yield
    finetune.clear_scan_cache()


def test_blank_studio_dir_resolves_to_the_bundled_copy():
    """An empty finetune.studio_dir must resolve to Codebase/finetune-studio,

    the vendored copy shipped in the repo, so a fresh clone works with zero
    owner-specific configuration (owner request 2026-08-21).
    """
    settings = Settings(finetune=FineTuneConfig(enabled=True, studio_dir=""))
    assert finetune.studio_dir(settings) == finetune.BUNDLED_STUDIO_DIR


def test_explicit_studio_dir_still_overrides_the_bundled_copy(tmp_path: Path):
    """A configured studio_dir must win over the bundled fallback."""
    override = tmp_path / "other-studio"
    settings = Settings(finetune=FineTuneConfig(enabled=True, studio_dir=str(override)))
    assert finetune.studio_dir(settings) == override


def test_blank_studio_dir_reports_unavailable_when_the_bundled_copy_is_missing(
    monkeypatch, tmp_path: Path
):
    """If the bundled directory is absent, studio_dir() must fall back to None,

    not silently point at a nonexistent path (e.g. a source checkout with the
    finetune-studio/ directory stripped out).
    """
    monkeypatch.setattr(finetune, "BUNDLED_STUDIO_DIR", tmp_path / "does-not-exist")
    settings = Settings(finetune=FineTuneConfig(enabled=True, studio_dir=""))
    assert finetune.studio_dir(settings) is None
    ok, reason = finetune.studio_available(settings)
    assert ok is False
    assert "bundled copy" in reason


# --------------------------------------------------------------------------- #
# AC-M13-2: the scanner (keyword finetune_scan)
# --------------------------------------------------------------------------- #


def test_finetune_scan_discovers_exactly_the_real_shapes(tmp_path, outputs_tree):
    """Exactly the servable .gguf files are discovered, and nothing else."""
    result = finetune.scan_outputs(_settings(tmp_path), use_cache=False)
    ids = sorted(item.id for item in result.items)
    assert ids == [
        "ft:DemoGPT-v4-vocab-regression",
        "ft:DemoGPT-v5",
        "ft:meta-bad",
        "ft:meta-good",
        "ft:two-quants:two-quants.f16",
        "ft:two-quants:two-quants.q4_k_m",
    ]
    # The log-only run and the zero-byte export are absent; the nested
    # merged_model/ artifact was not walked into.
    assert not any("still-exporting" in item.id for item in result.items)
    assert not any("run-20260815" in item.id for item in result.items)
    assert not any("leftover" in item.path for item in result.items)
    assert result.reason == ""


def test_finetune_scan_names_rows_after_the_run_folder(tmp_path, outputs_tree):
    """A mismatched filename still displays as its run folder, never the file stem."""
    result = finetune.scan_outputs(_settings(tmp_path), use_cache=False)
    row = next(i for i in result.items if i.run == "DemoGPT-v4-vocab-regression")
    assert row.name == "DemoGPT-v4-vocab-regression"
    assert Path(row.path).name == "DemoGPT.q4_k_m.gguf"


def test_finetune_scan_parses_quant_or_says_unknown(tmp_path):
    """Known quant tokens uppercase; anything else yields "" (rendered unknown)."""
    assert finetune.parse_quant("DemoGPT.q4_k_m.gguf") == "Q4_K_M"
    assert finetune.parse_quant("model.f16.gguf") == "F16"
    # "v5" is not a quant, so it must not be mistaken for one.
    assert finetune.parse_quant("DemoGPT-v5.gguf") == ""
    assert finetune.parse_quant("weird.zzz.gguf") == ""


def test_finetune_scan_reads_optional_run_meta_and_survives_a_bad_one(
    tmp_path, outputs_tree
):
    """run_meta.json enriches when valid and is ignored (never raised) when not."""
    result = finetune.scan_outputs(_settings(tmp_path), use_cache=False)
    good = next(i for i in result.items if i.run == "meta-good")
    bad = next(i for i in result.items if i.run == "meta-bad")
    assert good.base_model == "Qwen2.5-1.5B"
    assert good.recommended_prompt == "Be helpful."
    assert "epochs=2" in good.meta_note
    assert bad.base_model == ""
    assert finetune.describe_meta(bad) == "metadata: none"


def test_finetune_scan_drops_a_path_that_escapes_the_root(tmp_path, outputs_tree):
    """A symlink pointing outside the outputs root is never discovered."""
    outside = _write(tmp_path / "elsewhere" / "escape.q4_k_m.gguf")
    link = outputs_tree / "sneaky" / "escape.q4_k_m.gguf"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        # Windows refuses symlink creation without developer mode or elevation.
        # The guard itself is a pure function, so assert it directly instead of
        # skipping the escape case entirely.
        assert not finetune.path_is_confined(outside, outputs_tree)
        return
    result = finetune.scan_outputs(_settings(tmp_path), use_cache=False)
    assert not any(i.run == "sneaky" for i in result.items)
    assert not finetune.path_is_confined(link, outputs_tree)


def test_finetune_scan_missing_root_is_an_honest_empty_state(tmp_path):
    """A missing outputs root yields an empty list plus a reason, never a crash.

    Updated for DEC-M14-9: a blank finetune.outputs_dir now resolves to
    <data root>/finetune/outputs rather than <studio_dir>/outputs, so "unset"
    no longer means "no path at all" - it means a real, not-yet-created folder
    inside the one directory the user backs up. The honest empty state is
    therefore the missing-directory one, which UX Spec section 8 pins.
    """
    unset = Settings(finetune=FineTuneConfig(enabled=True), base_dir=tmp_path)
    assert finetune.outputs_root(unset) == tmp_path / "finetune" / "outputs"
    result = finetune.scan_outputs(unset, use_cache=False)
    assert result.items == ()
    assert "Could not read" in result.reason

    missing = finetune.scan_outputs(_settings(tmp_path), use_cache=False)
    assert missing.items == ()
    assert "Could not read" in missing.reason


def test_finetune_scan_caches_within_the_ttl(tmp_path, outputs_tree):
    """A repaint inside the TTL reuses the previous scan; a later one re-walks."""
    settings = _settings(tmp_path)
    first = finetune.scan_outputs(settings, now=100.0)
    _write(outputs_tree / "late-arrival" / "late-arrival.q4_k_m.gguf")
    cached = finetune.scan_outputs(settings, now=101.0)
    assert [i.id for i in cached.items] == [i.id for i in first.items]
    fresh = finetune.scan_outputs(settings, now=200.0)
    assert any(i.run == "late-arrival" for i in fresh.items)


# --------------------------------------------------------------------------- #
# AC-M13-3: dedup against the manual registry (keyword finetune_dedup)
# --------------------------------------------------------------------------- #


def _manual(model_id: str, location: str) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        description="",
        location=location,
        context_size=8192,
        gpu_layers=999,
    )


def test_finetune_dedup_folds_a_same_name_same_size_copy(tmp_path, outputs_tree):
    """A deployed copy outside the scan root folds into its manual row."""
    discovered = finetune.scan_outputs(_settings(tmp_path), use_cache=False).items
    original = next(i for i in discovered if i.run == "DemoGPT-v5")
    copy = _write(tmp_path / "AI" / "Models" / "DemoGPT-V5.Q4_K_M.gguf")
    assert copy.stat().st_size == original.size_bytes

    marked, matched = finetune.dedup_against_registry(
        discovered, [_manual("demogpt-v5", str(copy))]
    )
    folded = [i for i in marked if i.already_registered]
    assert [i.run for i in folded] == ["DemoGPT-v5"]
    assert matched == {"demogpt-v5"}
    assert not any(i.already_registered for i in marked if i.run != "DemoGPT-v5")


def test_finetune_dedup_keeps_a_different_size_file_separate(tmp_path, outputs_tree):
    """A same-named file of a different size is a different model; both show."""
    discovered = finetune.scan_outputs(_settings(tmp_path), use_cache=False).items
    other = _write(
        tmp_path / "AI" / "Models" / "DemoGPT-v5.q4_k_m.gguf", _PAYLOAD + b"different"
    )
    marked, matched = finetune.dedup_against_registry(
        discovered, [_manual("demogpt-v5", str(other))]
    )
    assert matched == set()
    assert not any(i.already_registered for i in marked)


def test_finetune_dedup_matches_an_exact_path_in_place(tmp_path, outputs_tree):
    """A manual row pointing straight at the discovered file folds by path identity."""
    discovered = finetune.scan_outputs(_settings(tmp_path), use_cache=False).items
    in_place = next(i for i in discovered if i.run == "meta-good")
    marked, matched = finetune.dedup_against_registry(
        discovered, [_manual("meta-good", in_place.path)]
    )
    assert matched == {"meta-good"}
    assert next(i for i in marked if i.run == "meta-good").already_registered


def test_finetune_dedup_claims_one_manual_row_only_once(tmp_path):
    """Byte-identical look-alikes do not all vanish into one manual row.

    The owner's four historical DemoGPT runs ship IDENTICAL copies of the same
    file. Folding all four into the single manual row would hide three real,
    distinct builds, so only the first claims the row.
    """
    root = tmp_path / "finetune" / "outputs"
    for run in ("DemoGPT", "DemoGPT-v1", "DemoGPT-v2", "DemoGPT-v4"):
        _write(root / run / "DemoGPT.q4_k_m.gguf")
    deployed = _write(tmp_path / "AI" / "Models" / "DemoGPT.Q4_K_M.gguf")

    discovered = finetune.scan_outputs(_settings(tmp_path), use_cache=False).items
    marked, matched = finetune.dedup_against_registry(
        discovered, [_manual("demogpt", str(deployed))]
    )
    assert matched == {"demogpt"}
    survivors = [i.run for i in marked if not i.already_registered]
    assert survivors == ["DemoGPT-v1", "DemoGPT-v2", "DemoGPT-v4"]


def test_finetune_dedup_skips_a_manual_row_whose_file_is_missing(tmp_path, outputs_tree):
    """An unreadable manual location asserts no match; duplication beats hiding."""
    discovered = finetune.scan_outputs(_settings(tmp_path), use_cache=False).items
    marked, matched = finetune.dedup_against_registry(
        discovered, [_manual("ghost", str(tmp_path / "gone" / "DemoGPT-v5.q4_k_m.gguf"))]
    )
    assert matched == set()
    assert not any(i.already_registered for i in marked)


# --------------------------------------------------------------------------- #
# The managed studio service spec
# --------------------------------------------------------------------------- #


def _installed_studio(tmp_path: Path) -> Settings:
    """A fixture studio with an app script and its own interpreter."""
    studio = tmp_path / "studio"
    (studio / "app").mkdir(parents=True, exist_ok=True)
    (studio / "app" / "app.py").write_text("# streamlit app\n", encoding="utf-8")
    interpreter = studio / ".venv" / "Scripts" / "python.exe"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_bytes(b"MZ")
    return _settings(tmp_path)


def test_finetune_studio_spec_is_loopback_only_and_carries_no_env(tmp_path):
    """The spec binds 127.0.0.1, disables telemetry, and injects no SECRET.

    M17.5: the child env now carries exactly one entry - the data-root PATH -
    so the studio's deploy step targets the running install's real models.yaml.
    A path is not a credential; the assertion is that nothing secret-shaped
    (keys, tokens, passwords) is injected, not that the env is literally empty."""
    settings = _installed_studio(tmp_path)
    spec = finetune.build_studio_spec(settings, log_path="logs/finetune_studio.log")
    assert spec.name == "finetune-studio"
    assert "--server.address" in spec.command
    assert spec.command[spec.command.index("--server.address") + 1] == "127.0.0.1"
    assert "0.0.0.0" not in spec.command
    assert spec.command[spec.command.index("--server.port") + 1] == "8501"
    assert spec.command[spec.command.index("--server.headless") + 1] == "true"
    assert spec.command[spec.command.index("--browser.gatherUsageStats") + 1] == "false"
    assert set(spec.env) == {"LOCITIZE_DATA_DIR"}
    assert spec.env["LOCITIZE_DATA_DIR"] == str(settings.data_dir)
    assert not any(
        k for k in spec.env
        if any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET", "PASS"))
    )
    # TCP readiness on purpose: Streamlit's HTTP health path is version-fragile.
    assert spec.health_path is None
    assert spec.append_log is True


def test_finetune_studio_uses_the_studios_own_interpreter(tmp_path):
    """The child runs on the studio's venv, never on the LOCITIZE interpreter."""
    settings = _installed_studio(tmp_path)
    spec = finetune.build_studio_spec(settings)
    assert spec.command[0].endswith(str(Path(".venv") / "Scripts" / "python.exe"))
    assert spec.command[1:4] == ["-m", "streamlit", "run"]


def test_finetune_studio_unavailable_states_name_the_missing_piece(tmp_path, monkeypatch):
    """Every unavailable case returns a concrete remedy, not a bare 'unavailable'."""
    disabled = Settings(finetune=FineTuneConfig(enabled=False), base_dir=tmp_path)
    ok, reason = finetune.studio_available(disabled)
    assert not ok and "finetune.enabled" in reason

    # A blank studio_dir normally resolves to the bundled copy shipped in the
    # repo; simulate a stripped-down checkout where that copy is absent so
    # this case still exercises the "not found" remedy.
    monkeypatch.setattr(finetune, "BUNDLED_STUDIO_DIR", tmp_path / "no-bundled-copy")
    unset = Settings(finetune=FineTuneConfig(enabled=True), base_dir=tmp_path)
    ok, reason = finetune.studio_available(unset)
    assert not ok and "LOCITIZE_FINETUNE_STUDIO_DIR" in reason

    missing_app = _settings(tmp_path)
    (tmp_path / "studio").mkdir(parents=True, exist_ok=True)
    ok, reason = finetune.studio_available(missing_app)
    assert not ok and "app.py" in reason

    # An unusable studio must raise a ValueError carrying that remedy, never
    # produce a half-built spec.
    with pytest.raises(ValueError):
        finetune.build_studio_spec(missing_app)


def test_finetune_studio_url_is_loopback(tmp_path):
    assert finetune.studio_url(_settings(tmp_path)) == "http://127.0.0.1:8501/"


# --------------------------------------------------------------------------- #
# Port reassignment (defect D-M13-1 / decision DEC-M13-2)
# --------------------------------------------------------------------------- #


class _FakeAllocator:
    """Stand-in for services.PortAllocator that touches no real sockets.

    `occupied` names the ports it refuses; anything else is handed back as-is
    except the preferred port, for which it returns `fallback`.
    """

    def __init__(self, occupied: set[int], fallback: int) -> None:
        self.occupied = occupied
        self.fallback = fallback
        self.calls: list[int] = []

    def ensure_free(self, port: int) -> int:
        self.calls.append(port)
        return self.fallback if port in self.occupied else port


def test_finetune_port_resolved_value_reaches_both_argv_and_spec(tmp_path):
    """D-M13-1: the reassigned port must land in --server.port AND ServiceSpec.port.

    This is the exact failure QA reproduced: the allocator resolved 8501 -> 8080
    while the child was still told to bind 8501. Asserting both in one test means
    the two can never drift apart unnoticed again.
    """
    settings = _installed_studio(tmp_path)
    allocator = _FakeAllocator(occupied={8501}, fallback=8080)
    spec = finetune.build_studio_spec(settings, None, allocator)
    argv_port = spec.command[spec.command.index("--server.port") + 1]
    assert allocator.calls == [8501]
    assert argv_port == "8080"
    assert spec.port == 8080
    # And neither carries the originally configured (occupied) port anywhere.
    assert "8501" not in spec.command


def test_finetune_port_default_path_is_unchanged_without_an_allocator(tmp_path):
    """Regression guard: no allocator means exactly the pre-M13 spec."""
    settings = _installed_studio(tmp_path)
    spec = finetune.build_studio_spec(settings)
    assert spec.command[spec.command.index("--server.port") + 1] == "8501"
    assert spec.port == 8501


def test_finetune_port_allocator_failure_propagates(tmp_path):
    """A spec that cannot get a port must raise, never ship a misconfigured one."""

    class _Exhausted:
        def ensure_free(self, port: int) -> int:
            raise PortUnavailableError("no free port in reserved range 8000-8099")

    settings = _installed_studio(tmp_path)
    with pytest.raises(PortUnavailableError):
        finetune.build_studio_spec(settings, None, _Exhausted())


def test_finetune_port_studio_url_reflects_the_resolved_port(tmp_path):
    """The URL follows the port the studio really bound, not the configured one."""
    settings = _settings(tmp_path)
    assert finetune.studio_url(settings, 8080) == "http://127.0.0.1:8080/"
    assert finetune.studio_url(settings) == "http://127.0.0.1:8501/"
    # The host stays a hardcoded literal no matter what the caller passes.
    assert finetune.studio_url(settings, 8080).startswith("http://127.0.0.1:")


def test_finetune_orphan_warning_is_the_verbatim_shipped_text():
    """The warning is the shipped wording, and it names the real limitation."""
    text = finetune.orphan_warning_text()
    shipped = (
        Path(finetune.__file__).resolve().parent / "finetune_warning.txt"
    ).read_text(encoding="utf-8").strip()
    assert text == shipped
    assert "may still be running" in text
    assert "container" in text.lower()


def test_finetune_active_run_follows_the_train_log_mtime(tmp_path):
    """A freshly written train.log means a run is live; a stale one does not."""
    root = tmp_path / "finetune" / "outputs"
    (root / "live").mkdir(parents=True, exist_ok=True)
    log = root / "live" / "train.log"
    log.write_text("step 1\n", encoding="utf-8")
    settings = _settings(tmp_path)
    assert finetune.active_run(settings) == "live"
    # Age the log past the window; the run is no longer considered live.
    stale = log.stat().st_mtime - 10_000
    os.utime(log, (stale, stale))
    assert finetune.active_run(settings) is None


def test_finetune_active_run_survives_a_log_mtime_ahead_of_the_clock(tmp_path):
    """A train.log timestamped AHEAD of time.time() still reads as live.

    Regression test for the clock-skew defect: the filesystem timestamp source is
    finer-grained than time.time()'s float seconds, so a log written moments ago
    routinely produces a negative age, and a network share can push the mtime
    seconds into the future. Rejecting a negative age suppressed the honesty
    warning on the exact case it exists for - a run that just wrote its log.
    """
    root = tmp_path / "finetune" / "outputs"
    (root / "live").mkdir(parents=True, exist_ok=True)
    log = root / "live" / "train.log"
    log.write_text("step 1\n", encoding="utf-8")
    settings = _settings(tmp_path)

    # Sub-second skew: the exact shape observed on real hardware.
    future = time.time() + 0.5
    os.utime(log, (future, future))
    assert finetune.active_run(settings) == "live"

    # Gross skew from a share with a badly set clock: still live, never dropped.
    future = time.time() + 120.0
    os.utime(log, (future, future))
    assert finetune.active_run(settings) == "live"


def test_finetune_active_run_is_deterministic_for_a_just_written_log(tmp_path):
    """Writing train.log and immediately asking must answer live, every time.

    The old code failed this roughly 1 run in 10 because it compared two clocks
    of different resolutions. Repeating the write/ask cycle makes the race the
    reviewer measured (19 negative ages in 200 writes) reproducible here.
    """
    root = tmp_path / "finetune" / "outputs"
    (root / "live").mkdir(parents=True, exist_ok=True)
    log = root / "live" / "train.log"
    settings = _settings(tmp_path)
    for step in range(200):
        log.write_text(f"step {step}\n", encoding="utf-8")
        assert finetune.active_run(settings) == "live", f"lost the run at step {step}"


# --------------------------------------------------------------------------- #
# Promotion: to_model, generated ids, and the models.yaml writer
# --------------------------------------------------------------------------- #


def test_finetune_to_model_is_a_normal_servable_model(tmp_path, outputs_tree):
    """A discovered model is an ordinary Model, marked with its source."""
    settings = _settings(tmp_path)
    item = next(
        i for i in finetune.scan_outputs(settings, use_cache=False).items
        if i.run == "DemoGPT-v5"
    )
    model = finetune.to_model(item, settings)
    assert model.source == "discovered"
    assert model.source_run == "DemoGPT-v5"
    assert model.status == "installed"
    assert model.location == item.path
    assert model.context_size == settings.finetune.default_context_size


def test_finetune_register_id_strips_the_prefix_and_cleans_the_name():
    assert finetune.register_id_for("ft:DemoGPT-v5") == "DemoGPT-v5"
    assert finetune.register_id_for("ft:sheng tutor.e2e") == "sheng-tutor-e2e"


def test_finetune_register_appends_a_real_row_and_refuses_duplicates(tmp_path):
    """append_model_entry writes a real row, preserves comments, refuses clashes."""
    gguf = _write(tmp_path / "outputs" / "DemoGPT-v5" / "DemoGPT-v5.q4_k_m.gguf")
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        "version: 1\n"
        "models:\n"
        "  # the owner's own comment must survive\n"
        "  - id: existing\n"
        '    name: "Existing"\n'
        '    description: ""\n'
        '    location: "/locitize-test/models/existing.gguf"\n'
        "    context_size: 8192\n"
        "    gpu_layers: 999\n"
        "    benchmark_score: null\n"
        "    status: installed\n",
        encoding="utf-8",
    )

    append_model_entry(
        tmp_path, "DemoGPT-v5", "DemoGPT-v5", str(gguf), quantization="Q4_K_M"
    )
    text = models_yaml.read_text(encoding="utf-8")
    assert "the owner's own comment must survive" in text
    # SEC-M13-5: the id is emitted double-quoted, like every other string scalar.
    assert 'id: "DemoGPT-v5"' in text

    import yaml

    rows = yaml.safe_load(text)["models"]
    assert [r["id"] for r in rows] == ["existing", "DemoGPT-v5"]
    assert rows[1]["location"] == str(gguf)

    with pytest.raises(ValueError) as clash:
        append_model_entry(tmp_path, "DemoGPT-v5", "again", str(gguf))
    assert "already exists" in str(clash.value)


def test_append_model_entry_quotes_a_yaml_significant_id(tmp_path):
    """SEC-M13-5: a YAML-significant id must land as ONE quoted scalar.

    `append_model_entry` is public API. Its current caller sanitizes the id, but
    a future one might not, so the writer itself must not be able to restructure
    the document. The id below would break an unquoted emitter outright.
    """
    gguf = _write(tmp_path / "outputs" / "odd" / "odd.q4_k_m.gguf")
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text("version: 1\nmodels: []\n", encoding="utf-8")
    hostile = "weird: id #2"

    append_model_entry(tmp_path, hostile, "Odd", str(gguf), quantization="Q4_K_M")

    import yaml

    rows = yaml.safe_load(models_yaml.read_text(encoding="utf-8"))["models"]
    assert [r["id"] for r in rows] == [hostile]
    assert rows[0]["location"] == str(gguf)


def test_finetune_register_refuses_a_location_that_does_not_exist(tmp_path):
    """A row pointing at nothing is refused rather than written."""
    (tmp_path / "models.yaml").write_text("version: 1\nmodels: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        append_model_entry(tmp_path, "ghost", "Ghost", str(tmp_path / "nope.gguf"))


def test_finetune_reserved_prefix_row_is_ignored_by_the_loader(tmp_path):
    """A hand-written models.yaml id in the ft: namespace is warned about and dropped."""
    from config import Config

    (tmp_path / "models.yaml").write_text(
        "version: 1\n"
        "models:\n"
        "  - id: ft:sneaky\n"
        '    name: "Sneaky"\n'
        '    location: "/locitize-test/models/x.gguf"\n'
        "    context_size: 8192\n"
        "    gpu_layers: 999\n",
        encoding="utf-8",
    )
    _settings_loaded, models, issues = Config.load(tmp_path)
    assert [m.id for m in models.models] == []
    assert any("reserved 'ft:' prefix" in str(i) for i in issues)


def test_finetune_empty_state_names_the_scanned_folder_and_the_key_that_moves_it(
    tmp_path,
):
    """The empty state has to explain a silent behaviour change (round 6, MEDIUM-6).

    Before DEC-M14-9 a blank finetune.outputs_dir meant <studio_dir>/outputs, so
    setting studio_dir alone was enough for discovery to find a user's training
    runs. It no longer is: the scan now looks in the data root. A user in that
    position sees an empty list, and the ONLY thing that can explain it to them
    is the empty state itself - so it must name the folder actually scanned and
    the key that repoints it.
    """
    settings = Settings(
        finetune=FineTuneConfig(enabled=True, studio_dir=str(tmp_path / "studio")),
        base_dir=tmp_path,
    )
    scanned = finetune.outputs_root(settings)
    scanned.mkdir(parents=True)

    result = finetune.scan_outputs(settings, use_cache=False)

    assert result.items == ()
    assert str(scanned) in result.reason, (
        "the empty state does not say which folder was scanned"
    )
    assert "finetune.outputs_dir" in result.reason, (
        "the empty state does not name the key that repoints the scan"
    )
    # studio_dir must NOT be offered as the fix: since DEC-M14-9 setting it
    # changes nothing about where discovery looks.
    assert "studio_dir" not in result.reason


def test_finetune_unconfigured_empty_state_does_not_send_the_user_to_studio_dir(
    tmp_path,
):
    """The 'not configured at all' reason names the one key that still works."""
    settings = Settings(finetune=FineTuneConfig(enabled=True), base_dir=tmp_path)
    object.__setattr__(settings, "data_dir", None)
    assert finetune.outputs_root(settings) is None

    reason = finetune.scan_outputs(settings, use_cache=False).reason
    assert "finetune.outputs_dir" in reason
    assert "studio_dir" not in reason


# --------------------------------------------------------------------------- #
# delete_run - the guarded permanent delete (M18.7)
# --------------------------------------------------------------------------- #


def _delete_settings(tmp_path):
    """Settings whose outputs root is a real temp dir with one run in it."""
    root = tmp_path / "outputs"
    run = root / "my-run"
    run.mkdir(parents=True)
    (run / "model.q4_k_m.gguf").write_bytes(b"g" * 64)
    settings = _settings(tmp_path)
    settings.finetune.outputs_dir = str(root)
    return settings, root, run


def test_delete_run_removes_the_folder(tmp_path):
    import finetune

    settings, _root, run = _delete_settings(tmp_path)
    ok, message = finetune.delete_run(settings, "my-run")
    assert ok and "deleted" in message
    assert not run.exists()


def test_delete_run_refuses_escape_and_the_root_itself(tmp_path):
    import finetune

    settings, root, run = _delete_settings(tmp_path)
    ok, message = finetune.delete_run(settings, "..")
    assert not ok and "outside the outputs root" in message
    ok, message = finetune.delete_run(settings, ".")
    assert not ok
    assert run.exists() and root.exists()  # nothing was touched


def test_delete_run_refuses_a_registered_runs_files(tmp_path):
    import finetune
    from config import Model

    settings, _root, run = _delete_settings(tmp_path)
    registered = Model(
        id="my-ft", name="My FT", description="", location=str(run / "model.q4_k_m.gguf"),
        context_size=8192, gpu_layers=999,
    )
    ok, message = finetune.delete_run(settings, "my-run", [registered])
    assert not ok and "my-ft" in message and "Models page" in message
    assert run.exists()  # refused, not deleted


def test_delete_run_already_gone_is_ok(tmp_path):
    import finetune

    settings, _root, _run = _delete_settings(tmp_path)
    ok, message = finetune.delete_run(settings, "never-existed")
    assert ok and "already gone" in message
