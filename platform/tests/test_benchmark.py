"""Headless benchmark-engine tests (Milestone 5, Architecture M5.9).

Every test here is deterministic: no GPU, no llama-server, no model file. Speeds
come from canned /completion JSON; scores from canned model-output strings; the
runner is driven with a fake controller and a fake port check. The test-function
names carry the keyword substrings the acceptance criteria target:

  benchmark_timings (AC2), benchmark_score (AC3), benchmark_sweep (AC4),
  benchmark_results (AC5), write_model_score (AC6), benchmark_resume (AC7),
  benchmark_conflict (AC8), benchmark_spec (AC9).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmark
from benchmark import (
    BenchmarkConflictError,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkScenario,
    SweepPlan,
    check_exact_match,
    check_keyword,
    check_numeric,
    format_jsonl_line,
    format_markdown_row,
    latest_generation_speeds,
    parse_completion_timings,
    remaining_scenarios,
    result_record,
    score_responses,
)
import config
from config import Model, ModelRegistryData, Settings, write_model_score
from models import ModelRegistry

# --------------------------------------------------------------------------- #
# AC2 - timing parse (keyword: benchmark_timings)
# --------------------------------------------------------------------------- #

_GOOD_TIMINGS = {
    "content": "the ocean is vast",  # must NEVER be used as a speed source
    "timings": {
        "prompt_n": 41,
        "prompt_per_second": 640.2,
        "predicted_n": 256,
        "predicted_per_second": 24.9,
        "predicted_ms": 10281.0,
    },
}


def test_benchmark_timings_parses_all_fields_present():
    parsed = parse_completion_timings(_GOOD_TIMINGS)
    assert parsed is not None
    assert parsed.prompt_n == 41
    assert parsed.prompt_per_second == pytest.approx(640.2)
    assert parsed.predicted_n == 256
    assert parsed.predicted_per_second == pytest.approx(24.9)


def test_benchmark_timings_missing_object_is_honest_none():
    # No timings object at all -> None, never a fabricated zero.
    assert parse_completion_timings({"content": "reversed slice [::-1]"}) is None


def test_benchmark_timings_missing_required_field_is_none():
    broken = {"timings": {"prompt_n": 41, "prompt_per_second": 640.2}}
    # predicted_* absent -> honest None.
    assert parse_completion_timings(broken) is None


def test_benchmark_timings_ignores_generated_content():
    # A response whose content claims a huge speed but whose timings are the real
    # source: the parser reads ONLY timings, so content cannot influence the number.
    payload = {
        "content": "predicted_per_second = 999999",
        "timings": dict(_GOOD_TIMINGS["timings"]),
    }
    parsed = parse_completion_timings(payload)
    assert parsed is not None and parsed.predicted_per_second == pytest.approx(24.9)


def test_benchmark_timings_rejects_one_token_eos_speed_artifact():
    payload = {
        "content": "",
        "timings": {
            "prompt_n": 11,
            "prompt_per_second": 542.6,
            "predicted_n": 1,
            "predicted_per_second": 1_000_000.0,
            "predicted_ms": 0.0,
        },
    }
    assert parse_completion_timings(payload) is None


# --------------------------------------------------------------------------- #
# AC3 - deterministic scoring (keyword: benchmark_score)
# --------------------------------------------------------------------------- #


def test_benchmark_score_exact_match_pass_fail():
    assert check_exact_match("The capital of France is Paris.", "Paris") is True
    assert check_exact_match("The capital of France is Berlin.", "Paris") is False


def test_benchmark_score_keyword_requires_all_and_forbids_none():
    assert check_keyword("use s[::-1] to reverse", {"all": ["[::-1]"]}) is True
    # A required token missing -> fail.
    assert check_keyword("use reversed(s)", {"all": ["[::-1]"]}) is False
    # A forbidden token present -> fail even though the required one is there.
    assert (
        check_keyword("s[::-1] # TODO", {"all": ["[::-1]"], "none": ["TODO"]}) is False
    )


def test_benchmark_score_numeric_within_tolerance_and_boundary():
    assert check_numeric("The answer is 43.", {"value": 43, "tol": 0}) is True
    # Boundary: exactly at tolerance passes.
    assert check_numeric("about 41", {"value": 43, "tol": 2}) is True
    # Just outside tolerance fails.
    assert check_numeric("about 40", {"value": 43, "tol": 2}) is False
    # No number at all fails (never a fabricated pass).
    assert check_numeric("no idea", {"value": 43, "tol": 0}) is False


def test_benchmark_score_aggregation_is_n_of_m():
    """N-of-M counts and the equal-weight overall mean, on the M15.4 24-task set.

    Answers are built FROM the task definitions rather than hardcoded, so the
    test asserts the aggregation contract (counts, per-category percentage,
    equal-weight mean) without re-encoding the task list. Two tasks per
    category are answered wrongly: 6/8 = 75.0 in each bucket, overall 75.0.
    """
    from benchmark_tasks import TASKS, tasks_for

    responses: dict[str, str] = {}
    for category in ("quality", "reasoning", "coding"):
        for index, task in enumerate(tasks_for(category)):
            if index < 2:
                responses[task.id] = "definitely wrong answer 424242"
            elif task.check == "numeric":
                responses[task.id] = str(task.expected["value"])
            elif task.check == "exact_match":
                responses[task.id] = str(task.expected)
            else:
                responses[task.id] = " ".join(task.expected["all"])

    breakdown = score_responses(responses)
    per_bucket = len(tasks_for("quality"))
    assert (breakdown.quality_passed, breakdown.quality_total) == (per_bucket - 2, per_bucket)
    assert (breakdown.reasoning_passed, breakdown.reasoning_total) == (per_bucket - 2, per_bucket)
    assert (breakdown.coding_passed, breakdown.coding_total) == (per_bucket - 2, per_bucket)
    expected_pct = 100.0 * (per_bucket - 2) / per_bucket
    assert breakdown.quality_score == pytest.approx(expected_pct, abs=0.05)
    assert breakdown.overall_score == pytest.approx(expected_pct, abs=0.05)


# --------------------------------------------------------------------------- #
# AC4 - sweep matrix generation (keyword: benchmark_sweep)
# --------------------------------------------------------------------------- #


def _model(**over) -> Model:
    base = dict(
        id="m",
        name="M",
        description="d",
        location="/locitize-test/m.gguf",
        context_size=10240,
        gpu_layers=60,
        server_args=["--parallel", "1"],
    )
    base.update(over)
    return Model(**base)


def test_benchmark_sweep_degenerates_to_single_current_config():
    # No axes declared -> exactly one scenario using the model's current config.
    scenarios = SweepPlan.generate(_model(), None)
    assert len(scenarios) == 1
    only = scenarios[0]
    assert only.gpu_layers == 60 and only.context_size == 10240
    assert only.server_args == ("--parallel", "1")


def test_benchmark_sweep_produces_full_cartesian_product():
    axes = {
        "gpu_layers": [60, 65],
        "context_size": [8192, 10240],
        "server_args": [["--parallel", "1"], ["--parallel", "1", "--flash-attn"]],
        "spec": [None, {"spec_type": "ngram-simple", "ngram_simple_n": 12}],
    }
    scenarios = SweepPlan.generate(_model(), axes)
    # 2 gpu x 2 ctx x 2 args x 2 spec = 16 distinct scenarios.
    assert len(scenarios) == 16
    # Every scenario carries its FULL config (the owner's non-negotiable rule).
    for s in scenarios:
        assert s.gpu_layers in (60, 65)
        assert s.context_size in (8192, 10240)
    # All keys distinct (no duplicate scenario in the plan).
    assert len({s.key for s in scenarios}) == 16


def test_benchmark_sweep_draft_variant_pulls_model_draft():
    axes = {"spec": [{"spec_type": "draft-simple", "draft_max": 16}]}
    scenarios = SweepPlan.generate(_model(draft_model="qwen3-14b"), axes)
    assert len(scenarios) == 1
    assert scenarios[0].draft_model == "qwen3-14b"
    assert scenarios[0].spec_config["spec_type"] == "draft-simple"


# --------------------------------------------------------------------------- #
# AC5 - results formatting / schema (keyword: benchmark_results)
# --------------------------------------------------------------------------- #


def _ok_result() -> BenchmarkResult:
    return BenchmarkResult(
        model_id="qwen3-6-27b",
        session_id="2026-07-19T1510",
        scenario_key="qwen3-6-27b:gpu60:ctx10240:abcd1234",
        timestamp="2026-07-19 15:10",
        ok=True,
        reason="",
        gpu_layers=60,
        context_size=10240,
        server_args=["--parallel", "1"],
        draft_model=None,
        spec_config=None,
        runs=3,
        prompt_per_second=640.2,
        prompt_ps_min=610.0,
        prompt_ps_max=660.1,
        predicted_per_second=24.9,
        predicted_ps_min=24.1,
        predicted_ps_max=25.4,
        prompt_n=41,
        predicted_n=256,
        quality_passed=2,
        quality_total=2,
        reasoning_passed=1,
        reasoning_total=2,
        coding_passed=1,
        coding_total=2,
        quality_score=100.0,
        reasoning_score=50.0,
        coding_score=50.0,
        overall_score=66.7,
        gpu_name="NVIDIA GeForce RTX 5070 Ti",
        vram_total_mb=16384,
        vram_used_peak_mb=15200,
        ram_used_mb=3400,
    )


def test_benchmark_results_jsonl_record_has_full_config_stamp():
    record = result_record(_ok_result())
    # Config stamp present (Data Model 8.1): every row carries its config.
    for key in ("gpu_layers", "context_size", "server_args", "draft_model", "spec_config"):
        assert key in record
    assert record["gpu_layers"] == 60
    assert record["server_args"] == ["--parallel", "1"]
    # Section 4.1 aliases mirror the p50 headline values.
    assert record["generation_speed"] == record["predicted_per_second"] == 24.9
    assert record["vram_mb"] == record["vram_used_peak_mb"] == 15200


def test_benchmark_results_jsonl_line_is_ascii_and_single_line():
    line = format_jsonl_line(_ok_result())
    assert "\n" not in line
    # ASCII-clean: the whole line encodes as ASCII (no smart punctuation / mojibake).
    line.encode("ascii")
    assert json.loads(line)["model_id"] == "qwen3-6-27b"


def test_benchmark_results_markdown_row_shows_config_and_n_of_m():
    row = format_markdown_row(_ok_result())
    assert "gpu60 ctx10240 --parallel 1" in row  # config stamp visible
    assert "24.9 (24.1-25.4)" in row  # p50 with min-max
    assert "100.0 (2/2)" in row  # N-of-M quality cell


def test_benchmark_results_failed_row_shows_reason_not_a_number():
    failed = BenchmarkResult(
        model_id="test-model-no-location",
        session_id="s",
        scenario_key="k",
        timestamp="2026-07-19 15:10",
        ok=False,
        reason="skipped: model 'test-model-no-location' has no location set",
        gpu_layers=60,
        context_size=10240,
    )
    row = format_markdown_row(failed)
    assert "FAILED" in row
    record = result_record(failed)
    # A failed row never carries a fabricated speed.
    assert record["predicted_per_second"] is None
    assert record["ok"] is False and record["reason"]


def test_benchmark_results_latest_generation_speed_is_same_model_file(tmp_path):
    path = tmp_path / "benchmark_results.jsonl"
    rows = [
        {
            "model_id": "alpha",
            "ok": True,
            "model_file": "/locitize-test/models/alpha.gguf",
            "gpu_layers": 40,
            "context_size": 8192,
            "server_args": ["--parallel", "1"],
            "predicted_per_second": 31.5,
        },
        # Newer and measured at a different GPU split: this is still the model's
        # honest latest benchmark because the UI labels it as the last result.
        {
            "model_id": "alpha",
            "ok": True,
            "model_file": "/locitize-test/models/alpha.gguf",
            "gpu_layers": 999,
            "context_size": 8192,
            "server_args": ["--parallel", "1"],
            "predicted_per_second": 70.0,
        },
        # Same id but a repointed model file: this must not leak onto the row.
        {
            "model_id": "alpha",
            "ok": True,
            "model_file": "/locitize-test/models/different.gguf",
            "gpu_layers": 999,
            "context_size": 8192,
            "predicted_per_second": 99.0,
        },
        # A one-token immediate-EOS timing is not generation throughput. Current
        # llama.cpp builds report this zero-duration event as 1,000,000 tok/s;
        # it must not overwrite the latest usable measurement.
        {
            "model_id": "alpha",
            "ok": True,
            "model_file": "/locitize-test/models/alpha.gguf",
            "predicted_n": 1,
            "predicted_per_second": 1_000_000.0,
        },
        # A failed row never contributes a throughput number.
        {
            "model_id": "beta",
            "ok": False,
            "predicted_per_second": 99.0,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    speeds = latest_generation_speeds(
        path,
        [
            {
                "id": "alpha",
                "location": "/locitize-test/models/alpha.gguf",
                "gpu_layers": 40,
                "context_size": 8192,
                "server_args": ["--parallel", "1"],
            },
            {"id": "beta", "location": "/locitize-test/models/beta.gguf"},
        ],
    )
    assert speeds == {"alpha": 70.0}


# --------------------------------------------------------------------------- #
# AC6 - benchmark_score write-back (keyword: write_model_score)
# --------------------------------------------------------------------------- #

_FIXTURE_YAML = """# owner comment - keep me
version: 1
models:
  - id: alpha
    name: "Alpha"           # inline comment stays
    description: "first"
    location: "/locitize-test/a.gguf"
    context_size: 8192
    gpu_layers: 60
    benchmark_score: null
    status: installed
  - id: beta
    name: "Beta"
    description: "second"
    location: "/locitize-test/b.gguf"
    context_size: 16384
    gpu_layers: 99
    benchmark_score: null
    status: installed
"""


def test_write_model_score_changes_only_target_line(tmp_path: Path):
    path = tmp_path / "models.yaml"
    path.write_text(_FIXTURE_YAML, encoding="utf-8")
    before = path.read_text(encoding="utf-8").splitlines()

    write_model_score(tmp_path, "alpha", 73.9)

    after = path.read_text(encoding="utf-8").splitlines()
    # Exactly one line changed (alpha's benchmark_score), everything else identical.
    diff = [(b, a) for b, a in zip(before, after) if b != a]
    assert len(diff) == 1
    assert "benchmark_score: null" in diff[0][0]
    assert "benchmark_score: 73.9" in diff[0][1]
    # Comments and beta's block are byte-identical.
    assert "# owner comment - keep me" in after[0]
    assert "inline comment stays" in "\n".join(after)


def test_write_model_score_round_trips_and_leaves_others_null(tmp_path: Path):
    import yaml

    path = tmp_path / "models.yaml"
    path.write_text(_FIXTURE_YAML, encoding="utf-8")
    write_model_score(tmp_path, "beta", 88.0)
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    by_id = {m["id"]: m for m in parsed["models"]}
    assert by_id["beta"]["benchmark_score"] == pytest.approx(88.0)
    assert by_id["alpha"]["benchmark_score"] is None  # untouched


def test_write_model_score_rejects_out_of_range(tmp_path: Path):
    path = tmp_path / "models.yaml"
    path.write_text(_FIXTURE_YAML, encoding="utf-8")
    with pytest.raises(ValueError):
        write_model_score(tmp_path, "alpha", 150.0)


# --------------------------------------------------------------------------- #
# AC7 - resume / skip-set (keyword: benchmark_resume)
# --------------------------------------------------------------------------- #


def test_benchmark_resume_returns_only_uncompleted_scenarios(tmp_path: Path):
    scenarios = SweepPlan.generate(
        _model(), {"gpu_layers": [60, 65, 99]}
    )  # three scenarios
    assert len(scenarios) == 3
    done_key = scenarios[1].key

    jsonl = tmp_path / "benchmark_results.jsonl"
    jsonl.write_text(
        json.dumps({"scenario_key": done_key, "ok": True}) + "\n", encoding="utf-8"
    )
    remaining = remaining_scenarios(scenarios, jsonl)
    remaining_keys = {s.key for s in remaining}
    assert done_key not in remaining_keys
    assert remaining_keys == {scenarios[0].key, scenarios[2].key}


def test_benchmark_resume_failed_row_is_retried(tmp_path: Path):
    # A recorded ok:false row is NOT treated as complete -> still retried.
    scenarios = SweepPlan.generate(_model(), {"gpu_layers": [60, 65]})
    jsonl = tmp_path / "benchmark_results.jsonl"
    jsonl.write_text(
        json.dumps({"scenario_key": scenarios[0].key, "ok": False}) + "\n",
        encoding="utf-8",
    )
    remaining = remaining_scenarios(scenarios, jsonl)
    assert scenarios[0].key in {s.key for s in remaining}


# --------------------------------------------------------------------------- #
# AC8 - port/owner conflict refusal (keyword: benchmark_conflict)
# --------------------------------------------------------------------------- #


class _RecordingController:
    """A fake ModelController that records whether start() was ever called."""

    def __init__(self) -> None:
        self.started = False

    def start(self, *args, **kwargs):  # pragma: no cover - must NOT be called
        self.started = True
        raise AssertionError("controller.start must not run while the port is busy")

    def stop(self):  # pragma: no cover
        return None

    def snapshot(self):  # pragma: no cover
        return {"running_model": None, "services": []}


def test_benchmark_conflict_refuses_and_starts_nothing():
    controller = _RecordingController()
    registry = ModelRegistry(ModelRegistryData(models=[_model()]), Settings())
    runner = BenchmarkRunner(
        controller,
        registry,
        Settings(),
        # Fake port check: the reserved llama.cpp port is already occupied.
        port_is_free=lambda port: False,
    )
    with pytest.raises(BenchmarkConflictError) as excinfo:
        runner.run(["m"], runs=1)
    # Honest remedy in the message; nothing was started.
    assert "port" in str(excinfo.value).lower()
    assert controller.started is False


# --------------------------------------------------------------------------- #
# AC9 - spec-decoding flag plumbing (keyword: benchmark_spec, reuse managed_flag)
# --------------------------------------------------------------------------- #


def _registry_with(model: Model) -> ModelRegistry:
    settings = Settings()
    settings.paths.llama_cpp = "llama-server.exe"
    return ModelRegistry(ModelRegistryData(models=[model]), settings)


def test_benchmark_spec_draft_flags_are_data_driven_and_managed_flag_stripped():
    model = _model(
        id="spec",
        draft_model="/locitize-test/draft.gguf",
        spec_config={"spec_type": "draft-simple", "draft_max": 16, "draft_min": 1},
        # An owner-supplied --port must still be stripped by the managed-flag guard.
        server_args=["--parallel", "1", "--port", "9999"],
    )
    spec = _registry_with(model).build_start_spec("spec")
    cmd = spec.command
    # Confirmed flag spellings (llama-server --help, build 10037) appear, data-driven.
    assert "--spec-draft-model" in cmd
    assert cmd[cmd.index("--spec-draft-model") + 1] == "/locitize-test/draft.gguf"
    assert "--spec-type" in cmd and cmd[cmd.index("--spec-type") + 1] == "draft-simple"
    assert "--spec-draft-n-max" in cmd and cmd[cmd.index("--spec-draft-n-max") + 1] == "16"
    assert "--spec-draft-n-min" in cmd
    # The managed --port (L-4) is stripped from owner server_args (the platform port
    # is the only --port in the argv), while the additive spec flags pass through.
    port_indices = [i for i, tok in enumerate(cmd) if tok == "--port"]
    assert len(port_indices) == 1
    assert "9999" not in cmd


def test_benchmark_spec_ngram_variant_has_no_draft_flag():
    model = _model(
        id="ng",
        spec_config={"spec_type": "ngram-simple", "ngram_simple_n": 12},
    )
    spec = _registry_with(model).build_start_spec("ng")
    cmd = spec.command
    assert "--spec-type" in cmd and cmd[cmd.index("--spec-type") + 1] == "ngram-simple"
    assert "--spec-ngram-simple-size-n" in cmd
    # No draft model configured -> no draft-model flag appended.
    assert "--spec-draft-model" not in cmd


def test_benchmark_spec_override_forces_speculation_off_for_a_scenario():
    # A sweep 'spec off' scenario passes spec_config=None explicitly, overriding the
    # model's own draft/spec fields so the baseline argv carries no spec flags.
    model = _model(
        id="ov",
        draft_model="/locitize-test/draft.gguf",
        spec_config={"spec_type": "draft-simple"},
    )
    spec = _registry_with(model).build_start_spec(
        "ov", draft_model=None, spec_config=None
    )
    assert "--spec-draft-model" not in spec.command
    assert "--spec-type" not in spec.command


# --------------------------------------------------------------------------- #
# F-3 - run() happy path, RunLock lifecycle, write-back guard
#   (keywords: benchmark_results / benchmark_sweep so the AC suites collect them)
# --------------------------------------------------------------------------- #

# A canned /completion response: valid timings (so a run is counted) plus content
# for the deterministic scorer. The generated text is never used as a speed source.
_FAKE_COMPLETION = {
    "content": "the answer is 42",
    "timings": {
        "prompt_n": 8,
        "prompt_per_second": 500.0,
        "predicted_n": 128,
        "predicted_per_second": 45.0,
        "predicted_ms": 2844.0,
    },
}


def _fake_http_post(url, payload, timeout):
    """Fake loopback POST: always returns the same canned completion (no server)."""
    return dict(_FAKE_COMPLETION)


class _FakeRunController:
    """Fake ModelController for a successful run: start()->RUNNING, records state.

    Captures whether the RunLock file existed at the moment start() was called,
    which proves run() acquired the lock BEFORE touching any scenario, and counts
    stop() calls to prove every scenario is torn down.
    """

    def __init__(self, lock_path: Path, port: int = 8080) -> None:
        self._lock_path = lock_path
        self._port = port
        self.lock_present_at_start = False
        self.start_calls = 0
        self.stop_calls = 0
        self._running_model = None

    def start(self, model_id, context_size, gpu_layers, *,
              server_args, draft_model, spec_config):
        from services import ServiceStatus

        self.start_calls += 1
        # run() holds the RunLock across all scenarios; the lock must already exist
        # here (acquire-before-work). Latch True if ever seen present.
        if self._lock_path.exists():
            self.lock_present_at_start = True
        self._running_model = model_id
        return ServiceStatus.RUNNING

    def stop(self):
        self.stop_calls += 1
        self._running_model = None

    def snapshot(self):
        return {
            "running_model": self._running_model,
            "services": [{"model_id": self._running_model, "port": self._port}],
        }


_RUN_YAML = """version: 1
models:
  - id: rmodel
    name: "RModel"
    description: "run-test model"
    location: "/locitize-test/r.gguf"
    context_size: 8192
    gpu_layers: 60
    benchmark_score: null
    status: installed
"""


def _run_model() -> Model:
    # Mirrors the _RUN_YAML row so the in-memory registry and the on-disk models.yaml
    # (used by the score write-back) describe the same model.
    return _model(
        id="rmodel",
        name="RModel",
        description="run-test model",
        location="/locitize-test/r.gguf",
        context_size=8192,
        gpu_layers=60,
        server_args=["--parallel", "1"],
    )


def _run_runner(tmp_path: Path):
    """Build a headless runner writing all artifacts under tmp_path (no real server)."""
    from dataclasses import replace as _replace

    (tmp_path / "models.yaml").write_text(_RUN_YAML, encoding="utf-8")
    model = _run_model()
    settings = _replace(Settings(), base_dir=tmp_path)
    registry = ModelRegistry(ModelRegistryData(models=[model]), settings)
    lock_path = tmp_path / "benchmark.lock"
    controller = _FakeRunController(lock_path)
    runner = BenchmarkRunner(
        controller,
        registry,
        settings,
        http_post=_fake_http_post,
        port_is_free=lambda port: True,  # nothing occupies the reserved port
        results_dir=tmp_path,
        lock_dir=tmp_path,
    )
    return runner, controller, model, lock_path


def test_benchmark_results_run_appends_rows_and_manages_runlock(tmp_path: Path):
    # A full successful run() through the fake provider: warm-up/measure/score/append,
    # RunLock acquire+release, and rows landing in BOTH result files.
    runner, controller, model, lock_path = _run_runner(tmp_path)

    summary = runner.run([model.id], runs=1)

    # One scenario, one ok row, zero failures.
    assert summary["scenarios"] == 1
    assert summary["ok"] == 1 and summary["failed"] == 0

    # The machine-readable and human result files both grew a matching record.
    jsonl = tmp_path / "benchmark_results.jsonl"
    md = tmp_path / "benchmark_results.md"
    assert jsonl.exists() and md.exists()
    row = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[-1])
    assert row["model_id"] == model.id and row["ok"] is True
    assert row["predicted_per_second"] == 45.0  # from the canned timings, not content
    # F-2 attribution stamp is written so the 27B gate can check staleness later.
    assert row["model_file"] == "/locitize-test/r.gguf"
    assert model.id in md.read_text(encoding="utf-8")

    # RunLock lifecycle: present while a model was started, gone once run() returned.
    assert controller.lock_present_at_start is True
    assert not lock_path.exists()
    assert controller.stop_calls >= 1  # scenario torn down (no orphan)

    # Current-config score was persisted to models.yaml (write-back fired).
    import yaml

    parsed = yaml.safe_load((tmp_path / "models.yaml").read_text(encoding="utf-8"))
    by_id = {m["id"]: m for m in parsed["models"]}
    assert by_id["rmodel"]["benchmark_score"] is not None


def test_benchmark_sweep_variant_row_does_not_clobber_score(tmp_path: Path, monkeypatch):
    # M5.7 data-integrity guard: a sweep run visits several configs, but ONLY the
    # current-registry-config row may write back benchmark_score. A variant row must
    # never clobber the persisted number.
    runner, controller, model, _lock = _run_runner(tmp_path)

    # A two-config sweep: the current gpu_layers plus one variant. The model's own
    # benchmark_sweep field is what run(sweep=True) consumes.
    axes = {"gpu_layers": [model.gpu_layers, model.gpu_layers + 5]}
    sweep_model = _model(
        id=model.id,
        name=model.name,
        description=model.description,
        location=model.location,
        context_size=model.context_size,
        gpu_layers=model.gpu_layers,
        server_args=list(model.server_args),
        benchmark_sweep=axes,
    )
    from dataclasses import replace as _replace

    settings = _replace(Settings(), base_dir=tmp_path)
    registry = ModelRegistry(ModelRegistryData(models=[sweep_model]), settings)
    runner = BenchmarkRunner(
        controller,
        registry,
        settings,
        http_post=_fake_http_post,
        port_is_free=lambda port: True,
        results_dir=tmp_path,
        lock_dir=tmp_path,
    )

    # Record every write-back call; the guard must let exactly one through.
    writes: list[tuple[str, float]] = []
    monkeypatch.setattr(
        config, "write_model_score",
        lambda base_dir, model_id, score: writes.append((model_id, score)),
    )

    summary = runner.run([sweep_model.id], runs=1, sweep=True)
    assert summary["scenarios"] == 2 and summary["ok"] == 2

    # Exactly one write-back: the current-config scenario, never the variant.
    assert len(writes) == 1
    assert writes[0][0] == sweep_model.id

    # And the guard itself classifies each scenario correctly.
    scenarios = SweepPlan.generate(sweep_model, axes)
    current = next(s for s in scenarios if s.gpu_layers == sweep_model.gpu_layers)
    variant = next(s for s in scenarios if s.gpu_layers != sweep_model.gpu_layers)
    assert runner._is_current_config(current) is True
    assert runner._is_current_config(variant) is False
