"""Tests for the context auto-tuner (autotune.py) and its models.yaml writer.

Split the same way the module is: the YaRN maths, the server_args merge and the
probe's decision-making are pure functions tested directly; the registry write is
tested against a real temp models.yaml the way test_config.py tests its siblings;
and the orchestrator is tested with both side effects injected.

What is deliberately NOT tested here is autotune.run_smoke_trial's subprocess
call. It starts a real llama-server on a real GPU for 20-30 seconds, which is not
something a test suite can do; it is left a thin wrapper over the already-tested
`launcher.py --smoke-start` path, the same treatment spawn_in_terminal gets
elsewhere in this suite. Every decision that chooses WHICH context to pass it is
covered below.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
import yaml

import autotune
from autotune import (
    CONTEXT_GRID,
    KV_CACHE_ARGS,
    MAX_YARN_FACTOR,
    Trial,
    autotune_model_context,
    compute_yarn_args,
    first_candidate,
    merge_server_args,
    plan_target_args,
    probe_cap,
    probe_ceiling,
)
from config import RegistryWriteError, write_model_tuning

# --------------------------------------------------------------------------- #
# YaRN parameter computation
# --------------------------------------------------------------------------- #


def test_no_yarn_flags_when_target_is_within_the_trained_window():
    """Inside the trained window there is nothing to extend, so no flags at all."""
    assert compute_yarn_args(262144, 131072) == []
    assert compute_yarn_args(262144, 262144) == []
    assert compute_yarn_args(40960, 8192) == []


def test_yarn_scale_is_ceil_of_target_over_this_models_own_native_window():
    """The 2026-08-22 manual result: native 262144, target 400000 -> scale 2."""
    assert compute_yarn_args(262144, 400000) == [
        "--rope-scaling",
        "yarn",
        "--rope-scale",
        "2",
        "--yarn-orig-ctx",
        "262144",
    ]


def test_yarn_orig_ctx_always_comes_from_the_model_not_a_constant():
    """A different model's native window must produce different numbers.

    The bug this guards is the reason the feature exists: reusing 262144 (or any
    other model's figure) would produce flags that are plausible-looking and
    wrong.
    """
    assert compute_yarn_args(40960, 65536)[-1] == "40960"
    assert compute_yarn_args(131072, 400000)[-1] == "131072"
    # ceil, not round: 400000/131072 is 3.05, and a scale of 3 would leave the
    # top of the context outside the interpolated range.
    assert compute_yarn_args(131072, 400000)[3] == "4"


def test_yarn_rejects_nonsense_inputs():
    with pytest.raises(ValueError):
        compute_yarn_args(0, 1000)
    with pytest.raises(ValueError):
        compute_yarn_args(1000, 0)


# --------------------------------------------------------------------------- #
# server_args merging
# --------------------------------------------------------------------------- #


def test_merge_appends_flags_that_are_absent():
    merged = merge_server_args(["--parallel", "1"], KV_CACHE_ARGS)
    assert merged == [
        "--parallel", "1", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
    ]


def test_merge_does_not_duplicate_flags_that_are_already_present():
    existing = ["--parallel", "1", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    assert merge_server_args(existing, KV_CACHE_ARGS) == existing


def test_merge_replaces_an_existing_value_in_place_and_keeps_other_flags():
    existing = ["--cache-type-k", "f16", "--chat-template", "chatml", "--parallel", "1"]
    merged = merge_server_args(existing, [("--cache-type-k", "q8_0")])
    assert merged == ["--cache-type-k", "q8_0", "--chat-template", "chatml", "--parallel", "1"]


def test_merge_coerces_yaml_integers_to_strings():
    """One real registry row carries ["--parallel", 1] as a YAML int."""
    assert merge_server_args(["--parallel", 1], []) == ["--parallel", "1"]


def test_merge_drops_stale_rope_flags_with_their_values():
    existing = [
        "--parallel", "1",
        "--rope-scaling", "yarn",
        "--rope-scale", "2",
        "--yarn-orig-ctx", "262144",
    ]
    assert merge_server_args(existing, [], drop_flags=autotune.ROPE_FLAGS) == [
        "--parallel", "1",
    ]


def test_plan_target_args_adds_kv_cache_and_yarn_when_the_target_exceeds_native():
    args = plan_target_args(["--parallel", "1"], 262144, 400000)
    assert args == [
        "--parallel", "1",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--rope-scaling", "yarn",
        "--rope-scale", "2",
        "--yarn-orig-ctx", "262144",
    ]


def test_plan_target_args_clears_previous_yarn_when_the_target_no_longer_needs_it():
    """A re-tune that lands inside the trained window must not keep old scaling."""
    existing = [
        "--parallel", "1",
        "--rope-scaling", "yarn",
        "--rope-scale", "2",
        "--yarn-orig-ctx", "262144",
    ]
    args = plan_target_args(existing, 262144, 131072)
    assert args == [
        "--parallel", "1", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
    ]


# --------------------------------------------------------------------------- #
# The bounded probe search
# --------------------------------------------------------------------------- #


class FakeMachine:
    """A machine whose real ceiling is known, so the search can be graded.

    Records every context size actually tried, which is what the "bounded number
    of real starts" requirement is asserted against - each of these would be a
    25-second GPU load in production.
    """

    def __init__(self, ceiling: int):
        self.ceiling = ceiling
        self.tried: list[int] = []

    def __call__(self, context_size: int) -> Trial:
        self.tried.append(context_size)
        ok = context_size <= self.ceiling
        return Trial(context_size, ok, "" if ok else "failed to create context")


def test_probe_finds_a_ceiling_below_the_real_failure_point():
    """The chosen value must be one that actually loaded, with room beneath the
    lowest value that actually failed - never sitting on the knife edge."""
    machine = FakeMachine(ceiling=430000)
    result = probe_ceiling(machine, native_context=262144, current_context=65536)
    assert result.ceiling is not None
    assert result.ceiling <= 430000  # verified to load
    assert result.first_failure is not None
    assert result.first_failure > result.ceiling
    assert len(result.trials) <= autotune.DEFAULT_MAX_TRIALS


def test_probe_respects_the_trial_budget_even_on_a_very_permissive_machine():
    machine = FakeMachine(ceiling=10_000_000)
    result = probe_ceiling(
        machine, native_context=262144, current_context=65536, max_trials=5
    )
    assert len(machine.tried) <= 5
    # Nothing failed, so the search stopped at the 4x policy cap rather than at a
    # real hardware limit - and says so by leaving first_failure unset.
    assert result.first_failure is None
    assert result.ceiling == probe_cap(262144)


def test_probe_never_exceeds_four_times_the_trained_window():
    """Past ~4x, YaRN degrades output, so a bigger number would be a worse one."""
    machine = FakeMachine(ceiling=10_000_000)
    probe_ceiling(machine, native_context=32768, current_context=8192)
    assert max(machine.tried) <= 32768 * MAX_YARN_FACTOR


def test_probe_starts_at_the_models_own_trained_window():
    """The most informative single trial: does what the model was built for fit?"""
    machine = FakeMachine(ceiling=131072)
    probe_ceiling(machine, native_context=131072, current_context=16384)
    assert machine.tried[0] == 131072


def test_probe_starts_at_the_configured_value_when_it_is_already_higher():
    """Re-tuning an already-raised model must not re-prove ground known to work."""
    machine = FakeMachine(ceiling=430000)
    probe_ceiling(machine, native_context=262144, current_context=400000)
    # Snapped to the context grid (400000 is not a multiple of 4096), but the
    # point stands: the opening bid is the already-configured value, not the
    # lower trained window.
    assert machine.tried[0] == 401408
    assert machine.tried[0] > 262144


def test_probe_searches_downward_when_even_the_native_window_will_not_fit():
    """A tight-VRAM model still gets a usable answer instead of a flat failure."""
    machine = FakeMachine(ceiling=40000)
    result = probe_ceiling(machine, native_context=262144, current_context=16384)
    assert result.ceiling is not None
    assert result.ceiling <= 40000
    # first_failure is the LOWEST value that failed, so after halving down from
    # the trained window it is well below it - not the opening 262144.
    assert result.first_failure is not None
    assert result.ceiling < result.first_failure <= 262144
    assert machine.tried[0] == 262144


def test_probe_reports_no_ceiling_when_nothing_loads_at_all():
    machine = FakeMachine(ceiling=0)
    result = probe_ceiling(machine, native_context=262144, current_context=16384)
    assert result.ceiling is None
    assert result.first_failure is not None
    assert len(result.trials) <= autotune.DEFAULT_MAX_TRIALS


def test_probe_candidates_are_snapped_to_the_context_grid():
    machine = FakeMachine(ceiling=300000)
    probe_ceiling(machine, native_context=131072, current_context=16384)
    # The opening bid comes straight from the model/config values, which are
    # already grid multiples here; every value the SEARCH invents must be too.
    for value in machine.tried[1:]:
        assert value % CONTEXT_GRID == 0


def test_probe_publishes_a_progress_line_per_trial():
    """The UI shows which value is being tried; without this it looks frozen."""
    lines: list[str] = []
    probe_ceiling(
        FakeMachine(ceiling=131072),
        native_context=131072,
        current_context=16384,
        progress=lines.append,
    )
    assert any("starting at context 131072" in line for line in lines)
    assert any("loaded" in line for line in lines)


def test_first_candidate_is_clamped_to_the_policy_cap():
    assert first_candidate(32768, 10_000_000) == probe_cap(32768)


# --------------------------------------------------------------------------- #
# The models.yaml writer
# --------------------------------------------------------------------------- #

REGISTRY = """# a comment above the list
version: 1
models:
  - id: alpha
    name: "Alpha"
    location: "models/alpha.gguf"
    context_size: 16384   # an older explanation of this value
    gpu_layers: 60
    benchmark_score: 83.3
    notes: "keep me"
    status: installed
    server_args: ["--parallel", "1"]   # trailing comment
    ready_timeout_s: 240

  - id: beta
    name: "Beta"
    location: "models/beta.gguf"
    context_size: 8192
    gpu_layers: 999
    notes: "do not touch"
    status: installed
"""


def _registry(tmp_path: Path, text: str = REGISTRY) -> Path:
    (tmp_path / "models.yaml").write_text(text, encoding="utf-8")
    return tmp_path


def _rows(tmp_path: Path) -> dict:
    parsed = yaml.safe_load((tmp_path / "models.yaml").read_text(encoding="utf-8"))
    return {row["id"]: row for row in parsed["models"]}


def test_writer_updates_only_context_size_and_server_args(tmp_path):
    base = _registry(tmp_path)
    write_model_tuning(base, "alpha", 262144, ["--parallel", "1", "--cache-type-k", "q8_0"])
    rows = _rows(base)
    assert rows["alpha"]["context_size"] == 262144
    assert rows["alpha"]["server_args"] == ["--parallel", "1", "--cache-type-k", "q8_0"]
    # Every other field on the row survives untouched.
    assert rows["alpha"]["gpu_layers"] == 60
    assert rows["alpha"]["benchmark_score"] == 83.3
    assert rows["alpha"]["notes"] == "keep me"
    assert rows["alpha"]["ready_timeout_s"] == 240


def test_writer_leaves_every_other_model_alone(tmp_path):
    base = _registry(tmp_path)
    write_model_tuning(base, "alpha", 32768, ["--parallel", "1"])
    rows = _rows(base)
    assert rows["beta"]["context_size"] == 8192
    assert rows["beta"]["notes"] == "do not touch"
    assert "server_args" not in rows["beta"]


def test_writer_inserts_server_args_when_the_row_has_none(tmp_path):
    base = _registry(tmp_path)
    write_model_tuning(base, "beta", 65536, ["--cache-type-k", "q8_0"])
    rows = _rows(base)
    assert rows["beta"]["server_args"] == ["--cache-type-k", "q8_0"]
    assert rows["beta"]["context_size"] == 65536
    assert rows["beta"]["gpu_layers"] == 999


def test_writer_preserves_the_files_comments_and_key_order(tmp_path):
    base = _registry(tmp_path)
    write_model_tuning(base, "alpha", 32768, ["--parallel", "1"])
    text = (base / "models.yaml").read_text(encoding="utf-8")
    assert "# a comment above the list" in text
    assert "# trailing comment" in text


def test_writer_replaces_the_stale_inline_comment_when_given_a_note(tmp_path):
    """A dated explanation of the OLD value must not stand beside the new one."""
    base = _registry(tmp_path)
    write_model_tuning(
        base, "alpha", 131072, ["--parallel", "1"], note="set by auto-tune 2026-08-22"
    )
    text = (base / "models.yaml").read_text(encoding="utf-8")
    assert "an older explanation of this value" not in text
    assert "set by auto-tune 2026-08-22" in text
    assert _rows(base)["alpha"]["context_size"] == 131072


def test_writer_refuses_an_unknown_model_and_leaves_the_file_alone(tmp_path):
    base = _registry(tmp_path)
    before = (base / "models.yaml").read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        write_model_tuning(base, "nope", 32768, ["--parallel", "1"])
    assert (base / "models.yaml").read_text(encoding="utf-8") == before


def test_writer_refuses_a_non_positive_context(tmp_path):
    base = _registry(tmp_path)
    with pytest.raises(ValueError):
        write_model_tuning(base, "alpha", 0, ["--parallel", "1"])
    with pytest.raises(ValueError):
        write_model_tuning(base, "alpha", -1, ["--parallel", "1"])


def test_writer_refuses_arguments_it_cannot_render_safely(tmp_path):
    """An argument with a quote or bracket would re-parse into something else."""
    base = _registry(tmp_path)
    with pytest.raises(ValueError):
        write_model_tuning(base, "alpha", 32768, ['--chat-template', 'he said "hi"'])
    with pytest.raises(ValueError):
        write_model_tuning(base, "alpha", 32768, ["--parallel", "1, 2"])


def test_writer_refuses_a_missing_registry_with_a_readable_message(tmp_path):
    with pytest.raises(RegistryWriteError) as excinfo:
        write_model_tuning(tmp_path, "alpha", 32768, ["--parallel", "1"])
    assert "models.yaml" in str(excinfo.value)


def test_writer_leaves_no_temp_files_behind(tmp_path):
    base = _registry(tmp_path)
    write_model_tuning(base, "alpha", 32768, ["--parallel", "1"])
    assert [p.name for p in base.iterdir()] == ["models.yaml"]


# --------------------------------------------------------------------------- #
# The orchestrator, with both side effects injected
# --------------------------------------------------------------------------- #


class FakeWriter:
    """Captures every registry write the orchestrator performs, in order."""

    def __init__(self, fail_on: int | None = None):
        self.calls: list[tuple[str, int, list[str]]] = []
        self.fail_on = fail_on

    def __call__(self, model_id: str, context_size: int, server_args: list[str]) -> None:
        self.calls.append((model_id, context_size, list(server_args)))
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise RegistryWriteError("disk on fire")


def _fixture_gguf(tmp_path: Path, architecture: str, context_length: int) -> Path:
    """Build a synthetic GGUF whose header carries a known native context.

    Written out here rather than imported from test_gguf_meta so this file does
    not depend on another test module's private helpers; the full byte-level
    coverage of the format lives there, this is just enough header for the
    orchestrator to read one real answer out of a real file.
    """
    import struct  # noqa: PLC0415 - only this fixture needs it

    def gstr(text: str) -> bytes:
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    kvs = (
        gstr("general.architecture") + struct.pack("<I", 8) + gstr(architecture)
        + gstr(f"{architecture}.context_length")
        + struct.pack("<I", 4)
        + struct.pack("<I", context_length)
    )
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 2)
    path = tmp_path / "model.gguf"
    path.write_bytes(header + kvs + b"\x00" * 32)
    return path


def test_orchestrator_end_to_end_writes_the_tuned_config(tmp_path):
    gguf = _fixture_gguf(tmp_path, "qwen35", 262144)
    machine = FakeMachine(ceiling=430000)
    writer = FakeWriter()
    lines: list[str] = []

    result = autotune_model_context(
        model_id="alpha",
        location=str(gguf),
        current_context=65536,
        current_server_args=["--parallel", "1"],
        write_tuning=writer,
        trial_fn=machine,
        progress=lines.append,
    )

    assert result.ok
    assert result.native_context == 262144
    assert result.architecture == "qwen35"
    assert result.chosen_context is not None and result.chosen_context <= 430000
    assert result.yarn_applied is True
    assert result.rope_scale == 2
    assert "--cache-type-k" in result.server_args
    assert "--yarn-orig-ctx" in result.server_args
    assert result.server_args[result.server_args.index("--yarn-orig-ctx") + 1] == "262144"
    # First write is the probe-time KV cache change at the UNCHANGED context; the
    # last is the tuned result.
    assert writer.calls[0][1] == 65536
    assert writer.calls[-1][1] == result.chosen_context
    assert any("native context: 262144" in line for line in lines)


def test_orchestrator_adds_no_yarn_when_the_ceiling_is_inside_the_trained_window(tmp_path):
    gguf = _fixture_gguf(tmp_path, "qwen2", 131072)
    writer = FakeWriter()
    result = autotune_model_context(
        model_id="alpha",
        location=str(gguf),
        current_context=16384,
        current_server_args=["--parallel", "1"],
        write_tuning=writer,
        trial_fn=FakeMachine(ceiling=70000),
    )
    assert result.ok
    assert result.yarn_applied is False
    assert result.rope_scale is None
    assert "--rope-scaling" not in result.server_args


def test_orchestrator_restores_previous_settings_when_nothing_loads(tmp_path):
    """A tune that finds no working context must leave the model exactly as it was."""
    gguf = _fixture_gguf(tmp_path, "qwen3", 40960)
    writer = FakeWriter()
    result = autotune_model_context(
        model_id="alpha",
        location=str(gguf),
        current_context=16384,
        current_server_args=["--parallel", "1"],
        write_tuning=writer,
        trial_fn=FakeMachine(ceiling=0),
    )
    assert not result.ok
    assert result.chosen_context is None
    assert result.server_args == ["--parallel", "1"]
    assert writer.calls[-1] == ("alpha", 16384, ["--parallel", "1"])
    assert "left unchanged" in result.detail


def test_orchestrator_rolls_back_when_the_written_config_fails_confirmation(tmp_path):
    """The confirmation start is what stops a broken config being left behind."""
    gguf = _fixture_gguf(tmp_path, "qwen3", 40960)
    writer = FakeWriter()
    calls: list[int] = []

    def flaky(context_size: int) -> Trial:
        calls.append(context_size)
        # Every probe passes; only the final confirmation start fails, which is
        # the case a probe-only design would ship straight into models.yaml.
        if len(calls) > autotune.DEFAULT_MAX_TRIALS or context_size in calls[:-1]:
            return Trial(context_size, False, "failed to create context")
        return Trial(context_size, True)

    result = autotune_model_context(
        model_id="alpha",
        location=str(gguf),
        current_context=16384,
        current_server_args=["--parallel", "1"],
        write_tuning=writer,
        trial_fn=flaky,
    )
    assert not result.ok
    assert "restored" in result.detail
    assert writer.calls[-1] == ("alpha", 16384, ["--parallel", "1"])


def test_orchestrator_reports_an_unreadable_model_file_without_writing(tmp_path):
    writer = FakeWriter()
    result = autotune_model_context(
        model_id="alpha",
        location=str(tmp_path / "missing.gguf"),
        current_context=16384,
        current_server_args=["--parallel", "1"],
        write_tuning=writer,
        trial_fn=FakeMachine(ceiling=999999),
    )
    assert not result.ok
    assert writer.calls == []
    assert "does not exist" in result.detail


def test_orchestrator_reports_a_write_failure_without_probing(tmp_path):
    gguf = _fixture_gguf(tmp_path, "qwen3", 40960)
    machine = FakeMachine(ceiling=999999)
    result = autotune_model_context(
        model_id="alpha",
        location=str(gguf),
        current_context=16384,
        current_server_args=["--parallel", "1"],
        write_tuning=FakeWriter(fail_on=1),
        trial_fn=machine,
    )
    assert not result.ok
    assert machine.tried == []
    assert "server_args" in result.detail


def test_orchestrator_bounds_the_total_number_of_real_starts(tmp_path):
    """Probe budget plus baseline plus one confirmation - never an open search.

    M15.4 added exactly one start to the fixed overhead: the throughput
    baseline at the current context, which is what the floor is derived from.
    """
    gguf = _fixture_gguf(tmp_path, "qwen35", 262144)
    machine = FakeMachine(ceiling=430000)
    autotune_model_context(
        model_id="alpha",
        location=str(gguf),
        current_context=65536,
        current_server_args=["--parallel", "1"],
        write_tuning=FakeWriter(),
        trial_fn=machine,
    )
    assert len(machine.tried) <= autotune.DEFAULT_MAX_TRIALS + 2


# --------------------------------------------------------------------------- #
# Cancellation (owner request 2026-08-22: Stop must interrupt a live auto-tune)
#
# An auto-tune holds the single ops worker for minutes, so "press Stop and
# nothing happens until it finishes" was not an acceptable answer. These cover
# the decision-making with injected trials, the same GPU-free way the probe
# tests above work; the one thing that genuinely needs a real subprocess (does a
# cancel actually kill the llama-server tree) is verified by hand on hardware,
# not here.
# --------------------------------------------------------------------------- #


class CancellingMachine:
    """A fake machine that trips the cancel event partway through the search.

    Models the real thing faithfully: run_smoke_trial notices the event WHILE a
    trial is in flight, kills the tree and returns a trial flagged canceled, so
    that is what this returns too.
    """

    def __init__(self, cancel: threading.Event, cancel_after: int):
        self.cancel = cancel
        self.cancel_after = cancel_after
        self.tried: list[int] = []

    def __call__(self, context_size: int) -> Trial:
        self.tried.append(context_size)
        if len(self.tried) >= self.cancel_after:
            self.cancel.set()
            return Trial(context_size, False, "canceled by the owner", canceled=True)
        return Trial(context_size, True)


def test_probe_stops_immediately_when_the_owner_cancels():
    """No further real starts are spent after a cancel - that is the whole point."""
    cancel = threading.Event()
    machine = CancellingMachine(cancel, cancel_after=2)
    result = probe_ceiling(
        machine, native_context=262144, current_context=65536, cancel=cancel
    )
    assert result.canceled is True
    assert len(machine.tried) == 2, "a canceled search must not keep loading models"


def test_probe_spends_nothing_at_all_when_cancelled_before_it_starts():
    """A cancel that beats the first trial must not fire up a 25-second load."""
    cancel = threading.Event()
    cancel.set()
    machine = FakeMachine(ceiling=430000)
    result = probe_ceiling(
        machine, native_context=262144, current_context=65536, cancel=cancel
    )
    assert result.canceled is True
    assert machine.tried == []


def test_a_cancelled_trial_is_not_counted_as_a_failure_of_that_context():
    """A canceled trial proved nothing about the machine, so it sets no bound.

    If cancellation were recorded as a failure it would become a known-bad upper
    bound and silently drag a later answer downward - reporting a ceiling the
    hardware never actually refused.
    """
    cancel = threading.Event()
    machine = CancellingMachine(cancel, cancel_after=1)
    result = probe_ceiling(
        machine, native_context=262144, current_context=65536, cancel=cancel
    )
    assert result.first_failure is None
    assert result.ceiling is None


def test_cancelling_restores_the_models_previous_settings(tmp_path):
    """Same rollback discipline as a failed tune: an abandoned run changes nothing."""
    gguf = _fixture_gguf(tmp_path, "qwen35", 262144)
    cancel = threading.Event()
    machine = CancellingMachine(cancel, cancel_after=2)
    writer = FakeWriter()

    result = autotune_model_context(
        model_id="alpha",
        location=str(gguf),
        current_context=65536,
        current_server_args=["--parallel", "1"],
        write_tuning=writer,
        trial_fn=machine,
        cancel=cancel,
    )

    assert result.ok is False
    assert result.canceled is True
    assert result.chosen_context is None
    assert result.server_args == ["--parallel", "1"]
    # The probe-time KV-cache write was undone: the LAST write put the original
    # context and the original args back exactly as they were found.
    assert writer.calls[-1] == ("alpha", 65536, ["--parallel", "1"])


def test_a_cancelled_tune_is_reported_as_cancelled_not_as_a_failure(tmp_path):
    """The owner stopped it; the wording must not invent a problem."""
    gguf = _fixture_gguf(tmp_path, "qwen35", 262144)
    cancel = threading.Event()
    machine = CancellingMachine(cancel, cancel_after=2)
    result = autotune_model_context(
        model_id="alpha",
        location=str(gguf),
        current_context=65536,
        current_server_args=[],
        write_tuning=FakeWriter(),
        trial_fn=machine,
        cancel=cancel,
    )
    assert result.canceled is True
    assert "cancel" in result.detail.lower()
    assert "left unchanged" in result.detail
    assert "fail" not in result.detail.lower()


def test_cancelling_before_anything_happens_writes_nothing_at_all(tmp_path):
    """A cancel that lands before step 1 has nothing to roll back and no work to do."""
    gguf = _fixture_gguf(tmp_path, "qwen35", 262144)
    cancel = threading.Event()
    cancel.set()
    writer = FakeWriter()
    machine = FakeMachine(ceiling=430000)
    result = autotune_model_context(
        model_id="alpha",
        location=str(gguf),
        current_context=65536,
        current_server_args=["--parallel", "1"],
        write_tuning=writer,
        trial_fn=machine,
        cancel=cancel,
    )
    assert result.canceled is True
    assert writer.calls == []
    assert machine.tried == []


def test_cancelling_during_the_confirmation_start_also_rolls_back(tmp_path):
    """The tuned values are on disk by then, but were never confirmed to start.

    LOCITIZE only keeps a config it has actually watched come up, so a cancel
    here restores the previous settings exactly as a failed confirmation does.
    """
    gguf = _fixture_gguf(tmp_path, "qwen35", 262144)
    cancel = threading.Event()
    calls: list[int] = []

    def trial(context_size: int) -> Trial:
        # The probe never tries the same context twice, so a repeat can only be
        # the confirmation start of the chosen value - exactly the moment this
        # test wants to cancel at, however many trials the search happened to
        # spend before getting there.
        confirming = context_size in calls
        calls.append(context_size)
        if confirming:
            cancel.set()
            return Trial(context_size, False, "canceled by the owner", canceled=True)
        return Trial(context_size, True)

    writer = FakeWriter()
    result = autotune_model_context(
        model_id="alpha",
        location=str(gguf),
        current_context=65536,
        current_server_args=["--parallel", "1"],
        write_tuning=writer,
        trial_fn=trial,
        cancel=cancel,
    )
    assert result.canceled is True
    assert result.chosen_context is None
    assert writer.calls[-1] == ("alpha", 65536, ["--parallel", "1"])


def test_run_smoke_trial_refuses_to_start_a_load_it_was_already_told_to_cancel():
    """Guards the cheapest case: no subprocess is spawned at all.

    This is the one run_smoke_trial branch that can be tested without a GPU,
    because it returns before reaching subprocess.Popen.
    """
    cancel = threading.Event()
    cancel.set()
    trial = autotune.run_smoke_trial("nonexistent-model", 40960, cancel=cancel)
    assert trial.canceled is True
    assert trial.ok is False
    assert "cancel" in trial.reason.lower()


def test_the_trial_interpreter_is_a_console_python_not_pythonw(monkeypatch, tmp_path):
    """Regression 2026-08-22: the trial must not inherit the GUI's pythonw.exe.

    LOCITIZE Desktop runs under pythonw.exe, which Windows gives no console. A
    trial subprocess started with sys.executable inherited that, and inside it
    ManagedProcess.stop()'s CTRL_BREAK could not be delivered - which is how
    every trial of the owner's 2026-08-22 auto-tune came back "became ready but
    did not shut down cleanly" while the identical command run from a terminal
    (python.exe, with a console) succeeded every time.
    """
    monkeypatch.setattr(autotune.sys, "platform", "win32")
    pythonw = tmp_path / "pythonw.exe"
    pythonw.write_bytes(b"")
    (tmp_path / "python.exe").write_bytes(b"")
    monkeypatch.setattr(autotune.sys, "executable", str(pythonw))
    assert autotune.console_python() == str(tmp_path / "python.exe")


def test_the_trial_interpreter_falls_back_when_there_is_no_sibling_console_python(
    monkeypatch, tmp_path
):
    """An unusual interpreter layout degrades to today's behaviour, not a crash."""
    monkeypatch.setattr(autotune.sys, "platform", "win32")
    pythonw = tmp_path / "pythonw.exe"
    pythonw.write_bytes(b"")  # no python.exe beside it
    monkeypatch.setattr(autotune.sys, "executable", str(pythonw))
    assert autotune.console_python() == str(pythonw)


def test_the_trial_interpreter_is_left_alone_when_it_is_already_a_console_python(
    monkeypatch, tmp_path
):
    """Nothing to fix in the terminal case; do not touch a working interpreter."""
    monkeypatch.setattr(autotune.sys, "platform", "win32")
    python = tmp_path / "python.exe"
    python.write_bytes(b"")
    monkeypatch.setattr(autotune.sys, "executable", str(python))
    assert autotune.console_python() == str(python)


# --------------------------------------------------------------------------- #
# M15.4: the throughput floor
# --------------------------------------------------------------------------- #


def test_floor_rejects_a_context_that_loads_but_crawls():
    """A loaded-but-spilled context is a search failure, with an honest reason."""
    from autotune import Trial, probe_ceiling

    def trial(ctx: int) -> Trial:
        # Everything loads; anything past 100k generates at spill speed.
        return Trial(context_size=ctx, ok=True,
                     tokens_per_second=200.0 if ctx <= 100_000 else 12.0)

    result = probe_ceiling(trial, 262144, 65536, floor_tokps=170.0)
    assert result.ceiling is not None and result.ceiling <= 100_000
    assert result.first_failure is not None
    slow = [t for t in result.trials if not t.ok]
    assert slow and "below the 170.0 tok/s floor" in slow[0].reason


def test_no_floor_preserves_the_load_only_behaviour():
    from autotune import Trial, probe_ceiling

    def trial(ctx: int) -> Trial:
        return Trial(context_size=ctx, ok=True, tokens_per_second=12.0)

    result = probe_ceiling(trial, 262144, 65536, floor_tokps=None)
    # Without a floor, slow-but-loading is still a ceiling (old contract).
    assert result.ceiling is not None


def test_trial_without_measurement_is_not_penalised_by_the_floor():
    from autotune import Trial, probe_ceiling

    def trial(ctx: int) -> Trial:
        return Trial(context_size=ctx, ok=True, tokens_per_second=None)

    result = probe_ceiling(trial, 262144, 65536, floor_tokps=170.0)
    # No measurement means no floor verdict - never fabricate a failure.
    assert result.ceiling is not None


def test_orchestrator_derives_floor_from_the_baseline(tmp_path):
    """The full tune measures a baseline and refuses a spilled 'ceiling'."""
    gguf = _fixture_gguf(tmp_path, "qwen35", 262144)
    lines: list[str] = []

    def trial(ctx: int):
        from autotune import Trial
        return Trial(context_size=ctx, ok=True,
                     tokens_per_second=200.0 if ctx <= 131072 else 13.0)

    outcome = autotune_model_context(
        model_id="alpha",
        location=str(gguf),
        current_context=65536,
        current_server_args=["--parallel", "1"],
        write_tuning=FakeWriter(),
        trial_fn=trial,
        progress=lines.append,
    )
    assert outcome.ok
    assert outcome.chosen_context is not None
    assert outcome.chosen_context <= 131072  # the spill zone was refused
    assert any("baseline 200.0 tok/s" in line for line in lines)
