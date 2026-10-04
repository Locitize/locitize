"""Benchmark engine for the LOCITIZE platform (Milestone 5).

Promotes the M1 record-only stub into a real runner (Architecture M5.1-M5.11,
Data Model sections 7-8). What it does and, importantly, what it does NOT do:

- It measures speed EXCLUSIVELY from the llama.cpp server's own /completion
  `timings` object (prompt_per_second / predicted_per_second). It never derives a
  tok/s number from the generated text (M5.2, Permission Matrix section 8).
- It owns NO process-spawn/kill code. Every model start/stop/switch goes through
  the injected ModelController / ServiceManager, so it inherits the proven
  D-M4-1 no-orphan teardown for free (M5.1).
- Its quality/reasoning/coding score means exactly "N of M deterministic local
  checks passed" (benchmark_tasks.py); it is NOT an LLM-judged rating (M5.3).
- All HTTP is loopback only (http://127.0.0.1:<resolved_port>/completion), the
  same fixed-host discipline as the readiness probe (SEC-1).

The pure functions here (parse_completion_timings, the checkers, score_responses,
SweepPlan.generate, the record/row formatters, the resume filter, scenario_key)
are the headless unit-test seams (M5.9): the whole scoring/plumbing surface is
validated with canned data and no GPU. Only run_scenario/run touch a real server.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from benchmark_tasks import CATEGORIES, TASKS, BenchmarkTask

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class BenchmarkConflictError(RuntimeError):
    """Raised when the runner refuses to start (port busy / another run active).

    The message is an owner-facing remedy, not a stack trace (M5.8): the runner
    never co-loads a second multi-GB model onto the 16GB card behind an
    already-serving one.
    """


# --------------------------------------------------------------------------- #
# 1. Timing parse - the single speed source (M5.2, keyword benchmark_timings)
# --------------------------------------------------------------------------- #


@dataclass
class CompletionTimings:
    """The exact fields taken from a /completion response `timings` object.

    Nothing here is ever computed from the generated text; these are the server's
    own authoritative per-request measurements.
    """

    prompt_n: int
    prompt_per_second: float
    predicted_n: int
    predicted_per_second: float
    predicted_ms: float


# The four required timings keys. Absence of any one is an honest failure (None),
# never a zero or a guess (M5.2).
_REQUIRED_TIMING_KEYS = (
    "prompt_n",
    "prompt_per_second",
    "predicted_n",
    "predicted_per_second",
)


def parse_completion_timings(response_json: dict[str, Any]) -> CompletionTimings | None:
    """Extract timings from a /completion response, or None on an honest failure.

    Returns a CompletionTimings only when the `timings` object is present AND
    every required numeric field is present and numeric. If the object or any
    required field is missing (a stripped/older build, RB3), returns None so the
    caller records a failed run rather than a fabricated zero. Deliberately does
    NOT read response_json["content"] - the generated text is never a speed source
    (M5.2 / Permission Matrix section 8).
    """
    timings = response_json.get("timings")
    if not isinstance(timings, dict):
        return None
    for key in _REQUIRED_TIMING_KEYS:
        value = timings.get(key)
        if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
    predicted_n = int(timings["predicted_n"])
    # A single predicted token is normally an immediate EOS. llama.cpp reports
    # that zero-duration event as 1,000,000 tok/s, which is a sentinel-like timing
    # artifact rather than generation throughput. A benchmark needs at least two
    # predicted tokens to measure an interval, so keep this out of both the run
    # summary and the desktop ranking instead of presenting a bogus winner.
    if predicted_n <= 1:
        return None
    return CompletionTimings(
        prompt_n=int(timings["prompt_n"]),
        prompt_per_second=float(timings["prompt_per_second"]),
        predicted_n=predicted_n,
        predicted_per_second=float(timings["predicted_per_second"]),
        # predicted_ms is contextual, not required; default 0.0 when absent.
        predicted_ms=float(timings.get("predicted_ms") or 0.0),
    )


# --------------------------------------------------------------------------- #
# 2. Deterministic scoring (M5.3, keyword benchmark_score)
# --------------------------------------------------------------------------- #

# Match a signed integer or decimal anywhere in the text; the LAST match is taken
# as the model's final answer (models often show working then the answer).
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _normalize(text: str) -> str:
    """Trim, collapse internal whitespace, and casefold for tolerant comparison."""
    return " ".join(text.split()).casefold()


def extract_final_number(text: str) -> float | None:
    """Return the last number appearing in `text`, or None if there is none."""
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:  # pragma: no cover - regex guarantees a numeric token
        return None


def check_exact_match(response_text: str, expected: str) -> bool:
    """True if the normalized response CONTAINS the normalized expected string.

    Containment (not full equality) because instruct models wrap a one-word answer
    in a sentence ("The capital of France is Paris.").
    """
    return _normalize(str(expected)) in _normalize(response_text)


def check_keyword(response_text: str, expected: dict[str, Any]) -> bool:
    """True if the response contains all required and none of the forbidden strings.

    Comparison is case-insensitive but NOT whitespace-collapsed, so a code token
    like "[::-1]" is matched verbatim.
    """
    lowered = response_text.casefold()
    required = expected.get("all", [])
    forbidden = expected.get("none", [])
    if not all(str(term).casefold() in lowered for term in required):
        return False
    if any(str(term).casefold() in lowered for term in forbidden):
        return False
    return True


def check_numeric(response_text: str, expected: dict[str, Any]) -> bool:
    """True if the final number in the response equals the expected value +/- tol."""
    value = extract_final_number(response_text)
    if value is None:
        return False
    target = float(expected["value"])
    tol = float(expected.get("tol", 0))
    return abs(value - target) <= tol


_CHECKERS: dict[str, Callable[[str, Any], bool]] = {
    "exact_match": check_exact_match,
    "keyword": check_keyword,
    "numeric": check_numeric,
}


def run_check(task: BenchmarkTask, response_text: str) -> bool:
    """Evaluate one task's deterministic checker against a model response."""
    checker = _CHECKERS.get(task.check)
    if checker is None:  # pragma: no cover - guarded by the fixed task set
        raise ValueError(f"unknown checker '{task.check}' for task '{task.id}'")
    return checker(response_text, task.expected)


@dataclass
class ScoreBreakdown:
    """Per-category "N of M" counts plus percentages and the overall mean (M5.3)."""

    quality_passed: int = 0
    quality_total: int = 0
    reasoning_passed: int = 0
    reasoning_total: int = 0
    coding_passed: int = 0
    coding_total: int = 0

    def _pct(self, passed: int, total: int) -> float:
        return round(100.0 * passed / total, 1) if total else 0.0

    @property
    def quality_score(self) -> float:
        return self._pct(self.quality_passed, self.quality_total)

    @property
    def reasoning_score(self) -> float:
        return self._pct(self.reasoning_passed, self.reasoning_total)

    @property
    def coding_score(self) -> float:
        return self._pct(self.coding_passed, self.coding_total)

    @property
    def overall_score(self) -> float:
        """Documented simple equal-weight mean of the three category pass-rates.

        Equal weight is a deliberate, stated choice (M5.3): no category is
        privileged, so the number is reproducible from the three visible
        percentages.
        """
        return round(
            (self.quality_score + self.reasoning_score + self.coding_score) / 3.0, 1
        )


def score_responses(
    responses: dict[str, str], tasks: tuple[BenchmarkTask, ...] = TASKS
) -> ScoreBreakdown:
    """Aggregate deterministic checks over the task set into a ScoreBreakdown.

    `responses` maps task id -> the model's returned text. A task with no response
    (the model was not asked, or the request failed) counts as a fail, never a
    fabricated pass. The "N of M" counts are the honest record of how many
    deterministic checks the model's outputs satisfied.
    """
    breakdown = ScoreBreakdown()
    for task in tasks:
        passed = run_check(task, responses.get(task.id, ""))
        if task.category == "quality":
            breakdown.quality_total += 1
            breakdown.quality_passed += int(passed)
        elif task.category == "reasoning":
            breakdown.reasoning_total += 1
            breakdown.reasoning_passed += int(passed)
        elif task.category == "coding":
            breakdown.coding_total += 1
            breakdown.coding_passed += int(passed)
    return breakdown


# --------------------------------------------------------------------------- #
# 3. Scenarios and sweep plan (M5.4, keyword benchmark_sweep)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BenchmarkScenario:
    """One fully-resolved config to benchmark. Every field travels into the row.

    server_args is stored as a tuple (hashable, so the dataclass is frozen/hashable
    and can key a set); it is converted to a list when passed to build_start_spec.
    """

    model_id: str
    gpu_layers: int
    context_size: int
    server_args: tuple[str, ...] = ()
    draft_model: str | None = None
    spec_config_json: str | None = None  # canonical JSON of spec_config, or None

    @property
    def spec_config(self) -> dict[str, Any] | None:
        """The spec_config mapping, decoded from its canonical JSON (or None)."""
        return json.loads(self.spec_config_json) if self.spec_config_json else None

    @property
    def key(self) -> str:
        """Stable scenario key: model_id + a short hash of the full config.

        The hash covers every config field, so no two distinct configs collide and
        a resume can reliably tell whether an exact scenario already ran (M5.8).
        """
        canonical = json.dumps(
            {
                "gpu_layers": self.gpu_layers,
                "context_size": self.context_size,
                "server_args": list(self.server_args),
                "draft_model": self.draft_model,
                "spec_config": self.spec_config,
            },
            sort_keys=True,
        )
        digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]
        return f"{self.model_id}:gpu{self.gpu_layers}:ctx{self.context_size}:{digest}"

    def config_stamp(self) -> str:
        """Human config summary for the markdown row (owner's 'config always shown')."""
        parts = [f"gpu{self.gpu_layers}", f"ctx{self.context_size}"]
        if self.server_args:
            parts.append(" ".join(self.server_args))
        spec = self.spec_config
        if self.draft_model:
            parts.append(f"draft={self.draft_model}")
        if spec:
            parts.append("spec=" + spec.get("spec_type", "on"))
        return " ".join(parts)


def _spec_to_json(spec: dict[str, Any] | None) -> str | None:
    """Canonical JSON for a spec_config mapping (sorted keys), or None."""
    return json.dumps(spec, sort_keys=True) if spec else None


class SweepPlan:
    """Generates the scenario list for a model (single-config or full sweep)."""

    @staticmethod
    def generate(model: Any, axes: dict[str, Any] | None) -> list[BenchmarkScenario]:
        """Cartesian product of declared axes, or the single current-config scenario.

        `model` is a config.Model. `axes` is the model's benchmark_sweep mapping (or
        the CLI-supplied equivalent), with optional keys gpu_layers (list[int]),
        context_size (list[int]), server_args (list[list[str]]), and spec (list of
        spec_config mappings, each of which may be None for 'speculation off'). When
        `axes` is falsy or declares nothing, the plan degenerates to exactly one
        scenario using the model's current registry config (M5.4). Pure function:
        no I/O, no GPU.
        """
        axes = axes or {}
        gpu_opts = _as_list(axes.get("gpu_layers"), model.gpu_layers)
        ctx_opts = _as_list(axes.get("context_size"), model.context_size)
        args_opts = (
            [list(v) for v in axes["server_args"]]
            if axes.get("server_args")
            else [list(model.server_args)]
        )
        # Each spec variant resolves to a (draft_model, spec_config) pair. A None
        # variant means speculation off; a "draft*" spec_type pulls in the model's
        # draft_model; an ngram spec_type needs no draft file.
        if axes.get("spec") is not None:
            spec_variants = list(axes["spec"])
        else:
            spec_variants = [model.spec_config]

        scenarios: list[BenchmarkScenario] = []
        seen: set[str] = set()
        for gpu in gpu_opts:
            for ctx in ctx_opts:
                for args in args_opts:
                    for variant in spec_variants:
                        draft, spec = _resolve_spec_variant(model, variant)
                        scenario = BenchmarkScenario(
                            model_id=model.id,
                            gpu_layers=int(gpu),
                            context_size=int(ctx),
                            server_args=tuple(str(a) for a in args),
                            draft_model=draft,
                            spec_config_json=_spec_to_json(spec),
                        )
                        # De-dup identical scenarios (e.g. two axes collapsing to the
                        # same config) so a row is never benchmarked twice in one plan.
                        if scenario.key not in seen:
                            seen.add(scenario.key)
                            scenarios.append(scenario)
        return scenarios


def _as_list(value: Any, fallback: Any) -> list[Any]:
    """Return value as a non-empty list, else [fallback] (sweep axis helper)."""
    if isinstance(value, list) and value:
        return value
    return [fallback]


def _resolve_spec_variant(
    model: Any, variant: dict[str, Any] | None
) -> tuple[str | None, dict[str, Any] | None]:
    """Map a sweep spec variant to (draft_model, spec_config).

    None -> speculation off. A variant whose spec_type starts with "draft" uses the
    model's draft_model field (the draft gguf); an ngram variant needs no draft.
    """
    if not variant:
        return None, None
    spec_type = str(variant.get("spec_type", ""))
    draft = model.draft_model if spec_type.startswith("draft") else None
    return draft, variant


# --------------------------------------------------------------------------- #
# 4. Result record (Data Model section 8.1) and formatting (keyword benchmark_results)
# --------------------------------------------------------------------------- #


@dataclass
class BenchmarkResult:
    """One scenario run's full record (Data Model section 8.1).

    Superset of the M1 section-4.1 fields. When `ok` is False (a start/timing
    failure or a skipped-with-remedy scenario), every speed/score field stays None
    and `reason` explains why - never a fabricated number.
    """

    model_id: str
    session_id: str
    scenario_key: str
    timestamp: str
    ok: bool
    reason: str
    # Config stamp (always present, even for a failed row).
    gpu_layers: int
    context_size: int
    server_args: list[str] = field(default_factory=list)
    draft_model: str | None = None
    spec_config: dict[str, Any] | None = None
    # Model-file attribution stamp (F-2): the exact file/quant this row measured, so
    # a later staleness gate (verify_27b_target) can refuse a row whose model was
    # since repointed at a different file or quant. None on legacy rows.
    model_file: str | None = None
    quantization: str | None = None
    # Measurement (None unless ok).
    runs: int = 0
    prompt_per_second: float | None = None
    prompt_ps_min: float | None = None
    prompt_ps_max: float | None = None
    predicted_per_second: float | None = None
    predicted_ps_min: float | None = None
    predicted_ps_max: float | None = None
    prompt_n: int | None = None
    predicted_n: int | None = None
    # Deterministic "N of M" scores (None unless ok).
    quality_passed: int | None = None
    quality_total: int | None = None
    reasoning_passed: int | None = None
    reasoning_total: int | None = None
    coding_passed: int | None = None
    coding_total: int | None = None
    quality_score: float | None = None
    reasoning_score: float | None = None
    coding_score: float | None = None
    overall_score: float | None = None
    # Hardware snapshot (None when nvidia-smi absent - honest, never fabricated).
    gpu_name: str | None = None
    vram_total_mb: float | None = None
    vram_used_peak_mb: float | None = None
    ram_used_mb: float | None = None
    notes: str = ""

    def config_stamp(self) -> str:
        """Human config summary reused by the markdown row."""
        parts = [f"gpu{self.gpu_layers}", f"ctx{self.context_size}"]
        if self.server_args:
            parts.append(" ".join(self.server_args))
        if self.draft_model:
            parts.append(f"draft={self.draft_model}")
        if self.spec_config:
            parts.append("spec=" + str(self.spec_config.get("spec_type", "on")))
        return " ".join(parts)


def result_record(result: BenchmarkResult) -> dict[str, Any]:
    """Serialize a BenchmarkResult to the JSONL object shape (Data Model 8.2).

    Includes both the M5 field names and the section-4.1 aliases (prompt_speed ==
    prompt_per_second, generation_speed == predicted_per_second, memory_mb ==
    ram_used_mb, vram_mb == vram_used_peak_mb) so a reader of either vocabulary
    finds what they expect.
    """
    return {
        "model_id": result.model_id,
        "session_id": result.session_id,
        "scenario_key": result.scenario_key,
        "ok": result.ok,
        "reason": result.reason,
        "timestamp": result.timestamp,
        "gpu_layers": result.gpu_layers,
        "context_size": result.context_size,
        "server_args": list(result.server_args),
        "draft_model": result.draft_model,
        "spec_config": result.spec_config,
        # F-2 attribution stamp: the model file/quant actually measured.
        "model_file": result.model_file,
        "quantization": result.quantization,
        "runs": result.runs,
        "prompt_per_second": result.prompt_per_second,
        "prompt_ps_min": result.prompt_ps_min,
        "prompt_ps_max": result.prompt_ps_max,
        "predicted_per_second": result.predicted_per_second,
        "predicted_ps_min": result.predicted_ps_min,
        "predicted_ps_max": result.predicted_ps_max,
        "prompt_n": result.prompt_n,
        "predicted_n": result.predicted_n,
        "quality_passed": result.quality_passed,
        "quality_total": result.quality_total,
        "reasoning_passed": result.reasoning_passed,
        "reasoning_total": result.reasoning_total,
        "coding_passed": result.coding_passed,
        "coding_total": result.coding_total,
        "quality_score": result.quality_score,
        "reasoning_score": result.reasoning_score,
        "coding_score": result.coding_score,
        "overall_score": result.overall_score,
        "gpu_name": result.gpu_name,
        "vram_total_mb": result.vram_total_mb,
        "vram_used_peak_mb": result.vram_used_peak_mb,
        "ram_used_mb": result.ram_used_mb,
        # Section 4.1 aliases (p50 headline values).
        "prompt_speed": result.prompt_per_second,
        "generation_speed": result.predicted_per_second,
        "memory_mb": result.ram_used_mb,
        "vram_mb": result.vram_used_peak_mb,
        "notes": result.notes,
    }


def format_jsonl_line(result: BenchmarkResult) -> str:
    """One compact ASCII JSON line for docs/benchmark_results.jsonl (Data Model 8.2)."""
    return json.dumps(result_record(result), ensure_ascii=True)


def _n_of_m(score: float | None, passed: int | None, total: int | None) -> str:
    """Render 'PP.P (n/m)' for a category cell, or '-' when the row failed."""
    if score is None or total is None:
        return "-"
    return f"{score:.1f} ({passed}/{total})"


def format_markdown_row(result: BenchmarkResult) -> str:
    """One human table row for docs/benchmark_results.md (Data Model 8.3).

    Includes the config stamp so no visible row lacks its config. A failed row
    (ok False) shows the honest reason in the speed columns rather than a number.
    """
    name = result.model_id
    config = result.config_stamp()
    if not result.ok:
        gen = f"FAILED: {result.reason}"[:60]
        prompt = "-"
        ram = vram = "-"
    else:
        prompt = f"{result.prompt_per_second:.1f}"
        gen = (
            f"{result.predicted_per_second:.1f} "
            f"({result.predicted_ps_min:.1f}-{result.predicted_ps_max:.1f})"
        )
        ram = "-" if result.ram_used_mb is None else f"{result.ram_used_mb:.0f}"
        vram = (
            "-" if result.vram_used_peak_mb is None else f"{result.vram_used_peak_mb:.0f}"
        )
    quality = _n_of_m(result.quality_score, result.quality_passed, result.quality_total)
    reasoning = _n_of_m(
        result.reasoning_score, result.reasoning_passed, result.reasoning_total
    )
    coding = _n_of_m(result.coding_score, result.coding_passed, result.coding_total)
    overall = "-" if result.overall_score is None else f"{result.overall_score:.1f}"
    return (
        f"| {name} | {config} | {prompt} | {gen} | {ram} | {vram} | "
        f"{quality} | {reasoning} | {coding} | {overall} |"
    )


# The human-file header, stating the honest meaning of the scores (M5.3/M5.7).
def resolve_results_dir(settings: Any) -> Path:
    """Return the directory benchmark results and reports are written under.

    <data root>/reports, never <install>/docs (DEC-M14-9). A benchmark result is
    a measurement of THIS user's machine: they cannot get it back without
    re-running an hour of benchmarks, so it is user data and lives in the folder
    they back up. The install tree's docs/ ships documentation and is read-only
    at runtime (invariant W1).

    One function, so the runner and the two scripts/verify_*.py helpers that
    read those files can never disagree about where they are.
    """
    return Path(settings.data_dir) / "reports"


RESULTS_MD_HEADER = (
    "# LOCITIZE Benchmark Results\n\n"
    "Scores are the fraction of deterministic local checks passed (exact-match /\n"
    "keyword / numeric), NOT an LLM-judged rating. Speeds are from the server's own\n"
    "/completion timings (p50 of N runs).\n"
)

_MD_TABLE_HEADER = (
    "| Model | Config | Prompt tok/s | Gen tok/s (p50, min-max) | RAM MB | "
    "VRAM MB | Quality | Reasoning | Coding | Overall |\n"
    "|-------|--------|-------------:|-------------------------:|-------:|"
    "--------:|--------:|----------:|-------:|--------:|"
)


def render_markdown_section(session_id: str, gpu_name: str | None, results: list[BenchmarkResult]) -> str:
    """Render one appendable run-set section (heading + table) for the .md file."""
    hw = gpu_name or "GPU unknown"
    lines = [f"\n## Run {session_id} ({hw})\n", _MD_TABLE_HEADER]
    for result in results:
        lines.append(format_markdown_row(result))
    return "\n".join(lines) + "\n"


def latest_generation_speeds(
    jsonl_path: Path | str, model_rows: list[dict[str, Any]]
) -> dict[str, float]:
    """Return each model's latest valid generation throughput in tok/s.

    The desktop's benchmark column is a speed surface, so its source must be the
    measured ``predicted_per_second`` field in the append-only benchmark record --
    never ``benchmark_score`` (which is a deterministic quality percentage).

    A historical row is accepted only when its model file still matches the visible
    model row. The column is explicitly labelled *Last gen tok/s*: GPU split and
    context can legitimately change after a run, but the last real measurement is
    still useful benchmark history. Repointing an id to a different GGUF invalidates
    the value. Malformed/failed/non-finite rows are ignored, and a missing report
    yields an empty mapping.
    """
    targets = {
        str(row.get("id", "")): row
        for row in model_rows
        if str(row.get("id", ""))
    }
    if not targets:
        return {}

    latest: dict[str, float] = {}
    try:
        lines = Path(jsonl_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return latest

    for line in lines:
        try:
            record = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or record.get("ok") is not True:
            continue
        model_id = str(record.get("model_id", ""))
        target = targets.get(model_id)
        if target is None or not _benchmark_record_matches_model_file(record, target):
            continue
        raw_predicted_n = record.get("predicted_n")
        if raw_predicted_n is not None:
            if isinstance(raw_predicted_n, bool):
                continue
            try:
                if int(raw_predicted_n) <= 1:
                    continue
            except (TypeError, ValueError):
                continue
        raw_speed = record.get("predicted_per_second", record.get("generation_speed"))
        if isinstance(raw_speed, bool):
            continue
        try:
            speed = float(raw_speed)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(speed) or speed < 0:
            continue
        latest[model_id] = speed
    return latest


def _benchmark_record_matches_model_file(
    record: dict[str, Any], model_row: dict[str, Any]
) -> bool:
    """Whether a saved result still describes the row's current GGUF file."""
    saved_file = str(record.get("model_file") or "").strip()
    current_file = str(model_row.get("location") or "").strip()
    if saved_file and current_file:
        if _normalized_model_path(saved_file) != _normalized_model_path(current_file):
            return False
    return True


def _normalized_model_path(value: str) -> str:
    """Normalize slash/case differences without requiring the file to exist."""
    return os.path.normcase(os.path.normpath(value.replace("/", os.sep)))


# --------------------------------------------------------------------------- #
# 5. Resume / skip-set (M5.8, keyword benchmark_resume)
# --------------------------------------------------------------------------- #


def completed_scenario_keys(jsonl_path: Path | str) -> set[str]:
    """Read the set of successfully-completed scenario keys from the JSONL file.

    A pure read of the append-only record: a scenario is "completed" only when a
    line records that key with ok True (a failed row is left for a retry). A
    missing/empty file yields an empty set. Malformed lines are skipped rather than
    crashing the resume (the record is append-only and could be mid-write).
    """
    path = Path(jsonl_path)
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("ok") and record.get("scenario_key"):
            done.add(str(record["scenario_key"]))
    return done


def remaining_scenarios(
    scenarios: list[BenchmarkScenario], jsonl_path: Path | str
) -> list[BenchmarkScenario]:
    """Return the scenarios not yet successfully completed in the JSONL (M5.8)."""
    done = completed_scenario_keys(jsonl_path)
    return [s for s in scenarios if s.key not in done]


# --------------------------------------------------------------------------- #
# 6. Variance helpers (M5.2 step 4)
# --------------------------------------------------------------------------- #


def summarize(values: list[float]) -> tuple[float, float, float]:
    """Return (p50 median, min, max) of a non-empty list, each rounded to 1 dp."""
    return (
        round(statistics.median(values), 1),
        round(min(values), 1),
        round(max(values), 1),
    )


# --------------------------------------------------------------------------- #
# 7. Run lock (M5.8) - an exclusive marker so two runs cannot overlap
# --------------------------------------------------------------------------- #


class RunLock:
    """A best-effort exclusive lock file created atomically (O_CREAT | O_EXCL).

    Prevents two benchmark sessions from overlapping (which would try to co-load
    two models). A stale lock is reported, not silently stolen (M5.8). Used as a
    context manager so the lock is always released.
    """

    def __init__(
        self,
        lock_path: Path | str,
        on_reclaim: "Callable[[str], None] | None" = None,
    ) -> None:
        self._path = Path(lock_path)
        self._acquired = False
        self._on_reclaim = on_reclaim

    def _holder_alive(self) -> bool:
        """True unless the recorded holder PID verifiably no longer exists.

        Unreadable or garbage lock contents answer True: when we cannot tell,
        the lock stays respected, because stealing a live run's lock co-loads
        two multi-GB models and that is the failure M5.8 exists to prevent.
        """
        try:
            pid = int(self._path.read_text().strip())
        except (OSError, ValueError):
            return True
        try:
            import psutil

            return psutil.pid_exists(pid)
        except Exception:  # noqa: BLE001 - no liveness signal -> respect the lock
            return True

    def __enter__(self) -> "RunLock":
        for attempt in (1, 2):
            try:
                fd = os.open(str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError as exc:
                # M15.4: a run killed mid-flight (timeout, task manager, power)
                # leaves its lock behind, and before this check every later
                # benchmark was blocked until a human deleted the file - it
                # happened twice in one working session. The lock is reclaimed
                # ONLY when its recorded holder PID verifiably no longer exists,
                # and the reclaim is announced, so this is still "reported, not
                # silently stolen" - just no longer a manual chore.
                if attempt == 1 and not self._holder_alive():
                    if self._on_reclaim is not None:
                        self._on_reclaim(
                            f"reclaiming stale benchmark lock at {self._path} "
                            f"(its holder process is gone)"
                        )
                    try:
                        self._path.unlink()
                    except OSError:
                        pass
                    continue
                raise BenchmarkConflictError(
                    f"a benchmark lock already exists at {self._path}; another run "
                    f"may be active. If none is, delete the file and retry."
                ) from exc
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
        self._acquired = True
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._acquired:
            try:
                self._path.unlink()
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# 8. The runner (M5.2/M5.8) - reuses the injected ModelController exclusively
# --------------------------------------------------------------------------- #

# The completion request knobs are fixed for comparable, deterministic runs
# (M5.2 step 3 / M5.3): temperature 0, a fixed seed, cache_prompt off so every run
# re-does prefill (a cold, comparable prompt measurement).
_N_PREDICT = 256
_SEED = 42
_TIMED_PROMPT = "Write a short paragraph about the ocean."


class BenchmarkRunner:
    """Drives one benchmark session over a set of models/scenarios.

    Collaborators are injected so the whole thing is headless-testable (M5.9):
    - `controller`  : the existing ModelController (start/stop/switch, no orphans).
    - `registry`    : the ModelRegistry (build_start_spec, model lookups).
    - `http_post`   : POST json to a loopback URL -> response dict (tests fake it).
    - `port_is_free`: real-bind check for the conflict guard (tests fake it).
    - `gpu_provider`/`sys_provider`: the health.py providers for VRAM/RAM sampling.
    - `clock`/`sleep`: injectable so tests do not really wait.
    The runner itself contains NO process spawn/kill code (M5.1).
    """

    def __init__(
        self,
        controller: Any,
        registry: Any,
        settings: Any,
        *,
        http_post: Callable[[str, dict[str, Any], float], dict[str, Any]] | None = None,
        port_is_free: Callable[[int], bool] | None = None,
        gpu_provider: Any = None,
        sys_provider: Any = None,
        results_dir: Path | str | None = None,
        lock_dir: Path | str | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        out: Callable[[str], None] = print,
    ) -> None:
        self._controller = controller
        self._registry = registry
        self._settings = settings
        self._http_post = http_post or _default_http_post
        self._port_is_free = port_is_free or _default_port_is_free
        self._gpu = gpu_provider
        self._sys = sys_provider
        self._results_dir = (
            Path(results_dir) if results_dir else resolve_results_dir(settings)
        )
        # The lock lives in the resolved log dir (honors LOCITIZE_LOG_DIR) so a test
        # never drops a lock into the tracked tree.
        from logger import resolve_log_dir

        self._lock_dir = Path(lock_dir) if lock_dir else resolve_log_dir(settings)
        self._clock = clock
        self._sleep = sleep
        self._out = out

    # ---- guards ----------------------------------------------------------- #

    def assert_can_run(self) -> None:
        """Refuse (BenchmarkConflictError) if the llama.cpp port is already in use.

        M5.8 port/owner conflict policy: if a GUI or terminal session is already
        serving a model on the reserved port, benchmarking would try to allocate a
        second port and co-load a second multi-GB model, OOMing the card. So we
        refuse with a remedy and start NOTHING.
        """
        port = self._settings.ports.llama_cpp
        if not self._port_is_free(port):
            raise BenchmarkConflictError(
                f"a model appears to be running on port {port}; stop it or close "
                f"LOCITIZE before benchmarking (the benchmark will not co-load a second "
                f"model on the 16GB card)."
            )
        self._assert_headroom()

    # Guard thresholds (M15.3). A benchmark run on a machine already out of
    # memory measures the machine's distress, not the model's speed, and then
    # records that number as if it were the model's. This session's own history
    # proved it: the same model+config measured 16.6 and 214.3 tok/s ten minutes
    # apart, the slow row taken while system RAM sat at 98%. These checks refuse
    # (RAM) or warn (VRAM) BEFORE a model is loaded, so a compromised number is
    # either never produced or at least never produced silently.
    RAM_FLOOR_MB = 4096.0
    VRAM_OTHERS_WARN_MB = 1500.0

    def _assert_headroom(self) -> None:
        """Refuse on RAM starvation; warn on unusual VRAM held by other apps."""
        if self._sys is not None:
            avail = self._sys.ram_available_mb()
            if avail is not None and avail < self.RAM_FLOOR_MB:
                raise BenchmarkConflictError(
                    f"only {avail:.0f} MB of system RAM is free (< "
                    f"{self.RAM_FLOOR_MB:.0f} MB floor); a run now would measure "
                    f"paging, not the model. Close applications and retry."
                )
        if self._gpu is not None:
            gpus = self._gpu.gpus()
            if gpus:
                used_by_others = gpus[0].vram_total_mb - gpus[0].vram_free_mb
                if used_by_others > self.VRAM_OTHERS_WARN_MB:
                    self._out(
                        f"  WARNING: {used_by_others:.0f} MB of VRAM is already in "
                        f"use by other applications; results may understate this "
                        f"model's speed."
                    )

    # ---- one throughput probe (M15.3) ------------------------------------- #

    def probe_context_throughput(
        self,
        model_id: str,
        context_size: int,
        server_args: list[str] | None = None,
        gpu_layers: int | None = None,
    ) -> float | None:
        """One timed completion at an explicit context. Returns tok/s, or None.

        This is the measurement the context auto-tune lacked: a context that
        LOADS but has spilled out of VRAM answers a completion 10-15x slower,
        and only a real generation exposes that. One discarded warm-up, one
        timed run - a probe ranks configurations, it does not certify them, so
        the full 3-run benchmark remains the number of record.

        Never raises for a start failure or timeout; None means "not usable at
        this context", which for a ceiling search is an answer, not an error.
        The model is always torn down before returning (same guarantee as
        run_scenario).
        """
        model = self._registry.get(model_id)
        if model is None:
            return None
        try:
            status = self._controller.start(
                model_id,
                context_size,
                gpu_layers if gpu_layers is not None else model.gpu_layers,
                server_args=list(
                    server_args if server_args is not None else model.server_args
                ),
            )
        except ValueError:
            return None
        from services import ServiceStatus

        if status is not ServiceStatus.RUNNING:
            self._controller.stop()
            return None
        try:
            port = self._resolved_port(model_id)
            if port is None:
                return None
            url = f"http://127.0.0.1:{port}/completion"
            self._post_completion(url, _TIMED_PROMPT)  # warm-up, discarded
            timings = parse_completion_timings(
                self._post_completion(url, _TIMED_PROMPT) or {}
            )
            return timings.predicted_per_second if timings else None
        finally:
            self._controller.stop()

    # ---- one scenario ----------------------------------------------------- #

    def run_scenario(
        self, scenario: BenchmarkScenario, session_id: str, runs: int
    ) -> BenchmarkResult:
        """Benchmark one fully-resolved scenario. Real path (needs a live server).

        Records an honest failed row (ok False, non-empty reason, null metrics) for
        any start failure or skip-with-remedy, and never fabricates a number.
        """
        model = self._registry.get(scenario.model_id)
        model_name = model.name if model is not None else scenario.model_id
        base = self._new_result(scenario, session_id)

        # Start via the existing controller (no new spawn code). A ValueError means
        # a config guard fired (missing location / draft) -> skipped-with-remedy.
        try:
            status = self._controller.start(
                scenario.model_id,
                scenario.context_size,
                scenario.gpu_layers,
                server_args=list(scenario.server_args),
                draft_model=scenario.draft_model,
                spec_config=scenario.spec_config,
            )
        except ValueError as exc:
            return replace(base, ok=False, reason=f"skipped: {exc}")

        from services import ServiceStatus

        if status is not ServiceStatus.RUNNING:
            self._controller.stop()
            return replace(
                base,
                ok=False,
                reason=f"model did not start (status={getattr(status, 'value', status)})",
            )

        try:
            port = self._resolved_port(scenario.model_id)
            if port is None:
                return replace(base, ok=False, reason="could not resolve server port")
            return self._measure(base, scenario, model_name, port, runs)
        finally:
            # Always tear the model down before the next scenario (no orphan, no
            # co-load). stop() confirms the child is gone (D-M4-1).
            self._controller.stop()

    def _measure(
        self,
        base: BenchmarkResult,
        scenario: BenchmarkScenario,
        model_name: str,
        port: int,
        runs: int,
    ) -> BenchmarkResult:
        """Warm-up + timed /completion runs + scoring + hardware sampling."""
        url = f"http://127.0.0.1:{port}/completion"

        # Warm-up (discarded): pays the first-token / graph-build cost once so it
        # does not skew the measured runs (M5.2 step 2).
        self._out(f"  warm-up ({model_name})...")
        self._post_completion(url, _TIMED_PROMPT)

        prompt_ps: list[float] = []
        predicted_ps: list[float] = []
        last_prompt_n = last_predicted_n = 0
        vram_peak = ram_peak = None
        gpu_name = vram_total = None
        for i in range(max(1, runs)):
            self._out(f"  run {i + 1}/{max(1, runs)} ({model_name})...")
            response = self._post_completion(url, _TIMED_PROMPT)
            timings = parse_completion_timings(response) if response is not None else None
            if timings is None:
                # Honest: a run with no timings is not counted; if none succeed the
                # scenario is a failed row.
                continue
            prompt_ps.append(timings.prompt_per_second)
            predicted_ps.append(timings.predicted_per_second)
            last_prompt_n = timings.prompt_n
            last_predicted_n = timings.predicted_n
            gpu_name, vram_total, vram_used, ram_used = self._sample_hardware()
            vram_peak = _peak(vram_peak, vram_used)
            ram_peak = _peak(ram_peak, ram_used)

        if not predicted_ps:
            return replace(
                base,
                ok=False,
                reason="no /completion run returned usable multi-token server timings",
            )

        # Deterministic scoring: one /completion per task, checked machine-side.
        responses: dict[str, str] = {}
        for task in TASKS:
            resp = self._post_completion(url, task.prompt)
            responses[task.id] = _content_of(resp)
        breakdown = score_responses(responses)

        prompt_p50, prompt_min, prompt_max = summarize(prompt_ps)
        gen_p50, gen_min, gen_max = summarize(predicted_ps)
        return replace(
            base,
            ok=True,
            reason="",
            runs=len(predicted_ps),
            prompt_per_second=prompt_p50,
            prompt_ps_min=prompt_min,
            prompt_ps_max=prompt_max,
            predicted_per_second=gen_p50,
            predicted_ps_min=gen_min,
            predicted_ps_max=gen_max,
            prompt_n=last_prompt_n,
            predicted_n=last_predicted_n,
            quality_passed=breakdown.quality_passed,
            quality_total=breakdown.quality_total,
            reasoning_passed=breakdown.reasoning_passed,
            reasoning_total=breakdown.reasoning_total,
            coding_passed=breakdown.coding_passed,
            coding_total=breakdown.coding_total,
            quality_score=breakdown.quality_score,
            reasoning_score=breakdown.reasoning_score,
            coding_score=breakdown.coding_score,
            overall_score=breakdown.overall_score,
            gpu_name=gpu_name,
            vram_total_mb=vram_total,
            vram_used_peak_mb=vram_peak,
            ram_used_mb=ram_peak,
            notes=scenario.config_stamp(),
        )

    # ---- whole session ---------------------------------------------------- #

    def run(
        self,
        model_ids: list[str],
        *,
        runs: int = 3,
        sweep: bool = False,
        resume: bool = False,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Run every scenario for the given models; append results; write back.

        Returns a summary dict (session_id, counts, per-scenario outcomes). Refuses
        (BenchmarkConflictError) if the port is busy or a run-lock is held, and
        never starts a model in that case (M5.8).
        """
        self.assert_can_run()
        session_id = session_id or time.strftime("%Y-%m-%dT%H%M")
        scenarios = self._plan(model_ids, sweep=sweep)
        jsonl_path = self._results_dir / "benchmark_results.jsonl"
        if resume:
            scenarios = remaining_scenarios(scenarios, jsonl_path)

        results: list[BenchmarkResult] = []
        gpu_name = None
        lock_path = self._lock_dir / "benchmark.lock"
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        with RunLock(lock_path, on_reclaim=self._out):
            for index, scenario in enumerate(scenarios, start=1):
                self._out(
                    f"[{index}/{len(scenarios)}] {scenario.model_id} - "
                    f"{scenario.config_stamp()}"
                )
                result = self.run_scenario(scenario, session_id, runs)
                self._append_jsonl(jsonl_path, result)
                results.append(result)
                gpu_name = gpu_name or result.gpu_name
                # Write back the score for the model's CURRENT-registry-config
                # scenario only (M5.7): the persisted number matches how the model
                # is actually configured to run.
                if result.ok and self._is_current_config(scenario):
                    self._write_back_score(result)

        self._append_markdown(session_id, gpu_name, results)
        return {
            "session_id": session_id,
            "scenarios": len(scenarios),
            "ok": sum(1 for r in results if r.ok),
            "failed": sum(1 for r in results if not r.ok),
            "results": [result_record(r) for r in results],
        }

    # ---- planning + helpers ---------------------------------------------- #

    def _plan(self, model_ids: list[str], sweep: bool) -> list[BenchmarkScenario]:
        """Build the scenario list across the requested models."""
        scenarios: list[BenchmarkScenario] = []
        for model_id in model_ids:
            model = self._registry.get(model_id)
            if model is None:
                self._out(f"  (skipping unknown model '{model_id}')")
                continue
            axes = model.benchmark_sweep if sweep else None
            scenarios.extend(SweepPlan.generate(model, axes))
        return scenarios

    def _is_current_config(self, scenario: BenchmarkScenario) -> bool:
        """True if a scenario matches the model's current registry config (M5.7)."""
        model = self._registry.get(scenario.model_id)
        if model is None:
            return False
        current = SweepPlan.generate(model, None)[0]
        return scenario.key == current.key

    def _write_back_score(self, result: BenchmarkResult) -> None:
        """Persist overall_score to models.yaml benchmark_score (M5.7)."""
        from config import RegistryWriteError, write_model_score

        if result.overall_score is None:
            return
        try:
            write_model_score(self._settings.data_dir, result.model_id, result.overall_score)
            self._out(
                f"  wrote benchmark_score {result.overall_score} for {result.model_id}"
            )
        # DEC-M14-11: config's chokepoint hands back one typed error whose text
        # is already a finished sentence, so this line frames it instead of
        # decorating an errno. OSError is not caught: it can no longer reach here.
        except (ValueError, RegistryWriteError) as exc:
            self._out(f"  (score write-back skipped for {result.model_id}: {exc})")

    def _new_result(self, scenario: BenchmarkScenario, session_id: str) -> BenchmarkResult:
        """A config-stamped, not-yet-measured result (mutated via replace())."""
        # Stamp the exact model file/quant this row will measure (F-2) so a later
        # staleness gate can reject a row whose registry entry has since changed.
        model = self._registry.get(scenario.model_id)
        return BenchmarkResult(
            model_id=scenario.model_id,
            session_id=session_id,
            scenario_key=scenario.key,
            timestamp=time.strftime("%Y-%m-%d %H:%M"),
            ok=False,
            reason="",
            gpu_layers=scenario.gpu_layers,
            context_size=scenario.context_size,
            server_args=list(scenario.server_args),
            draft_model=scenario.draft_model,
            spec_config=scenario.spec_config,
            model_file=model.location if model is not None else None,
            quantization=(model.quantization or None) if model is not None else None,
        )

    def _resolved_port(self, model_id: str) -> int | None:
        """Find the running service's resolved port via the controller snapshot."""
        snapshot = self._controller.snapshot()
        for svc in snapshot.get("services", []):
            if svc.get("model_id") == model_id and svc.get("port"):
                return int(svc["port"])
        return None

    def _post_completion(self, url: str, prompt: str) -> dict[str, Any] | None:
        """POST a fixed-knob /completion request; return the parsed dict or None."""
        payload = {
            "prompt": prompt,
            "n_predict": _N_PREDICT,
            "temperature": 0,
            "seed": _SEED,
            "cache_prompt": False,
        }
        try:
            return self._http_post(url, payload, 120.0)
        except OSError:
            return None

    def _sample_hardware(self) -> tuple[str | None, float | None, float | None, float | None]:
        """Sample (gpu_name, vram_total_mb, vram_used_mb, ram_used_mb), all measured.

        Absent nvidia-smi -> vram fields None (honest). RAM via the system provider.
        """
        gpu_name = vram_total = vram_used = None
        if self._gpu is not None:
            gpus = self._gpu.gpus()
            if gpus:
                gpu = gpus[0]
                gpu_name = gpu.name
                vram_total = gpu.vram_total_mb
                vram_used = gpu.vram_total_mb - gpu.vram_free_mb
        ram_used = None
        if self._sys is not None:
            ram_used = self._sys.ram_total_mb() - self._sys.ram_available_mb()
        return gpu_name, vram_total, vram_used, ram_used

    def _append_jsonl(self, path: Path, result: BenchmarkResult) -> None:
        """Append one JSON line (append-only machine record, Data Model 8.2)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(format_jsonl_line(result) + "\n")

    def _append_markdown(
        self, session_id: str, gpu_name: str | None, results: list[BenchmarkResult]
    ) -> None:
        """Append one run-set section to the human results file (append-only)."""
        if not results:
            return
        path = self._results_dir / "benchmark_results.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(RESULTS_MD_HEADER, encoding="utf-8")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(render_markdown_section(session_id, gpu_name, results))


def _content_of(response: dict[str, Any] | None) -> str:
    """The generated text of a /completion response (used ONLY for scoring, not speed)."""
    if not isinstance(response, dict):
        return ""
    return str(response.get("content", ""))


def _peak(current: float | None, sample: float | None) -> float | None:
    """Running max of two optional floats (peak VRAM/RAM tracking)."""
    if sample is None:
        return current
    if current is None:
        return sample
    return max(current, sample)


def _default_http_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Real loopback POST of a JSON body, returning the parsed JSON response.

    Loopback-only by construction (the URL is built from 127.0.0.1 + the resolved
    port; SEC-1). Imported lazily so importing this module stays cheap and side
    effect free.
    """
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as handle:  # noqa: S310 - loopback only
        return json.loads(handle.read().decode("utf-8"))


def _default_port_is_free(port: int) -> bool:
    """Real loopback-bind free check (same discipline as PortAllocator)."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False
