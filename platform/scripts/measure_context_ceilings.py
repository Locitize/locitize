"""Measured context-ceiling finder - OWNER RUN ONLY (M15.3).

Finds, for each registered model, the largest context window that holds real
generation throughput, by MEASURING it. This tool exists because two computed
approaches failed in one session: the load-only auto-tune chose contexts that
started fine and generated at 6% of baseline speed (the KV cache had spilled
out of VRAM), and a KV-arithmetic formula mispredicted in both directions
across architectures. The ceiling is an empirical fact about one model on one
machine; this measures it and writes down only what it measured.

Method, per model:
  1. Baseline: one probe at the model's CURRENT registry context and args.
  2. Ladder upward (q8_0 KV cache added, which halves cache size; measured
     within noise of fp16 on this machine): one probe per rung.
  3. A rung passes only if its throughput >= FLOOR x baseline. First failing
     rung ends the climb; the best passing rung is the model's ceiling.
  4. Rungs are capped at the model's native trained window - this tool never
     proposes YaRN extrapolation.

State is persisted after every probe to <data root>/reports/ctx_ceilings.json,
so a killed run resumes where it stopped. Nothing is written to models.yaml
unless --apply is passed, and then only through config.write_model_tuning with
a note that says the number was measured and when.

Identical-family dedup: models in the same GROUP share file size, architecture
and quant, so one representative is probed and its result applied to all.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM_DIR))

import config  # noqa: E402
import gguf_meta  # noqa: E402
from benchmark import BenchmarkRunner, RunLock  # noqa: E402
from health import DefaultSystemInfoProvider, NvidiaSmiGpuInfoProvider  # noqa: E402
from launcher import Launcher  # noqa: E402
from logger import resolve_log_dir  # noqa: E402

FLOOR = 0.85          # a rung must keep 85% of baseline throughput
LADDER = (12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072, 196608, 262144)
KV_ARGS = ("--cache-type-k", "q8_0", "--cache-type-v", "q8_0")

# One representative is probed per group; the result applies to every member.
# Fill this with YOUR registry's duplicate families (same weights re-registered
# under several ids - e.g. multiple snapshots or fine-tune versions of one
# base): {"representative-id": ("other-id", ...)}. Empty by default because
# duplicate groups are a property of a specific registry, not of LOCITIZE.
GROUPS: dict[str, tuple[str, ...]] = {}

# Never probed: verified this session, partial-offload compromise configs, or
# a model already at its native window.
# Fill with YOUR model ids to exclude: already-verified configs, deliberate
# partial-offload compromises, or models already at their native window.
# Empty by default for the same reason as GROUPS above.
SKIP: set[str] = set()


def probe_args(model) -> list[str]:
    """The model's args plus q8_0 KV (added once, never duplicated)."""
    args = [str(a) for a in (model.server_args or ["--parallel", "1"])]
    if "--cache-type-k" not in args:
        args.extend(KV_ARGS)
    return args


def state_rows_for_apply(state: dict, only: str | None):
    """Yield only state rows included by this invocation's model scope."""
    for model_id, row in state.items():
        if only is None or model_id == only:
            yield model_id, row


def context_candidates(native: int) -> tuple[int, ...]:
    """Return the fixed ladder plus the model's exact native endpoint."""
    return tuple(sorted({ctx for ctx in (*LADDER, native) if ctx > 0 and ctx <= native}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure_context_ceilings")
    parser.add_argument("--budget", type=float, default=480.0,
                        help="seconds of probing before a clean stop (resumable)")
    parser.add_argument("--apply", action="store_true",
                        help="write measured ceilings to models.yaml")
    parser.add_argument("--only", default=None, help="probe just this model id")
    args = parser.parse_args(argv)

    settings, models, _ = config.Config.load()
    launcher = Launcher()
    log_path = resolve_log_dir(settings) / "benchmark_llama.log"
    registry, manager, controller = launcher._build_controller(
        settings, models, log_path=log_path
    )
    import atexit

    atexit.register(manager.stop_all)
    gpu_provider = NvidiaSmiGpuInfoProvider()
    runner = BenchmarkRunner(
        controller, registry, settings,
        gpu_provider=gpu_provider,
        sys_provider=DefaultSystemInfoProvider(),
        out=lambda line: print(line, flush=True),
    )
    runner.assert_can_run()

    # Total VRAM on the working card, used to refuse models that cannot be
    # fully resident (see the partial-offload guard in the loop). 0 when
    # nvidia-smi is unavailable, which disables the guard rather than guessing.
    vram_total_mb = max(
        (g.vram_total_mb for g in gpu_provider.gpus() or []), default=0.0
    )

    state_path = config.resolve_data_dir() / "reports" / "ctx_ceilings.json"
    state: dict = json.loads(state_path.read_text()) if state_path.exists() else {}

    reps = [m for m in registry.launchable()
            if m.id not in SKIP
            and not any(m.id in members for members in GROUPS.values())]
    if args.only:
        reps = [m for m in reps if m.id == args.only]

    started = time.monotonic()
    lock_path = resolve_log_dir(settings) / "benchmark.lock"
    with RunLock(lock_path):
        for model in reps:
            row = state.setdefault(model.id, {})
            if row.get("done"):
                continue
            if time.monotonic() - started > args.budget:
                print(f"[budget reached - resumable; {sum(1 for m in reps if not state.get(m.id, {}).get('done'))} model(s) left]")
                break

            # Partial-offload guard (defect found 2026-09-01). The FLOOR is
            # RELATIVE to the model's own baseline, so a model whose weights do
            # not fit the card measures its ceiling against an already-spilled
            # baseline: holding 85% of 5.5 tok/s is trivial at any context, and
            # the ladder climbs to the native window. That is how
            # qwen3-8-27b-q4_k_m (17107MB on a 16303MB card) came away with the
            # highest context in the registry at the lowest throughput in it.
            # The number was true and the conclusion was useless, so refuse to
            # rank these rather than record a flattering ceiling. Weights alone
            # exceeding total VRAM is the unambiguous case; a model that fits
            # bare but spills once KV is added is still measured normally.
            need_mb = config.model_vram_need_mb(model)
            if vram_total_mb and need_mb > vram_total_mb:
                row["done"] = True
                row["verdict"] = (
                    f"partial-offload compromise: weights {need_mb:.0f}MB exceed "
                    f"{vram_total_mb:.0f}MB of VRAM, so any baseline is already "
                    f"spilled and a relative-floor ceiling is meaningless; left "
                    f"unchanged"
                )
                print(f"== {model.id}: {row['verdict']}", flush=True)
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps(state, indent=1))
                continue

            native = gguf_meta.read_gguf_header(model.location).context_length or 0
            if "baseline" not in row:
                print(f"== {model.id}: baseline at ctx {model.context_size}", flush=True)
                row["baseline"] = runner.probe_context_throughput(
                    model.id, model.context_size
                )
                row["baseline_ctx"] = model.context_size
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps(state, indent=1))
            base = row.get("baseline")
            if not base:
                row["done"] = True
                row["verdict"] = "baseline probe failed; left unchanged"
                state_path.write_text(json.dumps(state, indent=1))
                continue

            floor = base * FLOOR
            best = row.get("best", row["baseline_ctx"])
            for ctx in context_candidates(native):
                if ctx <= best or ctx > native:
                    continue
                if str(ctx) in row.get("probed", {}):
                    continue
                if time.monotonic() - started > args.budget:
                    break
                tokps = runner.probe_context_throughput(
                    model.id, ctx, server_args=probe_args(model)
                )
                row.setdefault("probed", {})[str(ctx)] = tokps
                print(f"   ctx {ctx:>7}: {tokps if tokps else 'FAILED':>8} tok/s "
                      f"(floor {floor:.1f})", flush=True)
                failed = tokps is None or tokps < floor
                if failed:
                    row["done"] = True
                else:
                    best = ctx
                    row["best"] = best
                # Persist only AFTER best/done are updated. Writing between the
                # probe and the best update loses a rung that verifiably passed:
                # the resume path skips any rung already in `probed`, so a kill
                # here followed by a failing next rung records the LOWER ceiling
                # and silently costs the owner a rung. Observed 2026-09-01 on
                # qwen3-8-27b-ud-q3_k_xl, whose 65536 had passed at 48.6 tok/s
                # (floor 41.7) while `best` still read 49152.
                state_path.write_text(json.dumps(state, indent=1))
                if failed:
                    break
            else:
                row["done"] = True
            if row.get("done"):
                row["verdict"] = (
                    f"ceiling {best} (baseline {base:.1f} tok/s at "
                    f"{row['baseline_ctx']})"
                )
                print(f"   -> {row['verdict']}", flush=True)
            state_path.write_text(json.dumps(state, indent=1))

    if args.apply:
        applied = 0
        for rep_id, row in state_rows_for_apply(state, args.only):
            if not row.get("done") or "best" not in row:
                continue
            best = int(row["best"])
            targets = [rep_id, *GROUPS.get(rep_id, ())]
            for tid in targets:
                model = registry.get(tid)
                if model is None or model.context_size >= best:
                    continue
                config.write_model_tuning(
                    settings.data_dir, tid, best, probe_args(model),
                    note=(f"ctx {model.context_size}->{best} measured by "
                          f"measure_context_ceilings (throughput floor "
                          f"{int(FLOOR * 100)}% of baseline)"
                          + (f"; probed via group representative {rep_id}"
                             if tid != rep_id else "")),
                )
                applied += 1
                print(f"APPLIED {tid}: ctx -> {best}")
        print(f"{applied} model(s) updated")

    remaining = sum(1 for m in reps if not state.get(m.id, {}).get("done"))
    print(f"\nstate: {state_path}  ({len(reps) - remaining}/{len(reps)} done)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
