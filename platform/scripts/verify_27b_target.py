"""AC15 CAMPAIGN GATE: is any CURRENT 27B config at or above 40 tok/s? (M5.6)

Reads <data root>/reports/benchmark_results.jsonl (DEC-M14-9), selects the successful 27B rows (model_id in
qwen3-6-27b), and exits 0 iff at least one row that is still
ATTRIBUTABLE to the model's current registry config reaches 40.0 tok/s generation.

F-2 rigor guard: a >=40 row does NOT pass forever. A row counts only when its
recorded model file/quant still equals the model's CURRENT models.yaml entry
(config-stamp equality on the fields that decide the number - the model file path
and its quantization) AND that model file still exists on disk. A row whose model
was since repointed at a different file or quant, or whose file is gone, is a stale
or mislabeled measurement and is refused (never a silent permanent pass). Among the
attributable rows the most recent one is chosen, and the gate prints exactly which
row satisfied or failed, with its timestamp and full config, plus why any candidate
was rejected - so an unmet or unattributable target is reported honestly.

This is deliberately its own criterion, separate from AC14's engine proof: the
owner's hard target may not be reachable on a 16GB card even with all three levers
(Architecture risk RB1; baseline 24.9 tok/s). A nonzero exit here does not, by
itself, disqualify the rest of Milestone 5 - it tells the owner whether to accept
with the target unmet, extend the campaign, or relax the number.

Run from Codebase/platform: `python scripts/verify_27b_target.py [--json]`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parent.parent

# Import the platform config loader so the gate reads the SAME models.yaml the
# engine runs from (the registry is the source of truth for the current config).
sys.path.insert(0, str(PLATFORM_DIR))
from benchmark import resolve_results_dir  # noqa: E402  (path insert first)
from config import Config  # noqa: E402  (path insert must precede the import)



def load_live_config() -> tuple:
    """Load the live settings/registry once, from the real data root.

    DEC-M14-9: benchmark results live under the user's data root, not under
    <install>/docs, and the path is resolved through the runner's own helper so
    this gate can never read a different file from the one the benchmark wrote.
    Called from main() rather than at import time, so importing this module has
    no side effects.
    """
    settings, models, _issues = Config.load()
    return settings, models, resolve_results_dir(settings) / "benchmark_results.jsonl"

TARGET_TOK_S = 40.0
# The 27B model ids the campaign is fought over.
TARGET_MODEL_IDS = {"qwen3-6-27b"}


def load_27b_rows(path: Path) -> list[dict]:
    """Return every successful 27B result row from the append-only JSONL."""
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            record.get("model_id") in TARGET_MODEL_IDS
            and record.get("ok")
            and record.get("predicted_per_second") is not None
        ):
            rows.append(record)
    return rows


def current_configs(models) -> dict[str, dict]:
    """Map each 27B model id to its current registry {location, quantization}.

    `models` is the live registry loaded by load_live_config(). A model id absent
    here (or with an empty location) has no current config, so any historical row
    for it is treated as unattributable.
    """
    current: dict[str, dict] = {}
    for model in models.models:
        if model.id in TARGET_MODEL_IDS:
            current[model.id] = {
                "location": model.location or "",
                "quantization": model.quantization or "",
            }
    return current


def attribution_of(row: dict, current: dict[str, dict]) -> tuple[bool, str]:
    """Is this row still attributable to the model's current registry config?

    Returns (ok, reason). The row must (1) name a model with a current registry
    entry, (2) record a model_file equal to that entry's location (config-stamp
    equality on the file path), (3) record a quantization equal to the entry's,
    and (4) point at a model file that still exists on disk. Any miss means the
    number belongs to a config the model no longer runs, so it must not count.
    """
    model_id = row.get("model_id")
    entry = current.get(model_id)
    if entry is None:
        return False, f"no current registry entry for '{model_id}'"

    row_file = row.get("model_file")
    if not row_file:
        # Legacy row written before F-2 stamping: unattributable by construction.
        return False, "row has no recorded model_file (pre-attribution row)"
    if row_file != entry["location"]:
        return False, (
            f"model_file changed since this row: row={row_file!r} "
            f"current={entry['location']!r}"
        )

    row_quant = row.get("quantization") or ""
    if row_quant != entry["quantization"]:
        return False, (
            f"quantization changed since this row: row={row_quant!r} "
            f"current={entry['quantization']!r}"
        )

    if not Path(row_file).exists():
        return False, f"recorded model file no longer exists on disk: {row_file}"

    return True, "current config match, file present"


def _config_of(row: dict) -> str:
    """Human config stamp for a row (gpu/ctx/server_args/spec/quant)."""
    parts = [f"gpu{row.get('gpu_layers')}", f"ctx{row.get('context_size')}"]
    if row.get("server_args"):
        parts.append(" ".join(row["server_args"]))
    if row.get("quantization"):
        parts.append(f"quant={row['quantization']}")
    if row.get("draft_model"):
        parts.append(f"draft={row['draft_model']}")
    if row.get("spec_config"):
        parts.append("spec=" + str(row["spec_config"].get("spec_type", "on")))
    return " ".join(parts)


def _timestamp(row: dict) -> str:
    """Sort key for recency: the row timestamp (ISO-like text sorts chronologically)."""
    return str(row.get("timestamp") or "")


def main(argv: list[str] | None = None) -> int:
    as_json = "--json" in (argv if argv is not None else sys.argv[1:])
    _settings, models, jsonl_path = load_live_config()
    all_rows = load_27b_rows(jsonl_path)
    current = current_configs(models)

    # Partition into attributable (current-config) rows and rejected ones, keeping
    # the rejection reasons so the gate can explain itself honestly.
    attributable: list[dict] = []
    rejected: list[dict] = []
    for row in all_rows:
        ok, reason = attribution_of(row, current)
        annotated = dict(row)
        annotated["_attribution_reason"] = reason
        (attributable if ok else rejected).append(annotated)

    # Prefer the most recent attributable row (F-2: newest current-config number).
    best = None
    if attributable:
        best = max(attributable, key=_timestamp)
    best_speed = best["predicted_per_second"] if best else None
    passed = best_speed is not None and best_speed >= TARGET_TOK_S

    report = {
        "ok": passed,
        "target_tok_s": TARGET_TOK_S,
        "measured_27b_rows": len(all_rows),
        "attributable_rows": len(attributable),
        "rejected_rows": [
            {
                "model_id": r.get("model_id"),
                "timestamp": r.get("timestamp"),
                "predicted_per_second": r.get("predicted_per_second"),
                "reason": r["_attribution_reason"],
            }
            for r in rejected
        ],
        "best_predicted_per_second": best_speed,
        "best_config": _config_of(best) if best else None,
        "best_model_id": best.get("model_id") if best else None,
        "best_timestamp": best.get("timestamp") if best else None,
        "results_file": str(jsonl_path),
    }
    if as_json:
        print(json.dumps(report))
    elif best is None:
        print(
            f"FAIL: no attributable (current-config) 27B benchmark row in {jsonl_path}.\n"
            f"Total 27B rows seen: {len(all_rows)}; attributable: 0."
        )
        for r in rejected:
            print(
                f"  rejected: {r.get('model_id')} @ {r.get('timestamp')} "
                f"({r.get('predicted_per_second')} tok/s) - {r['_attribution_reason']}"
            )
        print(
            "Run a 27B benchmark against the current registry config "
            "(launcher.py --benchmark --model qwen3-6-27b [--sweep]) before this gate "
            "can pass."
        )
    elif passed:
        print(
            f"PASS: {best['model_id']} reached {best_speed:.1f} tok/s "
            f">= {TARGET_TOK_S} tok/s at [{_config_of(best)}] "
            f"(session row {best.get('timestamp')}, current-config match)."
        )
    else:
        print(
            f"FAIL (honest): best attributable 27B row is {best['model_id']} at "
            f"{best_speed:.1f} tok/s, below the {TARGET_TOK_S} tok/s target.\n"
            f"Row: {best.get('timestamp')} [{_config_of(best)}].\n"
            f"The campaign target is unmet on this hardware; extend the sweep "
            f"(draft or ngram spec-decoding) or relax the target."
        )
        if rejected:
            print(f"({len(rejected)} stale/mislabeled row(s) refused:)")
            for r in rejected:
                print(
                    f"  rejected: {r.get('model_id')} @ {r.get('timestamp')} "
                    f"({r.get('predicted_per_second')} tok/s) - {r['_attribution_reason']}"
                )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
