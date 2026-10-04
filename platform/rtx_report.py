"""RTX compatibility report (M18.2): the measured per-GPU matrix, shareable.

LOCITIZE's core discipline is measured-not-assumed: every model's context
ceiling and tok/s is probed on the machine's real GPU. This module renders that
data - already sitting in the data root's reports - into one shareable document:

    Model X on a GeForce RTX 5070 Ti (16GB): 48 tok/s, context 65536, fits.

That is the answer every local-AI user wants BEFORE downloading 14GB, and in
aggregate it is a per-GPU compatibility matrix nobody else has, because nobody
else measures. Sharing is a deliberate ACT (the owner runs --rtx-report and
chooses what to do with the file); nothing here transmits anything anywhere -
the egress ledger stays honest.

It also answers the follow-on question - "what would a bigger card unlock?" -
by re-fitting each model's weights against the next VRAM tiers, so the report
ends with the concrete upgrade payoff instead of a vague "more is better".

Pure functions over injected data; the CLI wrapper in launcher.py does the file
I/O. ASCII only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MB = 1024 * 1024
# Consumer VRAM tiers worth speaking to (GB). The report names only tiers ABOVE
# the machine's own card.
VRAM_TIERS_GB = (8, 12, 16, 24, 32, 48)
# A model "fits" a card when its weights leave this much margin for KV + compute.
FIT_MARGIN_MB = 2048.0


@dataclass
class ModelRow:
    """One model's measured facts, as the report consumes them."""

    model_id: str
    name: str
    weights_mb: float
    context_size: int
    baseline_tok_s: float | None  # measured; None = not measured
    quantization: str = ""


def fits(weights_mb: float, vram_mb: float, margin_mb: float = FIT_MARGIN_MB) -> bool:
    """Weights plus a working margin inside the card = a genuinely usable fit."""
    if weights_mb <= 0 or vram_mb <= 0:
        return False
    return weights_mb + margin_mb <= vram_mb


def unlocked_at(rows: list[ModelRow], current_vram_mb: float) -> dict[int, list[ModelRow]]:
    """Map each HIGHER VRAM tier (GB) -> models that do not fit today but would.

    Only tiers strictly above the current card appear, and a model is listed at
    the FIRST tier that fits it (no double counting). Empty dict = everything
    already fits, the honest no-upsell answer.
    """
    out: dict[int, list[ModelRow]] = {}
    not_fitting = [r for r in rows if not fits(r.weights_mb, current_vram_mb)]
    for row in sorted(not_fitting, key=lambda r: r.weights_mb):
        for tier in VRAM_TIERS_GB:
            tier_mb = tier * 1024.0
            if tier_mb <= current_vram_mb:
                continue
            if fits(row.weights_mb, tier_mb):
                out.setdefault(tier, []).append(row)
                break
    return out


def render_markdown(
    gpu_name: str,
    vram_total_mb: float | None,
    rows: list[ModelRow],
    generated_on: str,
) -> str:
    """The whole report as one markdown document."""
    lines: list[str] = []
    vram_text = f"{vram_total_mb / 1024:.0f} GB" if vram_total_mb else "unknown"
    lines.append("# LOCITIZE RTX compatibility report")
    lines.append("")
    lines.append(f"- **GPU:** {gpu_name or 'none detected'} ({vram_text} VRAM)")
    lines.append(f"- **Generated:** {generated_on}")
    lines.append(
        "- **Method:** every figure below was MEASURED on this machine by "
        "LOCITIZE - real llama.cpp generation runs, never estimated. Context is "
        "the largest window that held >=85% of baseline throughput."
    )
    lines.append("")
    lines.append("| Model | Quant | Weights | Context | Measured tok/s | Fits this GPU |")
    lines.append("|---|---|--:|--:|--:|:--:|")
    for row in sorted(rows, key=lambda r: -(r.baseline_tok_s or 0)):
        speed = f"{row.baseline_tok_s:.0f}" if row.baseline_tok_s else "-"
        fit = "yes" if (vram_total_mb and fits(row.weights_mb, vram_total_mb)) else "no"
        lines.append(
            f"| {row.name or row.model_id} | {row.quantization or '-'} "
            f"| {row.weights_mb / 1024:.1f} GB | {row.context_size:,} "
            f"| {speed} | {fit} |"
        )
    lines.append("")

    if vram_total_mb:
        unlocks = unlocked_at(rows, vram_total_mb)
        if unlocks:
            lines.append("## What a bigger card would unlock")
            lines.append("")
            for tier in sorted(unlocks):
                names = ", ".join(r.name or r.model_id for r in unlocks[tier])
                lines.append(f"- **{tier} GB:** {names}")
            lines.append("")
        else:
            lines.append(
                "Every registered model fits this GPU - nothing to unlock by "
                "upgrading."
            )
            lines.append("")
    lines.append(
        "_Generated locally by LOCITIZE. Nothing was transmitted; sharing this "
        "file is your call._"
    )
    lines.append("")
    return "\n".join(lines)
