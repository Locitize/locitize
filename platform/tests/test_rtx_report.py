"""Tests for the RTX compatibility report (rtx_report.py, M18.2).

The report's claims are the product's reputation, so the pure logic must be
exact: a fit means weights plus working margin inside the card; the upgrade
section lists a model at the FIRST tier that fits it and only tiers above the
current card; and an unmeasured model renders honestly as '-', never a number.
"""

from __future__ import annotations

import rtx_report
from rtx_report import ModelRow, fits, render_markdown, unlocked_at


def _row(mid, weights_gb, tok=None, ctx=8192, name=""):
    return ModelRow(
        model_id=mid,
        name=name or mid,
        weights_mb=weights_gb * 1024.0,
        context_size=ctx,
        baseline_tok_s=tok,
    )


def test_fits_requires_margin_not_just_weights():
    vram_16gb = 16 * 1024.0
    assert fits(12 * 1024.0, vram_16gb)          # 12GB + 2GB margin fits 16GB
    assert not fits(15 * 1024.0, vram_16gb)      # 15GB + margin does not
    assert not fits(0, vram_16gb) and not fits(1024, 0)


def test_unlocked_lists_model_at_first_fitting_tier_only():
    rows = [
        _row("fits-now", 10, tok=50),
        _row("needs-24", 16, tok=7),    # 16+2 margin -> first fits at 24GB
        _row("needs-32", 27, tok=None), # 27+2 -> not 24, fits at 32GB
    ]
    unlocks = unlocked_at(rows, current_vram_mb=16 * 1024.0)
    assert [r.model_id for r in unlocks[24]] == ["needs-24"]
    assert [r.model_id for r in unlocks[32]] == ["needs-32"]
    assert 48 not in unlocks  # no double counting at higher tiers
    assert all(tier > 16 for tier in unlocks)


def test_everything_fits_yields_empty_unlocks():
    assert unlocked_at([_row("small", 4, tok=200)], 16 * 1024.0) == {}


def test_markdown_renders_measured_and_unmeasured_honestly():
    text = render_markdown(
        "GeForce RTX 5070 Ti",
        16 * 1024.0,
        [_row("fast", 3, tok=239.4, ctx=131072), _row("unmeasured", 16)],
        "2026-08-30 10:00",
    )
    assert "GeForce RTX 5070 Ti" in text and "16 GB VRAM" in text
    assert "| 239 | yes |" in text          # measured, fits
    assert "| - | no |" in text             # unmeasured renders '-', not a number
    assert "131,072" in text                # context with thousands separator
    assert "What a bigger card would unlock" in text
    assert "MEASURED" in text               # the method line
    assert "Nothing was transmitted" in text  # the privacy line


def test_markdown_no_gpu_is_honest():
    text = render_markdown("", None, [_row("m", 3, tok=100)], "2026-08-30")
    assert "none detected" in text
    assert "| no |" in text  # nothing "fits" an unknown card
