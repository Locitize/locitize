"""Tests for context-cliff advice (M17.3)."""
from __future__ import annotations

import json

import context_advice


def test_no_warning_at_or_below_ceiling():
    c = {"m": 16384}
    assert context_advice.context_warning("m", 8192, c) == ""
    assert context_advice.context_warning("m", 16384, c) == ""


def test_warning_past_ceiling_names_the_number():
    c = {"m": 16384}
    w = context_advice.context_warning("m", 32768, c)
    assert "16384" in w and "32768" in w and "slowdown" in w


def test_unknown_model_gets_no_warning():
    assert context_advice.context_warning("x", 999999, {"m": 8192}) == ""


def test_load_ceilings_missing_file_is_empty(tmp_path):
    assert context_advice.load_ceilings(tmp_path) == {}


def test_load_ceilings_reads_best_and_skips_unmeasured(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "ctx_ceilings.json").write_text(json.dumps({
        "good": {"best": 32768},
        "failed": {"best": 0},
        "nobest": {"verdict": "x"},
    }))
    got = context_advice.load_ceilings(tmp_path)
    assert got == {"good": 32768}
