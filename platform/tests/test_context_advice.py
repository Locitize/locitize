"""Tests for context-cliff advice (M17.3)."""
from __future__ import annotations

import json
from pathlib import Path

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


def test_remove_ceiling_uses_atomic_replace_and_keeps_other_entries(
        tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    target = reports / "ctx_ceilings.json"
    other = {"best": 16384, "notes": ["keep", "all", "metadata"]}
    target.write_text(json.dumps({"replaced": {"best": 8192}, "other": other}))
    real_replace = context_advice.os.replace
    calls = []

    def record_replace(source, destination):
        calls.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(context_advice.os, "replace", record_replace)

    assert context_advice.remove_ceiling(tmp_path, "replaced") is True
    assert json.loads(target.read_text(encoding="utf-8")) == {"other": other}
    assert len(calls) == 1
    assert Path(calls[0][0]).parent == reports
    assert Path(calls[0][1]) == target
    assert not list(reports.glob("*.tmp"))


def test_remove_ceiling_missing_or_corrupt_report_is_a_no_op(tmp_path):
    assert context_advice.remove_ceiling(tmp_path, "model") is False
    assert not (tmp_path / "reports").exists()

    reports = tmp_path / "reports"
    reports.mkdir()
    target = reports / "ctx_ceilings.json"
    corrupt = b"{not JSON\n"
    target.write_bytes(corrupt)
    assert context_advice.remove_ceiling(tmp_path, "model") is False
    assert target.read_bytes() == corrupt


def test_remove_ceiling_replace_failure_keeps_original(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    target = reports / "ctx_ceilings.json"
    original = b'{"model": {"best": 8192}, "other": {"best": 4096}}\n'
    target.write_bytes(original)
    monkeypatch.setattr(
        context_advice.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("busy")),
    )

    assert context_advice.remove_ceiling(tmp_path, "model") is False
    assert target.read_bytes() == original
    assert not list(reports.glob("*.tmp"))
