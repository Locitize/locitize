"""Tests for the wizard's measured-context step (setup_env.measure_contexts, M17.7).

The step delegates the actual probing to scripts/measure_context_ceilings.py
(exercised live, and by that script's own path). What matters to verify here is
its HONEST DEGRADATION: a bad interpreter or an absent measured tuner must return
a non-fatal result and leave models at their safe defaults, never raise into the
wizard.
"""

from __future__ import annotations

from pathlib import Path

import setup_env
from scripts.measure_context_ceilings import context_candidates, state_rows_for_apply


def test_missing_interpreter_is_a_clean_failure(tmp_path):
    result = setup_env.measure_contexts(tmp_path / "no-such-python.exe")
    assert not result.ok
    assert "interpreter not found" in result.message


def test_absent_measured_tuner_skips_not_fails(tmp_path, monkeypatch):
    """If the measured tuner script is not shipped, models keep their defaults -
    a skipped success, not an error that blocks setup."""
    # A real interpreter path so the interpreter check passes...
    import sys

    fake_exe = Path(sys.executable)
    # ...but a BASE_DIR with no scripts/measure_context_ceilings.py under it.
    monkeypatch.setattr(setup_env, "BASE_DIR", tmp_path)
    result = setup_env.measure_contexts(fake_exe)
    assert result.ok
    assert result.skipped
    assert "safe defaults" in result.message


def test_only_scope_also_limits_rows_applied_from_resumable_state():
    state = {
        "older-complete-model": {"done": True, "best": 65536},
        "requested-model": {"done": True, "best": 32768},
    }

    assert list(state_rows_for_apply(state, "requested-model")) == [
        ("requested-model", state["requested-model"])
    ]
    assert list(state_rows_for_apply(state, None)) == list(state.items())


def test_context_candidates_include_an_off_ladder_native_endpoint():
    candidates = context_candidates(40960)
    assert candidates[-2:] == (32768, 40960)
    assert all(context <= 40960 for context in candidates)
