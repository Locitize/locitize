"""Process-specific Codex context policy without global configuration writes."""
import json
import os

import pytest
from session_launch import build_local_launch


def overrides(argv):
    """Read only CLI configuration pairs, preserving their typed JSON values."""
    return dict((argv[i + 1].split("=", 1)[0],
                 json.loads(argv[i + 1].split("=", 1)[1]))
                for i, value in enumerate(argv[:-1]) if value == "-c")


@pytest.mark.parametrize("resume", ["", "existing-session"])
def test_actual_context_process_override_preserves_files(tmp_path, resume):
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "user-default"\nmodel_context_window = 99999\n')
    before = config_path.read_bytes()
    environment = dict(os.environ)
    argv, _ = build_local_launch("codex", str(tmp_path), "registered", 8093,
                                 resume, context_size=16384)
    values = overrides(argv)
    assert values["model_context_window"] == 16384
    assert values["model_auto_compact_token_limit"] == 13107
    assert values["model_providers.locitize.wire_api"] == "responses"
    assert config_path.read_bytes() == before
    assert dict(os.environ) == environment


def test_unknown_context_leaves_model_defaults(tmp_path):
    argv, _ = build_local_launch("codex", str(tmp_path), "registered", 8093)
    assert "model_context_window" not in overrides(argv)
    assert "model_auto_compact_token_limit" not in overrides(argv)


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1, "8192", 8192.0])
def test_bad_context_rejected_without_coercion(tmp_path, invalid):
    with pytest.raises(ValueError, match="Measured local context"):
        build_local_launch("codex", str(tmp_path), "registered", 8093,
                           context_size=invalid)
