"""The list writers accept block-style lists in a hand-written models.yaml.

Owner-observed 2026-10-19: Auto-tune finished but "could not write the tuned
settings" because server_args was written as a block list (`- --parallel` lines
under `server_args:`), which the flow-list writer left dangling.
"""

import yaml

import config

_MODELS = """\
# owner's own comment
models:
  - id: "first"
    name: "First"
    location: "/locitize-test/models/first.gguf"
    context_size: 4096
    gpu_layers: 999
    capabilities:
    - tools
    - vision
    status: installed
    server_args:
    - --parallel
    - "1"

  - id: "second"
    name: "Second"
    location: "/locitize-test/models/second.gguf"
    context_size: 8192
    gpu_layers: 999
    status: installed
    server_args: ["--parallel", "2"]
"""


def _load(tmp_path):
    return yaml.safe_load((tmp_path / "models.yaml").read_text(encoding="utf-8"))["models"]


def test_tuning_replaces_a_block_style_server_args_list(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS, encoding="utf-8")
    config.write_model_tuning(tmp_path, "first", 16384, ["--parallel", "1", "--flash-attn"], note="tuned")
    rows = _load(tmp_path)
    assert rows[0]["context_size"] == 16384
    assert rows[0]["server_args"] == ["--parallel", "1", "--flash-attn"]
    assert rows[0]["capabilities"] == ["tools", "vision"]
    assert rows[1]["server_args"] == ["--parallel", "2"]
    text = (tmp_path / "models.yaml").read_text(encoding="utf-8")
    assert "# owner's own comment" in text


def test_capabilities_replace_a_block_style_list(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS, encoding="utf-8")
    config.write_model_capabilities(tmp_path, "first", ["reasoning"])
    rows = _load(tmp_path)
    assert rows[0]["capabilities"] == ["reasoning"]
    assert rows[0]["server_args"] == ["--parallel", "1"]
    assert rows[0]["status"] == "installed"
