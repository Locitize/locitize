"""Tests for setup_env.pair_mmproj - vision projector pairing (M18.18).

The rules under test, in priority order:
- a projector whose name starts with the model's stem pairs outright
  (proof of vision even when the arch hides it, e.g. Kimi-VL says deepseek2);
- sibling family dirs are searched for stem matches (LM Studio lays out
  "Name.Q6_K/" next to "Name/" holding the projector);
- a generic projector (mmproj-F16.gguf) pairs only when it is the single
  candidate in the model's OWN directory AND the model's arch says vision -
  never from a sibling dir, never next to a text-only model.
"""

from __future__ import annotations

import struct

import setup_env


def _gguf(path, arch="qwen2vl"):
    """Minimal GGUF v3 header declaring `arch` (for the generic-rule guard)."""

    def _s(text: str) -> bytes:
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    kv = _s("general.architecture") + struct.pack("<I", 8) + _s(arch)  # 8 = string
    body = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)  # no tensors
    body += struct.pack("<Q", 1) + kv + b"\x00" * 16
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def test_stem_match_wins_even_for_hidden_arch(tmp_path):
    # Kimi-VL's header says deepseek2; the stem-named projector is the proof.
    model = _gguf(tmp_path / "d" / "Kimi-VL-A3B-Instruct.Q6_K.gguf", "deepseek2")
    proj = tmp_path / "d" / "Kimi-VL-A3B-Instruct.mmproj-Q8_0.gguf"
    proj.write_bytes(b"g")
    assert setup_env.pair_mmproj(model) == str(proj)


def test_generic_requires_vision_arch(tmp_path):
    # Text-arch model + one generic projector in the dir -> refused.
    model = _gguf(tmp_path / "d" / "TextModel.Q4.gguf", "qwen3")
    (tmp_path / "d" / "mmproj-F16.gguf").write_bytes(b"g")
    assert setup_env.pair_mmproj(model) == ""
    # Vision arch -> accepted (gemma-4-E2B's HF snapshot layout).
    vmodel = _gguf(tmp_path / "d2" / "SomeVL.Q4.gguf", "qwen2vl")
    proj = tmp_path / "d2" / "mmproj-F16.gguf"
    proj.write_bytes(b"g")
    assert setup_env.pair_mmproj(vmodel) == str(proj)


def test_generic_refused_when_allow_generic_off(tmp_path):
    vmodel = _gguf(tmp_path / "d" / "SomeVL.Q4.gguf", "qwen2vl")
    (tmp_path / "d" / "mmproj-F16.gguf").write_bytes(b"g")
    assert setup_env.pair_mmproj(vmodel, allow_generic=False) == ""


def test_sibling_family_dir_stem_only(tmp_path):
    # LM Studio layout: model in "Name.Q6_K/", projector in sibling "Name/".
    model = _gguf(tmp_path / "Qwen3-VL-8B.Q6_K" / "Qwen3-VL-8B.Q6_K.gguf")
    proj = tmp_path / "Qwen3-VL-8B" / "Qwen3-VL-8B.mmproj-Q8_0.gguf"
    proj.parent.mkdir()
    proj.write_bytes(b"g")
    assert setup_env.pair_mmproj(model) == str(proj)


def test_generic_never_pairs_from_sibling_dir(tmp_path):
    # The mis-pair this guards against: a generic projector placed for one
    # model must not be grabbed by a different model in a family sibling dir.
    model = _gguf(tmp_path / "Solo-VL.Q4" / "Solo-VL.Q4.gguf", "qwen2vl")
    (tmp_path / "Solo-VL.Q4-x" / "sub").mkdir(parents=True)
    (tmp_path / "Solo-VL.Q4-x" / "mmproj-F16.gguf").write_bytes(b"g")
    assert setup_env.pair_mmproj(model) == ""


def test_no_candidates_is_empty(tmp_path):
    model = _gguf(tmp_path / "d" / "Alone.Q4.gguf")
    assert setup_env.pair_mmproj(model) == ""
