"""Tests for GGUF capability detection (M17.2), using synthetic headers."""
from __future__ import annotations

import struct

import gguf_meta


def _write_gguf(path, arch, chat_template=None):
    """Minimal valid-enough GGUF: header + arch KV + optional chat_template KV."""
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))       # version
        f.write(struct.pack("<Q", 0))       # tensor count
        kvs = [("general.architecture", 8, arch)]
        if chat_template is not None:
            kvs.append(("tokenizer.chat_template", 8, chat_template))
        f.write(struct.pack("<Q", len(kvs)))
        for key, vtype, val in kvs:
            kb = key.encode()
            f.write(struct.pack("<Q", len(kb))); f.write(kb)
            f.write(struct.pack("<I", vtype))   # 8 = string
            vb = val.encode()
            f.write(struct.pack("<Q", len(vb))); f.write(vb)


def test_detects_tools_from_template(tmp_path):
    p = tmp_path / "m.gguf"
    _write_gguf(p, "qwen3", "{% if tools %}{{ tool_call }}{% endif %}")
    assert "tools" in gguf_meta.detect_capabilities(p)


def test_detects_reasoning_from_template(tmp_path):
    p = tmp_path / "m.gguf"
    _write_gguf(p, "qwen3", "reply in <think> ... </think> then answer")
    assert "reasoning" in gguf_meta.detect_capabilities(p)


def test_detects_vision_from_arch(tmp_path):
    p = tmp_path / "m.gguf"
    _write_gguf(p, "qwen2vl", "plain")
    assert "vision" in gguf_meta.detect_capabilities(p)


def test_plain_chat_model_has_no_special_caps(tmp_path):
    p = tmp_path / "m.gguf"
    _write_gguf(p, "llama", "{{ messages }}")
    assert gguf_meta.detect_capabilities(p) == []


def test_ablation_that_strips_tools_is_visible(tmp_path):
    # The real finding: same arch, a template WITHOUT the tool marker -> no tools.
    with_tools = tmp_path / "a.gguf"
    without = tmp_path / "b.gguf"
    _write_gguf(with_tools, "qwen3", "{{ tool_call }}")
    _write_gguf(without, "qwen3", "{{ messages }}")
    assert "tools" in gguf_meta.detect_capabilities(with_tools)
    assert "tools" not in gguf_meta.detect_capabilities(without)


def test_unreadable_file_yields_empty_not_error(tmp_path):
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"not a gguf")
    assert gguf_meta.detect_capabilities(p) == []
