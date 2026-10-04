"""Tests for the minimal GGUF metadata header reader (gguf_meta.py).

Every fixture here is a synthetic GGUF header built byte by byte from the spec
(https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), never a real model
file: the header is the only part this reader touches, so a few hundred bytes
exercises it exactly as a 15GB .gguf would, without the 15GB.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from gguf_meta import (
    GGUF_MAGIC,
    GgufError,
    read_gguf_header,
    read_native_context_length,
)

# GGUF metadata value type enum values used by the builders below.
T_UINT32 = 4
T_STRING = 8
T_ARRAY = 9
T_FLOAT32 = 6


def _gguf_string(text: str) -> bytes:
    """uint64 byte length followed by the UTF-8 bytes."""
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_string(key: str, value: str) -> bytes:
    return _gguf_string(key) + struct.pack("<I", T_STRING) + _gguf_string(value)


def _kv_uint32(key: str, value: int) -> bytes:
    return _gguf_string(key) + struct.pack("<I", T_UINT32) + struct.pack("<I", value)


def _kv_string_array(key: str, values: list[str]) -> bytes:
    """An array of strings - the shape a tokenizer vocabulary takes."""
    body = struct.pack("<I", T_STRING) + struct.pack("<Q", len(values))
    for value in values:
        body += _gguf_string(value)
    return _gguf_string(key) + struct.pack("<I", T_ARRAY) + body


def _kv_float_array(key: str, values: list[float]) -> bytes:
    """An array of fixed-width elements - skipped in one read by the parser."""
    body = struct.pack("<I", T_FLOAT32) + struct.pack("<Q", len(values))
    for value in values:
        body += struct.pack("<f", value)
    return _gguf_string(key) + struct.pack("<I", T_ARRAY) + body


def _build_gguf(
    kvs: list[bytes],
    *,
    version: int = 3,
    magic: bytes = GGUF_MAGIC,
    tensor_count: int = 7,
    kv_count: int | None = None,
    trailer: bytes = b"\x00" * 64,
) -> bytes:
    """Assemble a whole synthetic GGUF prefix, with a junk 'tensor' trailer."""
    header = magic + struct.pack("<I", version)
    header += struct.pack("<Q", tensor_count)
    header += struct.pack("<Q", len(kvs) if kv_count is None else kv_count)
    return header + b"".join(kvs) + trailer


def _write(tmp_path: Path, data: bytes, name: str = "model.gguf") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


def test_reads_architecture_and_context_length(tmp_path):
    """The two keys the auto-tuner needs come back with their real values."""
    path = _write(
        tmp_path,
        _build_gguf(
            [
                _kv_string("general.architecture", "qwen3"),
                _kv_string("general.name", "Qwen3 14B"),
                _kv_uint32("qwen3.context_length", 40960),
                _kv_uint32("qwen3.block_count", 40),
            ]
        ),
    )
    header = read_gguf_header(path)
    assert header.architecture == "qwen3"
    assert header.context_length == 40960
    assert header.version == 3
    assert header.tensor_count == 7
    assert read_native_context_length(path) == 40960


def test_context_length_is_matched_to_the_architecture_not_the_first_match(tmp_path):
    """A '<other>.context_length' key must not be mistaken for this model's.

    Guards the specific silent-wrong-answer bug this feature exists to prevent:
    reporting some other architecture's number as the native window would produce
    wrong YaRN parameters that still 'look' plausible.
    """
    path = _write(
        tmp_path,
        _build_gguf(
            [
                _kv_uint32("llama.context_length", 4096),
                _kv_string("general.architecture", "qwen3"),
                _kv_uint32("qwen3.context_length", 262144),
            ]
        ),
    )
    assert read_gguf_header(path).context_length == 262144


def test_skips_large_arrays_without_reading_them_as_values(tmp_path):
    """A tokenizer-sized array between the wanted keys is stepped over correctly."""
    path = _write(
        tmp_path,
        _build_gguf(
            [
                _kv_string_array("tokenizer.ggml.tokens", [f"tok{i}" for i in range(500)]),
                _kv_float_array("tokenizer.ggml.scores", [0.5] * 500),
                _kv_string("general.architecture", "qwen2"),
                _kv_uint32("qwen2.context_length", 131072),
            ]
        ),
    )
    header = read_gguf_header(path)
    assert (header.architecture, header.context_length) == ("qwen2", 131072)


def test_stops_early_once_both_keys_are_known(tmp_path):
    """A truncated tail is harmless when the wanted keys came first.

    This is the behaviour that keeps the read in the millisecond range on a real
    multi-gigabyte file: everything after the answer is never touched. The proof
    is that a file whose declared kv_count runs off the end still parses.
    """
    data = _build_gguf(
        [
            _kv_string("general.architecture", "qwen3"),
            _kv_uint32("qwen3.context_length", 32768),
        ],
        kv_count=99,  # claims far more entries than the file actually contains
        trailer=b"",
    )
    header = read_gguf_header(_write(tmp_path, data))
    assert header.context_length == 32768


def test_missing_context_length_is_a_none_not_an_error(tmp_path):
    """An absent context_length is a legitimate state the caller decides about."""
    path = _write(
        tmp_path,
        _build_gguf([_kv_string("general.architecture", "mystery")]),
    )
    header = read_gguf_header(path)
    assert header.architecture == "mystery"
    assert header.context_length is None
    # ... but the auto-tuner's convenience wrapper refuses rather than guess.
    with pytest.raises(GgufError) as excinfo:
        read_native_context_length(path)
    assert "mystery.context_length" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Refusals - every one of these must name the file and say what to do next
# --------------------------------------------------------------------------- #


def test_missing_file_refuses_with_the_real_path(tmp_path):
    missing = tmp_path / "not-here.gguf"
    with pytest.raises(GgufError) as excinfo:
        read_gguf_header(missing)
    assert str(missing) in str(excinfo.value)
    assert "models.yaml" in str(excinfo.value)


def test_wrong_magic_refuses(tmp_path):
    path = _write(tmp_path, b"NOTAGGUF" + b"\x00" * 64, name="fake.gguf")
    with pytest.raises(GgufError) as excinfo:
        read_gguf_header(path)
    assert "not a GGUF model file" in str(excinfo.value)


def test_unsupported_version_refuses(tmp_path):
    path = _write(tmp_path, _build_gguf([], version=1))
    with pytest.raises(GgufError) as excinfo:
        read_gguf_header(path)
    assert "version 1" in str(excinfo.value)


def test_truncated_header_refuses_rather_than_returning_a_guess(tmp_path):
    """A half-downloaded file must fail honestly, not report a partial answer."""
    full = _build_gguf(
        [
            _kv_string("general.architecture", "qwen3"),
            _kv_uint32("qwen3.context_length", 40960),
        ],
        trailer=b"",
    )
    path = _write(tmp_path, full[: len(full) - 6])
    with pytest.raises(GgufError) as excinfo:
        read_gguf_header(path)
    assert "truncated" in str(excinfo.value)


def test_absurd_string_length_is_refused_not_allocated(tmp_path):
    """A corrupt length prefix must not become a multi-gigabyte allocation."""
    body = struct.pack("<Q", 1 << 40) + b"junk"
    data = GGUF_MAGIC + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", 1) + body
    with pytest.raises(GgufError) as excinfo:
        read_gguf_header(_write(tmp_path, data))
    assert "not plausible" in str(excinfo.value)


def test_absurd_kv_count_is_refused(tmp_path):
    data = GGUF_MAGIC + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", 1 << 40)
    with pytest.raises(GgufError) as excinfo:
        read_gguf_header(_write(tmp_path, data))
    assert "not plausible" in str(excinfo.value)


def test_missing_architecture_key_refuses(tmp_path):
    path = _write(tmp_path, _build_gguf([_kv_string("general.name", "nameless")]))
    with pytest.raises(GgufError) as excinfo:
        read_gguf_header(path)
    assert "general.architecture" in str(excinfo.value)
