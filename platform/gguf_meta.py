"""Minimal, dependency-free GGUF metadata header reader.

Why this file exists
--------------------
The context auto-tuner (autotune.py) has to know a model's REAL native training
context window - the value llama.cpp calls `n_ctx_train` - before it can decide
whether a requested context needs YaRN rope scaling and, if so, what scale to
use. There were three ways to get that number and only one of them is cheap:

1. Start llama-server and parse its startup log. Correct, but costs 20-30s and a
   VRAM allocation for what should be an instant lookup.
2. Import a model library (torch/transformers/gguf+numpy). Correct, but pulls a
   heavy dependency into a desktop app that otherwise needs none for this.
3. Read the GGUF file's own metadata header. GGUF puts a small, self-describing
   key/value block at the very front of the file, so the answer is a few
   kilobytes in and takes milliseconds - the whole multi-GB tensor payload is
   never touched.

This module is option 3: a direct implementation of the GGUF header layout
documented at https://github.com/ggml-org/ggml/blob/master/docs/gguf.md, reading
only what is needed to answer "what is this model's architecture, and what
context length was it trained for".

File layout it parses (GGUF v2/v3, little-endian):

    magic       4 bytes, "GGUF"
    version     uint32
    tensor_count    uint64
    kv_count        uint64
    then kv_count entries of:
        key         uint64 length + UTF-8 bytes
        value_type  uint32 (the GGUF_METADATA_VALUE_TYPE_* enum)
        value       type-dependent payload

The key we want is `<architecture>.context_length` (e.g. `qwen3.context_length`),
where `<architecture>` is itself the string value of `general.architecture`.

Everything here is read-only. Nothing in this module writes, starts, or
downloads anything.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

# The four magic bytes every GGUF file starts with.
GGUF_MAGIC = b"GGUF"

# Versions this reader understands. v1 used 32-bit lengths/counts throughout and
# was dropped by llama.cpp long ago; rather than guess at a format we cannot test
# against a real file, we refuse it with an honest message.
SUPPORTED_VERSIONS = (2, 3)

# GGUF_METADATA_VALUE_TYPE_* enum values from the spec.
_T_UINT8 = 0
_T_INT8 = 1
_T_UINT16 = 2
_T_INT16 = 3
_T_UINT32 = 4
_T_INT32 = 5
_T_FLOAT32 = 6
_T_BOOL = 7
_T_STRING = 8
_T_ARRAY = 9
_T_UINT64 = 10
_T_INT64 = 11
_T_FLOAT64 = 12

# Fixed-width scalar types: enum -> (struct format, byte width). Anything not in
# here (STRING, ARRAY) needs its own length-prefixed handling below.
_SCALARS: dict[int, tuple[str, int]] = {
    _T_UINT8: ("<B", 1),
    _T_INT8: ("<b", 1),
    _T_UINT16: ("<H", 2),
    _T_INT16: ("<h", 2),
    _T_UINT32: ("<I", 4),
    _T_INT32: ("<i", 4),
    _T_FLOAT32: ("<f", 4),
    _T_BOOL: ("<?", 1),
    _T_UINT64: ("<Q", 8),
    _T_INT64: ("<q", 8),
    _T_FLOAT64: ("<d", 8),
}

# Sanity ceilings. A corrupt or truncated file can present an absurd length
# prefix; without a ceiling this reader would happily try to allocate it. These
# are far above anything a real GGUF contains (the largest real string is a chat
# template, tens of kilobytes; the largest real array is a tokenizer vocabulary,
# low hundreds of thousands of entries).
_MAX_STRING_BYTES = 64 * 1024 * 1024
_MAX_ARRAY_LEN = 50_000_000
_MAX_KV_COUNT = 1_000_000

# Read buffer size. The header sits at the front of a multi-gigabyte file, so a
# generous buffer means the whole parse is typically one or two disk reads.
_BUFFER_BYTES = 1024 * 1024

# The metadata key holding the model architecture name, and the suffix of the
# per-architecture key holding the trained context window.
KEY_ARCHITECTURE = "general.architecture"
CONTEXT_LENGTH_SUFFIX = ".context_length"


class GgufError(Exception):
    """A GGUF file could not be read as GGUF.

    One type for every cause (missing file, wrong magic, unsupported version,
    truncated header, key absent) because every caller does the same thing with
    it: show the message. Each message is written as a finished sentence naming
    the real path, matching this codebase's convention that an error the owner
    reads must also tell them what to do next.
    """


@dataclass
class GgufHeader:
    """The handful of header facts the auto-tuner actually uses."""

    path: str
    version: int
    tensor_count: int
    kv_count: int
    architecture: str
    # None when the file genuinely carries no `<arch>.context_length` key. That
    # is a legitimate state (some converted GGUFs omit it), not a parse failure,
    # so it is a None rather than an exception - the caller decides what to do.
    context_length: int | None


def read_gguf_header(path: str | Path) -> GgufHeader:
    """Read one GGUF file's architecture and native context length.

    Opens the file, validates the magic and version, then walks the metadata
    key/value block until both wanted keys are known, and stops there. Because
    real GGUF writers emit `general.architecture` and the model's dimension keys
    before the tokenizer arrays, the early stop means the (multi-megabyte)
    vocabulary is normally never even read.

    Raises GgufError for anything that makes the file unreadable AS GGUF; a
    merely absent context_length key returns a header with context_length=None.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise GgufError(
            f"LOCITIZE could not read the model file at {file_path} because it "
            f"does not exist. Check the model's `location` in models.yaml, then "
            f"try again."
        )

    try:
        with open(file_path, "rb", buffering=_BUFFER_BYTES) as handle:
            return _parse_header(handle, file_path)
    except OSError as exc:
        cause = (exc.strerror or "the file could not be opened").strip().rstrip(".")
        raise GgufError(
            f"LOCITIZE could not read the model file at {file_path}: {cause}. "
            f"Close any program using the file, then try again."
        ) from exc


def read_native_context_length(path: str | Path) -> int:
    """Return the model's trained context window, or raise GgufError.

    Convenience wrapper for the auto-tuner's one real question. Turns the
    "header parsed fine but carries no context_length" case into a refusal,
    because a tuner that cannot learn the native window must not proceed to
    invent YaRN parameters from a guess.
    """
    header = read_gguf_header(path)
    if header.context_length is None:
        raise GgufError(
            f"The model file at {header.path} carries no "
            f"'{header.architecture}{CONTEXT_LENGTH_SUFFIX}' value in its GGUF "
            f"header, so LOCITIZE cannot tell what context window it was trained "
            f"for. Set this model's context_size by hand instead of auto-tuning."
        )
    return header.context_length


# --------------------------------------------------------------------------- #
# Header parsing internals
# --------------------------------------------------------------------------- #


def _parse_header(handle: BinaryIO, file_path: Path) -> GgufHeader:
    """Walk the fixed prefix then the KV block; stop as soon as we have enough."""
    magic = _read_exact(handle, 4, file_path, "the magic bytes")
    if magic != GGUF_MAGIC:
        raise GgufError(
            f"The file at {file_path} is not a GGUF model file (it starts with "
            f"{magic!r}, not {GGUF_MAGIC!r}). Point this model's `location` at a "
            f"'.gguf' file, then try again."
        )

    version = _unpack("<I", _read_exact(handle, 4, file_path, "the format version"))
    if version not in SUPPORTED_VERSIONS:
        raise GgufError(
            f"The file at {file_path} uses GGUF format version {version}, which "
            f"LOCITIZE does not read (it understands versions "
            f"{', '.join(str(v) for v in SUPPORTED_VERSIONS)}). Re-download the "
            f"model in a current GGUF build, or set its context_size by hand."
        )

    tensor_count = _unpack("<Q", _read_exact(handle, 8, file_path, "the tensor count"))
    kv_count = _unpack("<Q", _read_exact(handle, 8, file_path, "the metadata count"))
    if kv_count > _MAX_KV_COUNT:
        raise GgufError(
            f"The GGUF header at {file_path} claims {kv_count} metadata entries, "
            f"which is not plausible for a real model file - the file is most "
            f"likely corrupt or was truncated mid-download. Re-download it."
        )

    architecture = ""
    context_length: int | None = None
    # The context_length key name depends on the architecture value, which we
    # may not have seen yet, so we remember every "*.context_length" we pass and
    # resolve the right one at the end. In practice general.architecture comes
    # first and there is exactly one such key, but relying on ordering would be
    # a silent-wrong-answer bug on a file written in another order.
    context_candidates: dict[str, int] = {}

    for _ in range(kv_count):
        key = _read_string(handle, file_path)
        value_type = _unpack(
            "<I", _read_exact(handle, 4, file_path, f"the type of '{key}'")
        )

        if key == KEY_ARCHITECTURE:
            architecture = str(_read_value(handle, value_type, file_path))
        elif key.endswith(CONTEXT_LENGTH_SUFFIX):
            value = _read_value(handle, value_type, file_path)
            if isinstance(value, int) and not isinstance(value, bool):
                context_candidates[key] = value
        else:
            _skip_value(handle, value_type, file_path)

        # Early stop: everything after this point is tokenizer/vocabulary bulk we
        # have no use for, and skipping it is what keeps this read in the
        # millisecond range on a 15GB file.
        if architecture and f"{architecture}{CONTEXT_LENGTH_SUFFIX}" in context_candidates:
            context_length = context_candidates[f"{architecture}{CONTEXT_LENGTH_SUFFIX}"]
            break

    if context_length is None and architecture:
        context_length = context_candidates.get(f"{architecture}{CONTEXT_LENGTH_SUFFIX}")

    if not architecture:
        raise GgufError(
            f"The GGUF header at {file_path} carries no '{KEY_ARCHITECTURE}' "
            f"entry, so LOCITIZE cannot tell which model family it belongs to. "
            f"Set this model's context_size by hand instead of auto-tuning."
        )

    return GgufHeader(
        path=str(file_path),
        version=version,
        tensor_count=tensor_count,
        kv_count=kv_count,
        architecture=architecture,
        context_length=context_length,
    )


def _read_exact(handle: BinaryIO, count: int, file_path: Path, what: str) -> bytes:
    """Read exactly `count` bytes or raise a truncation error naming `what`."""
    data = handle.read(count)
    if len(data) != count:
        raise GgufError(
            f"The GGUF header at {file_path} ends before {what}, so the file is "
            f"truncated or still downloading. Re-download the model, then try "
            f"again."
        )
    return data


def _unpack(fmt: str, data: bytes) -> Any:
    """struct.unpack for a single value, kept short because it is used everywhere."""
    return struct.unpack(fmt, data)[0]


def _read_string(handle: BinaryIO, file_path: Path) -> str:
    """Read one gguf_string_t: a uint64 byte length followed by UTF-8 bytes."""
    length = _unpack("<Q", _read_exact(handle, 8, file_path, "a string length"))
    if length > _MAX_STRING_BYTES:
        raise GgufError(
            f"The GGUF header at {file_path} declares a {length}-byte metadata "
            f"string, which is not plausible for a real model file - the file is "
            f"most likely corrupt. Re-download it."
        )
    raw = _read_exact(handle, length, file_path, "the end of a string value")
    # errors="replace" rather than strict: a single bad byte in some unrelated
    # metadata string must not make an otherwise readable header unreadable, and
    # we never write these strings anywhere - they are compared or displayed.
    return raw.decode("utf-8", errors="replace")


def _read_value(handle: BinaryIO, value_type: int, file_path: Path) -> Any:
    """Read one typed metadata value and return it as a Python object.

    Only used for the two keys we care about, so arrays are returned as a list
    (bounded by _MAX_ARRAY_LEN); everything else takes the cheaper _skip_value
    path and is never materialized.
    """
    if value_type in _SCALARS:
        fmt, width = _SCALARS[value_type]
        return _unpack(fmt, _read_exact(handle, width, file_path, "a scalar value"))
    if value_type == _T_STRING:
        return _read_string(handle, file_path)
    if value_type == _T_ARRAY:
        item_type = _unpack(
            "<I", _read_exact(handle, 4, file_path, "an array element type")
        )
        length = _unpack("<Q", _read_exact(handle, 8, file_path, "an array length"))
        _guard_array_length(length, file_path)
        return [_read_value(handle, item_type, file_path) for _ in range(length)]
    raise GgufError(
        f"The GGUF header at {file_path} uses metadata value type {value_type}, "
        f"which is not part of the GGUF specification LOCITIZE reads. The file "
        f"is most likely corrupt; re-download it."
    )


def _skip_value(handle: BinaryIO, value_type: int, file_path: Path) -> None:
    """Advance past one typed value without building a Python object for it.

    Uses sequential read-and-discard rather than seek() on purpose: the file is
    opened with a large read buffer, and seek() throws that buffer away on every
    call. For a tokenizer array of a hundred thousand short strings, buffered
    reads are the difference between one disk hit and a hundred thousand.
    """
    if value_type in _SCALARS:
        _read_exact(handle, _SCALARS[value_type][1], file_path, "a scalar value")
        return
    if value_type == _T_STRING:
        length = _unpack("<Q", _read_exact(handle, 8, file_path, "a string length"))
        if length > _MAX_STRING_BYTES:
            raise GgufError(
                f"The GGUF header at {file_path} declares a {length}-byte "
                f"metadata string, which is not plausible for a real model file "
                f"- the file is most likely corrupt. Re-download it."
            )
        _read_exact(handle, length, file_path, "the end of a string value")
        return
    if value_type == _T_ARRAY:
        item_type = _unpack(
            "<I", _read_exact(handle, 4, file_path, "an array element type")
        )
        length = _unpack("<Q", _read_exact(handle, 8, file_path, "an array length"))
        _guard_array_length(length, file_path)
        if item_type in _SCALARS:
            # Fixed-width elements: one read for the whole run, not one per item.
            _read_exact(
                handle,
                _SCALARS[item_type][1] * length,
                file_path,
                "the end of an array value",
            )
            return
        for _ in range(length):
            _skip_value(handle, item_type, file_path)
        return
    raise GgufError(
        f"The GGUF header at {file_path} uses metadata value type {value_type}, "
        f"which is not part of the GGUF specification LOCITIZE reads. The file "
        f"is most likely corrupt; re-download it."
    )


def _guard_array_length(length: int, file_path: Path) -> None:
    """Refuse an array length no real model file would contain."""
    if length > _MAX_ARRAY_LEN:
        raise GgufError(
            f"The GGUF header at {file_path} declares an array of {length} "
            f"elements, which is not plausible for a real model file - the file "
            f"is most likely corrupt. Re-download it."
        )


# --------------------------------------------------------------------------- #
# Capability detection (M17.2)
# --------------------------------------------------------------------------- #
# What a model can DO, read from its own GGUF instead of hand-entered. This is
# the scan that, done ad-hoc during a benchmark review, caught two ablated
# models that had silently lost tool-calling - exactly the kind of fact the
# registry should carry rather than a human rediscover. Best-effort and never
# raises: an unreadable file yields an empty capability set, not an error.

def detect_capabilities(path: str | Path) -> list[str]:
    """Return the capabilities a GGUF file advertises, in a stable order.

    - "tools": the chat template renders tool calls (the marker every
      tool-calling template carries). This is what decides whether a model can
      drive an agent, and it is invisible without reading the template.
    - "vision": the architecture is a known multimodal one, OR an mmproj
      projector is required (callers pair this with the model row's mmproj).
    - "reasoning"/"thinking": the template emits a dedicated think channel.
    Detection is metadata-only; it never runs the model.
    """
    import struct

    caps: list[str] = []
    try:
        header = read_gguf_header(path)
    except Exception:  # noqa: BLE001 - a bad file has no capabilities to report
        return caps

    arch = (header.architecture or "").lower()
    if any(tag in arch for tag in ("vl", "vision", "clip", "mllama", "gemma3", "gemma4")):
        # gemma3/qwen-vl families carry vision in the arch name; a text-only
        # gemma still won't match "gemma3" unless the file says so.
        if "vl" in arch or "vision" in arch or "clip" in arch or "mllama" in arch:
            caps.append("vision")

    # The chat template is the authoritative source for tools + thinking, and it
    # is a single KV string - read just it rather than the whole tensor table.
    template = ""
    try:
        with open(path, "rb") as fh:
            if fh.read(4) == b"GGUF":
                struct.unpack("<I", fh.read(4))
                fh.read(8)
                n_kv, = struct.unpack("<Q", fh.read(8))
                for _ in range(n_kv):
                    key = _read_string(fh, Path(path))
                    vtype, = struct.unpack("<I", fh.read(4))
                    if key == "tokenizer.chat_template":
                        template = str(_read_value(fh, vtype, Path(path)) or "")
                        break
                    _skip_value(fh, vtype, Path(path))
    except Exception:  # noqa: BLE001 - template is optional
        template = ""

    low = template.lower()
    if "tool_call" in low or ("tools" in low and "function" in low):
        caps.append("tools")
    if "<think>" in low or "reasoning_content" in low or "thinking" in low:
        caps.append("reasoning")

    # Stable de-duplicated order for a comparable registry value.
    order = {"vision": 0, "tools": 1, "reasoning": 2}
    return sorted(set(caps), key=lambda c: order.get(c, 9))
