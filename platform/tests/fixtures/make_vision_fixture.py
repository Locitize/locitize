"""Generate the deterministic vision test image (AC15 ground truth).

Builder-generated, NOT downloaded (the same "generate our own ground truth"
discipline as the M3 known_phrase.wav): a pure-stdlib PNG writer draws a large,
saturated RED square centered on a white background. No PIL/numpy dependency -- the
whole image is built from zlib + struct so it reproduces byte-for-byte on any
machine and needs nothing installed.

The unambiguous, deterministic visual property is the color RED (RGB 255,0,0) as a
bold centered square. A vision-language model reliably describes this with the word
"red", which is the keyword AC15's verifier asserts against the real model answer.

Run from Codebase/platform:  python tests/fixtures/make_vision_fixture.py
It rewrites tests/fixtures/vision_red_square.png in place (idempotent).
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

# Image geometry. A square canvas with a centered red square large enough that the
# model's attention lands on it, on a white margin so the shape reads as a square.
_SIZE = 480
_MARGIN = 90  # white border thickness; the red square is _SIZE - 2*_MARGIN per side

_WHITE = (255, 255, 255)
_RED = (255, 0, 0)  # pure saturated red -> reliably named "red"


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    """Assemble one PNG chunk: length + tag + data + CRC32(tag+data)."""
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def build_png_bytes() -> bytes:
    """Return the full PNG file bytes for the red-square-on-white fixture."""
    # Build raw RGB scanlines; each row is prefixed with filter byte 0 (no filter).
    rows = bytearray()
    for y in range(_SIZE):
        rows.append(0)  # filter type 0 (None) for this scanline
        for x in range(_SIZE):
            in_square = (_MARGIN <= x < _SIZE - _MARGIN) and (
                _MARGIN <= y < _SIZE - _MARGIN
            )
            r, g, b = _RED if in_square else _WHITE
            rows.extend((r, g, b))

    signature = b"\x89PNG\r\n\x1a\n"
    # IHDR: width, height, bit depth 8, color type 2 (truecolor RGB), no interlace.
    ihdr = struct.pack(">IIBBBBB", _SIZE, _SIZE, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(rows), 9)
    return (
        signature
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def main() -> int:
    out = Path(__file__).resolve().parent / "vision_red_square.png"
    out.write_bytes(build_png_bytes())
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
