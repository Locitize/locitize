"""Send model files to the Windows Recycle Bin, every name at once.

Why this file exists
--------------------
Owner report 2026-10-20: models deleted from the Models page kept coming
back, and the drive space never came free. Every model file on the owner's
machine was one of 2-4 NTFS hard links to the same data (fsutil hardlink
list): the data root's models folder, the shared Models folder,
LM Studio's library, the Hugging Face cache and an old checkout. Delete
unlinked only the data-root name, so the data stayed on disk, the Models
folder name came straight back as a "Discovered in Models folder" row, and
nothing ever reached the Recycle Bin.

So a Delete here finds every name the file has (file_names) and moves all of
them to the Recycle Bin together (send_to_recycle_bin). Emptying the bin then
frees the space. Nothing in this module deletes permanently.

Off Windows, file_names returns just the given path and send_to_recycle_bin
refuses, so a caller never silently falls back to a permanent delete.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

_MAX_PATH_CHARS = 32768

# shellapi.h
_FO_DELETE = 0x0003
_FOF_SILENT = 0x0004
_FOF_NOCONFIRMATION = 0x0010
_FOF_ALLOWUNDO = 0x0040
_FOF_NOERRORUI = 0x0400
# A file bigger than the Recycle Bin's size limit is otherwise destroyed
# without a word when FOF_NOCONFIRMATION is set - a 25 GB model easily is on
# a small drive. This flag makes Windows ask first, so the owner decides.
_FOF_WANTNUKEWARNING = 0x4000


def _key(path: str) -> str:
    # realpath, not abspath: the owner's shared Models folder is reached through
    # a junction, so the given path and FindFirstFileNameW's volume-relative
    # name were the same file under two spellings. Listing both made
    # SHFileOperationW move the first and then fail on the second with
    # code 0x2 (seen 2026-10-20 deleting a Models-folder file).
    return os.path.normcase(os.path.realpath(path))


def file_names(path: str | Path) -> list[str]:
    """Every hard-link name of the file at `path`, the given path first.

    Uses FindFirstFileNameW/FindNextFileNameW, which report names relative to
    the volume root, prefixed here with GetVolumePathNameW's answer. Any
    failure (not Windows, missing file, a file system without hard links)
    answers with just the given path.
    """
    given = str(path)
    names = [given]
    if sys.platform != "win32":
        return names
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.FindFirstFileNameW.restype = ctypes.c_void_p
        kernel32.FindFirstFileNameW.argtypes = [
            ctypes.c_wchar_p, ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong), ctypes.c_wchar_p,
        ]
        kernel32.FindNextFileNameW.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong), ctypes.c_wchar_p,
        ]
        kernel32.FindClose.argtypes = [ctypes.c_void_p]
        kernel32.GetVolumePathNameW.argtypes = [
            ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_ulong,
        ]

        full = os.path.abspath(given)
        volume = ctypes.create_unicode_buffer(_MAX_PATH_CHARS)
        if not kernel32.GetVolumePathNameW(full, volume, _MAX_PATH_CHARS):
            return names
        root = volume.value.rstrip("\\/")

        buffer = ctypes.create_unicode_buffer(_MAX_PATH_CHARS)
        length = ctypes.c_ulong(_MAX_PATH_CHARS)
        handle = kernel32.FindFirstFileNameW(full, 0, ctypes.byref(length), buffer)
        if handle in (None, ctypes.c_void_p(-1).value):
            return names
        try:
            found = [root + buffer.value]
            while True:
                length = ctypes.c_ulong(_MAX_PATH_CHARS)
                if not kernel32.FindNextFileNameW(handle, ctypes.byref(length), buffer):
                    break
                found.append(root + buffer.value)
        finally:
            kernel32.FindClose(handle)
    except (OSError, AttributeError, ValueError):
        return names

    seen = {_key(given)}
    for name in found:
        if _key(name) not in seen:
            seen.add(_key(name))
            names.append(name)
    return names


class _SHFILEOPSTRUCTW(ctypes.Structure):
    _fields_ = [
        ("hwnd", ctypes.c_void_p),
        ("wFunc", ctypes.c_uint),
        ("pFrom", ctypes.c_wchar_p),
        ("pTo", ctypes.c_wchar_p),
        ("fFlags", ctypes.c_ushort),
        ("fAnyOperationsAborted", ctypes.c_int),
        ("hNameMappings", ctypes.c_void_p),
        ("lpszProgressTitle", ctypes.c_wchar_p),
    ]


def send_to_recycle_bin(paths: list[str]) -> None:
    """Move files to the Recycle Bin in one operation, or raise OSError.

    Only regular files are accepted: a directory or a missing path is refused
    before anything moves, so one bad entry never leaves a half-done delete.
    """
    if not paths:
        return
    if sys.platform != "win32":
        raise OSError("the Recycle Bin is only available on Windows")
    unique: dict[str, str] = {}
    for path in paths:
        if not os.path.isfile(path):
            raise OSError(f"{path} is not a file")
        unique.setdefault(_key(path), path)
    paths = list(unique.values())
    operation = _SHFILEOPSTRUCTW()
    operation.wFunc = _FO_DELETE
    # A list of full paths, each NUL-terminated, ending with an extra NUL.
    operation.pFrom = "".join(os.path.abspath(p) + "\0" for p in paths) + "\0"
    operation.fFlags = (
        _FOF_ALLOWUNDO | _FOF_NOCONFIRMATION | _FOF_SILENT
        | _FOF_NOERRORUI | _FOF_WANTNUKEWARNING
    )
    shell32 = ctypes.WinDLL("shell32")
    shell32.SHFileOperationW.argtypes = [ctypes.POINTER(_SHFILEOPSTRUCTW)]
    code = shell32.SHFileOperationW(ctypes.byref(operation))
    if code != 0:
        raise OSError(f"Windows refused the move (code {code:#x})")
    if operation.fAnyOperationsAborted:
        raise OSError("the move was cancelled")
    left = [p for p in paths if os.path.exists(p)]
    if left:
        raise OSError(f"{left[0]} is still there")
