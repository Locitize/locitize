"""GGUF export without Docker: merge a LoRA adapter, convert with llama.cpp's own
convert_hf_to_gguf.py, quantize with the llama-quantize that locitize installed.

Everything here is plain standard library so it can be unit tested without torch.
The merge step (which needs torch) lives in train.py and is passed in as a
callable.

Version discipline: the converter script and the quantizer must come from the SAME
llama.cpp release, because GGUF metadata keys and tensor layouts move between
builds. Setup installs a pinned release (setup_env.LLAMA_CPP_TAG). This module
asks the installed llama-quantize/llama-server which build it is (`--version`
prints "build NNNNN") and fetches the converter source for exactly that tag,
including the matching gguf-py library, so no separately versioned `gguf` pip
package can drift against it. PINNED_TAG is only the fallback when the build
number cannot be read; a test keeps it equal to setup_env.LLAMA_CPP_TAG.
"""

import io
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

PINNED_TAG = "b11406"
LLAMA_REPO = "ggml-org/llama.cpp"
QUANTS = ("f16", "q8_0", "q4_k_m")

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class ExportError(Exception):
    """Export could not finish; the message is plain language for the UI."""


def data_root():
    explicit = (os.environ.get("LOCITIZE_DATA_DIR") or "").strip()
    if explicit:
        return explicit
    local = (os.environ.get("LOCALAPPDATA") or "").strip()
    if local:
        return os.path.join(local, "LOCITIZE")
    return os.path.join(os.path.expanduser("~"), ".locitize")


def _exe(name):
    return name + (".exe" if os.name == "nt" else "")


def find_llama_quantize(extra_dirs=()):
    """Locate llama-quantize: explicit env var, then the folder locitize's setup
    installs llama.cpp into (<data>/bin/llama.cpp), then PATH."""
    explicit = (os.environ.get("LOCITIZE_LLAMA_QUANTIZE") or "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit
    name = _exe("llama-quantize")
    roots = [*extra_dirs, os.path.join(data_root(), "bin", "llama.cpp")]
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        direct = os.path.join(root, name)
        if os.path.isfile(direct):
            return direct
        for dirpath, _dirs, files in os.walk(root):
            if name in files:
                return os.path.join(dirpath, name)
    return shutil.which("llama-quantize") or ""


def installed_build(quantize_path):
    """The llama.cpp build number of the installed binaries, or None.

    llama-quantize has no --version, so ask its sibling llama-server (same
    archive, same build)."""
    server = os.path.join(os.path.dirname(quantize_path), _exe("llama-server"))
    if not os.path.isfile(server):
        return None
    try:
        proc = subprocess.run(
            [server, "--version"], capture_output=True, text=True, timeout=60,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"build\s+(\d+)", (proc.stdout or "") + (proc.stderr or ""))
    return int(match.group(1)) if match else None


def converter_tag(quantize_path):
    build = installed_build(quantize_path) if quantize_path else None
    return f"b{build}" if build else PINNED_TAG


def converter_dir(tag, root=None):
    return os.path.join(root or os.path.join(data_root(), "finetune", "tools"),
                        f"llama.cpp-{tag}")


def converter_url(tag):
    return f"https://codeload.github.com/{LLAMA_REPO}/zip/refs/tags/{tag}"


def _wanted_member(relative):
    # Recent llama.cpp split the converter into a sibling `conversion/` package
    # (convert_hf_to_gguf.py fails with "No module named 'conversion'" without it).
    return relative == "convert_hf_to_gguf.py" or (
        relative.endswith(".py")
        and (relative.startswith("gguf-py/gguf/") or relative.startswith("conversion/"))
    )


def extract_converter(archive_bytes, dest):
    """Unpack only convert_hf_to_gguf.py, conversion/ and gguf-py/gguf/*.py from a tag archive.
    Returns the converter script path."""
    os.makedirs(dest, exist_ok=True)
    found = False
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            relative = info.filename.split("/", 1)[1] if "/" in info.filename else ""
            if not _wanted_member(relative):
                continue
            target = os.path.normpath(os.path.join(dest, *relative.split("/")))
            if not os.path.abspath(target).startswith(os.path.abspath(dest) + os.sep):
                continue  # never write outside dest
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            found = found or relative == "convert_hf_to_gguf.py"
    script = os.path.join(dest, "convert_hf_to_gguf.py")
    if not found or not os.path.isfile(script):
        raise ExportError("The llama.cpp download did not contain the GGUF converter.")
    return script


def ensure_converter(tag, root=None, opener=None, say=print):
    """Return the path of convert_hf_to_gguf.py for `tag`, downloading it once."""
    dest = converter_dir(tag, root)
    script = os.path.join(dest, "convert_hf_to_gguf.py")
    if os.path.isfile(script) and os.path.isdir(os.path.join(dest, "gguf-py", "gguf"))             and os.path.isdir(os.path.join(dest, "conversion")):
        return script
    say(f"Downloading the llama.cpp {tag} GGUF converter (one time, about 30 MB)")
    call = opener or urllib.request.urlopen
    try:
        with call(converter_url(tag), timeout=120) as resp:
            data = resp.read()
    except Exception as exc:  # noqa: BLE001 - any network failure is one message
        raise ExportError(
            f"Could not download the GGUF converter for llama.cpp {tag}: {exc}. "
            "Check your internet connection and try again."
        ) from exc
    return extract_converter(data, dest)


def build_convert_cmd(python_exe, script, model_dir, outfile, outtype="f16"):
    return [python_exe, script, model_dir, "--outfile", outfile, "--outtype", outtype]


def build_quantize_cmd(quantize_exe, src_gguf, dst_gguf, quant):
    return [quantize_exe, src_gguf, dst_gguf, quant.upper()]


def _run(cmd, say, what):
    say(f"{what}: {' '.join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd))}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", creationflags=_NO_WINDOW,
    )
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tail.append(line)
            tail = tail[-12:]
    code = proc.wait()
    if code != 0:
        raise ExportError(f"{what} failed (exit {code}): " + " | ".join(tail[-4:]))


def export_gguf(merged_dir, out_dir, run_name, quant, python_exe=None,
                quantize_exe=None, say=print, runner=None, converter=None):
    """Convert a merged HF model dir to GGUF. Returns the final .gguf path.

    `runner(cmd, say, what)` and `converter` are injectable for tests."""
    if quant not in QUANTS:
        raise ExportError(f"Unknown GGUF type {quant!r}; choose one of {', '.join(QUANTS)}.")
    run = runner or _run
    python_exe = python_exe or sys.executable
    f16 = os.path.join(out_dir, f"{run_name}.f16.gguf")
    final = os.path.join(out_dir, f"{run_name}.{quant}.gguf")

    quantize_exe = quantize_exe or find_llama_quantize()
    if quant != "f16" and not quantize_exe:
        raise ExportError(
            "llama-quantize was not found. It is installed with llama.cpp by locitize "
            "setup; run setup again or choose f16 (no quantizing needed)."
        )
    script = converter or ensure_converter(converter_tag(quantize_exe), say=say)
    run(build_convert_cmd(python_exe, script, merged_dir, f16, "f16"), say,
        "Converting to GGUF (f16)")
    if quant == "f16":
        return f16
    run(build_quantize_cmd(quantize_exe, f16, final, quant), say, f"Quantizing to {quant}")
    try:
        os.remove(f16)
    except OSError:
        pass
    return final
