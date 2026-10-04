"""Build a self-contained Windows beta from tracked source and a local Python runtime.

No downloads or signing occur here. Release files are copied from the working
tree only when tracked by Git. Python/Qt and their notices are included; optional
AI stacks and model weights are not. An existing output is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "platform"))
from release_info import VERSION  # noqa: E402

PACKAGES = ("PySide6", "PySide6_Addons", "PySide6_Essentials", "shiboken6", "PyYAML", "psutil", "Pillow")


def copy_runtime(target: Path):
    base = Path(sys.base_prefix)
    runtime = target / "runtime"
    runtime.mkdir()
    for pattern in ("*.exe", "*.dll", "LICENSE*.txt"):
        for path in base.glob(pattern):
            shutil.copy2(path, runtime / path.name)
    for name in ("DLLs", "tcl", "include", "libs"):
        if (base / name).is_dir():
            shutil.copytree(base / name, runtime / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(base / "Lib", runtime / "Lib", ignore=shutil.ignore_patterns(
        "site-packages", "__pycache__", "*.pyc", "test", "tests", "idlelib"))
    packages = runtime / "Lib" / "site-packages"
    packages.mkdir()
    versions = {}
    for name in PACKAGES:
        dist = importlib.metadata.distribution(name)
        versions[name] = dist.version
        for record in dist.files or []:
            relative = Path(str(record))
            if ".." in relative.parts or "__pycache__" in relative.parts or relative.suffix == ".pyc":
                continue
            source = Path(dist.locate_file(record))
            if source.is_file():
                destination = packages / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    return versions


def build(output: Path, make_zip=True):
    if sys.platform != "win32":
        raise RuntimeError("Build Windows releases on Windows")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if make_zip and (output.parent / (output.name + ".zip")).exists():
        raise FileExistsError("The release archive already exists")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    output.mkdir(parents=True)
    for name in tracked:
        if not name:
            continue
        rel = Path(name)
        if not (name.startswith(("platform/", "finetune-studio/")) or name in ("README.md", "LICENSE", "THIRD_PARTY.md", "SECURITY.md")):
            continue
        if any(p in {"tests", "__pycache__", "locitize-data", "logs", "reports", "tmp"} for p in rel.parts):
            continue
        src = ROOT / rel
        if src.is_file():
            dst = output / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    versions = copy_runtime(output)
    compiler = Path(os.environ["WINDIR"]) / "Microsoft.NET" / "Framework64" / "v4.0.30319" / "csc.exe"
    subprocess.run([str(compiler), "/nologo", "/target:winexe", "/reference:System.Windows.Forms.dll",
                    f"/win32icon:{ROOT / 'platform' / 'assets' / 'locitize_launcher.ico'}",
                    f"/out:{output / 'LOCITIZE.exe'}", str(ROOT / "platform" / "scripts" / "LocitizeLauncher.cs")], check=True)
    shutil.copy2(ROOT / "platform" / "scripts" / "install_release.ps1", output / "Install LOCITIZE.ps1")
    (output / "Install LOCITIZE.bat").write_text(
        '@echo off\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install LOCITIZE.ps1"\nif errorlevel 1 pause\n', encoding="ascii")
    entry = (
        'Option Explicit\nDim shell, fso, root, command\n'
        'Set shell = CreateObject("WScript.Shell")\n'
        'Set fso = CreateObject("Scripting.FileSystemObject")\n'
        'root = fso.GetParentFolderName(WScript.ScriptFullName)\n'
        'command = """" & root & "\\runtime\\pythonw.exe"" -B -E -s """ & root & "\\platform\\release_entry.py"""\n'
        'shell.Run command, 0, False\n'
    )
    (output / "LOCITIZE.vbs").write_text(entry, encoding="ascii")
    (output / "Start LOCITIZE.bat").write_text(
        '@echo off\n"%~dp0runtime\\python.exe" -B -E -s "%~dp0platform\\release_entry.py" %*\n', encoding="ascii")
    manifest = {"version": VERSION, "channel": "unsigned-beta", "python": sys.version.split()[0],
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_dirty": bool(subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT)),
                "packages": versions, "files": {}}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                manifest["files"][path.relative_to(output).as_posix()] = hashlib.file_digest(handle, "sha256").hexdigest()
    (output / "release-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if make_zip:
        archive = output.parent / (output.name + ".zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as package:
            for path in output.rglob("*"):
                if path.is_file():
                    package.write(path, output.name + "/" + path.relative_to(output).as_posix())
        with archive.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / f"LOCITIZE-{VERSION}")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    result = build(args.output, not args.no_zip)
    print(json.dumps({"version": result["version"], "files": len(result["files"]), "output": str(args.output)}))
