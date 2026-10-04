"""GPU llama.cpp build switch helpers (CUDA <-> CPU). Used by gpu_switch.ps1."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PLATFORM = Path(__file__).resolve().parent.parent
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))


def _probe(want_cuda: bool, tag: str = "") -> dict:
    from config import Config, resolve_data_dir
    from setup_env import find_llama_server, installed_llama_build, resolve_llama_release

    settings, _models, _issues = Config.load()
    data = resolve_data_dir(install_dir=settings.base_dir)
    bin_dir = Path(data) / "bin"
    current = settings.paths.llama_cpp or ""
    if not current:
        found = find_llama_server([bin_dir / "llama.cpp"])
        current = found or ""
    build = installed_llama_build(current) if current else None
    asset, err = resolve_llama_release(want_cuda, tag=tag)
    return {
        "data_dir": str(data),
        "bin_dir": str(bin_dir),
        "current_path": current,
        "current_build": build,
        "want_cuda": want_cuda,
        "error": err or "",
        "asset_name": asset.name if asset else "",
        "asset_size_mb": (asset.size // (1024 * 1024)) if asset else 0,
        "companions": [c.name for c in (asset.companions if asset else ())],
        "settings_llama": settings.paths.llama_cpp or "",
    }


def _apply(want_cuda: bool, tag: str = "") -> int:
    from config import Config, resolve_data_dir
    from setup_env import install_llama_cpp

    settings, _models, _issues = Config.load()
    data = resolve_data_dir(install_dir=settings.base_dir)
    bin_dir = Path(data) / "bin"
    target = bin_dir / "llama.cpp"
    if target.exists():
        stamp = __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = bin_dir / f"llama.cpp.bak-{stamp}"
        print(f"archiving {target} -> {backup}")
        shutil.move(str(target), str(backup))

    result, server = install_llama_cpp(
        bin_dir,
        want_cuda,
        confirm_unverified=True,
        tag=tag,
        say=print,
    )
    print(result.message)
    if not result.ok or not server:
        return 1

    # Persist into live settings.yaml via the existing setup_env writer.
    try:
        from setup_env import write_settings_paths

        repo = PLATFORM.parent
        py = repo / ".venv" / "Scripts" / "python.exe"
        if not py.is_file():
            py = Path(sys.executable)
        wr = write_settings_paths(py, data, {"llama_cpp": server})
        print(wr.message)
    except Exception as exc:  # noqa: BLE001
        print(f"INSTALLED_SERVER={server}")
        print(f"could not auto-write settings.yaml ({exc}); set paths.llama_cpp manually.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LOCITIZE llama.cpp CUDA/CPU switch")
    parser.add_argument("--target", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    want = args.target == "cuda"
    info = _probe(want, args.tag)
    if args.apply:
        if info.get("error"):
            print(json.dumps(info, indent=2))
            return 1
        print(json.dumps({k: info[k] for k in ("data_dir", "bin_dir", "current_path", "asset_name")}, indent=2))
        return _apply(want, args.tag)
    print(json.dumps(info, indent=2))
    return 1 if info.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
