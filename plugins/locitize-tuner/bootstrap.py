"""Install only the standalone tuning helper into an isolated per-user root.

Python is a prerequisite. Runtime binaries use LoCiTiZe's checked/digest path;
existing GGUF weights are registered in place, never uploaded or copied.
"""
from __future__ import annotations

import argparse
import json
import os
import platform  # Load stdlib before adding a bundled runtime to sys.path.
from pathlib import Path
import shutil
import subprocess
import sys
import venv


def data_dir():
    """Resolve a helper-specific root; never inherit the full app's data root.

    The default lives in the user profile, not AppData: the Codex desktop app is
    a Store (MSIX) app, and Windows redirects its helpers' AppData writes into
    the app's private package folder, where the Codex CLI cannot see them.
    """
    configured = os.environ.get("LOCITIZE_TUNER_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".locitize-tuner"


def legacy_data_dirs():
    """Earlier AppData installs: the plain path, then Codex's redirected copy."""
    local = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local")))
    found = [local / "LOCITIZE-Tuner"]
    found += sorted(local.glob("Packages/OpenAI.Codex_*/LocalCache/Local/LOCITIZE-Tuner"))
    return [path for path in found if (path / "models.yaml").is_file() and (path / "environment").is_dir()]


def migrate_legacy(root):
    """Copy an earlier install into the default root once; the old copy stays.

    Staged then renamed, so an interrupted copy never looks like an install.
    Lock files are process-local and are not carried over.
    """
    if os.environ.get("LOCITIZE_TUNER_DATA_DIR") or root.exists():
        return None
    sources = legacy_data_dirs()
    if not sources:
        return None
    staging = root.with_name(root.name + ".migrating")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(sources[0], staging, ignore=shutil.ignore_patterns("*.lock"))
    staging.rename(root)
    return sources[0]


def reconfigure(root, package):
    """Point saved runtime paths at the migrated root; stdout stays untouched."""
    command = [str(helper_python(root)), str(package / "bootstrap.py"), "--configure", "--skip-discovery"]
    result = subprocess.run(command, cwd=package, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError("Migrated helper could not be reconfigured: " + result.stderr[-500:])


def helper_python(root):
    """Return the private interpreter installed by this helper's bootstrap."""
    return root / "environment" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def prepare(root, package, skip_runtime=False, skip_discovery=False, roots=None):
    """Reuse checked acquisition and atomic registry writers inside private data."""
    runtime = package / "runtime"
    os.environ["LOCITIZE_DATA_DIR"] = str(root)
    sys.path.insert(0, str(runtime))
    import config
    import setup_env
    import modelhub
    import egress_log

    config.ensure_user_config(root, runtime)
    egress_log.configure(root)
    server = ""
    if not skip_runtime:
        gpu, _ = setup_env.detect_nvidia()
        result, server = setup_env.install_llama_cpp(root / "bin", gpu,
            confirm_unverified=False, local_only=True,
            say=lambda message: print(message, file=sys.stderr))
        if not result.ok or not server:
            raise RuntimeError("Runtime acquisition failed: " + result.message)
        written = setup_env.write_settings_paths(sys.executable, root, {"llama_cpp": server})
        if not written.ok:
            raise RuntimeError("Could not save acquired runtime settings")
    imported = paired = 0
    if not skip_discovery:
        for entry in setup_env.find_local_models(roots=roots):
            identity = modelhub.registry_id_for(entry["name"])
            # Strict same-folder pairing; the projector also stays in place.
            projector = setup_env.pair_mmproj(entry["path"])
            _, registered, _ = config.Config.load(root)
            existing = next((model for model in registered.models if model.id == identity), None)
            if existing is not None:
                # Attach a projector found later; never replace one already set.
                if projector and not existing.mmproj:
                    config.write_model_mmproj(root, identity, projector)
                    paired += 1
                continue
            # Preserve original location; fitting/tuning is a later measured job.
            config.append_model_entry(root, identity, Path(entry["name"]).stem,
                entry["path"], context_size=4096, gpu_layers=999,
                description="Discovered local model for tuning helper.",
                mmproj=projector)
            imported += 1
            paired += bool(projector)
    return {"state": "ready", "registered_models": imported,
            "vision_projectors_paired": paired,
            "runtime_acquired": bool(server),
            "runtime_setup_skipped": skip_runtime,
            "discovery_skipped": skip_discovery}


def main(argv=None):
    """Create private dependencies, then run configuration in that interpreter."""
    parser = argparse.ArgumentParser(description="Set up the standalone LoCiTiZe helper")
    parser.add_argument("--skip-runtime", action="store_true", help="Inspect only; model serving will remain unavailable")
    parser.add_argument("--skip-discovery", action="store_true", help="Leave the private registry empty")
    parser.add_argument("--models-root", action="append", help="Local folder to scan; weights stay in place")
    parser.add_argument("--configure", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11 or newer is required")
    package = Path(__file__).resolve().parent
    if not (package / "runtime" / "local_tuning_mcp.py").is_file():
        raise RuntimeError("Use the built standalone archive; its runtime payload is missing")
    root = data_dir()
    if not args.configure and migrate_legacy(root) is not None:
        print("Moved the existing helper install to " + str(root), file=sys.stderr)
    if args.configure:
        print(json.dumps(prepare(root, package, args.skip_runtime, args.skip_discovery, args.models_root)))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    python = helper_python(root)
    if not python.is_file():
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(root / "environment")
    # Reuse the existing toolchain installer, not a second custom HTTP client.
    sys.path.insert(0, str(package / "runtime"))
    import setup_env
    result = setup_env.pip_install(python, ["-r", str(package / "requirements-helper.txt")],
                                 on_output=lambda line: print(line, file=sys.stderr))
    if not result.ok:
        raise RuntimeError("Helper dependency installation failed: " + result.message)
    command = [str(python), str(package / "bootstrap.py"), "--configure"]
    if args.skip_runtime:
        command.append("--skip-runtime")
    if args.skip_discovery:
        command.append("--skip-discovery")
    for folder in args.models_root or []:
        command.extend(["--models-root", folder])
    return subprocess.call(command, cwd=package)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
