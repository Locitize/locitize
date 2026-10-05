"""Headless setup for AI agents (M18.3): the wizard's work, no GUI.

The setup wizard is a tkinter window a person clicks through. An AI agent
setting locitize up for a user has no mouse - so this script performs the same
core sequence headlessly, reusing the exact setup_env/config functions the
wizard calls (never a parallel implementation):

  1. seed the data root (settings.yaml + models.yaml from the shipped templates)
  2. find or install llama.cpp (GPU-matched, digest-verified download) and
     record its path in settings.yaml
  3. discover the GGUF models already on this machine and register them
     (hardlinked into the models dir - no copies; re-runs never duplicate)
  4. install Open WebUI (the rich chat app) into its own .webui-venv and
     enable it, as the wizard does when its Open WebUI box is ticked (the
     default). Open WebUI is separately licensed third-party software; pass
     --no-openwebui to skip it.

It then makes sure the shared model store folder exists and installs the
claude-local shim (claude-local.cmd/.ps1 in ~/.local/bin), which runs Claude
Code against the model locitize is serving.

Run it from the platform venv AFTER `pip install -r requirements.txt`:

    .venv\\Scripts\\python platform\\scripts\\agent_setup.py

Idempotent: every step detects what already exists and skips it, so re-running
is always safe. Exit 0 = usable install (a machine with no models still exits 0
- models can be imported or downloaded later); exit 1 = a hard step failed.

Other optional features (voice, vision, fine-tune studio) stay wizard/
user-driven - add them later with locitize.vbs --setup. ASCII only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM_DIR))

import config  # noqa: E402
import modelhub  # noqa: E402
import setup_env  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent_setup.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-openwebui",
        action="store_true",
        help="skip installing Open WebUI (chat then uses llama.cpp's built-in UI)",
    )
    args = parser.parse_args(argv)
    say = print
    # Packaged install: run inside the app's own platform venv (created from the
    # bundled runtime), exactly as release_entry.py does for the GUI.
    from runtime_layout import ensure_platform_venv
    venv_python = ensure_platform_venv(PLATFORM_DIR)
    if venv_python and Path(sys.prefix).resolve() != venv_python.parent.parent.resolve():
        import subprocess
        return subprocess.call([str(venv_python), "-B", "-E", "-s", str(Path(__file__).resolve()),
                                *(argv if argv is not None else sys.argv[1:])])
    data_root = setup_env.default_data_root(setup_env.BASE_DIR)

    # -- 1. seed config ------------------------------------------------------
    actions = config.ensure_user_config(data_root, setup_env.BASE_DIR)
    say(f"[1/4] config seeded at {data_root} "
        f"({', '.join(a.kind for a in actions) or 'already present'})")

    # -- 2. llama.cpp --------------------------------------------------------
    has_gpu, gpu_name = setup_env.detect_nvidia()
    say(f"      GPU: {gpu_name or 'none detected (CPU build will be used)'}")
    server = setup_env.find_llama_server()
    if server:
        say(f"[2/4] llama.cpp found: {server}")
    else:
        result, server = setup_env.install_llama_cpp(
            data_root / "bin", has_gpu, confirm_unverified=False, say=say
        )
        if not server:
            say(f"[2/4] FAILED: {result.message}")
            return 1
        say(f"[2/4] llama.cpp installed: {server}")
    written = setup_env.write_settings_paths(
        sys.executable, data_root, {"llama_cpp": server}
    )
    if not written.ok:
        say(f"      FAILED to record path: {written.message}")
        return 1

    # -- 3. the user's models ------------------------------------------------
    found = setup_env.find_local_models(say=say)
    models_dir = data_root / "models"
    imported = skipped = failed = 0
    if not found:
        say("[3/4] no local GGUF models found; use the Models page (or the "
            "user's own files) to add some")
    for entry in found:
        location = setup_env.place_into_models_dir(entry["path"], models_dir)
        mmproj_src = setup_env.pair_mmproj(entry["path"])
        mmproj = (setup_env.place_into_models_dir(mmproj_src, models_dir)
                  if mmproj_src else "")
        result = setup_env.register_model_via_venv(
            sys.executable,
            {
                "data_root": str(data_root),
                "mmproj": mmproj,
                "model_id": modelhub.registry_id_for(entry["name"]),
                "name": Path(entry["name"]).stem,
                "location": location,
                "description": "Imported by headless agent setup.",
                "notes": f"Found at {entry['path']} during agent setup scan.",
            },
        )
        if result.ok:
            imported += 1
        elif "already exists" in result.message:
            skipped += 1
        else:
            failed += 1
            say(f"      {entry['name']}: {result.message[:100]}")
    if found:
        say(f"[3/4] models: {imported} imported, {skipped} already registered, "
            f"{failed} failed")

    # -- 4. Open WebUI -------------------------------------------------------
    # Not a hard step: the core install works without it (chat falls back to
    # llama.cpp's built-in UI), so a failure is reported, not fatal.
    if args.no_openwebui:
        say("[4/4] Open WebUI skipped (--no-openwebui)")
    else:
        say("[4/4] installing Open WebUI (large download; separately licensed "
            "third-party software)")
        webui_venv = setup_env.REPO_DIR / ".webui-venv"
        made = setup_env.create_venv(webui_venv)
        if not made.ok:
            say(f"      FAILED to create {webui_venv}: {made.message}")
        else:
            webui_exe = webui_venv / ("Scripts/python.exe" if sys.platform == "win32"
                                      else "bin/python")
            installed = setup_env.pip_install(
                webui_exe, setup_env.PIP_SETS["webui_pip"]
            )
            if not installed.ok:
                say(f"      FAILED: {installed.message[-200:]}")
            else:
                enabled = setup_env.enable_openwebui_via_venv(
                    sys.executable, data_root
                )
                say(f"      {'ok' if enabled.ok else 'note'}: {enabled.message}")

    say("")
    say("Done. Verify with: launcher.py --health --json")
    store = setup_env.ensure_model_store(say)
    say(f"model store: {store.message}")
    shim = setup_env.install_claude_local(say)
    say(f"claude-local: {shim.message}")
    say("Launch the desktop with: locitize.vbs")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
