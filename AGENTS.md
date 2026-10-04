# LOCITIZE - agent guide

You are an AI agent asked to set up, run, or work on LOCITIZE for a user.
This file is your runbook. Everything here is executable as written, on Windows.

## What this is

A Windows-first local AI platform: it serves the user's own GGUF models through
llama.cpp (OpenAI-compatible, loopback-only), with voice (whisper.cpp in,
Kokoro out), vision, a fine-tune studio, optional Open WebUI chat, and a Qt
desktop that manages it all. Nothing leaves the machine unless the user asks;
the egress ledger records LOCITIZE's own outbound connections (installers,
Open WebUI and launched coding tools make their own - see SECURITY.md).

- Requirements: Windows 10/11, Python 3.11+, ~2 GB disk for the toolchain.
  NVIDIA GPU optional but recommended (CUDA build auto-selected when present).
- Layout: everything lives under `platform/`. User data (settings.yaml,
  models.yaml, downloaded binaries, logs) lives in a data root: the
  `LOCITIZE_DATA_DIR` env var if set, else `platform/locitize-data/` when that
  directory exists, else `%LOCALAPPDATA%\LOCITIZE` (the fresh-install default).
  The install tree is never written at runtime.

## Set it up for the user

**Preferred: the guided wizard (user clicks, you supervise).**
Tell the user to double-click `platform\LOCITIZE.vbs`. On a fresh machine it
opens the setup wizard, which installs the venv and dependencies, downloads the
right llama.cpp build for the GPU (digest-verified), finds the GGUF models the
user already has on disk and registers them, measures each model's best context
on the actual GPU, and drops a Desktop shortcut. `LOCITIZE.vbs --setup` reopens
the wizard later to add features.

**Headless (you do it; no GUI):** three commands from the repository root.

```bat
python -m venv .venv
.venv\Scripts\pip install -r platform\requirements.txt PySide6
.venv\Scripts\python platform\scripts\agent_setup.py
```

`agent_setup.py` performs the wizard's core sequence with the same functions
the wizard uses: seeds the config, finds or downloads the GPU-matched
llama.cpp build (digest-verified), discovers + registers the GGUF models
already on the machine (hardlinked, never copied), and installs and enables
Open WebUI as the chat app (separately licensed; skip it with
`--no-openwebui`). It is idempotent - re-run it any time; existing pieces are
detected and skipped. Other optional features (voice, vision, fine-tune
studio) are added later via `LOCITIZE.vbs --setup`.

## Verify the setup

```bat
.venv\Scripts\python platform\launcher.py --health --json
```

Exit 0, with every probe reporting `"status": "PASS"`, means the platform is healthy. Then prove
end-to-end serving with one real model start (picks the model, waits for
readiness, cleans up):

```bat
.venv\Scripts\python platform\launcher.py --service-status --json
.venv\Scripts\python platform\launcher.py --smoke-start <model-id> --json
```

The full acceptance harness (real chat completion, TTS, discovery, privacy
ledger) is:

```bat
cd platform && ..\.venv\Scripts\python scripts\acceptance.py
```

## Run it

- Desktop app (what the user wants): `platform\LOCITIZE.vbs` (no console) or
  `platform\LOCITIZE.bat` (visible console, for debugging).
- Everything is also CLI-driven via `platform\launcher.py`: `--health`,
  `--benchmark --model <id>`, `--gpu` / `--gpu-free` (see what holds VRAM, free
  LOCITIZE's own), `--rtx-report` (measured per-GPU compatibility matrix),
  `--egress` (the privacy ledger).

## Rules when changing code

- Run the tests: `cd platform && ..\.venv\Scripts\python -m pytest -q`.
  The suite (1300+ tests) must pass; it includes gates that refuse machine-specific
  paths and untracked imports in the shipped tree.
- Never hand-edit `models.yaml`/`settings.yaml` with string surgery: go through
  the chokepoint writers (`config.append_model_entry`,
  `config.write_model_tuning`, `setup_env.write_settings_paths`, ...) - they are targeted, atomic, and
  comment-preserving.
- Measured, not assumed: anything performance-shaped (context sizes, tok/s)
  gets measured on the real machine (`scripts/measure_context_ceilings.py`),
  never computed from a formula.
- Shipped code and UI text are plain ASCII. No emojis, no smart quotes.
- Outbound network only through `modelhub.open_checked` (host-allowlisted,
  ledger-recorded). Do not add a second HTTP path.
