# locitize

Run local AI and continue your coding work from one Windows desktop.

locitize combines model management, local coding sessions, chat, voice, vision,
and fine-tuning. Session Portal's local readers now live inside locitize's native
Sessions page. There is one product to install.

## Install the unsigned beta

1. Extract the complete locitize release ZIP to a folder.
2. Run **Install locitize.bat**. It verifies the included file hashes, installs
   under your user account, and creates Desktop and Start menu shortcuts.
3. Open **locitize**. The first-run wizard guides model discovery and optional
   component installation. No separate Python installation is needed.
4. Open Models, select an installed GGUF, and start it. Check health before
   launching a coding session.

The beta is unsigned. File hashes detect corruption but do not prove publisher
identity. Production release checks are listed in
[release readiness](platform/docs/release-readiness.md).

Windows 10/11 x64 is the target. A compatible NVIDIA GPU is optional; available
RAM, VRAM and model size determine what can run. locitize preserves its measured
memory guards. Model weights and optional voice/chat/training stacks are separate
downloads, so their disk requirements are additional to the desktop package.

## Your daily workflow

- **Home:** choose models, sessions, or chat.
- **Models:** discover, download, tune, benchmark, start, and stop GGUF models.
- **Sessions:** find locally stored histories from Codex, Claude Code, OpenCode,
  Gemini, Qwen, Copilot, and Grok. Search metadata or bounded transcript previews,
  pin sessions, add titles/tags/notes, archive, restore, or export text.
- **New coding session / Resume with local model:** choose Codex, Claude Code,
  or OpenCode, a project folder, and a registered model. locitize waits for model
  readiness before opening the terminal. These tools must be installed separately.
- **Resume original:** uses that tool's original backend settings, which may
  reach a cloud service. It is distinct from local-model resume.
- **Release model:** use after finishing local coding terminals to allow model
  switching again. Keep locitize open while the terminals need its model.
- **Chat, Talk, Voice Setup, Vision, Memory, Fine-tune:** retain the existing
  capabilities and their optional dependencies.

Local coding launches use process-specific settings. They do not rewrite the
tools' global configuration or disable their permission prompts. Open WebUI's
request-selected model remains authoritative unless a coding session has reserved
the running model.

## Session migration and data

Sessions > Import / backup > Import Session Portal details imports titles,
notes, tags, pins, and archived state from a selected legacy data folder.
Ambiguous cross-provider identities are skipped. Original histories and legacy
files are preserved. Annotation imports keep existing locitize edits.

Session backups contain locitize annotations; they do not back up the original
tools' transcripts. Exported transcripts can contain sensitive information.
locitize does not upload session histories during discovery or search.

Fresh installs use %LOCALAPPDATA%\locitize. LOCITIZE_DATA_DIR overrides this;
source checkouts also recognize platform/locitize-data. Packaged optional Python
environments live in the data directory, separated by release version.

Updates install side by side and redirect shortcuts. To roll back, open the
previous version's locitize.exe. Remove an old version's installation directory
only after closing it; your data directory is separate. This beta does not yet
provide a registered Windows uninstaller or automatic updates.

## Develop and verify

From the repository root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r platform/requirements-core.lock -r platform/requirements-dev.txt
.venv\Scripts\python platform/launcher.py --health --json
Set-Location platform
..\.venv\Scripts\python -m pytest -q
```

The optional full voice stack remains in platform/requirements.txt.
Source desktop entry: platform\locitize.vbs.
Release build: run platform\scripts\build_release.py using the prepared venv on
Windows; see [release instructions](platform/docs/release-readiness.md).
New source modules must be tracked before building, because the package uses
Git's tracked file list.

See [architecture](platform/docs/architecture.md),
[security boundaries](SECURITY.md), and [third-party notices](THIRD_PARTY.md).
