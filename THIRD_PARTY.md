# Third-party software

LOCITIZE's own code is licensed as described in [LICENSE](LICENSE). The desktop
beta bundles Python, PySide6/Qt, shiboken6, PyYAML, psutil and Pillow from the build
environment. Their distribution metadata and license files accompany the package.
Session readers derived from Session Portal are vendored under session_core,
with their MIT license in platform/session_core/LICENSE. Other optional components
arrive during setup and remain governed by their upstream licenses.

This inventory is not completed legal clearance for public distribution. The
release gate includes the Qt license/source obligations and dependency review.
See https://doc.qt.io/qt-6/licensing.html and https://docs.openwebui.com/license/.

| Component | Role in LOCITIZE | Obtained | Upstream license (see upstream) |
|---|---|---|---|
| llama.cpp (`llama-server`) | Model serving engine (the built-in inference backend) | Downloaded at setup from the official GitHub releases, digest-verified | MIT |
| whisper.cpp | Speech-to-text | Downloaded at setup from the official GitHub releases | MIT |
| Kokoro (weights + voices) | Text-to-speech | Downloaded at setup from the official distribution | Apache-2.0 (weights per upstream) |
| Open WebUI | Optional rich chat frontend | Installed at setup via `pip install open-webui` into a dedicated venv | Upstream's own license (BSD-3-style with branding conditions in recent releases) |
| Python | Packaged interpreter | Bundled from the build environment | PSF license and bundled component notices |
| Session Portal readers | Local session discovery | Vendored from revision 1909c9eb98339e947f23885776bf3346602a355b | MIT; notice preserved in session_core/LICENSE |
| PySide6 / Qt / shiboken6 | Desktop UI toolkit | Bundled dynamically linked libraries | See included license metadata and upstream Qt licensing; component terms vary |
| psutil | System memory readings | Installed via pip | BSD-3-Clause |
| PyYAML | Config parsing | Installed via pip | MIT |
| Pillow | Image handling (vision fixtures, icons) | Installed via pip | MIT-CMU |
| torch (CPU build) | Kokoro text-to-speech runtime | Installed via pip (pinned CPU wheel) | BSD-3-Clause |
| kokoro (pip package) | Text-to-speech engine code | Installed via pip (pulls its own stack) | Apache-2.0 |
| Model weights (GGUF files) | The models a user runs | Chosen and downloaded by the user (their own files, or via Hugging Face search they explicitly initiate) | Each model's own license; LOCITIZE ships **no** model weights and no model list |

Principles this repository holds to:

- **Bundled attribution.** The desktop runtime and Session Portal readers retain
  their notices. platform/assets/kokoro/config.json is an unmodified Apache-2.0
  architecture descriptor; it is not a model weight file.
- **Isolation.** Heavy third-party stacks (Open WebUI, the fine-tune studio) live
  in dedicated virtual environments so their dependency trees never mix with the
  platform's.
- **No bundled weights.** LOCITIZE ships no models and names no models; users
  bring their own or download what they choose, under those models' licenses.
- **Attribution stands.** Nothing in LOCITIZE removes or obscures any upstream
  project's name, license, or notices.

If you believe any entry here is inaccurate or incomplete, please open an issue.
