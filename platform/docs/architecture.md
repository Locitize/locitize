# locitize architecture

This repository is the build source of truth for this release.

## Desktop and model lifecycle

desktop.py renders Qt views. gui_controller.py serializes operations through
command/result queues. services.py owns managed processes and ModelController;
config.py owns typed settings and atomic, comment-preserving configuration writes.
The existing model router uses the model requested by Open WebUI.

The Sessions page queues local coding launches through the same controller.
Readiness precedes terminal creation. A model reservation prevents a conflicting
switch or GUI stop while local terminals depend on that model. The user releases
the reservation after finishing those terminals. Reservations are conservative:
the launcher does not infer that a detached terminal has closed.

## Session integration

session_core/providers contains seven local-file readers adapted from Session
Portal at revision 1909c9eb98339e947f23885776bf3346602a355b (MIT). The legacy GUI,
standalone packaging, and remote AMP reader are not imported.

session_service.py handles discovery, bounded transcript reads, original resume
commands, and inert text export. sessions_ui.py performs discovery, previews,
transcript searches and import/export work on a two-worker Qt pool. Preview
generation tokens reject stale results; transcript search supports cancellation.

session_store.py owns schema-versioned SQLite annotations keyed by provider and
session ID. Source histories remain provider-owned. Imports validate before
writing, preserve existing records, and create a recovery database copy.
Backups cover annotations only; imports never relocate original transcripts.

session_launch.py constructs per-process Codex, Claude Code, and OpenCode local
configuration. Both Sessions and the existing Chat launcher use it. It does not
grant tool permissions or impose process-wide network isolation.

## Packaging

release_entry.py runs using a bundled ordinary Python interpreter so child
services retain their normal subprocess contracts. runtime_layout.py places
managed virtual environments in the user's data root, separated by release
version. Source checkouts retain their existing layout.

scripts/build_release.py copies tracked source, Python, core dependencies and
license metadata; compiles a small native launcher; and emits a manifest, ZIP and
SHA-256 file. Optional AI stacks and weights are not bundled. The installer verifies
files before and after copying, stages in a unique directory, then publishes the
versioned installation and shortcuts. Existing versions and user data are retained.

## Verification boundaries

Unit and Qt functional tests exercise state changes and failure paths.
Packaging and installation smoke checks cover the local Windows machine.
Hardware, actual CLI/model compatibility, clean-machine installation, publisher
signing, and optional full-stack flows require separate evidence; see
release-readiness.md.
