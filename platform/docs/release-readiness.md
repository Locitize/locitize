# locitize 1.0.4 release readiness

Status: unsigned internal beta candidate. Not approved as a production release.
This work uses the repository build/test/package process; no vault workflow.

## Evidence collected on 2026-09-07

- Full regression suite: 1701 passed, 2 skipped. Existing fake-service shutdown
  fixtures emit a closed-log-stream diagnostic after the suite; no test failed.
- New integration code passes Ruff; staged changes pass whitespace checks.
- Install-tree write and machine-specific-path gates pass without exceptions.
- Live discovery: 663 local sessions across five detected tools, no reader errors
  in the measured scan (about five seconds).
- Qt functional actions cover search, transcript matches, notes, pins,
  archive/restore and preview. Rendered Home/Sessions layouts checked at 1366x850.
- Live desktop: Home opened and its Open Sessions button navigated successfully.
- Installer fixtures verify corruption/traversal rejection, successful staged
  installation, prior-version preservation and repeat-install refusal.
- Bundled Python 3.11.7, Qt and core library imports pass; bundled Tcl reports
  8.6.12. The native launcher compiles successfully.
- Latest workstation health still fails the RAM guard: 2591 MB available versus
  4096 MB required. All 16 registered model paths resolve. This is not a passed
  live inference/coding acceptance run.

The runtime version reflects the available local build environment. Updating to
a supported patched runtime and auditing its dependencies remain release work.

## Implemented

- One desktop: Home and native Sessions alongside existing model/chat capabilities.
- Seven local session readers; no AMP remote discovery.
- Provider-scoped persistent title, tags, notes, pin and archive state.
- Metadata search, cancellable bounded transcript search, inert preview/export.
- Non-destructive Session Portal annotation import and backup/restore.
- New/resumed local coding with project/model selection and readiness checks.
- Process-specific CLI configuration shared with the existing Chat launcher.
- Model reservation, explicit release, and close confirmation.
- Minimal user-reviewable version diagnostics and corrected privacy claims.
- Bundled Python/Qt Windows package, native launcher, verified side-by-side
  installation, per-version user environments, manifest and archive checksums.
- Automated coverage for annotations, migration, UI actions, launch validation,
  failure handling and model lifecycle protection.

## Build and distribution

Prepare the venv with requirements-core.lock and requirements-dev.txt. Track all
intended source files. From the repository root:

```powershell
.venv\Scripts\python platform/scripts/build_release.py --output dist/locitize-1.0.4-windows
```

The builder never downloads components or signs output. It refuses an existing
output directory. release-manifest.json records the source commit, dirty-source
flag, actual runtime versions and every packaged file hash. An uncommitted build
must be identified using its manifest; it must not be presented as a clean tag.

Extract the ZIP before installation. Run Install locitize.bat, then the new
shortcut. Portable use via locitize.exe is also supported. User data stays outside
the install tree. Older versions remain available for manual rollback.
Never distribute a QA directory that has been changed after manifest generation.

## Release gates still requiring evidence

1. Real inference plus a complete request/response through each supported coding
   CLI (new and resumed sessions), including tool-call compatibility. The current
   workstation had less than the required 4096 MB free RAM at the initial check.
2. Fresh Windows user/machine install, upgrade, rollback, restricted install-tree
   writes, optional dependency installs, and paths with spaces/non-ASCII text.
3. CPU-only and additional NVIDIA hardware checks; measured model performance.
   No model capability or speed should be inferred from successful loading.
4. Full optional voice, vision, chat, fine-tuning and phone microphone/autoplay
   acceptance. Existing tests are not substitute evidence for a new release.
5. Trusted publisher signing, clean dependency vulnerability review, license
   compliance bundle and review (including Qt obligations), and a distribution
   channel with authenticated release metadata.
6. Registered uninstall and a tested update/support policy; beta feedback from
   independent users and resolution of release-blocking defects.

The existing local self-signed certificate is not a trusted public publisher
identity. No public publishing, external beta invitations or production signing
has been performed.

## Product enhancements still open

- Recent sessions on Home and richer project organization.
- Explicit per-provider scan controls and scalable incremental transcript search.
- More precise detached-terminal lifecycle tracking.
- User-facing privacy/network inspection covering external tools, if pursued.
- Provider format compatibility fixtures for all seven tools.
- Model capability probes for coding tool use, guided recovery, and a simpler
  onboarding path for first-time users.
- Public download/support content, screenshots of the final release and release
  notes based on independent acceptance evidence.

Backlog entries stay visible until implemented or deliberately removed from the
release scope. The beta package is a concrete review artifact, not proof that
these remaining gates have passed.
