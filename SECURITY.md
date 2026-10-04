# Security boundaries

LOCITIZE-managed inference services bind to loopback. A separately configured
proxy such as Tailscale Serve can expose a service beyond the machine; LOCITIZE
does not treat that proxy as a loopback privacy guarantee.

LOCITIZE's model/release download path uses modelhub.open_checked, a host
allowlist, redirect checks, and an egress ledger. Verified downloads are checked
against available publisher hashes. Explicitly accepted unverifiable downloads
are identified as such.

Package installers, optional pip/npm environments, vendor coding CLI installers,
browsers, Open WebUI plugins, and launched coding tools can make their own network
requests. The LOCITIZE egress ledger is not a machine-wide firewall or a complete
record of those processes. Pointing inference at localhost does not disable a
tool's telemetry, extensions, authentication, or network tools.

The Sessions page reads existing local histories when opened. It does not invoke
a remote discovery CLI; AMP remote discovery is excluded. Transcript text is
untrusted data and rendered as plain text. Reading a transcript never executes
instructions in it. Export uses a fenced text block.

Annotation storage is separate from provider histories. Archive is a LOCITIZE
view preference, not deletion of source files. Backups and exports can contain
private notes and source content; the user selects where to save them.
SQLite storage is not encrypted by LOCITIZE and uses the current Windows user's
filesystem access. No analytics or crash-upload endpoint is added by this release.

Local coding launch arguments validate identities and project existence.
Per-process configuration selects the local endpoint without rewriting global
provider settings. Existing CLI permission prompts remain enabled.
Resume original deliberately retains the provider's own backend, possibly cloud.

The unsigned beta's manifest protects against accidental corruption when its
manifest is trusted; it is not publisher authentication. Public distribution
still needs signing and dependency/license review.

Report reproducible security findings using the repository's private security
advisory facility where available. Avoid publishing real session histories,
credentials, or personal paths in a public issue.
