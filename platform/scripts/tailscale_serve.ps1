<#
.SYNOPSIS
  Idempotent Tailscale Serve for Locitize (tailnet only - no Funnel).

.DESCRIPTION
  Phone-friendly HTTPS (locked 2026-09-24):
    Locitize / Open WebUI (main) -> https://<magicdns>/          (127.0.0.1:8096)
    Locitize / Open WebUI (alt)  -> https://<magicdns>:4443/     (127.0.0.1:8096)
    Agent Portal (own link)      -> https://<magicdns>:8443/     (127.0.0.1:4200)

  Locitize keeps the root URL. Agent Portal has its own port. Do NOT put Portal on :443.
  OWUI boot env (WEBUI_URL=https://<magicdns> root, WEBUI_AUTH=False, cookies Secure,
  ENABLE_LOGIN_FORM=False, FORWARDED_ALLOW_IPS=127.0.0.1, DATA_DIR) is set by
  platform/webui.py _build_openwebui_env on every launcher start — not by this script.
  Does NOT bind 0.0.0.0 or open Windows firewall. Does NOT enable Funnel.

  SECURITY: Open WebUI runs with no login. Serving it makes EVERY device on your
  tailnet (including shared nodes) an Open WebUI admin, and an Open WebUI admin
  can run code on this PC through Tools/Functions. Applying the map therefore
  requires -AllowNoLogin. Restrict access with tailnet ACLs, and never share this
  machine's node with people you would not give your PC to.

.PARAMETER StatusOnly
  Print current serve status and probe URLs; do not change config.

.PARAMETER Reset
  Clear all serve handlers first, then apply the Locitize map.

.PARAMETER AllowNoLogin
  Required to apply the map: confirms you accept that every tailnet device gets
  admin access to Open WebUI (see SECURITY above).
#>
[CmdletBinding()]
param(
  [switch]$StatusOnly,
  [switch]$Reset,
  [switch]$AllowNoLogin
)

$ErrorActionPreference = "Stop"

function Assert-Tailscale {
  if (-not (Get-Command tailscale -ErrorAction SilentlyContinue)) {
    throw "tailscale CLI not found on PATH"
  }
  $st = & tailscale status --json | ConvertFrom-Json
  if (-not $st.BackendState -or $st.BackendState -ne "Running") {
    $raw = & tailscale status 2>&1 | Out-String
    if ($raw -notmatch "100\.") {
      throw "tailscale does not appear to be up. Run: tailscale up"
    }
  }
}

function Get-MagicDns {
  $st = & tailscale status --json | ConvertFrom-Json
  $dns = $st.Self.DNSName
  if (-not $dns) { throw "Could not read MagicDNS name from tailscale status" }
  return $dns.TrimEnd(".")
}

function Set-LocitizeServe {
  param([string]$MagicDns)

  if ($Reset) {
    & tailscale serve reset
    if ($LASTEXITCODE -ne 0) { throw "tailscale serve reset failed (exit $LASTEXITCODE)" }
  }

  # Locitize / Open WebUI on root — do not steal this for Agent Portal
  & tailscale serve --bg --yes --https=443 "http://127.0.0.1:8096"
  if ($LASTEXITCODE -ne 0) { throw "failed to serve Open WebUI on :443 (exit $LASTEXITCODE)" }

  & tailscale serve --bg --yes --https=4443 "http://127.0.0.1:8096"
  if ($LASTEXITCODE -ne 0) { throw "failed to serve Open WebUI on :4443 (exit $LASTEXITCODE)" }

  # Agent Portal — own link only
  & tailscale serve --bg --yes --https=8443 "http://127.0.0.1:4200"
  if ($LASTEXITCODE -ne 0) { throw "failed to serve portal on :8443 (exit $LASTEXITCODE)" }

  # Drop leftover workaround ports if present
  & tailscale serve --https=9443 off 2>$null | Out-Null
}

function Test-ServeUrl {
  param([string]$Url, [string]$ExpectHint)
  try {
    $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 15
    $ok = $r.StatusCode -eq 200
    $hint = if ($ExpectHint -and ($r.Content -match $ExpectHint)) { " match=$ExpectHint" } else { "" }
    return "[PASS] $Url -> $($r.StatusCode)$hint"
  } catch {
    return "[FAIL] $Url -> $($_.Exception.Message)"
  }
}

Assert-Tailscale
$magic = Get-MagicDns
$owuiRoot = "https://$magic/"
$owuiAlt = "https://${magic}:4443/"
$portalPhone = "https://${magic}:8443/"

Write-Host "locitize Tailscale Serve"
Write-Host "MagicDNS: $magic"
Write-Host "Mode:     tailnet only (no Funnel)"
Write-Host ""

if (-not $StatusOnly -and -not $AllowNoLogin) {
  Write-Warning ("Open WebUI has no login. Serving it gives every device on your tailnet " +
    "admin access, which can run code on this PC. Re-run with -AllowNoLogin to accept " +
    "that, and restrict the tailnet with ACLs.")
  exit 1
}

if (-not $StatusOnly) {
  Set-LocitizeServe -MagicDns $magic
  Write-Host "Applied serve map:"
  Write-Host "  $owuiRoot     -> 127.0.0.1:8096  (Locitize / Open WebUI - main)"
  Write-Host "  $owuiAlt -> 127.0.0.1:8096  (Locitize / Open WebUI - alt)"
  Write-Host "  $portalPhone -> 127.0.0.1:4200  (Agent Portal - own link)"
  Write-Host ""
}

Write-Host "=== tailscale serve status ==="
& tailscale serve status
Write-Host ""

$json = & tailscale serve status --json | ConvertFrom-Json
if ($json.AllowFunnel -and (@($json.AllowFunnel.PSObject.Properties).Count -gt 0)) {
  Write-Host "[WARN] AllowFunnel entries present - disable Funnel (tailnet-only required)."
}

Write-Host "=== probes ==="
$lines = @(
  (Test-ServeUrl -Url $owuiRoot -ExpectHint "."),
  (Test-ServeUrl -Url $owuiAlt -ExpectHint "."),
  (Test-ServeUrl -Url $portalPhone -ExpectHint "Agent Portal"),
  (Test-ServeUrl -Url "${portalPhone}api/health" -ExpectHint "needs_login|ok")
)
$lines | ForEach-Object { Write-Host $_ }

$failed = @($lines | Where-Object { $_ -like "[FAIL]*" }).Count
Write-Host ""
Write-Host "Phone (same tailnet):"
Write-Host "  Locitize:  $owuiRoot"
Write-Host "  Portal:    $portalPhone"
Write-Host ""
if ($failed -gt 0) {
  Write-Host "Overall: FAIL ($failed probe(s))"
  exit 1
}
Write-Host "Overall: PASS"
exit 0
