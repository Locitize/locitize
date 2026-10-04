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
  ENABLE_LOGIN_FORM=False, FORWARDED_ALLOW_IPS=*, DATA_DIR) is set by
  platform/webui.py _build_openwebui_env on every launcher start — not by this script.
  Does NOT bind 0.0.0.0 or open Windows firewall. Does NOT enable Funnel.

.PARAMETER StatusOnly
  Print current serve status and probe URLs; do not change config.

.PARAMETER Reset
  Clear all serve handlers first, then apply the Locitize map.
#>
[CmdletBinding()]
param(
  [switch]$StatusOnly,
  [switch]$Reset
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

Write-Host "LOCITIZE Tailscale Serve"
Write-Host "MagicDNS: $magic"
Write-Host "Mode:     tailnet only (no Funnel)"
Write-Host ""

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
