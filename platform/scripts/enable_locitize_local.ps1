# enable_locitize_local.ps1 - OWNER-RUN, ELEVATED opt-in for the locitize.local hostname.
#
# The LOCITIZE platform NEVER edits the hosts file or binds a privileged port on its
# own (Permission Matrix section 7, Architecture G4). This script is the explicit,
# owner-executed, "Run as Administrator" opt-in that adds the loopback hosts entry
# so http://locitize.local:8085/ resolves. It performs no network access and touches
# only the single hosts line.
#
# Usage (from an elevated PowerShell):
#   powershell -ExecutionPolicy Bypass -File .\scripts\enable_locitize_local.ps1
#
# To reach chat WITHOUT any elevation at all, skip this entirely and use the Chat
# button (http://127.0.0.1:<port>/) - it always works and needs nothing.

$ErrorActionPreference = "Stop"

# Refuse to run unless truly elevated; we never silently attempt a privileged edit.
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "This script must be run as Administrator (right-click PowerShell -> Run as administrator)."
    Write-Host "No change was made. To use chat without elevation, use the Chat button instead."
    exit 1
}

$hostsPath = "$env:SystemRoot\System32\drivers\etc\hosts"
$hostname = "locitize.local"
$line = "127.0.0.1`t$hostname"

$existing = Get-Content -Path $hostsPath -ErrorAction SilentlyContinue
if ($existing -match "\s$hostname(\s|$)") {
    Write-Host "hosts already contains an entry for $hostname; no change made."
    exit 0
}

# Append the single loopback mapping. This is the ONLY line this script writes.
Add-Content -Path $hostsPath -Value $line
Write-Host "Added '$line' to $hostsPath."
Write-Host "You can now reach LOCITIZE chat at http://$hostname`:8085/ when the proxy is enabled."
