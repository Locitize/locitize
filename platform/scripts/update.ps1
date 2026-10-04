# LOCITIZE - update entry point.
# Delegates to install_or_update.ps1 (single production path). Extra switches pass through.

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "install_or_update.ps1") @args
exit $LASTEXITCODE
