# LOCITIZE - start the platform.
# Prefer the project venv next to this repo. Default: desktop (production).
# Pass -Terminal for the interactive menu. Extra args forward to launcher.py.

param(
    [switch]$Desktop,
    [switch]$Terminal
)

$ErrorActionPreference = "Stop"
$platformDir = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $platformDir

function Resolve-LocitizePython {
    foreach ($c in @(
        (Join-Path $repoRoot ".venv\Scripts\python.exe"),
        (Join-Path $platformDir ".venv\Scripts\python.exe"),
        (Join-Path $platformDir "runtime\python.exe")
    )) { if (Test-Path -LiteralPath $c) { return $c } }
    return "python"
}

$py = Resolve-LocitizePython
$forward = @($args)
if ($Terminal) {
    $mode = @("--terminal")
} else {
    # Production default matches locitize.bat
    $mode = @("--desktop")
    $Desktop = $true
}

Push-Location $platformDir
try {
    Write-Host "Starting locitize via $py $($mode -join ' ') ..."
    & $py launcher.py @mode @forward
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
