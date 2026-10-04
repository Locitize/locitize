# LOCITIZE single install/update path (production).
#
# Canonical roots (resolve at runtime; never hard-code a drive letter):
#   DEV_ROOT  = $env:LOCITIZE_DEV_ROOT or the git checkout root
#   Alias     = $env:SystemDrive\locitize (optional junction -> DEV_ROOT)
#   DATA_ROOT = <DEV_ROOT>\platform\locitize-data
#               (or %LOCITIZE_DATA_DIR% when set)
#   BIN_ROOT  = <DATA_ROOT>\bin
#
# Resolves python relative to this script (repo\.venv, platform\.venv,
# bundled runtime, else PATH). No manual venv hunting.
#
# Usage:
#   .\platform\scripts\install_or_update.ps1
#   .\platform\scripts\install_or_update.ps1 -NoPull
#   .\platform\scripts\install_or_update.ps1 -Restart
#   .\platform\scripts\install_or_update.ps1 -SkipDeps

param(
    [switch]$NoPull,
    [switch]$SkipDeps,
    [switch]$Restart,
    [switch]$SkipHealth
)

$ErrorActionPreference = "Stop"
$platformDir = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $platformDir

Write-Host "LOCITIZE install/update"
Write-Host "  DEV_ROOT     = $repoRoot"
Write-Host "  platform     = $platformDir"
if ($env:LOCITIZE_DATA_DIR) {
    Write-Host "  DATA_ROOT    = $env:LOCITIZE_DATA_DIR  (LOCITIZE_DATA_DIR)"
} else {
    $portable = Join-Path $platformDir "locitize-data"
    if (Test-Path -LiteralPath $portable) {
        Write-Host "  DATA_ROOT    = $portable  (portable locitize-data)"
    } else {
        Write-Host "  DATA_ROOT    = $env:LOCALAPPDATA\LOCITIZE  (default)"
    }
}

function Resolve-LocitizePython {
    $candidates = @(
        (Join-Path $repoRoot ".venv\Scripts\python.exe"),
        (Join-Path $platformDir ".venv\Scripts\python.exe"),
        (Join-Path $platformDir "runtime\python.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) { return $c }
    }
    return "python"
}

$py = Resolve-LocitizePython
Write-Host "  python       = $py"

if (-not $NoPull) {
    if (Test-Path -LiteralPath (Join-Path $repoRoot ".git")) {
        Write-Host ""
        Write-Host "[git] fetch + fast-forward pull ..."
        Push-Location $repoRoot
        try {
            git fetch --quiet origin 2>$null
            git pull --ff-only
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "git pull --ff-only failed (local commits or diverged). Continuing with deps."
            }
        } finally {
            Pop-Location
        }
    } else {
        Write-Host ""
        Write-Host "[git] no .git at DEV_ROOT - skip pull (packaged install?)"
    }
}

if (-not $SkipDeps) {
    $req = Join-Path $platformDir "requirements.txt"
    if (-not (Test-Path -LiteralPath $req)) { throw "missing $req" }
    Write-Host ""
    Write-Host "[deps] pip install -r requirements.txt ..."
    Push-Location $platformDir
    try {
        & $py -m pip install --upgrade -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
}

if ($Restart) {
    Write-Host ""
    Write-Host "[restart] stop then start ..."
    & (Join-Path $PSScriptRoot "stop.ps1")
    Start-Sleep -Seconds 2
    & (Join-Path $PSScriptRoot "start.ps1") -Desktop
}

if (-not $SkipHealth) {
    Write-Host ""
    Write-Host "[health] stack liveness ..."
    & (Join-Path $PSScriptRoot "health.ps1")
    $healthCode = $LASTEXITCODE
    if ($healthCode -ne 0) {
        Write-Warning "stack health reported failures (exit $healthCode). Start desktop via LOCITIZE.bat if services are down."
        exit $healthCode
    }
}

Write-Host ""
Write-Host "install/update complete."
