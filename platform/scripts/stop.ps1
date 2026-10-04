# LOCITIZE - stop helper for production restart.
# Frees GPU/listeners owned by LOCITIZE (same classification as Offload GPU),
# then asks any launcher.py --desktop process to exit. Does not kill foreign apps.

$ErrorActionPreference = "Continue"
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
Push-Location $platformDir
try {
    Write-Host "Freeing LOCITIZE GPU holders (launcher --gpu-free) ..."
    & $py launcher.py --gpu-free
} catch {
    Write-Warning "gpu-free: $_"
}

Write-Host "Stopping launcher.py --desktop processes (if any) ..."
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.CommandLine -and
        ($_.CommandLine -match 'launcher\.py') -and
        ($_.CommandLine -match '--desktop')
    } |
    ForEach-Object {
        Write-Host ("  stopping pid {0}" -f $_.ProcessId)
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

Write-Host "stop complete. Remaining listeners (if any) may be orphans; check health.ps1."
Pop-Location
