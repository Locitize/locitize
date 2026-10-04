# LOCITIZE production stack health (ops).
# Probes 127.0.0.1: 8080 llama, 8093 router, 8096 OWUI, 8091 whisper, 8092 kokoro, 4200 portal.
# Exit 0 only when every required endpoint PASSes. Does NOT fail on low RAM/VRAM
# (use python launcher.py --health for the full hardware ladder).

$ErrorActionPreference = "Stop"
$platformDir = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $platformDir

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
Push-Location $platformDir
try {
    & $py launcher.py --stack-health @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
