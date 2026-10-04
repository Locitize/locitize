# LOCITIZE GPU build switch (llama.cpp CUDA <-> CPU) - documented + testable.
#
# Runtime VRAM free (no rebuild):
#   LOCITIZE.bat --gpu
#   LOCITIZE.bat --gpu-free
#   Desktop header button: "Offload GPU"
#
# Build switch (this script):
#   .\platform\scripts\gpu_switch.ps1 -Target cuda          # dry-run (default)
#   .\platform\scripts\gpu_switch.ps1 -Target cpu -DryRun
#   .\platform\scripts\gpu_switch.ps1 -Target cuda -Apply   # replace binary
#
# After -Apply: restart via LOCITIZE.bat (boot order llama -> router -> OWUI -> portal),
# then .\platform\scripts\health.ps1

param(
    [ValidateSet("cuda", "cpu")]
    [string]$Target = "cuda",
    [switch]$Apply,
    [switch]$DryRun,
    [string]$Tag = ""
)

if (-not $Apply) { $DryRun = $true }

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
$lib = Join-Path $PSScriptRoot "gpu_switch_lib.py"
$argList = @($lib, "--target", $Target)
if ($Tag) { $argList += @("--tag", $Tag) }
if ($Apply) { $argList += "--apply" }

Write-Host "LOCITIZE GPU switch ($Target)  dry-run=$DryRun apply=$Apply"
& $py @argList
$code = $LASTEXITCODE
if ($DryRun -and -not $Apply) {
    Write-Host ""
    Write-Host "Dry-run only. Re-run with -Apply to replace the llama.cpp build under DATA_ROOT\bin."
    Write-Host "Verify afterwards:  .\platform\scripts\health.ps1"
    Write-Host "GPU status:         LOCITIZE.bat --gpu"
}
exit $code
