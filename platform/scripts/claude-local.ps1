# claude-local - run Claude Code against the model locitize is serving.
$ErrorActionPreference = "SilentlyContinue"
$installedDataRoot = ""

function Find-LlamaServer([int]$candidate) {
    if ($candidate -lt 1 -or $candidate -gt 65535) { return $null }
    try {
        $props = Invoke-RestMethod -Uri "http://127.0.0.1:$candidate/props" -TimeoutSec 2
        $ctx = $props.default_generation_settings.n_ctx
        if ($null -eq $ctx -or [int64]$ctx -le 0) { return $null }
        $models = Invoke-RestMethod -Uri "http://127.0.0.1:$candidate/v1/models" -TimeoutSec 2
        $id = $models.data[0].id
        if (-not $id) { return $null }
        return [PSCustomObject]@{
            Port = $candidate
            Model = [IO.Path]::GetFileNameWithoutExtension(($id -split "[\\/]")[-1])
            Context = [int64]$ctx
        }
    } catch { return $null }
}

$server = $null
$stateFiles = @()
if ($env:LOCITIZE_DATA_DIR) {
    $stateFiles += Join-Path $env:LOCITIZE_DATA_DIR "run\model-service.json"
}
if ($installedDataRoot) {
    $stateFiles += Join-Path $installedDataRoot "run\model-service.json"
}
if ($env:LOCALAPPDATA) {
    $stateFiles += Join-Path $env:LOCALAPPDATA "LOCITIZE\run\model-service.json"
}
$sourcePlatform = Split-Path -Parent $PSScriptRoot
if (Test-Path (Join-Path $sourcePlatform "launcher.py")) {
    $stateFiles += Join-Path $sourcePlatform "run\model-service.json"
    $stateFiles += Join-Path $sourcePlatform "locitize-data\run\model-service.json"
}

foreach ($stateFile in ($stateFiles | Select-Object -Unique)) {
    if (-not (Test-Path $stateFile -PathType Leaf)) { continue }
    try {
        $state = Get-Content -Raw $stateFile | ConvertFrom-Json
        $server = Find-LlamaServer ([int]$state.running_port)
    } catch { $server = $null }
    if ($server) { break }
}

if (-not $server) {
    foreach ($candidate in 8080..8099) {
        $server = Find-LlamaServer $candidate
        if ($server) { break }
    }
}

if (-not $server) {
    Write-Host "No locitize model is running."
    Write-Host "Start one on the Models page (or locitize.vbs), then run claude-local again."
    exit 1
}

Write-Host "Routing Claude Code to local model `"$($server.Model)`" on 127.0.0.1:$($server.Port)"
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:$($server.Port)"
$env:ANTHROPIC_API_KEY = "locitize-local"
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"
$env:CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL = "1"
$env:DISABLE_TELEMETRY = "1"
$env:DISABLE_ERROR_REPORTING = "1"
$env:DISABLE_AUTOUPDATER = "1"
$env:CLAUDE_CODE_MAX_CONTEXT_TOKENS = "$($server.Context)"
& claude --model $server.Model --strict-mcp-config --tools "Bash,Edit,Read,Write,Glob,Grep" @args
exit $LASTEXITCODE
