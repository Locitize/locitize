# claude-local - run Claude Code against the model locitize is serving.
$ErrorActionPreference = "SilentlyContinue"

$port = $null; $model = $null; $ctx = $null
foreach ($p in 8080..8099) {
    $h = $null
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$p/health" -TimeoutSec 1 } catch { continue }
    if (-not $h) { continue }
    try { $m = Invoke-RestMethod -Uri "http://127.0.0.1:$p/v1/models" -TimeoutSec 2 } catch { continue }
    $id = $m.data[0].id
    if (-not $id) { continue }
    $model = [IO.Path]::GetFileNameWithoutExtension(($id -split "[\\/]")[-1])
    try {
        $props = Invoke-RestMethod -Uri "http://127.0.0.1:$p/props" -TimeoutSec 2
        $ctx = $props.default_generation_settings.n_ctx
    } catch {}
    $port = $p
    break
}

if (-not $port) {
    Write-Host "No locitize model is running."
    Write-Host "Start one on the Models page (or locitize.vbs), then run claude-local again."
    exit 1
}

Write-Host "Routing Claude Code to local model `"$model`" on 127.0.0.1:$port"
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:$port"
$env:ANTHROPIC_API_KEY = "locitize-local"
if ($ctx) { $env:CLAUDE_CODE_MAX_CONTEXT_TOKENS = "$ctx" }
& claude --model $model --strict-mcp-config --tools "Bash,Edit,Read,Write,Glob,Grep" @args
exit $LASTEXITCODE
