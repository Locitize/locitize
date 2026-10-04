# claude-local - run Claude Code against the model LOCITIZE is serving.
#
# Safe by scope: the Anthropic overrides live only in THIS process and the
# claude it starts, so plain `claude` in any other window still talks to
# Anthropic. Finds the running llama-server on LOCITIZE's loopback range
# (8080-8099), reads the loaded model's id and real context size from the
# server itself, and refuses honestly when no model is running.
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
    Write-Host "No LOCITIZE model is running."
    Write-Host "Start one on the Models page (or LOCITIZE.vbs), then run claude-local again."
    exit 1
}

Write-Host "Routing Claude Code to local model `"$model`" on 127.0.0.1:$port"
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:$port"
$env:ANTHROPIC_API_KEY = "locitize-local"
if ($ctx) { $env:CLAUDE_CODE_MAX_CONTEXT_TOKENS = "$ctx" }
# Same trimmed tool allowlist as the in-app harness (harness_launch.py):
# the full tool schema alone can overflow a small local context window.
& claude --model $model --strict-mcp-config --tools "Bash,Edit,Read,Write,Glob,Grep" @args
exit $LASTEXITCODE
