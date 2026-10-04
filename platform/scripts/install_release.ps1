param([string]$InstallRoot = "", [switch]$NoShortcuts)
$ErrorActionPreference = 'Stop'
function Get-PackageHash([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}
$packageRoot = $PSScriptRoot
$manifestPath = Join-Path $packageRoot 'release-manifest.json'
if (-not (Test-Path -LiteralPath $manifestPath)) { throw 'Extract the complete locitize release first.' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.version -notmatch '^[0-9A-Za-z.-]+$') { throw 'Invalid release version.' }
if (-not $InstallRoot) { $InstallRoot = Join-Path $env:LOCALAPPDATA 'Programs\LOCITIZE' }
$installBase = [IO.Path]::GetFullPath($InstallRoot)
$installTarget = [IO.Path]::GetFullPath((Join-Path $installBase $manifest.version))
if (-not $installTarget.StartsWith($installBase.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid install destination.' }
if (Test-Path -LiteralPath $installTarget) { throw 'This version is already installed. Its files have been preserved.' }
# Validate every package file before installation. A digest checks integrity;
# it does not replace a trusted publisher signature.
foreach ($entry in $manifest.files.PSObject.Properties) {
    $sourcePath = [IO.Path]::GetFullPath((Join-Path $packageRoot $entry.Name))
    if (-not $sourcePath.StartsWith($packageRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid package path.' }
    if ((Get-PackageHash $sourcePath) -ne $entry.Value) { throw "Package integrity check failed: $($entry.Name)" }
}
$stagingTarget = [IO.Path]::GetFullPath((Join-Path $installBase ('.staging-' + [Guid]::NewGuid().ToString('N'))))
if (-not $stagingTarget.StartsWith($installBase.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid staging destination.' }
New-Item -ItemType Directory -Path $stagingTarget -Force | Out-Null
foreach ($entry in $manifest.files.PSObject.Properties) {
    $destination = Join-Path $stagingTarget $entry.Name
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $packageRoot $entry.Name) -Destination $destination
    if ((Get-PackageHash $destination) -ne $entry.Value) { throw 'Installed file failed integrity verification. Staging files were retained for diagnosis.' }
}
Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $stagingTarget 'release-manifest.json')
# Both resolved targets were checked under installBase before this directory move.
Move-Item -LiteralPath $stagingTarget -Destination $installTarget
if (-not $NoShortcuts) {
    $shell = New-Object -ComObject WScript.Shell
    foreach ($folder in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
        $shortcut = $shell.CreateShortcut((Join-Path $folder 'locitize.lnk'))
        $shortcut.TargetPath = Join-Path $installTarget 'locitize.exe'
        $shortcut.WorkingDirectory = $installTarget
        $shortcut.IconLocation = Join-Path $installTarget 'locitize.exe'
        $shortcut.Save()
    }
}
Write-Output "locitize $($manifest.version) installed at $installTarget"
Write-Output 'Models and session data stay in the existing locitize data directory.'
