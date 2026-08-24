$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$binary = Join-Path $root 'tools\mediamtx\mediamtx.exe'
$config = Join-Path $root 'mediamtx.yml'

if (-not (Test-Path -LiteralPath $binary)) {
    throw "MediaMTX binary missing at $binary. Place mediamtx.exe there, then rerun this script."
}

& $binary $config

