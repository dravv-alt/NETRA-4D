$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

$media = Join-Path $root 'tools\mediamtx\mediamtx.exe'
if (Test-Path -LiteralPath $media) {
    $mediaConfig = Join-Path $root 'mediamtx.yml'
    Start-Process -FilePath $media -ArgumentList "`"$mediaConfig`"" -WorkingDirectory $root -WindowStyle Hidden
}

Start-Process -FilePath 'python' -ArgumentList 'scripts/run_server.py' -WorkingDirectory $root -WindowStyle Hidden
Start-Process -FilePath 'npm.cmd' -ArgumentList 'run dev -- --host 127.0.0.1' -WorkingDirectory (Join-Path $root 'Frontend_Updated') -WindowStyle Hidden

Write-Host 'NETRA-4D API: http://127.0.0.1:8000/api/health'
Write-Host 'Frontend:     http://127.0.0.1:5173'
Write-Host 'RTSP ingest:  rtsp://127.0.0.1:8554/<path> (when MediaMTX binary is present)'
