@echo off
setlocal EnableExtensions

rem NETRA-4D local launcher: API + frontend + optional MediaMTX RTSP relay.
rem Run this file from Explorer or a Command Prompt. Each service gets its own window.

set "ROOT=%~dp0"
cd /d "%ROOT%"

if exist "%ROOT%.venv_gpu\Scripts\python.exe" (
    set "NETRA_PYTHON=%ROOT%.venv_gpu\Scripts\python.exe"
) else if exist "C:\Python312\python.exe" (
    set "NETRA_PYTHON=C:\Python312\python.exe"
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [NETRA] Python 3.12+ was not found. Install Python or create .venv_gpu.
        exit /b 1
    )
    set "NETRA_PYTHON=python"
)

where npm.cmd >nul 2>&1
if errorlevel 1 (
    echo [NETRA] npm was not found on PATH.
    exit /b 1
)

if not exist "%ROOT%Frontend_Updated\node_modules" (
    echo [NETRA] Frontend dependencies are missing.
    echo         Run: cd Frontend_Updated ^&^& npm install
    exit /b 1
)

if exist "%ROOT%tools\mediamtx\mediamtx.exe" (
    start "NETRA-4D MediaMTX" /D "%ROOT%" "%ROOT%tools\mediamtx\mediamtx.exe" "%ROOT%mediamtx.yml"
)

start "NETRA-4D API" /D "%ROOT%" "%NETRA_PYTHON%" "%ROOT%scripts\run_server.py"
start "NETRA-4D Frontend" /D "%ROOT%Frontend_Updated" npm.cmd run dev -- --host 127.0.0.1

echo.
echo NETRA-4D services are starting in separate windows.
echo API:      http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:5173
if exist "%ROOT%tools\mediamtx\mediamtx.exe" echo RTSP:     rtsp://127.0.0.1:8554/ ^<stream-name^>
echo.
echo Close the API and Frontend windows to stop those services.
endlocal
