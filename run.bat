@echo off
REM ============================================================================
REM  FCC Coverage-Change Audit - one-command launcher (Windows / work laptop)
REM
REM  Double-click this file, or from a terminal:
REM     run.bat                     ->  download + analyze ALL providers/techs
REM     run.bat download            ->  only pre-fetch raw data from the FCC API
REM     run.bat run --current 2025-12-31 --prior 2025-06-30
REM
REM  It creates a local virtual environment, installs dependencies once, then
REM  runs the pipeline. Re-running reuses the same environment.
REM ============================================================================
setlocal
cd /d "%~dp0"

REM 1) Create the virtual environment on first run.
if not exist ".venv\Scripts\python.exe" (
    echo [setup] creating virtual environment...
    python -m venv .venv || (echo ERROR: could not create venv. Is Python installed and on PATH? & pause & exit /b 1)
    echo [setup] installing dependencies...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo ERROR: pip install failed. & pause & exit /b 1)
)

set "PYTHONPATH=src"

REM 2) Run. With no arguments, do a full national run that deletes each raw file
REM    after processing so disk usage stays bounded.
if "%~1"=="" (
    echo [run] full pipeline: all providers, all configured technologies
    ".venv\Scripts\python.exe" -m fcc_audit.cli run --cleanup-raw
) else (
    ".venv\Scripts\python.exe" -m fcc_audit.cli %*
)

echo.
echo Done. Outputs are in data\outputs\  (open dashboard\index.html for the map)
pause
endlocal
