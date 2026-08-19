@echo off
REM ============================================================================
REM  FCC Coverage-Change Audit - one-command launcher (Windows / work laptop)
REM
REM  Double-click this file, or from a terminal:
REM     run.bat                     ->  full national batched run + final web build
REM     run.bat build-web           ->  rebuild web from completed national batches
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

REM 2) Run. With no arguments, use the crash-safe geographic batch runner.
REM    It validates all 50 states + DC, then builds the final web bundle once.
if "%~1"=="" (
    echo [run] full national pipeline: geographic batches + validated web build
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_overnight.ps1"
) else (
    ".venv\Scripts\python.exe" -m fcc_audit.cli %*
)
if errorlevel 1 (
    echo.
    echo ERROR: pipeline failed or produced an incomplete provider/service set.
    exit /b 1
)

echo.
echo Done. Outputs are in data\outputs\  Open the map with:
echo   .venv\Scripts\python.exe -m fcc_audit.cli serve
pause
endlocal
