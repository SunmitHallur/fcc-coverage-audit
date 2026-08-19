@echo off
setlocal
cd /d "%~dp0"
set STATES=%~1
if "%STATES%"=="" set STATES=01,02

if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  .venv\Scripts\pip install -q -r requirements.txt
)
set PYTHONPATH=src

echo === Processing states: %STATES% ===
REM Quotes keep comma-separated FIPS as a single --states value (cmd can otherwise
REM split on commas and only the first state is processed).
REM Do NOT pass --build-web here: that would replace the national web site with
REM this batch only. Run build-web after ALL batches succeed.
".venv\Scripts\python.exe" -m fcc_audit.cli run --states "%STATES%" --workers 6
if errorlevel 1 (
  echo ERROR: batch failed or was incomplete.
  exit /b 1
)

echo === Done. Next steps ===
echo   After ALL national batches succeed:
echo     .venv\Scripts\python.exe -m fcc_audit.cli build-web
echo   Preview partial:
echo     .venv\Scripts\python.exe -m fcc_audit.cli build-web --allow-incomplete
echo   View locally:
echo     .venv\Scripts\python.exe -m fcc_audit.cli serve
