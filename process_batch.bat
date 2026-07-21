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
".venv\Scripts\python.exe" -m fcc_audit.cli run --states "%STATES%" --cleanup-raw --build-web
if errorlevel 1 (
  echo ERROR: batch failed or was incomplete; web bundle was not rebuilt.
  exit /b 1
)

echo === Done. View the website locally ===
echo   cd web
echo   ..\.venv\Scripts\python.exe -m http.server 8000
echo   then open http://localhost:8000 in your browser
