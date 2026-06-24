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
".venv\Scripts\python.exe" -m fcc_audit.cli run --states %STATES% --cleanup-raw --build-web

echo === Done. View the website locally ===
echo   cd web
echo   ..\.venv\Scripts\python.exe -m http.server 8000
echo   then open http://localhost:8000 in your browser
