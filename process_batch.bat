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

echo.
echo === Done. Next steps ===
echo   git add web/public/data config/pipeline.yaml
echo   git commit -m "Add batch results for states %STATES%"
echo   git push   # Vercel auto-deploys web/
