# Overnight national run (Windows / PowerShell).
#
# Runs the configured providers x services (config/pipeline.yaml) across all 50
# states + DC, in crash-safe STATE BATCHES: each batch completes and saves its
# own results, so if the machine dies or you run out of time, every finished
# batch is preserved. At the end it merges everything into the web bundle.
#
# Prereqs (one time): a .venv with `pip install -r requirements.txt`, and a .env
# with REDSHIFT_HOST/DB/USER/PASSWORD. Then just:  .\run_overnight.ps1
# (If PowerShell blocks it:  powershell -ExecutionPolicy Bypass -File .\run_overnight.ps1)

$ErrorActionPreference = "Continue"   # a bad batch must not abort the whole night
$env:PYTHONPATH = "src"
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }   # fall back to PATH python

# Largest states first, so the highest-population coverage lands earliest if you
# run short on time. All 50 states + DC, in batches of ~5.
$batches = @(
  "06,48,12,36,42",      # CA TX FL NY PA
  "17,39,13,37,26",      # IL OH GA NC MI
  "34,51,53,04,25",      # NJ VA WA AZ MA
  "47,18,29,55,08",      # TN IN MO WI CO
  "24,27,01,45,22",      # MD MN AL SC LA
  "21,40,09,28,05",      # KY OK CT MS AR
  "20,19,49,32,35",      # KS IA UT NV NM
  "54,31,16,15,23,41",   # WV NE ID HI ME OR
  "33,44,30,10,46",      # NH RI MT DE SD
  "38,02,50,56,11"       # ND AK VT WY DC
)

$start = Get-Date
$i = 0
foreach ($b in $batches) {
  $i++
  Write-Host ("=== [{0}/{1}] states {2}  (elapsed {3:hh\:mm})" -f `
    $i, $batches.Count, $b, ((Get-Date) - $start)) -ForegroundColor Cyan
  & $py -m fcc_audit.cli run --states $b
}

Write-Host "=== building web bundle ===" -ForegroundColor Cyan
& $py -m fcc_audit.cli build-web

Write-Host ("DONE in {0:hh\:mm}. Outputs in data\outputs\ ; serve web with: cd web; python -m http.server 8000" -f `
  ((Get-Date) - $start)) -ForegroundColor Green
