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

Set-Location $PSScriptRoot
$ErrorActionPreference = "Continue"   # record failed batches, then fail the run
$env:PYTHONPATH = "src"
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  Write-Host "=== creating Python virtual environment ===" -ForegroundColor Cyan
  python -m venv .venv
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  & $py -m pip install --upgrade pip
  & $py -m pip install -r requirements.txt
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$log = Join-Path $PSScriptRoot ("overnight_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Start-Transcript -Path $log | Out-Null

# Geographic batches keep neighboring coverage together. The pipeline also adds
# a 50 km neighboring-state context halo, then saves only each batch's target
# states, reducing false cross-border tower/new-site classifications.
$batches = @(
  "04,06,15,32",            # AZ CA HI NV
  "02,16,41,53",            # AK ID OR WA
  "08,30,35,49,56",         # CO MT NM UT WY
  "20,31,38,40,46",         # KS NE ND OK SD
  "05,22,28,48",            # AR LA MS TX
  "17,19,27,29,55",         # IL IA MN MO WI
  "18,21,26,39,47",         # IN KY MI OH TN
  "01,12,13,37,45",         # AL FL GA NC SC
  "10,11,24,51,54",         # DE DC MD VA WV
  "09,23,25,33,34,36,42,44,50" # CT ME MA NH NJ NY PA RI VT
)

$start = Get-Date
$i = 0
$failedBatches = @()
foreach ($b in $batches) {
  $i++
  Write-Host ("=== [{0}/{1}] states {2}  (elapsed {3:hh\:mm})" -f `
    $i, $batches.Count, $b, ((Get-Date) - $start)) -ForegroundColor Cyan
  # Quote the batch string so commas stay in ONE --states arg. Unquoted, some
  # shells/arg parsers split "20,19,49" into argv tokens and only the first FIPS
  # (Kansas for this batch) is honored — looks "national" in the log, one state in results.
  & $py -m fcc_audit.cli run --states "$b" --cleanup-raw
  if ($LASTEXITCODE -ne 0) {
    $failedBatches += $b
    Write-Warning "Batch failed or was incomplete: $b (exit $LASTEXITCODE)"
  }
}

if ($failedBatches.Count -gt 0) {
  $failedList = $failedBatches -join "; "
  Write-Error "National run incomplete; refusing to build a partial web bundle. Failed batches: $failedList"
  Stop-Transcript | Out-Null
  exit 1
}

Write-Host "=== building web bundle ===" -ForegroundColor Cyan
& $py -m fcc_audit.cli build-web
if ($LASTEXITCODE -ne 0) {
  $code = $LASTEXITCODE
  Stop-Transcript | Out-Null
  exit $code
}

Write-Host ("DONE in {0:hh\:mm}. Outputs in data\outputs\ ; serve web with: cd web; python -m http.server 8000" -f `
  ((Get-Date) - $start)) -ForegroundColor Green
Write-Host "Log: $log" -ForegroundColor Green
Stop-Transcript | Out-Null
