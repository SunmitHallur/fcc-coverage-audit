# Overnight national run (Windows / PowerShell).
#
# Runs the configured providers x services (config/pipeline.yaml) across all 50
# states + DC, in crash-safe STATE BATCHES: each batch completes and saves its
# own results, so if the machine dies or you run out of time, every finished
# batch is preserved. At the end it merges everything into the web bundle.
#
# Prereqs (one time): a .venv with `pip install -r requirements.txt`, and a .env
# with REDSHIFT_HOST/DB/USER/PASSWORD. Then just:  .\run_overnight.ps1
# Optional: .\run_overnight.ps1 -Publish   # commit + push web/public/data
# (If PowerShell blocks it:  powershell -ExecutionPolicy Bypass -File .\run_overnight.ps1)

param(
  [switch]$Publish,
  [string]$Backend = $(if ($env:FCC_AUDIT_BACKEND) { $env:FCC_AUDIT_BACKEND } else { "redshift" })
)

Set-Location $PSScriptRoot
$ErrorActionPreference = "Continue"   # record failed batches, then fail the run
$env:PYTHONPATH = "src"
$py = ".\.venv\Scripts\python.exe"
# Overnight national path is Redshift-first (config defaults to files for offline).
$backendArgs = @("--backend", $Backend)
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

# National Redshift prefetch once; batches then --skip-prefetch.
Write-Host "=== National prefetch (download) backend=$Backend ===" -ForegroundColor Cyan
& $py -m fcc_audit.cli @backendArgs download
if ($LASTEXITCODE -ne 0) {
  Write-Error "National prefetch failed; refusing overnight batches"
  Stop-Transcript | Out-Null
  exit $LASTEXITCODE
}

$i = 0
$failedBatches = @()
foreach ($b in $batches) {
  $i++
  Write-Host ("=== [{0}/{1}] states {2}  (elapsed {3:hh\:mm})" -f `
    $i, $batches.Count, $b, ((Get-Date) - $start)) -ForegroundColor Cyan
  # Quote the batch string so commas stay in ONE --states arg. Unquoted, some
  # shells/arg parsers split "20,19,49" into argv tokens and only the first FIPS
  # (Kansas for this batch) is honored — looks "national" in the log, one state in results.
  # Caches warm from national download; --workers 6 = unit-level CPU parallelism.
  & $py -m fcc_audit.cli @backendArgs run --states "$b" --workers 6 --skip-prefetch
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

Write-Host "=== building web bundle backend=$Backend ===" -ForegroundColor Cyan
& $py -m fcc_audit.cli @backendArgs build-web
if ($LASTEXITCODE -ne 0) {
  $code = $LASTEXITCODE
  Stop-Transcript | Out-Null
  exit $code
}

Write-Host ("DONE in {0:hh\:mm}. Outputs in data\outputs\ ; serve with: python -m fcc_audit.cli serve" -f `
  ((Get-Date) - $start)) -ForegroundColor Green
Write-Host "Log: $log" -ForegroundColor Green

if ($Publish) {
  Write-Host "=== -Publish: committing and pushing web bundle ===" -ForegroundColor Cyan
  git add web/public/data
  git commit -m ("Final overnight web bundle {0}" -f (Get-Date -Format "yyyy-MM-dd"))
  if ($LASTEXITCODE -eq 0) {
    git push origin HEAD
  }
} else {
  Write-Host "Skipping git publish (pass -Publish to commit + push web/public/data)." -ForegroundColor Yellow
}

Stop-Transcript | Out-Null
