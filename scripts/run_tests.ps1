Param(
  [string]$Pattern,
  [switch]$Quiet,
  [switch]$All,
  [switch]$Smoke,
  [switch]$Coverage
)

$ErrorActionPreference = 'Stop'

$venv = Join-Path (Join-Path $PSScriptRoot '..') '.venv'
$venvActivate = Join-Path (Join-Path $venv 'Scripts') 'Activate.ps1'
if (-not (Test-Path $venvActivate)) {
  Write-Error "Virtualenv nicht gefunden: $venvActivate"
}
. $venvActivate

$pytest = Join-Path (Join-Path $venv 'Scripts') 'pytest.exe'
if (-not (Test-Path $pytest)) {
  Write-Host 'Installiere pytest...'
  pip install pytest -q
}

if (-not $env:API_TOKENS) { $env:API_TOKENS = 'test' }

# Determine target set
$target = ''
if ($All) { $target = 'tests' }
elseif ($Smoke) { $target = 'tests/test_smoke_sync.py' }
elseif ($Pattern) { $target = $Pattern }
else { $target = 'tests' }

$argsList = @($target)
if ($Quiet) { $argsList += '-q' }
if ($Coverage) { $argsList = @('--maxfail','1','--disable-warnings','--cov=api','--cov-report=term-missing') + $argsList }

Write-Host "Running: $pytest $($argsList -join ' ')" -ForegroundColor Cyan
& $pytest @argsList
$exit = $LASTEXITCODE
if ($exit -ne 0) { Write-Error "Tests failed ($exit)" }
exit $exit
