Param(
  [string]$Pattern = "tests/test_story.py",
  [switch]$Quiet
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

$env:PYTEST_CURRENT_TEST = '1'
if (-not $env:API_TOKENS) { $env:API_TOKENS = 'test' }

$argsList = @($Pattern)
if ($Quiet) { $argsList += '-q' }

Write-Host "Running: $pytest $($argsList -join ' ')" -ForegroundColor Cyan
& $pytest @argsList
$exit = $LASTEXITCODE
if ($exit -ne 0) { Write-Error "Tests failed ($exit)" }
exit $exit
