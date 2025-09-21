Param(
  [string]$Port = '8099',
  [string]$BindHost = '127.0.0.1',
  [switch]$NoReload
)
$ErrorActionPreference='Stop'
Write-Host "[+] Quick Start (Port $Port)" -ForegroundColor Cyan
if (-not (Test-Path .venv)) { Write-Host '[INFO] Erstelle venv...' -ForegroundColor Yellow; py -3 -m venv .venv }
. ./.venv/Scripts/Activate.ps1
try { python -c "import groq" 2>$null } catch { Write-Host '[INFO] Installiere requirements...' -ForegroundColor Yellow; pip install -r requirements.txt }
# Respect existing .env via python-dotenv in app; do not override secrets here.
# If you want a default for local dev, uncomment the next line:
# if (-not $env:API_TOKENS) { $env:API_TOKENS='test' }
if (-not $env:LOG_LEVEL) { $env:LOG_LEVEL='INFO' }
$reload = ''
if (-not $NoReload.IsPresent) {
  $reload = '--reload'
}
Write-Host ("[START] http://{0}:{1}  Token=test" -f $BindHost,$Port) -ForegroundColor Green
python -m uvicorn api.app:app --host $BindHost --port $Port $reload
