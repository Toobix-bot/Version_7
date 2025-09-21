Param(
  [string]$Port = '8000',
  [string]$Host = '127.0.0.1'
)
$ErrorActionPreference='Stop'
Write-Host '[+] Quick Start' -ForegroundColor Cyan
if (-not (Test-Path .venv)) {
  Write-Host '[INFO] Erstelle venv...' -ForegroundColor Yellow
  py -3 -m venv .venv
}
. ./.venv/Scripts/Activate.ps1
try { python -c "import groq" 2>$null } catch { Write-Host '[INFO] Installiere requirements...' -ForegroundColor Yellow; pip install -r requirements.txt }
# Respect .env secrets; do not force API_TOKENS in script. Uncomment for local default.
# if (-not $env:API_TOKENS) { $env:API_TOKENS='test' }
if (-not $env:LOG_LEVEL) { $env:LOG_LEVEL='INFO' }
Write-Host ("[START] http://{0}:{1}  Token=test" -f $Host,$Port) -ForegroundColor Green
python -m uvicorn api.app:app --host $Host --port $Port
