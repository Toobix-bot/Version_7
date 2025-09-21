Param(
  [string]$Port = '8099',
  [string]$BindHost = '127.0.0.1'
)
$ErrorActionPreference = 'Stop'

# Prefer venv python, fallback to py launcher
$env:UI_AUTO_KEY_FROM_ENV = if ($env:UI_AUTO_KEY_FROM_ENV) { $env:UI_AUTO_KEY_FROM_ENV } else { '1' }
$venvPy = Join-Path (Resolve-Path .).Path ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
  $exe = $venvPy
  $args = @('-m','uvicorn','api.app:app','--host',$BindHost,'--port',$Port)
} else {
  $exe = 'py'
  $args = @('-3','-m','uvicorn','api.app:app','--host',$BindHost,'--port',$Port)
}

Write-Host ("[DETACHED START] http://{0}:{1} (using .env secrets)" -f $BindHost,$Port) -ForegroundColor Green

# Start a new minimized console window; do not inherit current console lifecycle
Start-Process -FilePath $exe -ArgumentList $args -WindowStyle Minimized | Out-Null