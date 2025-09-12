Param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Reload,
    [string]$ApiKey = "test",
    [string]$LogFile = "logs/app.log",
    [switch]$Wait,
    [switch]$Verbose
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $root

if (-not (Test-Path .venv)) { Write-Error 'Virtualenv (.venv) nicht gefunden.' }

# Aktivieren
$venvActivate = Join-Path $root '.venv/Scripts/Activate.ps1'
. $venvActivate

$env:PYTHONPATH = $root
if ($ApiKey) { $env:API_TOKEN = $ApiKey }
if ($LogFile) { $env:LOG_FILE = $LogFile }

# Auto load .env (simple KEY=VALUE lines)
$envPath = Join-Path $root ".env"
if (Test-Path $envPath) {
    Get-Content $envPath | ForEach-Object {
        if ($_ -match '^[#;]') { return }
        if ($_ -match '^(?<k>[A-Za-z_][A-Za-z0-9_]*)=(?<v>.*)$') {
                $k=$matches['k']; $v=$matches['v'].Trim('"')
                if (-not [string]::IsNullOrWhiteSpace($k)) { $env:$k = $v }
        }
    }
}

$reloadFlag = ''
if ($Reload.IsPresent) { $reloadFlag = '--reload' }

Write-Host ("Starte Server http://{0}:{1} (Reload={2}) LogFile={3}" -f $BindHost,$Port,$Reload.IsPresent,$LogFile)

try {
    $python = Join-Path $root '.venv/ScriptS/python.exe'
    if (-not (Test-Path $python)) { $python = Join-Path $root '.venv/Scripts/python.exe' }
    if (-not (Test-Path $python)) { throw "Python executable nicht gefunden unter .venv/Scripts" }

    $argsList = @('-m','uvicorn','api.app:app','--host', $BindHost,'--port', $Port)
    if ($Reload.IsPresent) { $argsList += '--reload' }

    if ($Verbose) {
        Write-Host "Python: $python"
        Write-Host ('Args : ' + ($argsList -join ' '))
    }

    & $python @argsList
} catch {
    Write-Host "[ERROR] $_" -ForegroundColor Red
} finally {
    if ($Wait) { Write-Host 'Beenden mit Enter...' ; [void][System.Console]::ReadLine() }
}
