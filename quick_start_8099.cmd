@echo off
setlocal ENABLEDELAYEDEXPANSION
REM Quick start on port 8099
if not exist .venv (
  echo [INFO] Creating venv...
  py -3 -m venv .venv || goto :fail
)
call .venv\Scripts\python -m pip -q install --upgrade pip >nul 2>&1
.venv\Scripts\python -c "import groq" 2>nul || (
  echo [INFO] Installing requirements...
  call .venv\Scripts\pip install -r requirements.txt || goto :fail
)
if "%API_TOKENS%"=="" set API_TOKENS=test
set LOG_LEVEL=INFO
set PORT=8099
echo [START] http://127.0.0.1:%PORT%  (Token: test)
call .venv\Scripts\python -m uvicorn api.app:app --host 127.0.0.1 --port %PORT% --reload
goto :eof
:fail
echo [ERROR] Startup failed.
exit /b 1
