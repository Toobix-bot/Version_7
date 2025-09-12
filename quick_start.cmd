@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Simple one-click starter
REM 1. Ensure venv exists
if not exist .venv (
  echo [INFO] Creating virtual environment...
  py -3 -m venv .venv || goto :fail
)

REM 2. Upgrade pip minimal
call .venv\Scripts\python -m pip -q install --upgrade pip >nul 2>&1

REM 3. Install requirements if needed (heuristic: check groq package)
.venv\Scripts\python -c "import groq" 2>nul || (
  echo [INFO] Installing requirements...
  call .venv\Scripts\pip install -r requirements.txt || goto :fail
)

REM 4. Default API token
if "%API_TOKENS%"=="" set API_TOKENS=test
set LOG_LEVEL=INFO

REM 5. Start server
echo [START] http://127.0.0.1:8000  (Token: test)
call .venv\Scripts\python -m uvicorn api.app:app --host 127.0.0.1 --port 8000

goto :eof
:fail
echo [ERROR] Startup failed.
exit /b 1
