@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Default values
set BIND_HOST=127.0.0.1
set PORT=8000
set API_KEY=test
set RELOAD=
set LOG_FILE=logs/app.log

:parse
if "%~1"=="" goto run
if /I "%~1"=="-BindHost" (set BIND_HOST=%~2& shift & shift & goto parse)
if /I "%~1"=="-Port" (set PORT=%~2& shift & shift & goto parse)
if /I "%~1"=="-ApiKey" (set API_KEY=%~2& shift & shift & goto parse)
if /I "%~1"=="-Reload" (set RELOAD=--reload& shift & goto parse)
if /I "%~1"=="-LogFile" (set LOG_FILE=%~2& shift & shift & goto parse)
shift
goto parse

:run
REM Move to repo root (script is scripts\)
REM Resolve script directory and go to project root
set SCRIPT_DIR=%~dp0
REM Remove trailing backslash if present
if "%SCRIPT_DIR:~-1%"=="\" set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
REM Determine project root (if parent has .venv use parent, else current)
set TARGET_DIR=%SCRIPT_DIR%..
pushd "%TARGET_DIR%"
if not exist .venv (
  REM maybe we are already in root (if someone executed from root directly)
  if exist "%SCRIPT_DIR%\.venv" (
	  popd
	  pushd "%SCRIPT_DIR%"
  ) else (
	  echo [ERROR] .venv not found at %CD%
	  popd
	  exit /b 1
  )
)

call .venv\Scripts\activate.bat
set PYTHONPATH=%CD%
set API_TOKEN=%API_KEY%
set LOG_FILE=%LOG_FILE%

REM Simple .env loader (KEY=VALUE)
if exist .env (
	for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
		echo %%A| findstr /R "^[#;]" >nul && (
			 rem comment line skip
		) || (
			 if NOT "%%A"=="" set %%A=%%B
		)
	)
)

echo Starting server http://%BIND_HOST%:%PORT% (Reload=%RELOAD%) LogFile=%LOG_FILE%
".venv\Scripts\python" -m uvicorn api.app:app --host %BIND_HOST% --port %PORT% %RELOAD%

popd
endlocal
