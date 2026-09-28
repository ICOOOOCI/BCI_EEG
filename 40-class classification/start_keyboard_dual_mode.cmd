@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "PYTHONDONTWRITEBYTECODE=1"
if not defined FBCCA_QUICK_ENTRY set "FBCCA_QUICK_ENTRY=1"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "EXIT_CODE=10"
if not exist "%PYTHON_EXE%" (
    echo [ENVIRONMENT] Project .venv is missing. Run setup_windows.cmd explicitly first.
    goto finished
)
"%PYTHON_EXE%" "%~dp0run_keyboard.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
:finished
echo Exit code: %EXIT_CODE%
if not "%FBCCA_NO_PAUSE%"=="1" pause
exit /b %EXIT_CODE%
