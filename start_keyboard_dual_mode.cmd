@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "PYTHONDONTWRITEBYTECODE=1"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0fbcca_keyboard_dual_mode.py" %*
) else (
    py -3.10 "%~dp0fbcca_keyboard_dual_mode.py" %*
)
set "EXIT_CODE=%ERRORLEVEL%"
:finished
echo Exit code: %EXIT_CODE%
if not "%FBCCA_NO_PAUSE%"=="1" pause
exit /b %EXIT_CODE%
