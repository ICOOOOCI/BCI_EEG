@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
py -3.10 "%~dp0setup_environment.py"
set "EXIT_CODE=%ERRORLEVEL%"
echo Setup exit code: %EXIT_CODE%
if not "%FBCCA_NO_PAUSE%"=="1" pause
exit /b %EXIT_CODE%
