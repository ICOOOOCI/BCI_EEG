@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "SCRIPT=%~dp0fbcca_keyboard_dual_mode.py"
set "EXIT_CODE=1"

if not exist "%SCRIPT%" (
    echo ERROR: fbcca_keyboard_dual_mode.py is missing.
    echo Extract the ZIP first. Keep the PY and CMD files in the same folder.
    goto finished
)

py -3.10 -c "import sys" >nul 2>&1
if not errorlevel 1 goto use_py_launcher

if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" goto use_local_python

python -c "import sys; assert sys.version_info[:2] == (3, 10)" >nul 2>&1
if not errorlevel 1 goto use_path_python

echo ERROR: Python 3.10 was not found.
echo Use the same Python 3.10 environment as your previous working keyboard.
echo You may also run fbcca_keyboard_dual_mode.py directly from that IDE.
goto finished

:use_py_launcher
py -3.10 "%SCRIPT%"
set "EXIT_CODE=%ERRORLEVEL%"
goto finished

:use_local_python
"%LOCALAPPDATA%\Programs\Python\Python310\python.exe" "%SCRIPT%"
set "EXIT_CODE=%ERRORLEVEL%"
goto finished

:use_path_python
python "%SCRIPT%"
set "EXIT_CODE=%ERRORLEVEL%"

:finished
echo.
echo Program ended. Review output or error details above.
echo Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
