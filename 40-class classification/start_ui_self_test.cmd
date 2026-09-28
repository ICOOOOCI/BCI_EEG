@echo off
call "%~dp0start_keyboard_dual_mode.cmd" --self-test-ui %*
exit /b %ERRORLEVEL%
