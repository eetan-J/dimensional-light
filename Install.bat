@echo off
rem Double-click to install Dimensional Light. Extra options are passed through,
rem e.g.  Install.bat -SkipUltra
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
pause
