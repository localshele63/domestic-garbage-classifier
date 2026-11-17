@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Website: http://127.0.0.1:7860
echo Please wait. The browser will open after the model loads.
python -u scripts\launch_app.py
if errorlevel 1 echo Startup failed. Please show the error above for troubleshooting.
pause
