@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Training with real GPU monitoring. Keep this window open.
echo Output will be saved in a new timestamped folder under outputs.
python -u scripts\train_with_gpu_monitor.py --epochs 12
if errorlevel 1 echo Training failed. Check the error above and training.log.
pause
