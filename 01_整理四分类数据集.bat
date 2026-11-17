@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 请把新的垃圾分类数据集 zip 或已解压文件夹路径拖到本窗口，然后按回车：
set /p DATASET_PATH=
python scripts\prepare_dataset.py --src "%DATASET_PATH%" --dst data\domestic_garbage
echo.
echo 如果没有报错，四分类数据已整理到 data\domestic_garbage
pause
