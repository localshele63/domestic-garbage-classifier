@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在训练生活垃圾四分类模型...
python train.py --data-dir data\domestic_garbage --model efficientnet_b0 --epochs 12 --batch-size 64 --num-workers 4 --lr 0.0001 --class-weight --label-smoothing 0 --amp --deduplicate --selection-metric balanced_accuracy --patience 4 --output-dir outputs\expanded_balanced_efficientnet_b0
pause
