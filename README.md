# 基于深度学习的垃圾分类图像识别系统

本项目使用 PyTorch 和 EfficientNet-B0 识别生活垃圾图像，通过 Gradio 接收图片并显示预测类别及四类 Softmax 分数。界面将这些分数显示为置信度。实际投放需结合物品状态和当地分类规则。

| 训练标签 | 类别 | 原始数据细类示例 |
| --- | --- | --- |
| `dry` | 干垃圾 | `trash`、`clothes`、`shoes` |
| `hazardous` | 有害垃圾 | `battery` |
| `recyclable` | 可回收物 | `paper`、`cardboard`、`plastic`、`metal`、各类 `glass` |
| `wet` | 湿垃圾 | `biological` |

## 安装与自检

在仓库根目录打开终端。论文实验使用 Python 3.9.9、PyTorch 2.8.0、Torchvision 0.23.0，记录的网页环境使用 Gradio 4.44.1。使用 GPU 训练时，请结合系统和显卡选择兼容的 PyTorch 与 Torchvision 安装包。

```bash
python -m pip install -r requirements.txt "gradio==4.44.1" "huggingface-hub==0.25.2"
python smoke_test.py
```

自检使用未训练的 MobileNet v2 和零张量，检查模型构建及输出维度。

## 整理数据和训练

请自备有权使用的垃圾分类图片。本仓库提供源码。运行识别需要同一次训练生成的模型检查点 `best_model.pth` 与类别文件 `class_names.json`。

将原始数据按细类放在目录中，也可以提供保留了目录结构的 ZIP 文件。将下面的 `path/to/raw-dataset` 换成本地路径。脚本会重新生成目标目录，请使用新的 `--dst`。整理后 `train` 与 `val` 中均需有 `dry`、`hazardous`、`recyclable`、`wet` 四个目录，且每类均需有图片。

```bash
python scripts/prepare_dataset.py --src "path/to/raw-dataset" --dst data/domestic_garbage
python train.py --data-dir data/domestic_garbage --model efficientnet_b0 --epochs 12 --batch-size 64 --num-workers 4 --lr 0.0001 --class-weight --label-smoothing 0 --deduplicate --selection-metric balanced_accuracy --patience 0 --output-dir outputs/expanded_balanced_efficientnet_b0
```

训练默认加载 ImageNet 预训练参数，首次获取参数需要联网。兼容 CUDA 的环境可在训练命令末尾添加 `--amp`。训练产物保存在 `--output-dir` 指定的目录。

## 网页识别、单张预测与评估

训练完成后运行以下命令，并打开 [本机识别页面](http://127.0.0.1:7860)。如果使用了其他输出目录，请替换两个模型文件路径。

```bash
python app.py --checkpoint outputs/expanded_balanced_efficientnet_b0/best_model.pth --class-names outputs/expanded_balanced_efficientnet_b0/class_names.json --single-model
```

命令行可执行单张预测，也可对包含四类子目录的 `val` 或 `test` 目录执行评估。用本地图片路径替换 `path/to/image.jpg`。

```bash
python predict.py --image "path/to/image.jpg" --checkpoint outputs/expanded_balanced_efficientnet_b0/best_model.pth --class-names outputs/expanded_balanced_efficientnet_b0/class_names.json
python evaluate.py --data-dir data/domestic_garbage/val --checkpoint outputs/expanded_balanced_efficientnet_b0/best_model.pth --class-names outputs/expanded_balanced_efficientnet_b0/class_names.json --out outputs/evaluation/val_predictions.csv
python scripts/analyze_results.py --eval-csv outputs/evaluation/val_predictions.csv --out-dir outputs/evaluation/analysis --split-name "验证集"
```

## 论文实验代码

`scripts/thesis_run.py` 保存论文主实验的数据清洗、固定留出测试集划分、训练和离线测试流程。通用数据整理脚本生成训练集与验证集，主实验脚本再对本地原实验数据执行去重和三集合划分。公开仓库中的程序可以处理自备数据，论文数值对应原实验的数据和检查点。

运行主实验脚本需要 Windows、支持 CUDA 的 NVIDIA GPU、可运行的 `nvidia-smi`、本地 `data/domestic_garbage` 数据，以及预先建立的 `outputs` 目录和仓库旁的 `毕业论文` 目录。脚本将训练、逐图预测和绘图结果保存到新建的 `outputs/thesis_日期_时间` 中，并在旁边的 `毕业论文/实验记录` 写入报告。

## 许可证

项目源码采用 [MIT 许可证](LICENSE)。第三方依赖适用各自的许可证。
