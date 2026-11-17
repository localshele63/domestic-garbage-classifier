import argparse
import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = ROOT_DIR / "vendor_packages"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import torch
from PIL import Image

from src.config import DEFAULT_DATA_DIR
from src.data import build_transforms
from src.inference import load_checkpoint


def parse_checkpoint(value):
    parts = value.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("模型参数格式应为 checkpoint:class_names")
    return parts[0], parts[1]


def main():
    parser = argparse.ArgumentParser(description="评估多个模型集成后的生活垃圾四分类准确率")
    parser.add_argument("--data-dir", default=str(Path(DEFAULT_DATA_DIR) / "val"))
    parser.add_argument("--model", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--out", default="outputs/domestic_four_class_ensemble/eval_results.csv")
    args = parser.parse_args()

    loaded = [load_checkpoint(checkpoint, class_names) for checkpoint, class_names in args.model]
    class_names = loaded[0][1]
    device = loaded[0][2]
    for _, names, _ in loaded:
        if names != class_names:
            raise ValueError("参与集成的模型类别顺序不一致。")

    models = [model for model, _, _ in loaded]
    transform = build_transforms(train=False)
    val_dir = Path(args.data_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    rows = []
    total = 0
    correct = 0
    per_class = {name: {"total": 0, "correct": 0} for name in class_names}

    for class_dir in sorted([p for p in val_dir.iterdir() if p.is_dir()]):
        true_name = class_dir.name
        if true_name not in class_names:
            print(f"跳过未知类别目录: {true_name}", file=sys.stderr)
            continue

        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in image_exts:
                continue
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                probs_list = [torch.softmax(model(tensor), dim=1)[0] for model in models]
                probs = torch.stack(probs_list).mean(dim=0)
                pred_idx = int(torch.argmax(probs).item())

            pred_name = class_names[pred_idx]
            confidence = float(probs[pred_idx].item())
            ok = pred_name == true_name

            total += 1
            correct += int(ok)
            per_class[true_name]["total"] += 1
            per_class[true_name]["correct"] += int(ok)
            rows.append([str(image_path), true_name, pred_name, f"{confidence:.4f}", int(ok)])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "true_class", "pred_class", "confidence", "correct"])
        writer.writerows(rows)

    print(f"总图片数: {total}")
    print(f"预测正确: {correct}")
    print(f"总体准确率: {correct / total * 100:.2f}%" if total else "总体准确率: 0.00%")
    print("各类别准确率:")
    for name, item in per_class.items():
        acc = item["correct"] / item["total"] * 100 if item["total"] else 0.0
        print(f"{name}: {item['correct']}/{item['total']} = {acc:.2f}%")
    print(f"详细结果已保存: {out_path}")


if __name__ == "__main__":
    main()
