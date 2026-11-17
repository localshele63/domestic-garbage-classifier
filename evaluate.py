import argparse
import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = ROOT_DIR.parent / "garbage_classifier_bishe" / "vendor_packages"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import torch
from PIL import Image

from src.config import DEFAULT_CHECKPOINT, DEFAULT_CLASS_NAMES, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from src.data import build_transforms
from src.inference import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="评估生活垃圾四分类模型")
    parser.add_argument("--data-dir", default=str(Path(DEFAULT_DATA_DIR) / "val"))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--class-names", default=DEFAULT_CLASS_NAMES)
    parser.add_argument("--out", default=str(Path(DEFAULT_OUTPUT_DIR) / "eval_results.csv"))
    args = parser.parse_args()

    model, class_names, device = load_checkpoint(args.checkpoint, args.class_names)
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
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
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
