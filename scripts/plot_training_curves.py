import argparse
import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT_DIR / "vendor_packages"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import matplotlib.pyplot as plt


def read_history(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["val_loss"]),
                    "train_acc": float(row["train_acc"]) * 100,
                    "val_acc": float(row["val_acc"]) * 100,
                }
            )
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return rows


def plot_loss(rows, out_path):
    epochs = [row["epoch"] for row in rows]
    plt.figure(figsize=(8, 5), dpi=160)
    plt.plot(epochs, [row["train_loss"] for row in rows], marker="o", label="Train loss")
    plt.plot(epochs, [row["val_loss"] for row in rows], marker="o", label="Validation loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_accuracy(rows, out_path):
    epochs = [row["epoch"] for row in rows]
    plt.figure(figsize=(8, 5), dpi=160)
    plt.plot(epochs, [row["train_acc"] for row in rows], marker="o", label="Train accuracy")
    plt.plot(epochs, [row["val_acc"] for row in rows], marker="o", label="Validation accuracy")
    plt.title("Training Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot training loss and accuracy curves.")
    parser.add_argument(
        "--history",
        default="outputs/domestic_four_class_efficientnet_b0/train_history.csv",
        help="Path to train_history.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/teacher_training_curves",
        help="Output directory for curve images",
    )
    args = parser.parse_args()

    rows = read_history(args.history)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_png = out_dir / "loss_curve.png"
    acc_png = out_dir / "accuracy_curve.png"
    loss_svg = out_dir / "loss_curve.svg"
    acc_svg = out_dir / "accuracy_curve.svg"

    plot_loss(rows, loss_png)
    plot_accuracy(rows, acc_png)
    plot_loss(rows, loss_svg)
    plot_accuracy(rows, acc_svg)

    best_val = max(rows, key=lambda row: row["val_acc"])
    note = [
        "# 训练曲线说明",
        "",
        f"训练日志来源：`{args.history}`",
        "",
        f"训练轮数：{len(rows)}",
        f"最佳验证集准确率：{best_val['val_acc']:.2f}%（第 {best_val['epoch']} 轮）",
        "",
        "生成文件：",
        "",
        "- loss_curve.png",
        "- accuracy_curve.png",
        "- loss_curve.svg",
        "- accuracy_curve.svg",
    ]
    (out_dir / "curve_note.md").write_text("\n".join(note), encoding="utf-8")
    print(f"曲线已生成: {out_dir}")
    print(f"最佳验证集准确率: {best_val['val_acc']:.2f}% epoch={best_val['epoch']}")


if __name__ == "__main__":
    main()
