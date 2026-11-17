import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT_DIR / "vendor_packages"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))
sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt

from src.categories import display_name


def read_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(path, classes, matrix):
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=25, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Domestic Garbage Classification Confusion Matrix")

    max_value = max(max(row) for row in matrix) if matrix else 0
    threshold = max_value * 0.55
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = "white" if value > threshold else "#1f2937"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="根据评估明细生成实验统计报告")
    parser.add_argument("--eval-csv", default="outputs/domestic_four_class/eval_results.csv")
    parser.add_argument("--out-dir", default="outputs/domestic_four_class/analysis")
    parser.add_argument("--split-name", default="验证集")
    args = parser.parse_args()

    rows = read_rows(args.eval_csv)
    if not rows:
        raise ValueError("评估明细为空，无法分析。")

    classes = sorted({row["true_class"] for row in rows} | {row["pred_class"] for row in rows})
    total = len(rows)
    correct = sum(int(row["correct"]) for row in rows)

    by_class = defaultdict(lambda: {"total": 0, "correct": 0})
    confusion = {cls: Counter() for cls in classes}
    for row in rows:
        true_class = row["true_class"]
        pred_class = row["pred_class"]
        ok = int(row["correct"])
        by_class[true_class]["total"] += 1
        by_class[true_class]["correct"] += ok
        confusion[true_class][pred_class] += 1

    out_dir = Path(args.out_dir)
    summary_rows = [
        {"metric": "val_total", "value": total},
        {"metric": "correct", "value": correct},
        {"metric": "wrong", "value": total - correct},
        {"metric": "accuracy", "value": f"{correct / total:.6f}"},
    ]
    write_csv(out_dir / "summary.csv", ["metric", "value"], summary_rows)

    per_class_rows = []
    for cls in classes:
        cls_total = by_class[cls]["total"]
        cls_correct = by_class[cls]["correct"]
        predicted_total = sum(confusion[other][cls] for other in classes)
        precision = cls_correct / predicted_total if predicted_total else 0.0
        recall = cls_correct / cls_total if cls_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_rows.append(
            {
                "class": cls,
                "class_cn": display_name(cls),
                "total": cls_total,
                "correct": cls_correct,
                "wrong": cls_total - cls_correct,
                "accuracy": f"{cls_correct / cls_total if cls_total else 0:.6f}",
                "precision": f"{precision:.6f}",
                "recall": f"{recall:.6f}",
                "f1": f"{f1:.6f}",
            }
        )
    write_csv(out_dir / "per_class_accuracy.csv", ["class", "class_cn", "total", "correct", "wrong", "accuracy", "precision", "recall", "f1"], per_class_rows)
    write_csv(out_dir / "wrong_predictions.csv", list(rows[0]), [row for row in rows if not int(row["correct"])])
    macro_f1 = sum(float(row["f1"]) for row in per_class_rows) / len(per_class_rows)

    matrix = [[confusion[true_cls][pred_cls] for pred_cls in classes] for true_cls in classes]
    confusion_rows = []
    for cls, values in zip(classes, matrix):
        row = {"true_class": cls}
        row.update({pred_cls: value for pred_cls, value in zip(classes, values)})
        confusion_rows.append(row)
    write_csv(out_dir / "confusion_matrix.csv", ["true_class"] + classes, confusion_rows)
    save_confusion_matrix(out_dir / "confusion_matrix.svg", classes, matrix)

    report_lines = [
        "# 生活垃圾四分类实验结果分析",
        "",
        f"{args.split_name}共有 {total} 张图片，其中预测正确 {correct} 张，预测错误 {total - correct} 张。",
        "",
        "准确率计算公式：",
        "",
        "```text",
        "准确率 = 预测正确样本数 / 评估样本总数 × 100%",
        "```",
        "",
        f"本次实验准确率为：{correct} / {total} × 100% = {correct / total * 100:.2f}%。",
        f"宏平均 F1 为：{macro_f1 * 100:.2f}%。",
        "",
        "| 类别 | 图片数量 | 正确数量 | 错误数量 | 召回率 | 精确率 | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in per_class_rows:
        report_lines.append(
            f"| {row['class_cn']} | {row['total']} | {row['correct']} | {row['wrong']} | {float(row['recall']) * 100:.2f}% | {float(row['precision']) * 100:.2f}% | {float(row['f1']) * 100:.2f}% |"
        )
    (out_dir / "experiment_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"total={total}")
    print(f"correct={correct}")
    print(f"accuracy={correct / total * 100:.2f}%")
    print(f"report={out_dir / 'experiment_report.md'}")


if __name__ == "__main__":
    main()
