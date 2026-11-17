import argparse
import random
import shutil
import sys
import zipfile
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CATEGORY_ALIASES = {
    "recyclable": {
        "cardboard",
        "paper",
        "plastic",
        "metal",
        "glass",
        "white-glass",
        "green-glass",
        "brown-glass",
        "white_glass",
        "green_glass",
        "brown_glass",
    },
    "hazardous": {"battery", "batteries"},
    "wet": {"biological", "organic", "food", "food-waste", "food_waste", "kitchen-waste", "kitchen_waste"},
    "dry": {"trash", "clothes", "shoes", "textile", "textiles"},
}

CLASS_CN = {
    "recyclable": "可回收物",
    "hazardous": "有害垃圾",
    "wet": "湿垃圾/厨余垃圾",
    "dry": "干垃圾/其他垃圾",
}


def normalize_name(name):
    return name.strip().lower().replace(" ", "-")


def target_class_for_folder(folder_name):
    normalized = normalize_name(folder_name)
    for target, aliases in CATEGORY_ALIASES.items():
        if normalized in aliases:
            return target
    return None


def extract_if_needed(src, work_dir):
    src = Path(src)
    if src.is_dir():
        return src

    if src.suffix.lower() != ".zip":
        raise ValueError("目前支持传入已解压数据集文件夹，或 .zip 压缩包。")

    work_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = work_dir / src.stem
    marker = extract_dir / ".extracted"
    if marker.exists():
        return extract_dir

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(extract_dir)
    marker.write_text("ok", encoding="utf-8")
    return extract_dir


def collect_images(raw_root):
    grouped = {name: [] for name in CATEGORY_ALIASES}
    for class_dir in sorted([p for p in raw_root.rglob("*") if p.is_dir()]):
        target = target_class_for_folder(class_dir.name)
        if target is None:
            continue
        images = [
            p
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and "__macosx" not in str(p).lower()
        ]
        grouped[target].extend(images)
    return grouped


def copy_split_dataset(grouped, out_dir, val_ratio, seed):
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    summary = []
    for target in ["recyclable", "hazardous", "wet", "dry"]:
        images = list(dict.fromkeys(grouped[target]))
        rng.shuffle(images)
        val_count = max(1, round(len(images) * val_ratio)) if images else 0
        splits = {"val": images[:val_count], "train": images[val_count:]}

        for split, paths in splits.items():
            class_dir = out_dir / split / target
            class_dir.mkdir(parents=True, exist_ok=True)
            for idx, src in enumerate(paths, start=1):
                dst = class_dir / f"{target}_{idx:05d}{src.suffix.lower()}"
                shutil.copy2(src, dst)

        summary.append(
            {
                "class": target,
                "cn": CLASS_CN[target],
                "total": len(images),
                "train": len(splits["train"]),
                "val": len(splits["val"]),
            }
        )
    return summary


def write_summary(out_dir, summary):
    lines = [
        "# 四分类数据集整理结果",
        "",
        "| 类别 | 中文名称 | 总数 | 训练集 | 验证集 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(f"| {row['class']} | {row['cn']} | {row['total']} | {row['train']} | {row['val']} |")
    Path(out_dir, "dataset_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="将多类别垃圾图片数据集整理为生活垃圾四分类数据集")
    parser.add_argument("--src", required=True, help="原始数据集文件夹或 .zip 压缩包")
    parser.add_argument("--dst", default="data/domestic_garbage")
    parser.add_argument("--work-dir", default="data/imported_raw")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_root = extract_if_needed(args.src, Path(args.work_dir))
    grouped = collect_images(raw_root)
    summary = copy_split_dataset(grouped, args.dst, args.val_ratio, args.seed)
    write_summary(args.dst, summary)

    print("四分类数据集整理完成")
    for row in summary:
        print(f"{row['class']}({row['cn']}): total={row['total']} train={row['train']} val={row['val']}")

    empty = [row for row in summary if row["total"] == 0]
    if empty:
        print("警告：以下类别没有找到图片，请确认原始数据集类别是否完整：", file=sys.stderr)
        for row in empty:
            print(f"- {row['class']}({row['cn']})", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
