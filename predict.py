import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = ROOT_DIR.parent / "garbage_classifier_bishe" / "vendor_packages"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

from src.categories import display_name
from src.config import DEFAULT_CHECKPOINT, DEFAULT_CLASS_NAMES
from src.inference import load_checkpoint, predict_image


def main():
    parser = argparse.ArgumentParser(description="预测单张生活垃圾图片")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--class-names", default=DEFAULT_CLASS_NAMES)
    args = parser.parse_args()

    model, class_names, device = load_checkpoint(args.checkpoint, args.class_names)
    results = predict_image(model, class_names, args.image, device, topk=4)

    print("预测结果：")
    for item in results:
        print(f"{display_name(item['class'])}: {item['confidence'] * 100:.2f}%")


if __name__ == "__main__":
    main()
