import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = ROOT_DIR.parent / "garbage_classifier_bishe" / "vendor_packages"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

from PIL import Image
import torch

from src.model import build_model


def main():
    model = build_model("mobilenet_v2", num_classes=4, pretrained=False)
    model.eval()
    image = Image.new("RGB", (224, 224), color=(120, 120, 120))
    tensor = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        logits = model(tensor)
    assert logits.shape == (1, 4)
    assert image.size == (224, 224)
    print("自检通过：模型构建和基础推理流程正常。")


if __name__ == "__main__":
    main()
