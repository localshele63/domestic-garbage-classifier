import json
from pathlib import Path

import torch
from PIL import Image

from .data import build_transforms
from .model import build_model


def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(checkpoint_path, class_names_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    class_names = load_class_names(class_names_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "mobilenet_v2")

    model = build_model(model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, class_names, device


def load_ensemble(model_specs, device=None):
    loaded = [load_checkpoint(checkpoint, class_names, device=device) for checkpoint, class_names in model_specs]
    class_names = loaded[0][1]
    device = loaded[0][2]
    models = []
    for model, names, _ in loaded:
        if names != class_names:
            raise ValueError("参与集成的模型类别顺序不一致。")
        models.append(model)
    return models, class_names, device


def predict_image(model, class_names, image, device, topk=4):
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    transform = build_transforms(train=False)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, k=min(topk, len(class_names)))

    results = []
    for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
        results.append({"class": class_names[idx], "confidence": float(prob)})
    return results


def predict_image_ensemble(models, class_names, image, device, topk=4):
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    transform = build_transforms(train=False)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs_list = [torch.softmax(model(tensor), dim=1)[0] for model in models]
        probs = torch.stack(probs_list).mean(dim=0)
        top_probs, top_indices = torch.topk(probs, k=min(topk, len(class_names)))

    results = []
    for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
        results.append({"class": class_names[idx], "confidence": float(prob)})
    return results
