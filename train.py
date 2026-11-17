import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = ROOT_DIR / "vendor_packages"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from src.data import create_dataloaders
from src.model import build_model


def run_one_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_total = torch.zeros(len(loader.dataset.classes), device=device)
    class_correct = torch.zeros_like(class_total)

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device, enabled=scaler is not None):
                outputs = model(images)
                loss = criterion(outputs, labels)

        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        class_total += torch.bincount(labels, minlength=len(class_total))
        class_correct += torch.bincount(labels[preds == labels], minlength=len(class_total))

    balanced_acc = (class_correct / class_total.clamp_min(1)).mean().item()
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1), balanced_acc


def main():
    parser = argparse.ArgumentParser(description="训练生活垃圾四分类图像识别模型")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", default="efficientnet_b0", choices=["mobilenet_v2", "resnet18", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--phase-log", help="Optional JSONL training phase timestamps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision")
    parser.add_argument("--deduplicate", action="store_true", help="Exclude identical images across and within splits")
    parser.add_argument("--selection-metric", choices=["accuracy", "balanced_accuracy"], default="accuracy")
    parser.add_argument("--no-pretrained", action="store_true", help="不加载 ImageNet 预训练权重")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight", action="store_true", help="按训练集类别数量自动设置损失权重")
    parser.add_argument("--patience", type=int, default=6, help="验证集准确率连续多少轮不提升时提前停止")
    args = parser.parse_args()

    def record_phase(phase, epoch=0):
        if args.phase_log:
            path = Path(args.phase_log)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"timestamp": time.time(), "phase": phase, "epoch": epoch}) + "\n")

    record_phase("setup")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.amp and device != "cuda":
        raise ValueError("--amp requires CUDA")
    scaler = torch.amp.GradScaler("cuda") if args.amp else None
    print(f"Using device: {device}")
    record_phase("data_loading")

    train_loader, val_loader, class_names = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        deduplicate=args.deduplicate,
    )
    print(f"Classes: {class_names}")
    if args.deduplicate:
        audit = train_loader.dataset.split_audit
        (output_dir / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print({key: value for key, value in audit.items() if not key.endswith("_samples")})
    counts = torch.bincount(torch.tensor(train_loader.dataset.targets), minlength=len(class_names)).float()
    config = vars(args).copy()
    config.update({"device": device, "torch_version": str(torch.__version__),
                   "train_counts": dict(zip(class_names, counts.int().tolist())),
                   "val_count": len(val_loader.dataset),
                   "augmentation": "random_resized_crop_0.65_1.0_rotation_180_flip_color_jitter"})
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    model = build_model(args.model, len(class_names), pretrained=not args.no_pretrained).to(device)
    class_weight = None
    if args.class_weight:
        class_weight = counts.sum() / (len(class_names) * counts)
        class_weight = class_weight.to(device)
        print(f"Class weights: {class_weight.detach().cpu().tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_acc = 0.0
    best_score = -1.0
    stale_epochs = 0

    with open(output_dir / "train_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_balanced_acc", "lr", "epoch_seconds", "train_samples", "val_samples"])

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            record_phase("train", epoch)
            train_loss, train_acc, _ = run_one_epoch(model, train_loader, criterion, device, optimizer, scaler)
            record_phase("validation", epoch)
            val_loss, val_acc, val_balanced_acc = run_one_epoch(model, val_loader, criterion, device, scaler=scaler)
            record_phase("checkpoint", epoch)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, val_balanced_acc, optimizer.param_groups[0]['lr'], time.perf_counter()-epoch_start, len(train_loader.dataset), len(val_loader.dataset)])
            f.flush()

            print(
                f"Epoch {epoch:02d}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} balanced_acc={val_balanced_acc:.4f}"
            )

            score = val_balanced_acc if args.selection_metric == "balanced_accuracy" else val_acc
            best_acc = max(best_acc, val_acc)
            if score > best_score:
                best_score = score
                stale_epochs = 0
                torch.save(
                    {
                        "model_name": args.model,
                        "class_names": class_names,
                        "model_state": model.state_dict(),
                        "val_acc": val_acc,
                        "val_balanced_acc": val_balanced_acc,
                        "training_config": config,
                        "epoch": epoch,
                    },
                    output_dir / "best_model.pth",
                )
            else:
                stale_epochs += 1

            scheduler.step(score)
            if args.patience > 0 and stale_epochs >= args.patience:
                print(f"验证集准确率连续 {args.patience} 轮未提升，提前停止训练。")
                break

    print(f"训练完成，最佳验证准确率: {best_acc:.4f}")
    print(f"模型文件: {output_dir / 'best_model.pth'}")
    record_phase("finished")


if __name__ == "__main__":
    main()
