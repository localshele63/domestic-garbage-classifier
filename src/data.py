from pathlib import Path
import hashlib

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import IMAGE_SIZE


def build_transforms(train=True):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(180),
                transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def deduplicate_splits(train_dataset, val_dataset):
    def unique_samples(dataset):
        seen = set()
        samples = []
        for path, label in dataset.samples:
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if digest not in seen:
                samples.append((path, label, digest))
                seen.add(digest)
        return samples, seen

    val_samples, val_hashes = unique_samples(val_dataset)
    train_samples, train_hashes = unique_samples(train_dataset)
    train_samples = [sample for sample in train_samples if sample[2] not in val_hashes]
    report = {"cross_split_duplicate_hashes": len(train_hashes & val_hashes)}
    for name, dataset, samples in [("train", train_dataset, train_samples), ("val", val_dataset, val_samples)]:
        report[name + "_removed"] = len(dataset.samples) - len(samples)
        dataset.samples = [(path, label) for path, label, _ in samples]
        dataset.imgs = dataset.samples
        dataset.targets = [label for _, label in dataset.samples]
        report[name + "_samples"] = [Path(path).relative_to(Path(dataset.root)).as_posix() for path, _ in dataset.samples]
    return report


def create_dataloaders(data_dir, batch_size=32, num_workers=0, deduplicate=False):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "数据目录需要包含 train 和 val 两个子目录，例如 data/domestic_garbage/train、data/domestic_garbage/val。"
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=build_transforms(train=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=build_transforms(train=False))
    if deduplicate:
        train_dataset.split_audit = deduplicate_splits(train_dataset, val_dataset)

    loader_options = {"num_workers": num_workers, "pin_memory": True,
                      "persistent_workers": num_workers > 0}
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_options)
    return train_loader, val_loader, train_dataset.classes
