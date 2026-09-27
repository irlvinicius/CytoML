"""Treina o classificador de 24 tipos cromossômicos (torchvision, fine-tuning ImageNet).

Uso:
    uv run python src/classification/train.py --backbone resnet50 --epochs 30
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))

import numpy as np
import torch
import torch.nn as nn
from dataset import ChromosomeCropDataset
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import models
from voc_to_yolo import DENVER_CLASSES

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "data" / "processed" / "classification" / "manifest.csv"
CROPS_ROOT = REPO_ROOT / "data" / "processed" / "classification"
RUNS_ROOT = REPO_ROOT / "models" / "classifier" / "runs"

NUM_CLASSES = len(DENVER_CLASSES)


def build_model(backbone: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Backbone desconhecido: {backbone!r}")


def class_weights_from_counts(counts: list[int]) -> torch.Tensor:
    """Peso ∝ 1/sqrt(freq): reduz o desbalanceamento (Y é ~4x mais raro que a mediana)
    sem exagerar a ponto de o modelo overfitar nas classes raras."""
    counts = np.array(counts, dtype=np.float64)
    weights = 1.0 / np.sqrt(counts)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)

    total_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_preds.append(logits.argmax(dim=1).detach().cpu())
        all_labels.append(labels.detach().cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), macro_f1


def train(
    backbone: str = "resnet50",
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 3e-4,
    image_size: int = 224,
    num_workers: int = 8,
    run_name: str | None = None,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = run_name or f"{backbone}-{epochs}epochs"
    run_dir = RUNS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ChromosomeCropDataset(MANIFEST, CROPS_ROOT, "train", image_size, train=True)
    val_ds = ChromosomeCropDataset(MANIFEST, CROPS_ROOT, "val", image_size, train=False)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    model = build_model(backbone).to(device)
    weights = class_weights_from_counts(train_ds.class_counts()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device=device.type) if device.type == "cuda" else None

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_f1", "val_loss", "val_f1"])

    best_f1, best_path = -1.0, run_dir / "best.pt"
    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss, val_f1 = run_epoch(model, val_loader, criterion, device, None, None)
        scheduler.step()

        print(
            f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f} train_f1={train_f1:.4f}  "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_f1, val_loss, val_f1])

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {"backbone": backbone, "num_classes": NUM_CLASSES, "state_dict": model.state_dict()},
                best_path,
            )

    print(f"Melhor macro-F1 (val): {best_f1:.4f} -- salvo em {best_path}")
    return best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    train(
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        num_workers=args.num_workers,
        run_name=args.run_name,
    )
