"""Treina o classificador de 24 tipos cromossômicos (torchvision, fine-tuning ImageNet).

A configuração padrão (`DEFAULT_CONFIG`) reproduz a rota (b) YOLOv8 + ResNet-50 de
Rosas-Alatriste et al. (Scientific Reports, 2026, seção "Route (b)"): Adam lr=1e-3, primeiras
~50 camadas congeladas, cross-entropy pura, ReduceLROnPlateau(factor=0.02, patience=3),
EarlyStopping(patience=15) e checkpoint pelo menor val_loss, até 100 épocas, batch 32, resize
esticado + augmentation do ImageDataGenerator, seed 0.

Qualquer campo pode ser sobrescrito (ex.: `train(resize="letterbox")`). `last.pt` é salvo a
cada época -- inclusive o estado dos geradores aleatórios -- e `resume=True` continua uma run
interrompida de onde parou, com a mesma sequência de embaralhamento/augmentation.

Uso:
    uv run python src/classification/train.py --run-name resnet50-paper
    uv run python src/classification/train.py --run-name resnet50-paper --resume
"""

import argparse
import csv
import json
import random
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

DEFAULT_CONFIG: dict = {
    "backbone": "resnet50",
    "epochs": 100,
    "batch_size": 32,
    "lr": 1e-3,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "scheduler": "plateau",
    "plateau_factor": 0.02,
    "plateau_patience": 3,
    "plateau_min_lr": 1e-6,
    "freeze": "paper",
    "class_weights": False,
    "label_smoothing": 0.0,
    "monitor": "val_loss",
    "early_stopping_patience": 15,
    "resize": "stretch",
    "augment": "paper",
    "seed": 0,
}

HISTORY_FIELDS = [
    "epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1", "lr",
]


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


def freeze_layers(model: nn.Module, backbone: str, mode: str) -> list[nn.Module]:
    """Congela as camadas iniciais e devolve os módulos congelados.

    `mode="paper"` reproduz `base.layers[:50]` do ResNet50 do Keras: stem (conv1/bn1) +
    conv2_x (`layer1`) + o primeiro bloco de conv3_x (`layer2[0]`). Como no Keras, o
    BatchNorm dessas camadas fica em modo inferência (ver `run_epoch`).
    """
    if mode == "none":
        return []
    if mode != "paper" or backbone != "resnet50":
        raise ValueError(f"freeze={mode!r} não suportado para backbone={backbone!r}")

    frozen = [model.conv1, model.bn1, model.layer1, model.layer2[0]]
    for module in frozen:
        for param in module.parameters():
            param.requires_grad = False
    return frozen


def class_weights_from_counts(counts: list[int]) -> torch.Tensor:
    """Peso ∝ 1/sqrt(freq): reduz o desbalanceamento (Y é ~4x mais raro que a mediana)
    sem exagerar a ponto de o modelo overfitar nas classes raras."""
    counts = np.array(counts, dtype=np.float64)
    weights = 1.0 / np.sqrt(counts)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def build_optimizer(params, config: dict) -> torch.optim.Optimizer:
    if config["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=config["lr"], weight_decay=config["weight_decay"])
    if config["optimizer"] == "adamw":
        return torch.optim.AdamW(params, lr=config["lr"], weight_decay=config["weight_decay"])
    raise ValueError(f"optimizer desconhecido: {config['optimizer']!r}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    if config["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    if config["scheduler"] == "plateau":
        # threshold_mode="abs" + 1e-4 equivale ao min_delta padrão do ReduceLROnPlateau do Keras.
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["plateau_factor"],
            patience=config["plateau_patience"],
            min_lr=config["plateau_min_lr"],
            threshold=1e-4,
            threshold_mode="abs",
        )
    raise ValueError(f"scheduler desconhecido: {config['scheduler']!r}")


def seed_everything(seed: int) -> torch.Generator:
    """Fixa as seeds de python/numpy/torch e devolve o gerador do embaralhamento do DataLoader
    (que também deriva as seeds de cada worker, onde roda a augmentation)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def rng_state(loader_generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": loader_generator.get_state(),
    }


def restore_rng_state(state: dict, loader_generator: torch.Generator) -> None:
    # estados de RNG precisam ser ByteTensor de CPU; torch.load(map_location="cuda") os move pra GPU
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    loader_generator.set_state(state["loader"].cpu())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    frozen: list[nn.Module] = (),
) -> tuple[float, float, float]:
    """Devolve (loss média, acurácia, macro-F1) da época."""
    train = optimizer is not None
    model.train(train)
    for module in frozen:
        module.eval()

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
    accuracy = float((preds == labels).mean())
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), accuracy, macro_f1


def resolve_config(overrides: dict) -> dict:
    unknown = set(overrides) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"parâmetros desconhecidos: {sorted(unknown)} (opções: {list(DEFAULT_CONFIG)})")
    config = {**DEFAULT_CONFIG, **overrides}
    if config["monitor"] not in ("val_loss", "val_f1"):
        raise ValueError(f"monitor desconhecido: {config['monitor']!r}")
    return config


def train(
    run_name: str | None = None,
    resume: bool = False,
    image_size: int = 224,
    num_workers: int = 8,
    **overrides,
) -> Path:
    config = resolve_config(overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = run_name or f"{config['backbone']}-{config['epochs']}epochs"
    run_dir = RUNS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path, history_path = run_dir / "best.pt", run_dir / "last.pt", run_dir / "history.csv"

    loader_generator = seed_everything(config["seed"])

    train_ds = ChromosomeCropDataset(
        MANIFEST, CROPS_ROOT, "train", image_size, train=True,
        resize=config["resize"], augment=config["augment"],
    )
    val_ds = ChromosomeCropDataset(
        MANIFEST, CROPS_ROOT, "val", image_size, train=False, resize=config["resize"],
    )
    loader_kwargs = {"batch_size": config["batch_size"], "num_workers": num_workers, "pin_memory": True}
    train_loader = DataLoader(train_ds, shuffle=True, generator=loader_generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = build_model(config["backbone"]).to(device)
    frozen = freeze_layers(model, config["backbone"], config["freeze"])
    weights = class_weights_from_counts(train_ds.class_counts()).to(device) if config["class_weights"] else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=config["label_smoothing"])
    optimizer = build_optimizer([p for p in model.parameters() if p.requires_grad], config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler(device=device.type) if device.type == "cuda" else None

    higher_is_better = config["monitor"] == "val_f1"
    best_score = -np.inf if higher_is_better else np.inf
    bad_epochs, start_epoch = 0, 1

    saved_config = {"image_size": image_size, **config}
    if resume and last_path.exists():
        previous = json.loads((run_dir / "config.json").read_text())
        if previous != saved_config:
            raise ValueError(
                f"A config atual difere da usada em {run_dir / 'config.json'} -- "
                "use os mesmos overrides para retomar, ou outro run_name."
            )
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state["scaler"] is not None:
            scaler.load_state_dict(state["scaler"])
        restore_rng_state(state["rng"], loader_generator)
        best_score, bad_epochs, start_epoch = state["best_score"], state["bad_epochs"], state["epoch"] + 1
        # descarta linhas de uma época que não chegou a ser salva em last.pt
        with open(history_path, newline="") as f:
            rows = list(csv.reader(f))
        with open(history_path, "w", newline="") as f:
            csv.writer(f).writerows([rows[0], *(r for r in rows[1:] if int(r[0]) < start_epoch)])
        print(f"Retomando {run_name} a partir da época {start_epoch} (melhor {config['monitor']}={best_score:.4f})")
    else:
        with open(history_path, "w", newline="") as f:
            csv.writer(f).writerow(HISTORY_FIELDS)
        (run_dir / "config.json").write_text(json.dumps(saved_config, indent=2))

    patience = config["early_stopping_patience"]
    if patience is not None and bad_epochs >= patience:
        print(f"Run já tinha parado por early stopping -- nada a retomar. Melhor checkpoint: {best_path}")
        return best_path

    for epoch in range(start_epoch, config["epochs"] + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, frozen
        )
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, device, None, None)
        if config["scheduler"] == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        print(
            f"epoch {epoch}/{config['epochs']}  lr={lr:.1e}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )
        score = val_f1 if higher_is_better else val_loss
        improved = score > best_score if higher_is_better else score < best_score
        if improved:
            best_score, bad_epochs = score, 0
            torch.save(
                {
                    "backbone": config["backbone"],
                    "num_classes": NUM_CLASSES,
                    "image_size": image_size,
                    "resize": config["resize"],
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                },
                best_path,
            )
        else:
            bad_epochs += 1

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "rng": rng_state(loader_generator),
                "best_score": best_score,
                "bad_epochs": bad_epochs,
            },
            last_path,
        )
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, lr]
            )

        if patience is not None and bad_epochs >= patience:
            print(f"Early stopping: {patience} épocas sem melhora em {config['monitor']}.")
            break

    print(f"Melhor {config['monitor']} (val): {best_score:.4f} -- salvo em {best_path}")
    return best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true", help="continua a partir de last.pt")
    parser.add_argument("--backbone", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    cli_overrides = {
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
    }
    train(
        run_name=args.run_name,
        resume=args.resume,
        image_size=args.image_size,
        num_workers=args.num_workers,
        **{k: v for k, v in cli_overrides.items() if v is not None},
    )
