"""Avalia um checkpoint do classificador no split de teste.

Reporta top-1/top-2, macro-F1, matriz de confusão 24x24 e acurácia por grupo de
Denver (A-G, X, Y) -- erro dentro do mesmo grupo é o caso mais comum e mais
tolerável (cromossomos de tamanho parecido). Salva as probabilidades softmax do
teste em test_probs.npy, usadas na Fase 5 para atribuir contagens com a restrição
de "2 por classe" em vez de só o argmax.

Uso:
    uv run python src/classification/evaluate.py --checkpoint models/classifier/runs/resnet50-30epochs/best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))

import numpy as np
import torch
import torch.nn.functional as functional
from dataset import ChromosomeCropDataset
from sklearn.metrics import confusion_matrix, f1_score, top_k_accuracy_score
from torch.utils.data import DataLoader
from train import CROPS_ROOT, MANIFEST, build_model
from voc_to_yolo import DENVER_CLASSES

DENVER_GROUPS = sorted({name[0] for name in DENVER_CLASSES})


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(ckpt["backbone"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    all_probs, all_labels = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_probs.append(functional.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def group_accuracy(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    class_group = np.array([name[0] for name in DENVER_CLASSES])
    label_groups = class_group[labels]
    pred_groups = class_group[preds]
    return {
        group: float((pred_groups[label_groups == group] == group).mean())
        for group in DENVER_GROUPS
        if (label_groups == group).any()
    }


def evaluate(checkpoint_path: Path, batch_size: int = 64, num_workers: int = 8) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(checkpoint_path, device)

    test_ds = ChromosomeCropDataset(MANIFEST, CROPS_ROOT, "test", train=False)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    probs, labels = predict(model, test_loader, device)
    preds = probs.argmax(axis=1)

    out_dir = checkpoint_path.parent
    np.save(out_dir / "test_probs.npy", probs)
    np.save(out_dir / "test_labels.npy", labels)

    metrics = {
        "top1": float((preds == labels).mean()),
        "top2": float(top_k_accuracy_score(labels, probs, k=2, labels=range(len(DENVER_CLASSES)))),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "group_accuracy": group_accuracy(labels, preds),
        "confusion_matrix": confusion_matrix(
            labels, preds, labels=range(len(DENVER_CLASSES))
        ).tolist(),
    }

    print(f"Top-1: {metrics['top1']:.4f}  Top-2: {metrics['top2']:.4f}  Macro-F1: {metrics['macro_f1']:.4f}")
    print("Acurácia por grupo de Denver:")
    for group, acc in sorted(metrics["group_accuracy"].items()):
        print(f"  {group}: {acc:.4f}")
    print(f"Probabilidades salvas em: {out_dir / 'test_probs.npy'}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    evaluate(args.checkpoint, batch_size=args.batch_size, num_workers=args.num_workers)
