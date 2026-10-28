"""Validação cruzada do detector (YOLO26m): um modelo por fold, todos avaliados no mesmo teste.

Usa os mesmos folds do classificador (`src/preprocessing/pipeline.py::generate_kfold_splits`): para
cada fold `i`, treina com as listas `fold_i_train.txt`/`fold_i_val.txt` -- o fold `i` é a validação
(escolha do `best.pt`) -- e avalia o `best.pt` no split de teste fixo. Hiperparâmetros iguais aos do
`models/yolo/yolo26m/notebooks/02_train.ipynb`. O resultado final (precision, recall, F1, mAP@50 e
mAP@50-95) é a média ± desvio padrão entre os folds, gravado em `cv_summary.csv`.

Retomável: folds já avaliados (`test_metrics.json`) são pulados, e um fold interrompido continua de
`weights/last.pt`. Os folds precisam ter sido gerados nesta máquina (os `.txt` têm caminhos
absolutos). Rodar dentro do tmux:

    PYTHONUNBUFFERED=1 uv run --group yolo python src/detection/cross_validate.py \\
        | tee models/yolo/yolo26m/runs/yolo26m-cv.log
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "classification"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))

import torch
import yaml
from metrics import mean_std
from pipeline import PROCESSED_ROOT
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_YAML = REPO_ROOT / "data" / "dataset.yaml"
FOLDS_DIR = PROCESSED_ROOT / "folds"
YOLO_ROOT = REPO_ROOT / "models" / "yolo" / "yolo26m" / "runs"
BASE_WEIGHTS = YOLO_ROOT / "yolo26m.pt"  # baixado pelo 01_setup.ipynb

METRICS = ("precision", "recall", "f1", "map50", "map50_95")


def fold_dataset_yaml(fold: int) -> Path:
    """`dataset.yaml` do fold, com caminhos absolutos (o Ultralytics exige `path` absoluto)."""
    base = yaml.safe_load(DATASET_YAML.read_text())
    config = {
        "path": str(DATASET_YAML.parent.resolve()),
        "train": str(FOLDS_DIR / f"fold_{fold}_train.txt"),
        "val": str(FOLDS_DIR / f"fold_{fold}_val.txt"),
        "test": str(PROCESSED_ROOT / "images" / "test"),
        "nc": base["nc"],
        "names": base["names"],
    }
    out = FOLDS_DIR / f"fold_{fold}.yaml"
    out.write_text(yaml.safe_dump(config, sort_keys=False))
    return out


def training_finished(last: Path) -> bool:
    # Ao terminar, o Ultralytics remove o otimizador do last.pt e marca epoch=-1
    # (`strip_optimizer`); retomar um checkpoint assim dá erro.
    return torch.load(last, map_location="cpu", weights_only=False).get("epoch") == -1


def evaluate_on_test(weights: Path, data: Path, project: Path, name: str) -> dict:
    metrics = YOLO(weights).val(data=str(data), split="test", project=str(project), name=name, exist_ok=True)
    precision, recall = float(metrics.box.mp), float(metrics.box.mr)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def summarize(run_dir: Path) -> list[dict]:
    rows = [
        {"fold": path.parent.name, **json.loads(path.read_text())}
        for path in sorted(run_dir.glob("fold*/test_metrics.json"))
    ]
    stats = {name: mean_std([row[name] for row in rows]) for name in METRICS}
    rows += [
        {"fold": "media", **{name: mean for name, (mean, _std) in stats.items()}},
        {"fold": "desvio", **{name: std for name, (_mean, std) in stats.items()}},
    ]
    with open(run_dir / "cv_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def cross_validate(
    run_name: str = "yolo26m-cv",
    folds: list[int] | None = None,
    epochs: int = 125,
    imgsz: int = 640,
    batch: int = -1,
) -> list[dict]:
    if not BASE_WEIGHTS.exists():
        raise FileNotFoundError(f"{BASE_WEIGHTS} não existe -- rode o 01_setup.ipynb do yolo26m primeiro.")
    folds = folds if folds is not None else sorted(
        int(p.stem.split("_")[1]) for p in FOLDS_DIR.glob("fold_*_val.txt")
    )
    run_dir = YOLO_ROOT / run_name

    for fold in folds:
        fold_dir = run_dir / f"fold{fold}"
        if (fold_dir / "test_metrics.json").exists():
            print(f"fold {fold}: já avaliado, pulando.")
            continue
        print(f"===== fold {fold} =====")
        data = fold_dataset_yaml(fold)
        last = fold_dir / "weights" / "last.pt"
        if not last.exists():
            YOLO(BASE_WEIGHTS).train(
                data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, device=0,
                project=str(run_dir), name=f"fold{fold}", exist_ok=True, seed=0,
            )
        elif not training_finished(last):
            YOLO(last).train(resume=True)

        metrics = evaluate_on_test(fold_dir / "weights" / "best.pt", data, run_dir, f"fold{fold}-test")
        (fold_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    rows = summarize(run_dir)
    print(f"{'':8s}" + "".join(f"{name:>12s}" for name in METRICS))
    for row in rows:
        print(f"{row['fold']:8s}" + "".join(f"{row[name]:>12.4f}" for name in METRICS))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", default="yolo26m-cv")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="default: todos os gerados")
    parser.add_argument("--epochs", type=int, default=125)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="-1 = AutoBatch (~60%% da VRAM)")
    args = parser.parse_args()

    cross_validate(args.run_name, folds=args.folds, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
