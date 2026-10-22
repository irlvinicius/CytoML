"""Validação cruzada do classificador: um modelo por fold, todos avaliados no mesmo teste.

Para cada fold `i` de `folds.csv` (ver `src/preprocessing/pipeline.py::generate_kfold_splits`), treina
com a configuração padrão (`train.DEFAULT_CONFIG`) nos outros folds, usando o fold `i` como validação
(early stopping e escolha do checkpoint), e avalia o melhor checkpoint no split de teste fixo. O
resultado final é a média ± desvio padrão entre os folds, gravado em `cv_summary.csv`.

Avaliar no teste fixo -- e não no próprio fold de validação -- evita o viés de medir no mesmo
conjunto que escolheu o checkpoint; o desvio entre folds mede quanto o resultado depende da
partição de treino.

Retomável: folds já avaliados (`test_probs.npy` presente) são pulados, e um fold interrompido
continua de `last.pt`. Rodar dentro do tmux:

    PYTHONUNBUFFERED=1 uv run python src/classification/cross_validate.py --run-name resnet50-paper-cv \\
        | tee models/classifier/runs/resnet50-paper-cv.log
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))

import numpy as np
from evaluate import evaluate
from metrics import SCALAR_METRICS, classification_metrics, summarize_folds
from pipeline import PROCESSED_ROOT
from train import RUNS_ROOT, train

FOLDS_CSV = PROCESSED_ROOT / "folds" / "folds.csv"


def available_folds(folds_csv: Path = FOLDS_CSV) -> list[int]:
    with open(folds_csv, newline="") as f:
        return sorted({int(row["fold"]) for row in csv.DictReader(f)})


def summarize(run_dir: Path) -> list[dict]:
    """Métricas de teste por fold + linhas de média e desvio; grava `cv_summary.csv` em `run_dir`."""
    rows = []
    for fold_dir in sorted(run_dir.glob("fold*")):
        probs_path = fold_dir / "test_probs.npy"
        if not probs_path.exists():
            continue
        metrics = classification_metrics(np.load(probs_path), np.load(fold_dir / "test_labels.npy"))
        rows.append({"fold": fold_dir.name, **{name: metrics[name] for name in SCALAR_METRICS}})

    summary = summarize_folds(rows)
    rows += [
        {"fold": "media", **{name: mean for name, (mean, _std) in summary.items()}},
        {"fold": "desvio", **{name: std for name, (_mean, std) in summary.items()}},
    ]
    with open(run_dir / "cv_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", *SCALAR_METRICS])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def cross_validate(
    run_name: str,
    folds: list[int] | None = None,
    num_workers: int = 8,
    **overrides,
) -> list[dict]:
    folds = available_folds() if folds is None else folds
    run_dir = RUNS_ROOT / run_name
    for fold in folds:
        fold_dir = run_dir / f"fold{fold}"
        if (fold_dir / "test_probs.npy").exists():
            print(f"fold {fold}: já avaliado, pulando.")
            continue
        print(f"===== fold {fold} =====")
        best_path = train(run_name=run_name, fold=fold, resume=True, num_workers=num_workers, **overrides)
        evaluate(best_path, num_workers=num_workers)

    rows = summarize(run_dir)
    print(f"{'':8s}" + "".join(f"{name:>17s}" for name in SCALAR_METRICS))
    for row in rows:
        print(f"{row['fold']:8s}" + "".join(f"{row[name]:>17.4f}" for name in SCALAR_METRICS))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", default="resnet50-paper-cv")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="default: todos de folds.csv")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    cross_validate(args.run_name, folds=args.folds, num_workers=args.num_workers)
