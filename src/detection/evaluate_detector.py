"""Avaliação do detector no teste, com comparação justa contra o baseline do TCC1.

O baseline do TCC1 (`models/yolo/yolo26m/runs/baseline/best.pt`) foi treinado com imagens cruas na
partição oficial do dataset (`train.txt`). O split de teste atual (15%, `pipeline.py`) foi sorteado do
pool `train.txt + test.txt`, então ~75% dele estava no treino do baseline -- avaliá-lo ali o
favoreceria. Por isso a comparação entre os dois usa só `unseen_test_subset()`: as imagens do teste
atual que também estão no `test.txt` oficial, que nenhum dos modelos viu. Cada modelo pode receber a
imagem crua (`source="raw"`, como o baseline foi treinado) ou com mediana + CLAHE
(`source="processed"`, como os modelos do pipeline novo).

AUC-ROC não é uma métrica de detecção: não existe um conjunto fixo de negativos (qualquer região da
imagem sem cromossomo seria um), então não há taxa de falsos positivos definida. O equivalente usado
em detecção é a curva precisão × recall (cuja área é o AP) e a curva F1 × confiança, que mostra o
limiar de confiança que equilibra precisão e recall.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))

import numpy as np
import yaml
from pipeline import (
    PROCESSED_ROOT,
    RAW_ROOT,
    TEST_LIST,
    load_existing_pool,
    load_full_pool,
    split_train_val_test,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_YAML = REPO_ROOT / "data" / "dataset.yaml"
EVAL_ROOT = PROCESSED_ROOT.parent / "eval"
SOURCES = ("raw", "processed")


def unseen_test_subset() -> list[str]:
    """Imagens do teste atual que também estão no `test.txt` oficial (fora do treino do baseline)."""
    _train, _val, test = split_train_val_test(load_full_pool())
    official_test = set(load_existing_pool(TEST_LIST))
    return sorted(name for name in test if name in official_test)


def build_eval_set(
    name: str,
    names: list[str],
    source: str,
    eval_root: Path = EVAL_ROOT,
    raw_images: Path = RAW_ROOT / "JEPG",
    processed_root: Path = PROCESSED_ROOT,
) -> Path:
    """Monta `eval_root/<name>/{images,labels}/test` (imagens por link simbólico, labels do teste) e
    devolve o `dataset.yaml` correspondente. Os labels são os mesmos para as duas fontes: o filtro
    não muda o tamanho da imagem, então as coordenadas normalizadas valem para ambas."""
    if source not in SOURCES:
        raise ValueError(f"source desconhecido: {source!r} (opções: {SOURCES})")
    root = eval_root / name
    shutil.rmtree(root, ignore_errors=True)
    images, labels = root / "images" / "test", root / "labels" / "test"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    image_dir = raw_images if source == "raw" else processed_root / "images" / "test"
    for image_name in names:
        (images / f"{image_name}.jpg").symlink_to((image_dir / f"{image_name}.jpg").resolve())
        shutil.copy(processed_root / "labels" / "test" / f"{image_name}.txt", labels)

    base = yaml.safe_load(DATASET_YAML.read_text())
    # o Ultralytics exige `train`/`val` mesmo só validando no `test`; apontam pro próprio teste
    config = {
        "path": str(root.resolve()),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "nc": base["nc"],
        "names": base["names"],
    }
    data_yaml = root / "dataset.yaml"
    data_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
    return data_yaml


def processed_test_yaml() -> Path:
    """`dataset.yaml` do split de teste completo (imagens processadas), com `path` absoluto."""
    base = yaml.safe_load(DATASET_YAML.read_text())
    config = {**base, "path": str(DATASET_YAML.parent.resolve())}
    out = EVAL_ROOT / "test_processed.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(config, sort_keys=False))
    return out


def evaluate_detector(weights: Path, data_yaml: Path, name: str, project: Path) -> dict:
    """Valida `weights` no split `test` de `data_yaml`. Precision/recall são os do Ultralytics: no
    limiar de confiança que maximiza o F1. Devolve também as curvas PR e F1 × confiança."""
    from ultralytics import YOLO

    metrics = YOLO(weights).val(
        data=str(data_yaml), split="test", project=str(project), name=name, exist_ok=True, plots=True
    )
    box = metrics.box
    precision, recall = float(box.mp), float(box.mr)
    confidence, f1_curve = box.px, box.f1_curve.mean(axis=0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "best_conf": float(confidence[np.argmax(f1_curve)]),
        "curves": {
            "pr": (box.px, box.prec_values.mean(axis=0)),
            "f1_conf": (confidence, f1_curve),
        },
    }
