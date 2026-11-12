"""Avalia a contagem por tipo e o alerta de aneuploidias por metáfase (Fase 5).

Para cada metáfase de validação e de teste:

1. Gabarito: a contagem das anotações (XML) da própria metáfase, interpretada por
   `karyotype.interpret`. Usa a anotação, e não os CSVs do dataset, porque os CSVs descrevem o exame
   do paciente e uma metáfase isolada pode diferir dele (cromossomo perdido na preparação, mosaico).
2. Modo "anotacao": recortes das caixas anotadas -> classificador. Isola o erro do classificador.
3. Modo "detector": recortes das caixas do YOLO -> classificador. É o uso real da pipeline.

As probabilidades do classificador ficam em cache (`outputs/karyotype/probs_{split}.npz`). Sobre
elas, cada método gera a contagem prevista: argmax por recorte ou `karyotype.assign_classes` com
penalidade λ. λ é escolhido na validação e só então aplicado no teste: maior taxa de fórmula exata
no modo anotação (empate -> menor λ). O modo anotação é o critério porque λ corrige erros do
classificador; no modo detector, a mesma métrica continua subindo com λ alto à custa de esconder
desvios verdadeiros (alterações sinalizadas caem). O limiar de confiança do detector é o que
maximiza o F1 na validação.

Uso:
    uv run --group classifier --group yolo python src/classification/evaluate_karyotype.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "detection"))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from crops import clamp_box, crop_box
from dataset import preprocess
from evaluate import load_checkpoint_with_meta
from evaluate_detector import EVAL_ROOT, processed_test_yaml
from karyotype import NORMAL, assign_classes, count_by_class, interpret
from pipeline import PROCESSED_ROOT, RAW_ROOT
from voc_to_yolo import DENVER_CLASSES, parse_annotation

REPO_ROOT = Path(__file__).resolve().parents[2]
DETECTOR_CHECKPOINT = REPO_ROOT / "models" / "yolo" / "yolo26m" / "runs" / "preprocessed" / "best.pt"
CLASSIFIER_CHECKPOINT = REPO_ROOT / "models" / "classifier" / "runs" / "resnet50-paper" / "best.pt"
OUT_DIR = REPO_ROOT / "outputs" / "karyotype"
NUMBER_ABNORMALITIES = RAW_ROOT.parent / "number_abnormalities.csv"

MODES = ("anotacao", "detector")
PENALTIES = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0)


def detector_confidence(weights: Path) -> float:
    """Limiar de confiança que maximiza o F1 no split de validação."""
    from ultralytics import YOLO

    metrics = YOLO(weights).val(
        data=str(processed_test_yaml()), split="val", project=str(EVAL_ROOT),
        name="karyotype_conf", exist_ok=True, plots=False, verbose=False,
    )
    return float(metrics.box.px[np.argmax(metrics.box.f1_curve.mean(axis=0))])


@torch.no_grad()
def classify(crops: list[np.ndarray], classifier, meta: dict, device) -> np.ndarray:
    if not crops:
        return np.empty((0, len(DENVER_CLASSES)), dtype=np.float32)
    batch = torch.stack([preprocess(c, meta["image_size"], meta["resize"]) for c in crops])
    return functional.softmax(classifier(batch.to(device)), dim=1).cpu().numpy().astype(np.float32)


def crops_from_boxes(gray: np.ndarray, boxes) -> list[np.ndarray]:
    height, width = gray.shape[:2]
    crops = []
    for xmin, ymin, xmax, ymax in boxes:
        box = clamp_box(xmin, ymin, xmax, ymax, width, height)
        if box is not None:
            crops.append(crop_box(gray, box))
    return crops


def collect_probs(split: str, detector_weights: Path, classifier_weights: Path, conf: float) -> dict:
    """{imagem: {"gt": classes anotadas, "anotacao": probs, "detector": probs}} de um split."""
    from ultralytics import YOLO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = YOLO(str(detector_weights))
    classifier, meta = load_checkpoint_with_meta(classifier_weights, device)

    records = {}
    image_paths = sorted((PROCESSED_ROOT / "images" / split).glob("*.jpg"))
    for i, path in enumerate(image_paths, start=1):
        objects = parse_annotation(RAW_ROOT / "annotations" / f"{path.stem}.xml")["objects"]
        gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
        gt_boxes = [(o["xmin"], o["ymin"], o["xmax"], o["ymax"]) for o in objects]
        det_boxes = detector.predict(str(path), conf=conf, verbose=False)[0].boxes.xyxy.cpu().numpy()
        records[path.stem] = {
            "gt": [o["name"] for o in objects],
            "anotacao": classify(crops_from_boxes(gray, gt_boxes), classifier, meta, device),
            "detector": classify(crops_from_boxes(gray, det_boxes), classifier, meta, device),
        }
        if i % 100 == 0 or i == len(image_paths):
            print(f"  {split}: {i}/{len(image_paths)} metáfases")
    return records


def save_records(records: dict, path: Path) -> None:
    arrays = {}
    for name, record in records.items():
        arrays[f"{name}|gt"] = np.array(record["gt"])
        for mode in MODES:
            arrays[f"{name}|{mode}"] = record[mode]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_records(path: Path) -> dict:
    records: dict[str, dict] = {}
    with np.load(path) as data:
        for key in data.files:
            name, field = key.split("|")
            value = data[key]
            records.setdefault(name, {})[field] = value.tolist() if field == "gt" else value
    return records


def predicted_labels(probs: np.ndarray, penalty: float | None) -> list[str]:
    if penalty is None:
        return [DENVER_CLASSES[i] for i in probs.argmax(axis=1)]
    return assign_classes(probs, extra_penalty=penalty)


def per_image(records: dict, mode: str, penalty: float | None) -> pd.DataFrame:
    rows = []
    for name, record in sorted(records.items()):
        gt_counts = count_by_class(record["gt"])
        pred_counts = count_by_class(predicted_labels(record[mode], penalty))
        gt, pred = interpret(gt_counts), interpret(pred_counts)
        rows.append({
            "imagem": name,
            "gabarito": gt.formula,
            "status_gabarito": gt.status,
            "previsto": pred.formula,
            "status_previsto": pred.status,
            "achados": "; ".join(pred.findings),
            "total_correto": sum(pred_counts.values()) == sum(gt_counts.values()),
            "formula_exata": pred_counts == gt_counts,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    gt_normal = df[df.status_gabarito == NORMAL]
    gt_altered = df[df.status_gabarito != NORMAL]
    return {
        "metafases": len(df),
        "total_correto": df.total_correto.mean(),
        "formula_exata": df.formula_exata.mean(),
        # separa o erro do detector (total errado) do erro de classificação (total certo, tipos errados)
        "formula_exata_se_total_correto": df[df.total_correto].formula_exata.mean(),
        "normais": len(gt_normal),
        "alerta_falso": (gt_normal.status_previsto != NORMAL).mean(),
        "alteradas": len(gt_altered),
        "alteradas_sinalizadas": (gt_altered.status_previsto != NORMAL).mean(),
        "alteradas_formula_exata": gt_altered.formula_exata.mean(),
    }


def choose_penalty(val_records: dict) -> tuple[float | None, pd.DataFrame]:
    sweep = []
    for mode in MODES:
        for penalty in (None, *PENALTIES):
            summary = summarize(per_image(val_records, mode, penalty))
            sweep.append({"modo": mode, "lambda": "argmax" if penalty is None else penalty, **summary})
    table = pd.DataFrame(sweep)
    candidates = table[table.modo == "anotacao"]
    best = candidates.sort_values("formula_exata", ascending=False, kind="stable").iloc[0]["lambda"]
    return (None if best == "argmax" else float(best)), table


def dataset_numerical_cases(split_records: dict) -> dict[str, str]:
    """Metáfases do split listadas em `number_abnormalities.csv`, com a descrição do dataset."""
    with open(NUMBER_ABNORMALITIES, newline="") as f:
        rows = {Path(r["filename"]).stem: r for r in csv.DictReader(f)}
    return {
        name: f"{rows[name]['chromosome numbers']} cromossomos, par: {rows[name]['description (abnormal chromosome-pair)']}"
        for name in sorted(rows.keys() & split_records.keys())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--detector", type=Path, default=DETECTOR_CHECKPOINT)
    parser.add_argument("--classifier", type=Path, default=CLASSIFIER_CHECKPOINT)
    parser.add_argument("--conf", type=float, default=None, help="padrão: o que maximiza o F1 na validação")
    parser.add_argument("--recompute", action="store_true", help="ignora o cache de probabilidades")
    args = parser.parse_args()

    records = {}
    for split in ("val", "test"):
        cache = OUT_DIR / f"probs_{split}.npz"
        if cache.exists() and not args.recompute:
            records[split] = load_records(cache)
            continue
        if args.conf is None:
            args.conf = detector_confidence(args.detector)
            print(f"Limiar de confiança do detector (máx. F1 na validação): {args.conf:.3f}")
        records[split] = collect_probs(split, args.detector, args.classifier, args.conf)
        save_records(records[split], cache)

    penalty, sweep = choose_penalty(records["val"])
    print("\n## Validação: escolha de λ (critério: fórmula exata no modo anotação)\n")
    print(sweep.to_markdown(index=False, floatfmt=".4f"))
    print(f"\nλ escolhido: {'argmax' if penalty is None else penalty}")

    summaries = []
    for mode in MODES:
        for label, value in (("argmax", None), (f"λ = {penalty}", penalty)):
            if label != "argmax" and penalty is None:
                continue
            df = per_image(records["test"], mode, value)
            df.to_csv(OUT_DIR / f"test_{mode}_{'argmax' if value is None else 'restricao'}.csv", index=False)
            summaries.append({"modo": mode, "metodo": label, **summarize(df)})
    print("\n## Teste: contagem e alerta por metáfase\n")
    print(pd.DataFrame(summaries).to_markdown(index=False, floatfmt=".4f"))

    print("\n## Teste: metáfases com alteração numérica no dataset (modo detector)\n")
    cases = dataset_numerical_cases(records["test"])
    detector_runs = {
        label: per_image(records["test"], "detector", value).set_index("imagem")
        for label, value in (("argmax", None), ("restricao", penalty))
    }
    rows = [
        {
            "imagem": name,
            "dataset": description,
            "gabarito (anotação)": detector_runs["argmax"].loc[name, "gabarito"],
            "argmax": detector_runs["argmax"].loc[name, "previsto"],
            "com restrição": detector_runs["restricao"].loc[name, "previsto"],
            "achados (com restrição)": detector_runs["restricao"].loc[name, "achados"],
        }
        for name, description in cases.items()
    ]
    print(pd.DataFrame(rows).to_markdown(index=False))

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(
        {"conf": args.conf, "lambda": penalty, "sweep_val": sweep.to_dict("records"), "test": summaries},
        indent=2, ensure_ascii=False, default=float,
    ))
    print(f"\nResumo salvo em {summary_path}")


if __name__ == "__main__":
    main()
