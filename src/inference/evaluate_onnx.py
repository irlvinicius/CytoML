"""Confere que a pipeline ONNX (`onnx_pipeline.py`, usada no site) reproduz a Tabela 5 do artigo.

Roda a pipeline nas metáfases de teste e resume com as mesmas funções de `evaluate_karyotype.py`
(`per_image` + `summarize`), comparando com o `outputs/karyotype/summary.json` (PyTorch + Ultralytics).

Duas entradas possíveis:
- `--source processed` (padrão): as imagens de `data/processed`, já filtradas, como na avaliação do artigo;
- `--source raw`: as imagens originais, com a mediana + CLAHE aplicada pela própria pipeline, que é
  o caminho do site.

Uso:
    uv run --no-active --group classifier python src/inference/evaluate_onnx.py --source raw
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "classification"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "detection"))

import cv2
import pandas as pd
from evaluate_karyotype import OUT_DIR, dataset_numerical_cases, per_image, summarize
from onnx_pipeline import CLASSIFIER_ONNX, DETECTOR_ONNX, EXTRA_PENALTY, KaryotypePipeline
from pipeline import PROCESSED_ROOT, RAW_ROOT
from voc_to_yolo import parse_annotation

ARTICLE_KEYS = ("total_correto", "formula_exata", "alerta_falso", "alteradas_sinalizadas")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("processed", "raw"), default="processed")
    parser.add_argument("--detector", type=Path, default=DETECTOR_ONNX)
    parser.add_argument("--classifier", type=Path, default=CLASSIFIER_ONNX)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="só as N primeiras metáfases")
    parser.add_argument("--cpu", action="store_true", help="força CPU (o site roda em CPU)")
    args = parser.parse_args()

    providers = ["CPUExecutionProvider"] if args.cpu else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    pipe = KaryotypePipeline(args.detector, args.classifier, providers=providers)

    names = sorted(p.stem for p in (PROCESSED_ROOT / "images" / args.split).glob("*.jpg"))[: args.limit]
    records, seconds = {}, []
    for i, name in enumerate(names, start=1):
        if args.source == "processed":
            path = PROCESSED_ROOT / "images" / args.split / f"{name}.jpg"
        else:
            path = RAW_ROOT / "JEPG" / f"{name}.jpg"
        image = cv2.imread(str(path))
        start = time.perf_counter()
        result = pipe.run(image, preprocessed=args.source == "processed")
        seconds.append(time.perf_counter() - start)
        gt = [o["name"] for o in parse_annotation(RAW_ROOT / "annotations" / f"{name}.xml")["objects"]]
        records[name] = {"gt": gt, "detector": result.probs}
        if i % 100 == 0 or i == len(names):
            print(f"  {i}/{len(names)} metáfases ({sum(seconds) / len(seconds):.2f} s por metáfase)")

    df = per_image(records, "detector", EXTRA_PENALTY)
    summary = summarize(df)
    tag = f"onnx_{args.source}_{args.detector.stem}_{args.classifier.stem}"
    df.to_csv(OUT_DIR / f"{args.split}_{tag}.csv", index=False)

    reference = None
    summary_path = OUT_DIR / "summary.json"
    if summary_path.exists() and args.split == "test" and args.limit is None:
        tests = json.loads(summary_path.read_text())["test"]
        reference = next(t for t in tests if t["modo"] == "detector" and t["metodo"] != "argmax")

    rows = [{"métrica": k, "onnx": summary[k], "artigo": reference[k] if reference else None} for k in ARTICLE_KEYS]
    print(f"\n## {args.split}: pipeline ONNX ({args.source}) x artigo\n")
    print(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"))

    cases = dataset_numerical_cases(records)
    if cases:
        print("\n## Aneuploidias registradas no dataset\n")
        print(df.set_index("imagem").loc[list(cases), ["gabarito", "previsto", "status_previsto"]].to_markdown())
    print(f"\nTempo médio por metáfase: {sum(seconds) / len(seconds):.2f} s ({providers[0]})")


if __name__ == "__main__":
    main()
