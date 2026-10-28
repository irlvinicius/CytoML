"""Extrai um recorte por cromossomo anotado, organizado por split e classe.

O split de cada imagem vem das pastas `data/processed/24_chromosomes_object/images/{train,val,test}`
(o mesmo que o detector usa), garantindo que recortes da mesma metáfase nunca fiquem em splits
diferentes. Os recortes saem da imagem já processada (mediana + CLAHE) e a classe real vem do XML
bruto (24 classes de Denver).

Saída em `data/processed/classification/`:
    {split}/{classe}/{imagem}_{idx}.png
    manifest.csv
"""

import argparse
import csv
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from crops import clamp_box, crop_box
from voc_to_yolo import DENVER_CLASS_TO_ID, parse_annotation

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "data" / "raw" / "24_chromosomes_object"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed" / "24_chromosomes_object"
OUT_ROOT = REPO_ROOT / "data" / "processed" / "classification"

SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = [
    "split",
    "image",
    "idx",
    "class_name",
    "class_id",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "path",
]


def list_split_images(split: str, processed_root: Path = PROCESSED_ROOT) -> list[str]:
    return sorted(p.stem for p in (processed_root / "images" / split).glob("*.jpg"))


def extract_image(
    task: tuple[str, str, Path, Path, Path],
) -> tuple[list[dict], int]:
    """Recorta todos os cromossomos de uma imagem. Devolve (linhas do manifest, nº de boxes descartadas)."""
    name, split, raw_root, processed_root, out_root = task

    annotation = parse_annotation(raw_root / "annotations" / f"{name}.xml")
    image = cv2.imread(str(processed_root / "images" / split / f"{name}.jpg"))
    if image is None:
        raise FileNotFoundError(f"Imagem processada não encontrada/ilegível: {name} ({split})")
    height, width = image.shape[:2]

    rows = []
    skipped = 0
    for idx, obj in enumerate(annotation["objects"]):
        box = clamp_box(obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"], width, height)
        if box is None:
            skipped += 1
            continue

        class_name = obj["name"]
        rel_path = Path(split) / class_name / f"{name}_{idx}.png"
        out_path = out_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), crop_box(image, box))

        rows.append(
            {
                "split": split,
                "image": name,
                "idx": idx,
                "class_name": class_name,
                "class_id": DENVER_CLASS_TO_ID[class_name],
                "xmin": box[0],
                "ymin": box[1],
                "xmax": box[2],
                "ymax": box[3],
                "path": rel_path.as_posix(),
            }
        )
    return rows, skipped


def run(
    raw_root: Path = RAW_ROOT,
    processed_root: Path = PROCESSED_ROOT,
    out_root: Path = OUT_ROOT,
    workers: int = 1,
) -> dict[str, int]:
    split_images = {split: list_split_images(split, processed_root) for split in SPLITS}

    seen: dict[str, str] = {}
    for split, names in split_images.items():
        for name in names:
            if name in seen:
                raise ValueError(f"Imagem '{name}' aparece em '{seen[name]}' e '{split}'.")
            seen[name] = split

    # Limpa a saída anterior: o mapeamento imagem -> split pode ter mudado entre execuções.
    shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = [
        (name, split, raw_root, processed_root, out_root)
        for split, names in split_images.items()
        for name in names
    ]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(extract_image, tasks, chunksize=16))
    else:
        results = [extract_image(task) for task in tasks]

    all_rows = [row for rows, _ in results for row in rows]
    skipped = sum(n for _, n in results)

    with open(out_root / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    counts = {"images": len(tasks), "crops": len(all_rows), "skipped": skipped}
    for split in SPLITS:
        counts[split] = sum(1 for row in all_rows if row["split"] == split)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=1, help="processos em paralelo (por imagem)")
    args = parser.parse_args()

    counts = run(workers=args.workers)
    print(f"Imagens processadas: {counts['images']}")
    print(f"Recortes: {counts['crops']} (descartados por bbox vazia: {counts['skipped']})")
    for split in SPLITS:
        print(f"  {split}: {counts[split]}")
