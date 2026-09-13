"""Orquestra o pré-processamento completo sobre `data/raw/24_chromosomes_object`.

O dataset vem com uma partição oficial em `train.txt`/`test.txt`, mas em
proporções pouco padronizadas (~61%/39% do pool total). Para ter uma divisão
treino/validação/teste nos moldes convencionais (70/15/15), as duas listas são
combinadas num pool único e reparticionadas do zero, embaralhadas com seed fixa, 
`test.txt` deixa de ser "o teste original dos autores do dataset" e passa a ser
apenas a fatia de 15% definida aqui.

Saída em `data/processed/24_chromosomes_object/{images,labels}/{train,val,test}`,
nos caminhos já esperados por `data/dataset.yaml`.

`generate_kfold_splits` particiona em k folds o pool de treino+validação (85%
do total), sempre mantendo o teste (15%) de fora — as imagens já processadas em
`images/train`/`images/val` são reaproveitadas, sem reprocessar nada: cada fold
só gera listas de caminhos (`.txt`, um por linha), no formato que o Ultralytics
aceita como `train`/`val` no lugar de uma pasta.
"""

import random
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filters import denoise_and_enhance
from voc_to_yolo import convert_annotation

RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "24_chromosomes_object"
PROCESSED_ROOT = Path(__file__).resolve().parents[2] / "data" / "processed" / "24_chromosomes_object"
TRAIN_LIST = RAW_ROOT.parent / "train.txt"
TEST_LIST = RAW_ROOT.parent / "test.txt"

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_SEED = 42
KFOLD_SEED = 42


def load_existing_pool(list_path: Path) -> list[str]:
    names = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    existing = {p.stem for p in (RAW_ROOT / "JEPG").glob("*.jpg")}
    return sorted(name for name in names if name in existing)


def load_full_pool() -> list[str]:
    return sorted(set(load_existing_pool(TRAIN_LIST)) | set(load_existing_pool(TEST_LIST)))


def split_train_val_test(pool: list[str]) -> tuple[list[str], list[str], list[str]]:
    shuffled = pool.copy()
    random.Random(SPLIT_SEED).shuffle(shuffled)
    n = len(shuffled)
    n_val = round(n * VAL_FRACTION)
    n_test = round(n * TEST_FRACTION)
    val_names = shuffled[:n_val]
    test_names = shuffled[n_val : n_val + n_test]
    train_names = shuffled[n_val + n_test :]
    return train_names, val_names, test_names


def process_split(names: list[str], split: str, **filter_kwargs) -> None:
    images_out = PROCESSED_ROOT / "images" / split
    labels_out = PROCESSED_ROOT / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for name in names:
        image_path = RAW_ROOT / "JEPG" / f"{name}.jpg"
        xml_path = RAW_ROOT / "annotations" / f"{name}.xml"

        image = cv2.imread(str(image_path))
        processed = denoise_and_enhance(image, **filter_kwargs)
        cv2.imwrite(str(images_out / f"{name}.jpg"), processed)

        convert_annotation(xml_path, labels_out / f"{name}.txt")


def run(**filter_kwargs) -> dict[str, int]:
    # Limpa saída de execuções anteriores: o mapeamento nome -> split pode ter
    # mudado (ex.: troca de proporção do split), então reprocessar por cima da
    # estrutura antiga deixaria arquivos de splits antigos misturados.
    for subdir in ("images", "labels"):
        shutil.rmtree(PROCESSED_ROOT / subdir, ignore_errors=True)

    pool = load_full_pool()
    train_names, val_names, test_names = split_train_val_test(pool)

    process_split(train_names, "train", **filter_kwargs)
    process_split(val_names, "val", **filter_kwargs)
    process_split(test_names, "test", **filter_kwargs)

    return {
        "pool": len(pool),
        "train": len(train_names),
        "val": len(val_names),
        "test": len(test_names),
    }


def make_folds(pool: list[str], k: int) -> list[list[str]]:
    shuffled = pool.copy()
    random.Random(KFOLD_SEED).shuffle(shuffled)
    return [shuffled[i::k] for i in range(k)]


def resolve_processed_image(name: str) -> Path:
    for split in ("train", "val"):
        candidate = PROCESSED_ROOT / "images" / split / f"{name}.jpg"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"'{name}' não está em images/train nem images/val — rode `run()` primeiro."
    )


def generate_kfold_splits(k: int = 5, out_dir: Path | None = None) -> list[dict[str, int]]:
    """Gera k folds de treino/validação sobre o pool de treino+validação (85%
    do total), sem reprocessar imagens. O teste (15%) nunca entra em nenhum
    fold. Cada fold vira dois arquivos `.txt` (um caminho de imagem processada
    por linha) em `out_dir` (default: `data/processed/24_chromosomes_object/folds/`).
    """
    out_dir = out_dir or (PROCESSED_ROOT / "folds")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = load_full_pool()
    train_names, val_names, _test_names = split_train_val_test(pool)
    train_val_pool = sorted(train_names + val_names)
    folds = make_folds(train_val_pool, k)

    counts = []
    for i in range(k):
        val_fold = folds[i]
        train_fold = [name for j, fold in enumerate(folds) if j != i for name in fold]

        train_txt = out_dir / f"fold_{i}_train.txt"
        val_txt = out_dir / f"fold_{i}_val.txt"
        train_txt.write_text("\n".join(str(resolve_processed_image(n)) for n in train_fold) + "\n")
        val_txt.write_text("\n".join(str(resolve_processed_image(n)) for n in val_fold) + "\n")

        counts.append({"fold": i, "train": len(train_fold), "val": len(val_fold)})

    return counts


if __name__ == "__main__":
    counts = run()
    print(f"Pool total (train.txt + test.txt filtrados): {counts['pool']}")
    print(f"Train: {counts['train']}")
    print(f"Val:   {counts['val']}")
    print(f"Test:  {counts['test']}")
