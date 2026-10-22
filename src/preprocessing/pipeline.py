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
`images/train`/`images/val` são reaproveitadas, sem reprocessar nada. Os folds
são estratificados por sexo (XX/XY) × tipo de cariótipo (normal, anomalia
numérica, anomalia estrutural) e agrupados por caso: metáfases do mesmo caso
caem sempre no mesmo fold (ver `case_id`). Saída em `folds/`: `folds.csv`
(imagem → fold, lido pelo classificador) e listas de caminhos
`fold_i_{train,val}.txt` (um por linha), no formato que o Ultralytics aceita
como `train`/`val` no lugar de uma pasta.
"""

import csv
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filters import denoise_and_enhance
from voc_to_yolo import convert_annotation, parse_annotation

RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "24_chromosomes_object"
PROCESSED_ROOT = Path(__file__).resolve().parents[2] / "data" / "processed" / "24_chromosomes_object"
TRAIN_LIST = RAW_ROOT.parent / "train.txt"
TEST_LIST = RAW_ROOT.parent / "test.txt"
KARYOTYPE_CSVS = {
    "numerica": RAW_ROOT.parent / "number_abnormalities.csv",
    "estrutural": RAW_ROOT.parent / "structural_abnormalities.csv",
}

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


def case_id(name: str) -> str:
    """Caso (paciente) de uma imagem, inferido do nome do arquivo.

    O dataset não documenta a nomenclatura, mas os nomes seguem `<caso><célula>[_n]`
    (ex.: `1050761`…`1050764`): as imagens com o mesmo prefixo compartilham a mesma
    anomalia nos CSVs, então tratamos o prefixo como o caso. Nomes fora desse
    padrão (ex.: `unknow395`) viram um caso próprio.
    """
    match = re.fullmatch(r"(\d+)\d(_\d+)?", name)
    return match.group(1) if match else name


def load_karyotype_categories() -> dict[str, str]:
    """Imagem -> "numerica"/"estrutural", pelos CSVs do dataset; ausente = normal."""
    categories = {}
    for category, csv_path in KARYOTYPE_CSVS.items():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                categories[Path(row["filename"].strip()).stem] = category
    return categories


def image_strata(names: list[str]) -> dict[str, str]:
    """Estrato de cada imagem: sexo pela anotação (XY se houver algum Y) × tipo de cariótipo."""
    categories = load_karyotype_categories()
    strata = {}
    for name in names:
        objects = parse_annotation(RAW_ROOT / "annotations" / f"{name}.xml")["objects"]
        sex = "XY" if any(obj["name"] == "Y" for obj in objects) else "XX"
        strata[name] = f"{sex}/{categories.get(name, 'normal')}"
    return strata


def make_folds(
    pool: list[str],
    k: int,
    strata: dict[str, str] | None = None,
    groups: dict[str, str] | None = None,
) -> list[list[str]]:
    """Divide `pool` em k folds estratificados, sem separar imagens do mesmo grupo.

    Cada grupo herda o estrato mais frequente entre suas imagens. Dentro de cada
    estrato, os grupos (do maior para o menor) vão para o fold com menos imagens
    daquele estrato -- no empate, o fold com menos imagens no total. Sem `strata`
    ou `groups`, todas as imagens ficam no mesmo estrato / cada uma no seu grupo.
    """
    strata = strata or {}
    groups = groups or {}

    members = defaultdict(list)
    for name in sorted(pool):
        members[groups.get(name, name)].append(name)
    group_ids = sorted(members)
    random.Random(KFOLD_SEED).shuffle(group_ids)

    by_stratum = defaultdict(list)
    for group in group_ids:
        stratum = Counter(strata.get(name, "") for name in members[group]).most_common(1)[0][0]
        by_stratum[stratum].append(group)

    folds = [[] for _ in range(k)]
    for stratum in sorted(by_stratum):
        stratum_sizes = [0] * k
        # sort estável: grupos do mesmo tamanho mantêm a ordem embaralhada
        for group in sorted(by_stratum[stratum], key=lambda g: -len(members[g])):
            i = min(range(k), key=lambda j: (stratum_sizes[j], len(folds[j]), j))
            folds[i].extend(members[group])
            stratum_sizes[i] += len(members[group])
    return folds


def resolve_processed_image(name: str) -> Path:
    for split in ("train", "val"):
        candidate = PROCESSED_ROOT / "images" / split / f"{name}.jpg"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"'{name}' não está em images/train nem images/val — rode `run()` primeiro."
    )


def generate_kfold_splits(k: int = 5, out_dir: Path | None = None) -> list[dict]:
    """Gera k folds de treino/validação sobre o pool de treino+validação (85%
    do total), sem reprocessar imagens. O teste (15%) nunca entra em nenhum
    fold. Grava em `out_dir` (default: `data/processed/24_chromosomes_object/folds/`)
    o `folds.csv` (imagem, caso, estrato, fold) e, por fold, dois `.txt` com um
    caminho de imagem processada por linha.
    """
    out_dir = out_dir or (PROCESSED_ROOT / "folds")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = load_full_pool()
    train_names, val_names, _test_names = split_train_val_test(pool)
    train_val_pool = sorted(train_names + val_names)
    strata = image_strata(train_val_pool)
    groups = {name: case_id(name) for name in train_val_pool}
    folds = make_folds(train_val_pool, k, strata=strata, groups=groups)

    with open(out_dir / "folds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "case", "stratum", "fold"])
        for i, fold in enumerate(folds):
            writer.writerows([name, groups[name], strata[name], i] for name in sorted(fold))

    counts = []
    for i in range(k):
        val_fold = folds[i]
        train_fold = [name for j, fold in enumerate(folds) if j != i for name in fold]

        train_txt = out_dir / f"fold_{i}_train.txt"
        val_txt = out_dir / f"fold_{i}_val.txt"
        train_txt.write_text("\n".join(str(resolve_processed_image(n)) for n in train_fold) + "\n")
        val_txt.write_text("\n".join(str(resolve_processed_image(n)) for n in val_fold) + "\n")

        counts.append(
            {"fold": i, "train": len(train_fold), "val": len(val_fold),
             **Counter(strata[n] for n in val_fold)}
        )

    return counts


def load_fold(fold: int, folds_csv: Path | None = None) -> tuple[list[str], list[str]]:
    """(imagens de treino, imagens de validação) do fold `fold`, lidas do `folds.csv`."""
    folds_csv = folds_csv or (PROCESSED_ROOT / "folds" / "folds.csv")
    if not folds_csv.exists():
        raise FileNotFoundError(f"{folds_csv} não existe -- rode `generate_kfold_splits()` primeiro.")
    with open(folds_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if fold not in {int(row["fold"]) for row in rows}:
        raise ValueError(f"fold {fold} não existe em {folds_csv}")
    train = [row["image"] for row in rows if int(row["fold"]) != fold]
    val = [row["image"] for row in rows if int(row["fold"]) == fold]
    return train, val


if __name__ == "__main__":
    counts = run()
    print(f"Pool total (train.txt + test.txt filtrados): {counts['pool']}")
    print(f"Train: {counts['train']}")
    print(f"Val:   {counts['val']}")
    print(f"Test:  {counts['test']}")
