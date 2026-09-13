"""Baixa o dataset de cariotipagem do Kaggle para data/raw/.

Requer credenciais da Kaggle API configuradas (arquivo ~/.kaggle/kaggle.json
ou variáveis de ambiente KAGGLE_USERNAME / KAGGLE_KEY). Veja:
https://www.kaggle.com/docs/api#authentication

O slug do dataset é informado via variável de ambiente KAGGLE_DATASET. Dataset
usado no projeto: aliabedimadiseh/chromosome-image-dataset-karyotype
(https://www.kaggle.com/datasets/aliabedimadiseh/chromosome-image-dataset-karyotype).

Uso:
    KAGGLE_DATASET="aliabedimadiseh/chromosome-image-dataset-karyotype" \
        uv run python src/data/download_dataset.py
"""

import os
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DEFAULT_DATASET = "aliabedimadiseh/chromosome-image-dataset-karyotype"


def main() -> None:
    dataset_slug = os.environ.get("KAGGLE_DATASET", DEFAULT_DATASET)

    from kaggle.api.kaggle_api_extended import KaggleApi

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Baixando dataset '{dataset_slug}' para {RAW_DIR} ...")
    api.dataset_download_files(dataset_slug, path=str(RAW_DIR), quiet=False)

    zip_path = RAW_DIR / f"{dataset_slug.split('/')[-1]}.zip"
    if zip_path.exists():
        print(f"Extraindo {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()

    _flatten_single_wrapper_dir(RAW_DIR)
    print(f"Dataset disponível em: {RAW_DIR}")


def _flatten_single_wrapper_dir(raw_dir: Path) -> None:
    """Alguns exports do Kaggle vêm com o conteúdo do zip dentro de uma única
    pasta extra (ex.: "Data/"), em vez de já na raiz de data/raw/. Se isso
    acontecer, sobe o conteúdo dessa pasta um nível e remove o wrapper.
    """
    entries = [p for p in raw_dir.iterdir() if p.name != ".gitkeep"]
    if len(entries) == 1 and entries[0].is_dir():
        wrapper = entries[0]
        for item in wrapper.iterdir():
            item.rename(raw_dir / item.name)
        wrapper.rmdir()


if __name__ == "__main__":
    main()
