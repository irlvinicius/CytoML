"""Baixa o dataset de cariotipagem do Kaggle para data/raw/.

Requer credenciais da Kaggle API configuradas (arquivo ~/.kaggle/kaggle.json
ou variáveis de ambiente KAGGLE_USERNAME / KAGGLE_KEY). Veja:
https://www.kaggle.com/docs/api#authentication

O slug do dataset (formato "usuario/nome-do-dataset", visível na URL
kaggle.com/datasets/usuario/nome-do-dataset) precisa ser informado via
variável de ambiente KAGGLE_DATASET, já que datasets diferentes podem estar
associados ao notebook de referência do projeto.

Uso:
    KAGGLE_DATASET="usuario/nome-do-dataset" uv run python src/data/download_dataset.py
"""

import os
import sys
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def main() -> None:
    dataset_slug = os.environ.get("KAGGLE_DATASET")
    if not dataset_slug:
        sys.exit(
            "Erro: defina a variável de ambiente KAGGLE_DATASET com o slug do "
            "dataset (ex.: KAGGLE_DATASET=usuario/nome-do-dataset).\n"
            "O slug fica visível na URL do dataset em kaggle.com/datasets/<slug>."
        )

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

    print(f"Dataset disponível em: {RAW_DIR}")


if __name__ == "__main__":
    main()
