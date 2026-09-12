# Dados

Este diretório não versiona dados brutos/processados (ver `.gitignore`), apenas
a estrutura de pastas e a configuração do dataset.

## Aquisição

Os dados vêm de um dataset público do Kaggle com imagens de metáfase e
anotações de classes cromossômicas (pastas `24_chromosomes_object` e
`single_chromosomes`, conforme descrito no artigo do projeto).

Para baixar:

```bash
KAGGLE_DATASET="usuario/nome-do-dataset" uv run python src/data/download_dataset.py
```

Substitua `usuario/nome-do-dataset` pelo slug real do dataset (visível na URL
`kaggle.com/datasets/usuario/nome-do-dataset`). Requer credenciais da Kaggle
API em `~/.kaggle/kaggle.json` (ou `KAGGLE_USERNAME`/`KAGGLE_KEY`).

O download extrai o conteúdo para `data/raw/`.

## Estrutura

- `raw/` — dados originais do Kaggle (anotações em Pascal VOC XML).
- `processed/` — dados convertidos para o formato YOLO (gerado pelo pipeline
  de pré-processamento, ver `src/`).
- `dataset.yaml` — configuração do dataset para o Ultralytics. O campo `path`
  é um placeholder: os notebooks de treino sobrescrevem esse valor em tempo de
  execução com o caminho absoluto real desta pasta.
