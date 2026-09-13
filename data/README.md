# Dados

Este diretório não versiona dados brutos/processados (ver `.gitignore`), apenas
a estrutura de pastas e a configuração do dataset.

## Aquisição

Os dados vêm de um dataset público do Kaggle com imagens de metáfase e
anotações de classes cromossômicas (pastas `24_chromosomes_object` e `single_chromosomes_object`, conforme descrito no artigo do projeto).

Dataset: [chromosome-image-dataset-karyotype](https://www.kaggle.com/datasets/aliabedimadiseh/chromosome-image-dataset-karyotype)
(slug `aliabedimadiseh/chromosome-image-dataset-karyotype`).

Para baixar:

```bash
KAGGLE_DATASET="aliabedimadiseh/chromosome-image-dataset-karyotype" uv run python src/data/download_dataset.py
```

Requer credenciais da Kaggle API em `~/.kaggle/kaggle.json` (ou
`KAGGLE_USERNAME`/`KAGGLE_KEY`).

O download extrai o conteúdo para `data/raw/`.

## Estrutura

- `raw/` — dados originais do Kaggle (anotações em Pascal VOC XML). Ver
  "Pastas do dataset original" abaixo para a diferença entre
  `24_chromosomes_object` e `single_chromosomes_object`.
- `processed/` — dados convertidos para o formato YOLO (gerado pelo pipeline
  de pré-processamento, ver `src/preprocessing/`).
- `dataset.yaml` — configuração do dataset para o Ultralytics. O campo `path`
  é um placeholder: os notebooks de treino sobrescrevem esse valor em tempo de
  execução com o caminho absoluto real desta pasta.

## Pastas do dataset original (`raw/`)

O Kaggle disponibiliza duas pastas, ambas com o **mesmo tipo de imagem**
(metáfase completa — a célula inteira, com todos os cromossomos espalhados;
apesar do nome, não são cromossomos já recortados individualmente). A
diferença está só na anotação:

- **`24_chromosomes_object`** (a única usada neste projeto): cada cromossomo
  anotado com uma classe real, na nomenclatura de grupo de Denver — A1, A2,
  A3, B4, B5, C6–C12, D13–D15, E16–E18, F19, F20, G21, G22, X, Y (24 classes,
  um par homólogo por classe). Usado como `nc: 1` (classe única) nas Fases de detecção/segmentação, mas a informação de classe real fica preservada em `DENVER_CLASSES` (`src/preprocessing/voc_to_yolo.py`) para
  reuso na Fase 4 (classificação dos 24 tipos).
- **`single_chromosomes_object`**: as mesmas imagens, anotadas com uma única
  classe genérica `"chromosomes"` para todos os cromossomos — sem
  diferenciação de tipo. Não usada neste projeto (redundante com
  `24_chromosomes_object` para detecção, e sem valor para a Fase 4).

Dentro de `24_chromosomes_object`, os arquivos `train.txt`/`test.txt` marcam a
partição oficial definida pelos autores do dataset (não usada diretamente
pelo pipeline) e `diff_image.txt` marca imagens consideradas
"difíceis" pelos autores (fundo escuro / sobreposição de cromossomos).
