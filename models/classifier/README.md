# Classificador de tipos cromossômicos

Classificador dos 24 tipos de cromossomo (nomenclatura de grupo de Denver: A1, A2, A3, B4, B5,
C6–C12, D13–D15, E16–E18, F19, F20, G21, G22, X, Y), a partir de recortes individuais. Complementa
o detector (`models/yolo/yolo26m/`), que localiza cromossomos numa metáfase mas não diz de qual
tipo é cada um — informação necessária para contar por classe e detectar aneuploidias depois.

## Status

Código completo e testado (extração de recortes, dataset, treino, avaliação, export ONNX). O treino
em si ainda não rodou — depende de GPU, feito na máquina de treino remota, fora deste repositório.

## Como funciona

```
detector (splits já existentes)
        │
        ▼
extract_crops.py  ──► data/processed/classification/{split}/{classe}/*.png + manifest.csv
        │
        ▼
dataset.py  (letterbox 224×224, augmentation no treino)
        │
        ▼
train.py  ──► models/classifier/runs/<run>/best.pt + history.csv
        │
        ▼
evaluate.py  ──► métricas + test_probs.npy
        │
        ▼
export_onnx.py  ──► models/classifier/runs/<run>/best.onnx
```

### Por que reaproveitar o split do detector

Cada imagem (metáfase) já está em `train`, `val` ou `test` por causa do pipeline do detector
(`src/preprocessing/pipeline.py`, split 70/15/15). `extract_crops.py` usa exatamente essa mesma
divisão, por imagem — nunca por recorte. Se cromossomos da mesma metáfase caíssem em splits
diferentes, o teste deixaria de ser independente e o resultado ficaria artificialmente bom.

### Letterbox em vez de esticar

Cromossomos são numerados por tamanho dentro do mesmo grupo (C6 é maior que C12, por exemplo) — o
comprimento do recorte é parte do sinal que separa as classes. Por isso o redimensionamento
(`src/classification/crops.py::letterbox`) preserva a proporção original e preenche o restante do
canvas 224×224 com a cor de fundo estimada da própria imagem, em vez de esticar para quadrado.

### Classe rara (Y) e peso de treino

Y é a classe mais rara (~2.600 cromossomos no dataset completo, contra ~10.000 da mediana). O treino
usa peso por classe `∝ 1/√frequência` (mitiga o desbalanceamento sem overfitar nas classes raras) e
`label_smoothing=0.1`.

## Como rodar

Requer o grupo de dependências `classifier` (torch, torchvision, onnx, onnxruntime, scikit-learn) e
os dados processados do detector já gerados (`data/processed/24_chromosomes_object/`, ver
`src/preprocessing/`).

```bash
uv sync --group classifier
```

### 1. Extrair os recortes

```bash
uv run python src/classification/extract_crops.py --workers 8
```

Gera `data/processed/classification/{train,val,test}/{classe}/*.png` e `manifest.csv`. Limpa a saída
anterior antes de rodar de novo (o mapeamento imagem → split pode ter mudado).

### 2. Treinar

```bash
uv run python src/classification/train.py --backbone resnet50 --epochs 30
```

Backbones disponíveis: `resnet50` (default) e `efficientnet_b0`, ambos com pesos ImageNet. Salva o
melhor checkpoint por macro-F1 na validação em `models/classifier/runs/<backbone>-<epochs>epochs/`
(`best.pt` + `history.csv`).

### 3. Avaliar no split de teste

```bash
uv run python src/classification/evaluate.py --checkpoint models/classifier/runs/resnet50-30epochs/best.pt
```

Reporta top-1/top-2, macro-F1, matriz de confusão 24×24 e acurácia por grupo de Denver (A–G, X, Y —
erro dentro do mesmo grupo é o caso mais comum e mais tolerável). Salva `test_probs.npy` e
`test_labels.npy` no mesmo diretório do checkpoint — as probabilidades (não só o argmax) são o que a
próxima fase (detecção de aneuploidias) usa para atribuir contagens por classe.

### 4. Exportar para ONNX

```bash
uv run python src/classification/export_onnx.py --checkpoint models/classifier/runs/resnet50-30epochs/best.pt
```

Exporta com batch dinâmico e confere a paridade numérica contra `onnxruntime` antes de considerar o
export válido.

### Notebooks

`models/classifier/notebooks/01_setup.ipynb`, `02_train.ipynb`, `03_eval.ipynb` fazem o mesmo fluxo
acima, com validação visual dos recortes e dos resultados — mesmo padrão dos notebooks do detector
(`models/yolo/yolo26m/notebooks/`).

## Estrutura

- `src/classification/crops.py` — recorte + letterbox (sem torch, roda no CI).
- `src/classification/extract_crops.py` — orquestra a extração (sem torch).
- `src/classification/dataset.py` — `Dataset` do torch sobre o manifest.
- `src/classification/train.py` — loop de treino.
- `src/classification/evaluate.py` — métricas no split de teste.
- `src/classification/export_onnx.py` — export + verificação de paridade.
- `tests/test_crops.py` — testes de recorte/letterbox/extração, sem depender do dataset real.

## Pendências

- Rodar o treino de verdade na máquina remota e registrar as métricas de teste aqui (ou num arquivo
  de resultados à parte).
- Cross-validation (k-fold): a extração já reaproveita os splits por imagem do detector, incluindo os
  folds gerados em `data/processed/24_chromosomes_object/folds/`; adaptar `train.py`/`evaluate.py`
  para rodar sobre eles fica para quando fizer sentido pagar o custo de treinar k vezes.
