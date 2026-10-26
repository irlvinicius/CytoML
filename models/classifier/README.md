# Classificador de tipos cromossômicos

Classificador dos 24 tipos de cromossomo (nomenclatura de grupo de Denver: A1, A2, A3, B4, B5,
C6–C12, D13–D15, E16–E18, F19, F20, G21, G22, X, Y), a partir de recortes individuais. Complementa
o detector (`models/yolo/yolo26m/`), que localiza cromossomos numa metáfase mas não diz de qual
tipo é cada um — informação necessária para contar por classe e detectar aneuploidias depois.

## Status

Código completo e testado (extração de recortes, dataset, treino, avaliação, export ONNX). Run
`resnet50-30epochs` (configuração anterior) treinada e avaliada; a run na configuração do paper
(`resnet50-paper`) é a próxima, na máquina de treino remota.

## Como funciona

```
detector (splits já existentes)
        │
        ▼
extract_crops.py  ──► data/processed/classification/{split}/{classe}/*.png + manifest.csv
        │
        ▼
dataset.py  (resize 224×224, augmentation no treino)
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

### Configuração de treino (padrão: paper de referência)

A configuração padrão de `src/classification/train.py` (`DEFAULT_CONFIG`) reproduz a rota (b)
YOLOv8 + ResNet-50 de Rosas-Alatriste et al. (*Scientific Reports*, 2026). Qualquer campo pode ser
sobrescrito (`train(resize="letterbox")`, ou pelo dicionário `OVERRIDES` do `02_train.ipynb`).

| | Padrão (paper) | Run anterior `resnet50-30epochs` |
|---|---|---|
| Resize | `stretch` — esticado (`target_size` do Keras) | `letterbox` |
| Augmentation | `paper` — rotação ≤30°, zoom ≤20% (x/y independentes), shift ≤10%, flip horizontal | `free_rotation` — flips + rotação livre 0–360° + jitter de escala/translação |
| Camadas congeladas | `paper` — ~50 primeiras do Keras = `conv1`, `bn1`, `layer1`, `layer2[0]` (BN em modo inferência) | `none` |
| Otimizador | Adam 1e-3, `ReduceLROnPlateau(0.02, patience=3, min_lr=1e-6)` | AdamW 3e-4, weight decay 1e-4, `cosine` |
| Loss | cross-entropy pura | peso de classe `∝ 1/√freq` + `label_smoothing=0.1` |
| Checkpoint / parada | menor `val_loss`, early stopping (patience 15), até 100 épocas | maior `val_f1`, sem early stopping, 30 épocas |
| Batch / seed | 32 / 0 | 64 / sem seed fixa |

Para reproduzir a run anterior: `train(epochs=30, batch_size=64, lr=3e-4, optimizer="adamw",
weight_decay=1e-4, scheduler="cosine", freeze="none", class_weights=True, label_smoothing=0.1,
monitor="val_f1", early_stopping_patience=None, resize="letterbox", augment="free_rotation")`.

**Letterbox vs. esticado.** Cromossomos são numerados por tamanho dentro do mesmo grupo (C6 é maior
que C12, por exemplo), então o comprimento do recorte é parte do sinal que separa as classes. O
letterbox (`crops.py::letterbox`) preserva a proporção e preenche o canvas com a cor de fundo
estimada da própria imagem; o esticado do paper distorce bastante recortes muito alongados
(cromossomo deitado na metáfase). Por isso `{"resize": "letterbox"}` é a ablação natural da
configuração do paper.

**Classe rara (Y).** Y é a classe mais rara (~2.600 cromossomos no dataset completo, contra ~10.000
da mediana). O paper treina num dataset balanceado e usa cross-entropy pura; aqui isso tende a
penalizar o Y — vale acompanhar a acurácia do grupo Y no `03_eval.ipynb`.

O que fica diferente do paper mesmo na configuração padrão: o treino usa recortes do próprio dataset
de metáfases (o paper treina no Dataset B, de cromossomos Q-band isolados) e o split 70/15/15 por
imagem do detector, em vez de 75/25 aleatório — evita que cromossomos da mesma metáfase caiam em
treino e validação. A normalização é a do torchvision (média/desvio do ImageNet), o equivalente do
`resnet50.preprocess_input` do Keras para pesos do torchvision.

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
uv run python src/classification/train.py --run-name resnet50-paper
```

Salva em `models/classifier/runs/<run-name>/`: `best.pt` (melhor checkpoint segundo o `monitor`),
`history.csv` (loss, acurácia e macro-F1 de treino/val por época), `config.json` e `last.pt`
(estado completo de treino, inclusive dos geradores aleatórios — fora do git). Se o treino for
interrompido, a mesma linha com `--resume` continua da última época salva, com a mesma sequência de
embaralhamento/augmentation da run sem interrupção; o resume recusa config diferente da salva em
`config.json`. Para treinos longos na máquina remota, rodar dentro de um `tmux`.

### 3. Avaliar no split de teste

```bash
uv run python src/classification/evaluate.py --checkpoint models/classifier/runs/resnet50-paper/best.pt
```

Usa o mesmo resize do treino (lido do checkpoint). Reporta top-1/top-2, macro-F1, matriz de confusão 24×24 e acurácia por grupo de Denver (A–G, X, Y —
erro dentro do mesmo grupo é o caso mais comum e mais tolerável). Salva `test_probs.npy` e
`test_labels.npy` no mesmo diretório do checkpoint — as probabilidades (não só o argmax) são o que a
próxima fase (detecção de aneuploidias) usa para atribuir contagens por classe.

### 4. Exportar para ONNX

```bash
uv run python src/classification/export_onnx.py --checkpoint models/classifier/runs/resnet50-30epochs/best.pt
```

Exporta com batch dinâmico e confere a paridade numérica contra `onnxruntime` antes de considerar o
export válido.

### 5. Validação cruzada (k-fold)

```bash
PYTHONUNBUFFERED=1 uv run python src/classification/cross_validate.py --run-name resnet50-paper-cv \
    | tee models/classifier/runs/resnet50-paper-cv.log
```

Treina um modelo por fold de `data/processed/24_chromosomes_object/folds/folds.csv` (gerado por
`src/preprocessing/pipeline.py::generate_kfold_splits`, 5 folds estratificados por sexo × tipo de
cariótipo e agrupados por caso, sobre o pool de treino+validação), com a configuração padrão. Cada
modelo usa o próprio fold como validação (early stopping e checkpoint) e é avaliado no **mesmo split
de teste fixo**; `cv_summary.csv` traz as métricas por fold e a média ± desvio padrão. Folds já
avaliados são pulados e um fold interrompido continua de `last.pt`, então basta repetir o comando.
Um fold isolado: `train.py --run-name resnet50-paper-cv --fold 0`. Custo: ~2h20 por fold na máquina
de treino (RTX 4070 Ti), ~12h os 5. O detector tem o equivalente em `src/detection/cross_validate.py`.

### Notebooks

Os pesos de todas as runs (inclusive os 5 folds) estão no Hugging Face -- ver "Pesos dos modelos" no README principal.

`models/classifier/notebooks/01_setup.ipynb`, `02_train.ipynb`, `03_eval.ipynb` fazem o mesmo fluxo
acima, com validação visual dos recortes e dos resultados — mesmo padrão dos notebooks do detector
(`models/yolo/yolo26m/notebooks/`).

Três notebooks adicionais, fora do fluxo de treino/avaliação, servem só pra inspecionar o modelo
já treinado:

- `04_demo.ipynb` — roda o classificador em recortes individuais do teste e mostra a predição ao
  lado da imagem (acerto/erro), sem métrica agregada.
- `05_karyotype_demo.ipynb` — monta uma metáfase inteira do teste no formato clássico de
  cariótipo (seções por grupo de Denver), usando as bounding boxes reais do XML — isola o
  classificador, sem passar pelo detector.
- `06_full_pipeline_demo.ipynb` — pipeline completo de ponta a ponta (imagem → YOLO →
  classificador → cariótipo montado), usando o detector de verdade em vez das boxes do XML. A
  anotação original só é usada no final, pra conferir a acurácia do resultado.

`07_cross_validation.ipynb` agrega a validação cruzada (passo 5): métricas de teste por fold, média ± desvio
padrão ao lado do split único (`resnet50-paper`), F1 por classe com barras de erro e curvas de validação
por fold. Tabelas e figuras também são salvas em `outputs/figures/classifier/` (só localmente, não versionadas).

`04_demo` ainda carrega `../runs/resnet50-30epochs/best.pt`; `05` e `06` usam `../runs/resnet50-paper/best.pt` e leem do checkpoint o resize usado no treino (`evaluate.load_checkpoint_with_meta` + `dataset.preprocess`), então funcionam com qualquer run.

O `06` também inclui o pós-processamento dos recortes para a visualização (`src/classification/postprocess.py`: a versão fiel ao paper e a adaptada, usada no cariótipo) e o alerta numérico por contagem (`src/classification/karyotype.py`). O cariótipo final segue o layout do exame (`src/classification/karyogram.py`): 4 linhas, homólogos lado a lado, cromossomos em pé e na escala real da metáfase. As metáfases com aneuploidia real do split de teste (`data/raw/number_abnormalities.csv`) são listadas no passo 1.

## Estrutura

- `src/classification/crops.py` — recorte + letterbox (sem torch, roda no CI).
- `src/classification/extract_crops.py` — orquestra a extração (sem torch).
- `src/classification/dataset.py` — `Dataset` do torch sobre o manifest.
- `src/classification/train.py` — loop de treino.
- `src/classification/evaluate.py` — métricas no split de teste.
- `src/classification/export_onnx.py` — export + verificação de paridade.
- `src/classification/cross_validate.py` — validação cruzada (um modelo por fold, avaliado no teste fixo).
- `src/classification/metrics.py` — métricas a partir das probabilidades salvas: acurácia, P/R/F1, AUC-ROC, IC por bootstrap (sem torch).
- `src/classification/postprocess.py` — limpeza dos recortes para o cariótipo (sem torch, roda no CI).
- `src/classification/karyotype.py` — contagem por classe e alerta numérico (sem torch, roda no CI).
- `src/classification/karyogram.py` — layout e desenho do cariograma no formato do exame.
- `tests/test_crops.py` — testes de recorte/letterbox/extração, sem depender do dataset real.
- `tests/test_postprocess.py` — testes do pós-processamento, da orientação em pé e do alerta numérico.
- `tests/test_karyogram.py` — testes do layout do cariograma.
- `tests/test_metrics.py` — testes das métricas e da agregação entre folds.

## Pendências

- Rodar o treino de verdade na máquina remota e registrar as métricas de teste aqui (ou num arquivo
  de resultados à parte).
- Subir os pesos da validação cruzada para o Hugging Face (ver "Pesos dos modelos" no README principal).
