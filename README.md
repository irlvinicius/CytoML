# CytoML

## Configurando em uma máquina nova

Passo a passo para deixar o ambiente pronto (ex.: máquina remota de treino
acessada via SSH), do zero até rodar o pipeline de pré-processamento:

```bash
# 1. Dependências de sistema (Debian/Ubuntu)
sudo apt-get update
sudo apt-get install -y git curl build-essential \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1

# 2. Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # ou reabrir o terminal, para "uv" entrar no PATH

# 3. Clonar o repositório e ir para a branch de trabalho
git clone https://github.com/irlvinicius/CytoML.git
cd CytoML
git checkout main

# 4. Sincronizar o ambiente Python (--all-groups para torch ultralytics do grupo "yolo")
uv sync --all-groups

# 5. Configurar credenciais da Kaggle API (necessárias para baixar o dataset)
#    Gerar o token em https://www.kaggle.com/settings -> API -> "Create New Token"
#    (baixa um arquivo kaggle.json com "username" e "key"). Copiar esse arquivo
#    para a máquina onde vai rodar (ex.: "scp kaggle.json usuario@ip:~/.kaggle/"),
#    ou colar o conteúdo manualmente com um editor de texto.
mkdir -p ~/.kaggle
# copiar kaggle.json para ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 6. Baixar o dataset — ver data/README.md para detalhes sobre as pastas
uv run python src/data/download_dataset.py

# 7. Rodar o pipeline de pré-processamento completo (gera data/processed/)
uv run python src/preprocessing/pipeline.py

# 8. (opcional) Rodar os notebooks de validação via Jupyter
uv run jupyter lab --no-browser --port 8888
# na máquina local, abrir um túnel SSH para acessar o Jupyter remoto:
# ssh -L 8888:localhost:8888 usuario@ip-da-maquina-remota
```

Se a máquina tiver GPU, confirme que o driver NVIDIA/CUDA já está instalado
antes do passo 4 — o `pyproject.toml` aponta para o índice `pytorch-cu128` do
PyTorch com CUDA; sem driver compatível, `torch.cuda.is_available()` (checado
em `models/yolo/yolo26m/notebooks/01_setup.ipynb`) retorna falso e o treino
cai para CPU.

## Pesos dos modelos

O git guarda só os pesos finais, via Git LFS, cada um com o `best.pt` e o `best.onnx`
(o formato usado pelo `cytoml-demo`):

- o detector final, YOLO26m treinado com mediana + CLAHE (`models/yolo/yolo26m/runs/preprocessed/`), que
  espera a imagem já pré-processada (`src/preprocessing/filters.py`);
- o classificador (`models/classifier/runs/resnet50-paper/`);
- o detector baseline do TCC1 (`models/yolo/yolo26m/runs/baseline/`), mantido para a comparação.

Os demais pesos, incluindo os modelos da validação cruzada e os artefatos de treino, ficam no
Hugging Face: [irlvinicius/KaryoVisio](https://huggingface.co/irlvinicius/KaryoVisio).

| Modelo | Onde | Resultado no teste |
|---|---|---|
| YOLO26m baseline (TCC1, imagens cruas) | git + HF `models/yolo26m-original/` | mAP@50-95 0,815¹ |
| YOLO26m pré-processado (mediana + CLAHE), detector final | git + HF `models/yolo26m-preprocessed/` | mAP@50-95 0,829¹ / 0,871 |
| YOLO26m, validação cruzada (5 folds) | HF `models/yolo26m-cv/fold{0..4}/` | mAP@50-95 0,826 ± 0,003¹ / 0,878 ± 0,004 |
| ResNet50 classificador (configuração do paper) | git `models/classifier/runs/resnet50-paper/` | acurácia 0,9848 (IC 95% 0,9830–0,9865) |
| ResNet50, validação cruzada (5 folds) | HF `models/resnet50-paper-cv/fold{0..4}/` | acurácia 0,9854 ± 0,0003 |

¹ Nas 185 imagens de teste fora do treino do baseline, cada modelo no formato de
imagem em que foi treinado (ver `src/detection/evaluate_detector.py`); o outro valor é
no teste completo (749 imagens). Detalhes em `models/yolo/yolo26m/notebooks/03_eval.ipynb`
e `models/classifier/notebooks/{03_eval,07_cross_validation}.ipynb`.

Para baixar (ex.: os 5 folds do classificador):

```bash
uvx --from huggingface_hub hf download irlvinicius/KaryoVisio --include "models/resnet50-paper-cv/*" --local-dir hf_models
mv hf_models/models/resnet50-paper-cv models/classifier/runs/
```

Os arquivos são baixados em `hf_models/models/<modelo>/` e precisam ser movidos para a pasta de
runs correspondente para os notebooks os encontrarem. Os comandos do detector estão no início de
`models/yolo/yolo26m/notebooks/03_eval.ipynb`.

## Desenvolvimento

Antes de abrir um PR, rode localmente os mesmos checks do CI:

```bash
uv sync --group dev
uv run ruff check .
uv run pytest tests/ -v
```