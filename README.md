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
git checkout feature/pre_processing   # ou develop, após o merge

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