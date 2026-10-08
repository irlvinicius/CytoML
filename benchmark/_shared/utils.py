from pathlib import Path
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

HF_REPO = "irlvinicius/KaryoVisio"
SHARED_DIR = Path(__file__).resolve().parent  # resolve() colapsa os ".." do sys.path
REPO_ROOT = SHARED_DIR.parent.parent  # benchmark/_shared -> raiz do repositório


def resolve_data_yaml(yaml_path) -> str:
    """Torna o `path:` do dataset.yaml absoluto, relativo à RAIZ DO REPO.

    O Ultralytics resolve `path:` relativo a ~/datasets (e não à pasta do YAML),
    então geramos um YAML temporário com o caminho já resolvido. Assim o mesmo
    dataset.yaml funciona em qualquer máquina, desde que o dataset esteja no
    mesmo lugar dentro do repo. Para usar outro local, defina a variável de
    ambiente KARYO_DATA_ROOT com o caminho (absoluto) do dataset.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    root = os.environ.get("KARYO_DATA_ROOT") or cfg["path"]
    root = Path(root).expanduser()
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {root}. Coloque o dataset nesse caminho "
            "(relativo à raiz do repo) ou defina KARYO_DATA_ROOT."
        )
    cfg["path"] = str(root)
    tmp = Path(tempfile.mkdtemp()) / "dataset_resolved.yaml"
    with open(tmp, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(tmp)


def load_eval_config(path=SHARED_DIR / "eval_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_weights(path_in_repo: str) -> str:
    """Baixa (uma vez, fica em cache) um peso do HF e devolve o caminho local."""
    return hf_hub_download(repo_id=HF_REPO, filename=path_in_repo)


def evaluate(name: str, weights: str, data: str, cfg: dict, out_dir, device=None) -> dict:
    """Roda model.val() e devolve as métricas principais + salva os plots em out_dir."""
    out_dir = Path(out_dir)
    model = YOLO(weights)
    m = model.val(
        data=resolve_data_yaml(data), split=cfg["split"], imgsz=cfg["imgsz"], batch=cfg["batch"],
        conf=cfg["conf"], iou=cfg["iou"], seed=cfg["seed"], device=device,
        plots=True, project=str(out_dir / "_runs"), name=name, exist_ok=True,
    )
    p, r = float(m.box.mp), float(m.box.mr)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    # copia os plots gerados (PR, confusion matrix...) para results/<modelo>/
    plots_dir = out_dir / name
    plots_dir.mkdir(parents=True, exist_ok=True)
    for f in Path(m.save_dir).glob("*.png"):
        shutil.copy(f, plots_dir / f.name)
    shutil.rmtree(out_dir / "_runs", ignore_errors=True)

    return {
        "Modelo": name,
        "mAP50": float(m.box.map50),
        "mAP50-95": float(m.box.map),
        "Precision": p,
        "Recall": r,
        "F1": f1,
        "Inferência (ms/img)": float(m.speed["inference"]),
    }


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("Modelo").round(4)


def plot_comparison(df: pd.DataFrame, save_path=None, metrics=None):
    """Gráfico de barras agrupado (uma barra por modelo, por métrica)."""
    metrics = metrics or ["mAP50", "mAP50-95", "Precision", "Recall", "F1"]
    ax = df[metrics].T.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor")
    ax.set_title("Comparação de métricas")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8, padding=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    return ax