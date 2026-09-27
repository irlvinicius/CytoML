"""Dataset torch sobre o manifest gerado por extract_crops.py.

Entrada: recorte de um cromossomo (grayscale) -> letterbox 224x224 -> replicado em
3 canais -> normalização ImageNet (backbones pré-treinados esperam esse formato).
Augmentation só é aplicada quando `train=True`.
"""

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))

import cv2
import numpy as np
import torch
from crops import letterbox
from torch.utils.data import Dataset
from voc_to_yolo import DENVER_CLASSES

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_manifest(manifest_path: Path, split: str) -> list[dict]:
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row["split"] == split]


def _random_rotate_scale_translate(
    image: np.ndarray, max_scale_jitter: float = 0.1, max_translate_frac: float = 0.08
) -> np.ndarray:
    """Rotação livre 0-360° (cromossomos não têm orientação canônica numa metáfase),
    com um pequeno jitter de escala/translação que aproxima variações do detector.

    Assume imagem em escala de cinza (2D) -- é assim que `__getitem__` sempre a lê,
    antes do letterbox e da conversão pra 3 canais.
    """
    height, width = image.shape[:2]
    angle = random.uniform(0, 360)
    scale = 1.0 + random.uniform(-max_scale_jitter, max_scale_jitter)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[0, 2] += random.uniform(-max_translate_frac, max_translate_frac) * width
    matrix[1, 2] += random.uniform(-max_translate_frac, max_translate_frac) * height

    border_value = float(np.median(image[[0, -1]]))
    return cv2.warpAffine(
        image, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=border_value
    )


class ChromosomeCropDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        crops_root: Path,
        split: str,
        image_size: int = 224,
        train: bool = False,
    ):
        self.rows = load_manifest(manifest_path, split)
        self.crops_root = Path(crops_root)
        self.image_size = image_size
        self.train = train
        self.classes = DENVER_CLASSES

    def __len__(self) -> int:
        return len(self.rows)

    def class_counts(self) -> list[int]:
        counts = [0] * len(self.classes)
        for row in self.rows:
            counts[int(row["class_id"])] += 1
        return counts

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image = cv2.imread(str(self.crops_root / row["path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(self.crops_root / row["path"])

        if self.train:
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
            if random.random() < 0.5:
                image = cv2.flip(image, 0)
            image = _random_rotate_scale_translate(image)

        canvas = letterbox(image, size=self.image_size)
        rgb = np.stack([canvas] * 3, axis=-1).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        return tensor, int(row["class_id"])
