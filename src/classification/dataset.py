"""Dataset torch sobre o manifest gerado por extract_crops.py.

Entrada: recorte de um cromossomo (grayscale) -> redimensionado para 224x224 -> replicado em
3 canais -> normalização ImageNet (backbones pré-treinados esperam esse formato).
Augmentation só é aplicada quando `train=True`.

Dois modos de redimensionamento e de augmentation:
- `resize="stretch"`, `augment="paper"` (usados por padrão no treino): reprodução do paper de
  referência (Rosas-Alatriste et al., 2026) -- estica para 224x224 e aplica rotação ≤30°,
  zoom ≤20%, shift ≤10% e flip horizontal depois do resize, na mesma ordem do
  `ImageDataGenerator` do Keras.
- `resize="letterbox"`, `augment="free_rotation"`: configuração anterior do projeto (preserva
  o comprimento do cromossomo; flips + rotação livre 0-360° antes do letterbox).
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
from crops import letterbox, random_affine, stretch
from torch.utils.data import Dataset
from voc_to_yolo import DENVER_CLASSES

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

RESIZE_MODES = ("letterbox", "stretch")
AUGMENT_MODES = ("free_rotation", "paper")


def resize_crop(image: np.ndarray, size: int, mode: str) -> np.ndarray:
    if mode == "letterbox":
        return letterbox(image, size=size)
    if mode == "stretch":
        return stretch(image, size=size)
    raise ValueError(f"resize desconhecido: {mode!r} (opções: {RESIZE_MODES})")


def to_tensor(canvas: np.ndarray) -> torch.Tensor:
    """Canvas grayscale já redimensionado -> tensor 3x224x224 normalizado (ImageNet)."""
    rgb = np.stack([canvas] * 3, axis=-1).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float()


def preprocess(image_gray: np.ndarray, image_size: int = 224, resize: str = "letterbox") -> torch.Tensor:
    """Pré-processamento de inferência (sem augmentation), igual ao usado na validação/teste."""
    return to_tensor(resize_crop(image_gray, image_size, resize))


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
        resize: str = "letterbox",
        augment: str = "free_rotation",
    ):
        if resize not in RESIZE_MODES:
            raise ValueError(f"resize desconhecido: {resize!r} (opções: {RESIZE_MODES})")
        if augment not in AUGMENT_MODES:
            raise ValueError(f"augment desconhecido: {augment!r} (opções: {AUGMENT_MODES})")
        self.rows = load_manifest(manifest_path, split)
        self.crops_root = Path(crops_root)
        self.image_size = image_size
        self.train = train
        self.resize = resize
        self.augment = augment
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

        if self.train and self.augment == "free_rotation":
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
            if random.random() < 0.5:
                image = cv2.flip(image, 0)
            image = _random_rotate_scale_translate(image)

        canvas = resize_crop(image, self.image_size, self.resize)

        if self.train and self.augment == "paper":
            canvas = random_affine(canvas, random)

        return to_tensor(canvas), int(row["class_id"])
