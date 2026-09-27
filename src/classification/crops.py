"""Recorte de cromossomos individuais e letterbox para entrada do classificador.

Sem dependência de torch: só numpy/cv2, para rodar tanto na extração quanto no CI.
"""

import cv2
import numpy as np

DEFAULT_SIZE = 224


def clamp_box(
    xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Arredonda a bbox para fora, limita aos bordos da imagem e devolve None se ficar vazia."""
    x0 = max(0, int(np.floor(xmin)))
    y0 = max(0, int(np.floor(ymin)))
    x1 = min(width, int(np.ceil(xmax)))
    y1 = min(height, int(np.ceil(ymax)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def crop_box(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


def background_color(image: np.ndarray) -> np.ndarray:
    """Mediana dos pixels da borda do recorte, por canal (estimativa da cor de fundo)."""
    ring = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    return np.median(ring, axis=0).astype(image.dtype)


def fit_size(height: int, width: int, size: int, scale_up: bool = True) -> tuple[int, int]:
    """Tamanho (altura, largura) após encaixar no canvas preservando a proporção."""
    scale = size / max(height, width)
    if not scale_up:
        scale = min(scale, 1.0)
    return max(1, round(height * scale)), max(1, round(width * scale))


def letterbox(image: np.ndarray, size: int = DEFAULT_SIZE, scale_up: bool = True) -> np.ndarray:
    """Encaixa a imagem em um canvas size x size sem distorcer a proporção.

    O espaço restante é preenchido com a cor de fundo estimada da própria imagem.
    Com `scale_up=False`, recortes menores que o canvas não são ampliados (mantém o
    tamanho absoluto em pixels, ao custo de menos resolução útil).
    """
    height, width = image.shape[:2]
    new_h, new_w = fit_size(height, width, size, scale_up)
    shrinking = new_w * new_h < width * height
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    canvas = np.empty((size, size, *image.shape[2:]), dtype=image.dtype)
    canvas[...] = background_color(image)

    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas
