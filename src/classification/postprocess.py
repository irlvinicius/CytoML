"""Limpeza dos recortes para a montagem do cariótipo (pós-processamento).

Duas versões:

- `clean_crop_paper`: porta fiel dos Algoritmos 1 e 2 de Rosas-Alatriste et al. (Scientific
  Reports, 2026, seção "Border detection, mask generation, and image enhancement for the karyotype
  template"; `image_enhancer.py` do repositório oficial) -- CLAHE -> blur -> threshold
  adaptativo -> morfologia -> maior contorno -> fundo branco -> recorte com margem -> nitidez 3x3.
- `clean_crop` (usada no cariótipo): adaptação para os nossos recortes. No paper o algoritmo
  roda em metáfases já filtradas sem sobreposição e em cromossomos isolados; nos nossos recortes,
  vizinhos e sobreposições são comuns, e o original falha de três formas: o threshold adaptativo
  trata as bandas claras como fundo (o cromossomo fica com "buracos" brancos), o maior contorno
  muitas vezes é um vizinho ou um fragmento, e a nitidez granula imagens que já passaram por
  CLAHE no pré-processamento. Por isso: threshold global (Otsu), objeto que ocupa o centro do
  recorte (o detector centraliza a caixa no cromossomo) e sem nitidez. Vizinhos que encostam no
  cromossomo continuam na imagem -- separá-los exigiria segmentação de instâncias.

Para o cariograma, `karyogram_crop` ainda deixa o cromossomo em pé, com ideias do KarySOM
(Marc & Czibula): gira e apara até a orientação "retrato" (aqui pelo eixo principal da máscara,
via PCA) e usa o fato de o centrômero ser a parte mais fina do cromossomo para pôr o braço curto
(p) para cima -- só quando o estreitamento é nítido. Cromossomos curvos ficam em pé, mas
continuam curvos (endireitar sem deformar é limitação apontada pelo próprio KarySOM).

Como no paper, isso serve só para a visualização: o classificador continua recebendo o recorte
cru, que é o que ele viu no treino. Sem dependência de torch, roda no CI.
"""

import math

import cv2
import numpy as np

SHARPEN_KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)


def _ellipse(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


# --- Versão fiel ao paper --------------------------------------------------------------------


def foreground_mask_paper(image_gray: np.ndarray) -> np.ndarray:
    """Máscara binária (0/255) do Algoritmo 1 do paper."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    blurred = cv2.GaussianBlur(clahe.apply(image_gray), (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=11, C=2
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _ellipse(3), iterations=1)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _ellipse(3), iterations=2)


def crop_largest_object(image_gray: np.ndarray, margin: int = 20) -> np.ndarray:
    """Algoritmo 1 do paper: isola o maior contorno, pinta o resto de branco e recorta em volta."""
    contours, _ = cv2.findContours(
        foreground_mask_paper(image_gray), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return image_gray.copy()
    largest = max(contours, key=cv2.contourArea)
    object_mask = np.zeros_like(image_gray)
    cv2.drawContours(object_mask, [largest], -1, 255, thickness=cv2.FILLED)
    return _whiten_and_crop(image_gray, object_mask, margin)


def enhance_quality(image_gray: np.ndarray) -> np.ndarray:
    """Algoritmo 2 do paper: convolução 3x3 com centro 9 e vizinhos -1 (soma 1, preserva o brilho)."""
    return cv2.filter2D(image_gray, ddepth=-1, kernel=SHARPEN_KERNEL)


def clean_crop_paper(image_gray: np.ndarray, margin: int = 20) -> np.ndarray:
    """Algoritmo 1 + Algoritmo 2 do paper, na mesma ordem."""
    return enhance_quality(crop_largest_object(image_gray, margin=margin))


# --- Versão adaptada (usada no cariótipo) ----------------------------------------------------


def foreground_mask(image_gray: np.ndarray, close: bool = False) -> np.ndarray:
    """Máscara (0/255) por threshold global de Otsu: separa cromossomo (escuro) do fundo (claro)
    sem cortar as bandas claras, ao contrário do threshold adaptativo do paper.

    Sem fechamento morfológico por padrão: ele gruda no cromossomo central pedaços de vizinhos a
    poucos pixels de distância. O custo é que uma banda muito clara atravessando o cromossomo
    inteiro pode parti-lo (~2% dos recortes numa amostra do teste, contra ~7% em que o
    fechamento grudava um vizinho) -- `_central_and_others` usa a versão fechada nesses casos.
    """
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if close:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _ellipse(5), iterations=2)
    return binary


def central_object_mask(image_gray: np.ndarray) -> np.ndarray | None:
    """Máscara preenchida do componente que mais ocupa a metade central do recorte (ou o maior,
    se nenhum tocar o centro). `None` se não houver objeto."""
    result = _central_and_others(image_gray)
    return None if result is None else result[0]


def grow_mask(central: np.ndarray, others: np.ndarray, dilate_px: int) -> np.ndarray:
    """Dilata a máscara do cromossomo central sem invadir pixels de outros objetos -- evita que
    uma faixa fina de um vizinho próximo entre na imagem (e vire uma lasca depois de girar)."""
    if dilate_px <= 0:
        return central
    return cv2.dilate(central, _ellipse(2 * dilate_px + 1)) & cv2.bitwise_not(others)


def _central_and_others(
    image_gray: np.ndarray, min_split_ratio: float = 0.3
) -> tuple[np.ndarray, np.ndarray] | None:
    """(objeto central preenchido, demais objetos). Se o objeto da máscara sem fechamento tiver
    menos de `min_split_ratio` da área do objeto com fechamento, o cromossomo provavelmente foi
    partido por uma banda clara -- usa a versão com fechamento."""
    plain = _select_central(foreground_mask(image_gray))
    if plain is None:
        return None
    closed = _select_central(foreground_mask(image_gray, close=True))
    if closed is not None and (plain[0] > 0).sum() < min_split_ratio * (closed[0] > 0).sum():
        return closed
    return plain


def _select_central(foreground: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    if count <= 1:
        return None
    height, width = foreground.shape[:2]
    center = labels[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    in_center = np.bincount(center.ravel(), minlength=count)
    in_center[0] = 0
    if in_center.max() > 0:
        label = int(in_center.argmax())
    else:
        label = int(stats[1:, cv2.CC_STAT_AREA].argmax()) + 1

    component = (labels == label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(component)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled, foreground & cv2.bitwise_not(filled)


def clean_crop(image_gray: np.ndarray, margin: int = 10, dilate_px: int = 3) -> np.ndarray:
    """Mantém só o cromossomo central (dilatado em `dilate_px` para não comer a borda suave),
    pinta o resto de branco e recorta em volta. Devolve o recorte original se não achar objeto."""
    result = _central_and_others(image_gray)
    if result is None:
        return image_gray.copy()
    return _whiten_and_crop(image_gray, grow_mask(*result, dilate_px), margin)


def principal_angle(mask: np.ndarray) -> float:
    """Ângulo (graus, eixo x da imagem, y para baixo) do eixo maior do objeto, via PCA dos pixels."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return 90.0
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(np.stack([xs, ys]).astype(np.float64)))
    vx, vy = eigenvectors[:, np.argmax(eigenvalues)]
    return math.degrees(math.atan2(vy, vx))


def rotate_upright(image_gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gira imagem e máscara para o eixo maior ficar vertical, sem cortar cantos, e apara na máscara."""
    height, width = image_gray.shape[:2]
    side = int(math.ceil(math.hypot(height, width))) + 2
    top, left = (side - height) // 2, (side - width) // 2
    padded_image = np.full((side, side), 255, dtype=image_gray.dtype)
    padded_mask = np.zeros((side, side), dtype=np.uint8)
    padded_image[top : top + height, left : left + width] = image_gray
    padded_mask[top : top + height, left : left + width] = mask

    matrix = cv2.getRotationMatrix2D((side / 2, side / 2), principal_angle(mask) - 90, 1.0)
    image_rot = cv2.warpAffine(padded_image, matrix, (side, side), flags=cv2.INTER_LINEAR, borderValue=255)
    mask_rot = cv2.warpAffine(padded_mask, matrix, (side, side), flags=cv2.INTER_NEAREST, borderValue=0)

    x, y, w, h = cv2.boundingRect(mask_rot)
    image_rot, mask_rot = image_rot[y : y + h, x : x + w], mask_rot[y : y + h, x : x + w]
    if w > h:  # objeto quase redondo: o PCA pode deixar deitado
        image_rot, mask_rot = cv2.rotate(image_rot, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(mask_rot, cv2.ROTATE_90_CLOCKWISE)
    return image_rot, mask_rot


def centromere_row(mask: np.ndarray, end_margin: float = 0.15, min_narrowing: float = 0.8) -> int | None:
    """Linha mais estreita do cromossomo em pé (ideia do KarySOM: o centrômero é a parte mais fina),
    longe das pontas. `None` se o estreitamento não for nítido (largura mínima acima de
    `min_narrowing` x a mediana) -- aí a orientação fica como está."""
    widths = (mask > 0).sum(axis=1).astype(np.float64)
    height = len(widths)
    lo, hi = int(height * end_margin), int(math.ceil(height * (1 - end_margin)))
    if hi - lo < 3:
        return None
    kernel = max(3, height // 12)
    smooth = np.convolve(widths, np.ones(kernel) / kernel, mode="same")
    interior = smooth[lo:hi]
    row = lo + int(np.argmin(interior))
    return row if smooth[row] < min_narrowing * np.median(interior) else None


def karyogram_crop(image_gray: np.ndarray, dilate_px: int = 3, margin: int = 3) -> np.ndarray:
    """Recorte para o cariograma: cromossomo central isolado em fundo branco, em pé, com o braço
    curto (p) para cima quando o centrômero é identificável. Mantém a escala em pixels."""
    result = _central_and_others(image_gray)
    if result is None:
        return image_gray.copy()
    mask = grow_mask(*result, dilate_px)
    isolated = np.where(mask > 0, image_gray, 255).astype(image_gray.dtype)
    upright, upright_mask = rotate_upright(isolated, mask)

    row = centromere_row(upright_mask)
    if row is not None and row > upright_mask.shape[0] / 2:
        upright = np.flipud(upright)
    return np.pad(upright, margin, constant_values=255)


def _whiten_and_crop(image_gray: np.ndarray, mask: np.ndarray, margin: int) -> np.ndarray:
    isolated = np.where(mask > 0, image_gray, 255).astype(image_gray.dtype)
    x, y, w, h = cv2.boundingRect(mask)
    height, width = image_gray.shape[:2]
    return isolated[
        max(0, y - margin) : min(height, y + h + margin),
        max(0, x - margin) : min(width, x + w + margin),
    ]
