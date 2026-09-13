"""Pré-processamento de imagem (Seção 4.3 do artigo TCC1): filtro de mediana
para ruído sal-e-pimenta + realce de contraste via CLAHE no canal L (LAB).

`clahe_clip=2.0` (valor original do artigo do TCC1) foi validado empiricamente via
`notebooks/02_filters_before_after.ipynb`: em amostras variadas, incluindo casos
de fundo escuro e sobreposição de cromossomos, sem sinal de estouro de contraste.
Aplicar CLAHE apenas no canal L do LAB (preservando crominância), com a mediana
antes, evita o estouro que ocorre ao aplicar CLAHE direto sobre a imagem inteira.
"""

import cv2
import numpy as np


def denoise_and_enhance(
    image_bgr: np.ndarray,
    median_ksize: int = 3,
    clahe_clip: float = 2.0,
    clahe_tile: tuple[int, int] = (8, 8),
) -> np.ndarray:
    denoised = cv2.medianBlur(image_bgr, median_ksize)

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def save_comparison(image_path, out_path, **filter_kwargs) -> None:
    import matplotlib.pyplot as plt

    original = cv2.imread(str(image_path))
    processed = denoise_and_enhance(original, **filter_kwargs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Processada ({filter_kwargs or 'defaults'})")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
