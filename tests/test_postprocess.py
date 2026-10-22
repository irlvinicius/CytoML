import cv2
import numpy as np
import pytest
from karyotype import count_by_class, expected_counts, numerical_alerts
from postprocess import (
    centromere_row,
    clean_crop,
    clean_crop_paper,
    crop_largest_object,
    enhance_quality,
    karyogram_crop,
)
from voc_to_yolo import DENVER_CLASSES


def banded_chromosome(neighbor: tuple[slice, slice] | None = None, band: int = 130) -> np.ndarray:
    """Fundo claro (240), cromossomo escuro (60) no centro com uma banda clara no meio."""
    image = np.full((120, 80), 240, dtype=np.uint8)
    image[20:100, 30:50] = 60
    image[55:65, 30:50] = band
    if neighbor is not None:
        image[neighbor] = 60
    return image


def test_clean_crop_keeps_light_band_and_removes_separate_neighbor():
    # vizinho maior que o cromossomo, sem encostar nele -- o "maior contorno" do paper pegaria ele
    image = banded_chromosome(neighbor=(slice(0, 118), slice(62, 80)))
    out = clean_crop(image, margin=5, dilate_px=0)

    assert out.shape[1] < image.shape[1]  # recortado em volta do cromossomo central
    assert (out == 60).any()
    assert (out == 130).any()  # banda clara preservada, não virou fundo
    assert out.shape[0] >= 80  # cromossomo inteiro, não um fragmento


def test_clean_crop_recovers_chromosome_split_by_very_light_bands():
    # três bandas quase da cor do fundo partem o cromossomo em 4 pedaços; o central fica com
    # ~25% da área, e a proteção contra corte volta para a máscara com fechamento
    image = banded_chromosome()
    for row in (39, 59, 79):
        image[row : row + 2, 30:50] = 236
    out = clean_crop(image, margin=0, dilate_px=0)
    assert out.shape[0] >= 80


def test_paper_version_picks_largest_object_even_if_it_is_the_neighbor():
    image = banded_chromosome(neighbor=(slice(0, 118), slice(62, 80)))
    out = crop_largest_object(image, margin=0)
    assert out.shape[1] <= 20  # recorte com a largura do vizinho, não do cromossomo central


def test_clean_crop_returns_copy_when_there_is_no_object():
    blank = np.full((40, 30), 240, dtype=np.uint8)
    out = clean_crop(blank)
    assert np.array_equal(out, blank) and out is not blank


def test_enhance_quality_preserves_flat_regions():
    flat = np.full((20, 20), 128, dtype=np.uint8)
    assert np.array_equal(enhance_quality(flat), flat)


def test_clean_crop_paper_returns_grayscale_uint8():
    out = clean_crop_paper(banded_chromosome())
    assert out.ndim == 2 and out.dtype == np.uint8


def test_count_by_class_fills_missing_classes_and_rejects_unknown():
    counts = count_by_class(["A1", "A1", "X"])
    assert counts["A1"] == 2 and counts["X"] == 1 and counts["G22"] == 0
    assert list(counts) == DENVER_CLASSES
    with pytest.raises(ValueError):
        count_by_class(["Z9"])


def normal_karyotype(sex: str) -> dict[str, int]:
    counts = {c: 2 for c in DENVER_CLASSES if c not in ("X", "Y")}
    counts.update(X=sex.count("X"), Y=sex.count("Y"))
    return counts


@pytest.mark.parametrize("sex", ["XX", "XY"])
def test_normal_karyotypes_have_no_alerts(sex):
    assert numerical_alerts(normal_karyotype(sex)) == {}


def test_trisomy_21_and_turner_are_flagged():
    down = normal_karyotype("XY") | {"G21": 3}
    assert numerical_alerts(down) == {"G21": {"observed": 3, "expected": 2}}

    turner = normal_karyotype("XX") | {"X": 1}
    assert numerical_alerts(turner) == {"X": {"observed": 1, "expected": 2}}


def test_klinefelter_is_flagged_as_extra_x():
    klinefelter = normal_karyotype("XY") | {"X": 2}
    assert expected_counts(klinefelter)["X"] == 1
    assert numerical_alerts(klinefelter) == {"X": {"observed": 2, "expected": 1}}


def dark_widths(image: np.ndarray) -> np.ndarray:
    return (image < 128).sum(axis=1)


def test_karyogram_crop_stands_diagonal_chromosome_upright():
    image = np.full((120, 120), 240, dtype=np.uint8)
    cv2.line(image, (25, 95), (95, 25), color=60, thickness=12)
    out = karyogram_crop(image)
    ys, xs = np.nonzero(out < 128)
    assert (ys.max() - ys.min()) > 3 * (xs.max() - xs.min())


def test_karyogram_crop_puts_short_arm_up_using_narrowest_row():
    # cromossomo vertical com o estreitamento (centrômero) perto de baixo -> deve ser invertido
    image = np.full((120, 60), 240, dtype=np.uint8)
    image[10:110, 20:40] = 60
    image[78:84, 20:26] = 240
    image[78:84, 34:40] = 240
    out = karyogram_crop(image, dilate_px=0)
    widths = dark_widths(out)
    rows = np.nonzero(widths)[0]
    narrow = rows[0] + int(np.argmin(widths[rows[0] + 10 : rows[-1] - 10])) + 10
    assert narrow < (rows[0] + rows[-1]) / 2


def test_centromere_row_is_none_without_clear_narrowing():
    mask = np.zeros((100, 40), dtype=np.uint8)
    mask[5:95, 10:30] = 255
    assert centromere_row(mask) is None
