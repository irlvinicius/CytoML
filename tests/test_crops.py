import csv

import cv2
import numpy as np
import pytest
from crops import background_color, clamp_box, crop_box, fit_size, letterbox
from extract_crops import extract_image, list_split_images, run


def write_annotation(path, width, height, objects):
    body = "".join(
        f"<object><name>{name}</name><bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin>"
        f"<xmax>{x1}</xmax><ymax>{y1}</ymax></bndbox></object>"
        for name, (x0, y0, x1, y1) in objects
    )
    path.write_text(
        f"<annotation><size><width>{width}</width><height>{height}</height>"
        f"<depth>3</depth></size>{body}</annotation>"
    )


@pytest.fixture
def dataset(tmp_path):
    """Mini dataset sintético: 3 imagens processadas (uma por split) + XMLs brutos."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    (raw / "annotations").mkdir(parents=True)

    specs = {
        "train": ("a", [("A1", (10, 10, 40, 60)), ("X", (50, 20, 90, 70))]),
        "val": ("b", [("Y", (0, 0, 30, 30)), ("A2", (200, 200, 210, 210))]),  # 2ª fora da imagem
        "test": ("c", [("G22", (5, 5, 25, 50))]),
    }
    for split, (name, objects) in specs.items():
        (processed / "images" / split).mkdir(parents=True)
        image = np.full((80, 100, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(processed / "images" / split / f"{name}.jpg"), image)
        write_annotation(raw / "annotations" / f"{name}.xml", 100, 80, objects)

    return raw, processed, tmp_path / "out"


def test_clamp_box_limits_to_image_and_rounds_outward():
    assert clamp_box(-5, 2.4, 30.2, 500, 100, 80) == (0, 2, 31, 80)


def test_clamp_box_returns_none_when_empty():
    assert clamp_box(200, 200, 210, 210, 100, 80) is None
    assert clamp_box(10, 10, 10, 30, 100, 80) is None


def test_crop_box_shape():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    assert crop_box(image, (10, 10, 40, 60)).shape == (50, 30, 3)


@pytest.mark.parametrize("shape", [(40, 100), (100, 40), (60, 60), (10, 30)])
def test_letterbox_output_is_square_and_keeps_aspect_ratio(shape):
    height, width = shape
    out = letterbox(np.zeros((height, width, 3), dtype=np.uint8), size=224)
    assert out.shape == (224, 224, 3)

    new_h, new_w = fit_size(height, width, 224)
    assert max(new_h, new_w) == 224
    assert new_w / new_h == pytest.approx(width / height, rel=0.02)


def test_letterbox_pads_with_background_and_centers_content():
    image = np.full((100, 40, 3), 255, dtype=np.uint8)
    image[3:-3, 3:-3] = 0  # miolo escuro, borda clara

    out = letterbox(image, size=224)
    new_h, new_w = fit_size(100, 40, 224)
    left = (224 - new_w) // 2

    assert background_color(image).tolist() == [255, 255, 255]
    assert (out[:, : left - 1] == 255).all()
    assert (out[:, left + new_w + 1 :] == 255).all()
    assert (out[:, left : left + new_w] < 128).any()


def test_letterbox_does_not_upscale_when_disabled():
    assert fit_size(30, 20, 224, scale_up=False) == (30, 20)
    assert fit_size(300, 100, 224, scale_up=False) == (224, 75)
    assert letterbox(np.zeros((30, 20), dtype=np.uint8), scale_up=False).shape == (224, 224)


def test_extract_image_writes_crops_and_rows(dataset):
    raw, processed, out = dataset
    rows, skipped = extract_image(("a", "train", raw, processed, out))

    assert skipped == 0
    assert [(r["class_name"], r["class_id"]) for r in rows] == [("A1", 0), ("X", 22)]
    assert rows[0]["path"] == "train/A1/a_0.png"
    crop = cv2.imread(str(out / rows[0]["path"]))
    assert crop.shape == (50, 30, 3)


def test_run_matches_annotation_counts_and_skips_empty_boxes(dataset):
    raw, processed, out = dataset
    counts = run(raw, processed, out)

    assert counts == {"images": 3, "crops": 4, "skipped": 1, "train": 2, "val": 1, "test": 1}
    with open(out / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert all((out / row["path"]).exists() for row in rows)

    split_of_image = {}
    for row in rows:
        assert split_of_image.setdefault(row["image"], row["split"]) == row["split"]


def test_run_is_identical_with_multiple_workers(dataset):
    raw, processed, out = dataset
    run(raw, processed, out, workers=1)
    serial = (out / "manifest.csv").read_text()
    run(raw, processed, out, workers=2)
    assert (out / "manifest.csv").read_text() == serial


def test_run_rejects_image_present_in_two_splits(dataset):
    raw, processed, out = dataset
    image = cv2.imread(str(processed / "images" / "train" / "a.jpg"))
    cv2.imwrite(str(processed / "images" / "val" / "a.jpg"), image)

    with pytest.raises(ValueError, match="aparece em"):
        run(raw, processed, out)


def test_list_split_images_is_sorted(dataset):
    _, processed, _ = dataset
    for name in ("z", "m"):
        cv2.imwrite(
            str(processed / "images" / "train" / f"{name}.jpg"), np.zeros((4, 4, 3), np.uint8)
        )
    assert list_split_images("train", processed) == ["a", "m", "z"]
