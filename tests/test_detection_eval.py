import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "detection"))

import numpy as np  # noqa: E402
from evaluate_detector import box_iou, build_eval_set, difficult_images, match_detections  # noqa: E402


@pytest.fixture
def fake_dataset(tmp_path):
    raw = tmp_path / "raw" / "JEPG"
    processed = tmp_path / "processed"
    (processed / "images" / "test").mkdir(parents=True)
    (processed / "labels" / "test").mkdir(parents=True)
    raw.mkdir(parents=True)
    for name in ("a", "b"):
        (raw / f"{name}.jpg").write_bytes(b"raw")
        (processed / "images" / "test" / f"{name}.jpg").write_bytes(b"processed")
        (processed / "labels" / "test" / f"{name}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    return raw, processed, tmp_path / "eval"


@pytest.mark.parametrize("source", ["raw", "processed"])
def test_build_eval_set_links_images_from_source(fake_dataset, source):
    raw, processed, eval_root = fake_dataset
    data_yaml = build_eval_set("subset", ["a"], source, eval_root, raw, processed)

    image = eval_root / "subset" / "images" / "test" / "a.jpg"
    assert image.read_bytes() == source.encode()
    assert (eval_root / "subset" / "labels" / "test" / "a.txt").exists()
    assert not (eval_root / "subset" / "images" / "test" / "b.jpg").exists()

    config = yaml.safe_load(data_yaml.read_text())
    assert Path(config["path"]).is_absolute()
    assert config["test"] == "images/test"
    assert config["nc"] == 1


def test_build_eval_set_rejects_unknown_source(fake_dataset):
    raw, processed, eval_root = fake_dataset
    with pytest.raises(ValueError):
        build_eval_set("subset", ["a"], "clahe", eval_root, raw, processed)


def test_box_iou():
    boxes = np.array([[0, 0, 10, 10]], dtype=float)
    others = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    assert box_iou(boxes, others) == pytest.approx(np.array([[1.0, 50 / 150, 0.0]]))


def test_match_detections_greedy_by_confidence():
    gt = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
    pred = np.array([[1, 0, 11, 10], [0, 0, 10, 10], [100, 100, 110, 110]], dtype=float)
    scores = np.array([0.6, 0.9, 0.8])
    is_tp, found = match_detections(pred, scores, gt)
    # a mais confiante fica com a anotação; a duplicata vira falso positivo, assim como a caixa
    # longe de tudo, e o segundo cromossomo não foi detectado
    assert is_tp.tolist() == [False, True, False]
    assert found.tolist() == [True, False]


def test_match_detections_empty():
    is_tp, found = match_detections(np.zeros((0, 4)), np.zeros(0), np.array([[0, 0, 1, 1]], dtype=float))
    assert is_tp.size == 0 and found.tolist() == [False]


def test_difficult_images_strips_lines(tmp_path):
    diff_list = tmp_path / "diff_image.txt"
    diff_list.write_text("103191\r\n103202\n\n")
    assert difficult_images(diff_list) == {"103191", "103202"}
