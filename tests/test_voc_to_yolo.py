from pathlib import Path

import pytest
from voc_to_yolo import DENVER_CLASS_TO_ID, DENVER_CLASSES, parse_annotation, to_yolo_line

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_annotation.xml"


def test_denver_classes_has_24_entries():
    assert len(DENVER_CLASSES) == 24
    assert DENVER_CLASS_TO_ID["A1"] == 0
    assert DENVER_CLASS_TO_ID["Y"] == 23


def test_parse_annotation_reads_size_and_objects():
    annotation = parse_annotation(FIXTURE)

    assert annotation["width"] == 524
    assert annotation["height"] == 869
    assert len(annotation["objects"]) == 2
    assert annotation["objects"][0]["name"] == "A1"
    assert annotation["objects"][1]["name"] == "X"


def test_to_yolo_line_round_trips_bbox():
    annotation = parse_annotation(FIXTURE)
    obj = annotation["objects"][0]
    line = to_yolo_line(obj, annotation["width"], annotation["height"])

    class_id, xc, yc, w, h = line.split()
    xc, yc, w, h = float(xc), float(yc), float(w), float(h)

    assert class_id == "0"
    assert 0 <= xc <= 1
    assert 0 <= yc <= 1
    assert 0 <= w <= 1
    assert 0 <= h <= 1

    xmin = (xc - w / 2) * annotation["width"]
    xmax = (xc + w / 2) * annotation["width"]
    ymin = (yc - h / 2) * annotation["height"]
    ymax = (yc + h / 2) * annotation["height"]

    assert xmin == pytest.approx(obj["xmin"], abs=0.5)
    assert xmax == pytest.approx(obj["xmax"], abs=0.5)
    assert ymin == pytest.approx(obj["ymin"], abs=0.5)
    assert ymax == pytest.approx(obj["ymax"], abs=0.5)
