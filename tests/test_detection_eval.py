import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "detection"))

from evaluate_detector import build_eval_set  # noqa: E402


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
