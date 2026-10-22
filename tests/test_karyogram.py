from karyogram import ROWS, karyogram_layout, pair_label
from voc_to_yolo import DENVER_CLASSES


def normal_sizes() -> dict[str, list[tuple[int, int]]]:
    # cromossomos maiores nos primeiros pares, como numa metáfase real
    return {c: [(120 - 4 * i, 20), (118 - 4 * i, 22)] for i, c in enumerate(DENVER_CLASSES) if c != "Y"}


def test_rows_follow_exam_layout_and_cover_all_classes():
    assert [c for row in ROWS for c in row] == DENVER_CLASSES[:22] + ["Y", "X"]
    assert sorted(c for row in ROWS for c in row) == sorted(DENVER_CLASSES)


def test_pair_label():
    assert [pair_label(c) for c in ("A1", "C12", "G22", "X", "Y")] == ["1", "12", "22", "X", "Y"]


def test_every_class_gets_a_label_even_without_chromosomes():
    layout = karyogram_layout(normal_sizes())
    assert sorted(label.class_name for label in layout.labels) == sorted(DENVER_CLASSES)
    assert not any(p.class_name == "Y" for p in layout.placements)


def test_chromosomes_share_the_row_baseline_and_do_not_overlap():
    layout = karyogram_layout(normal_sizes())
    for row in ROWS:
        placed = sorted((p for p in layout.placements if p.class_name in row), key=lambda p: p.x)
        assert len({p.y for p in placed}) == 1
        for a, b in zip(placed, placed[1:], strict=False):
            assert a.x + a.width <= b.x


def test_rows_are_stacked_top_to_bottom_and_centered():
    layout = karyogram_layout(normal_sizes())
    xmin, xmax, _, _ = layout.bounds
    baselines = []
    for row in ROWS:
        placed = [p for p in layout.placements if p.class_name in row]
        baselines.append(placed[0].y)
        left, right = min(p.x for p in placed), max(p.x + p.width for p in placed)
        assert abs((left - xmin) - (xmax - right)) < 1e-6
    assert baselines == sorted(baselines, reverse=True)


def test_trisomy_places_three_homologs_side_by_side():
    sizes = normal_sizes() | {"G21": [(60, 20)] * 3}
    placed = [p for p in karyogram_layout(sizes).placements if p.class_name == "G21"]
    assert [p.index for p in placed] == [0, 1, 2]
    assert placed[0].x < placed[1].x < placed[2].x
