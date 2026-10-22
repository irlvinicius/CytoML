from collections import Counter

import pytest
from pipeline import case_id, load_fold, make_folds, split_train_val_test


def test_split_proportions_are_70_15_15():
    pool = [str(i) for i in range(1000)]
    train, val, test = split_train_val_test(pool)

    assert len(train) + len(val) + len(test) == len(pool)
    assert len(train) == 700
    assert len(val) == 150
    assert len(test) == 150


def test_split_has_no_overlap_and_covers_pool():
    pool = [str(i) for i in range(997)]  # not evenly divisible, exercises rounding
    train, val, test = split_train_val_test(pool)

    train_set, val_set, test_set = set(train), set(val), set(test)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert train_set | val_set | test_set == set(pool)


def test_split_is_deterministic():
    pool = [str(i) for i in range(500)]
    assert split_train_val_test(pool) == split_train_val_test(pool)


def test_test_split_is_unchanged():
    # Regressão: o teste do classificador e do detector depende desta partição exata.
    _train, _val, test = split_train_val_test([f"img{i:02d}" for i in range(20)])
    assert test == ["img04", "img09", "img13"]


def test_make_folds_covers_pool_without_overlap():
    pool = [str(i) for i in range(203)]  # not evenly divisible by 5
    folds = make_folds(pool, k=5)

    assert len(folds) == 5

    union = set()
    for fold in folds:
        union |= set(fold)
    assert union == set(pool)
    assert sum(len(fold) for fold in folds) == len(pool)

    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            assert set(folds[i]).isdisjoint(folds[j])


def _stratified_pool():
    # 3 estratos desbalanceados, como no dataset (muitos normais, poucas anomalias)
    pool = [f"n{i}" for i in range(180)] + [f"a{i}" for i in range(23)] + [f"s{i}" for i in range(11)]
    strata = {name: name[0] for name in pool}
    return pool, strata


def test_make_folds_balances_each_stratum():
    pool, strata = _stratified_pool()
    folds = make_folds(pool, k=5, strata=strata)

    assert sorted(name for fold in folds for name in fold) == sorted(pool)
    for stratum in "nas":
        sizes = [sum(strata[name] == stratum for name in fold) for fold in folds]
        assert max(sizes) - min(sizes) <= 1, (stratum, sizes)
    totals = [len(fold) for fold in folds]
    assert max(totals) - min(totals) <= 1


def test_make_folds_never_splits_a_group():
    pool, strata = _stratified_pool()
    groups = {name: name[0] + str(int(name[1:]) // 4) for name in pool}  # grupos de até 4 imagens
    folds = make_folds(pool, k=5, strata=strata, groups=groups)

    fold_of = {name: i for i, fold in enumerate(folds) for name in fold}
    group_folds = {}
    for name, group in groups.items():
        assert group_folds.setdefault(group, fold_of[name]) == fold_of[name]

    for stratum in "nas":
        sizes = [sum(strata[name] == stratum for name in fold) for fold in folds]
        assert max(sizes) - min(sizes) <= 4, (stratum, sizes)


def test_make_folds_with_strata_is_deterministic():
    pool, strata = _stratified_pool()
    groups = {name: name[:2] for name in pool}
    assert make_folds(pool, 5, strata, groups) == make_folds(list(reversed(pool)), 5, strata, groups)


@pytest.mark.parametrize(
    ("name", "expected"),
    [("1050761", "105076"), ("1050764", "105076"), ("1050261_1", "105026"), ("103081", "10308"),
     ("unknow395", "unknow395")],
)
def test_case_id(name, expected):
    assert case_id(name) == expected


def test_load_fold_reads_folds_csv(tmp_path):
    folds_csv = tmp_path / "folds.csv"
    folds_csv.write_text(
        "image,case,stratum,fold\n"
        "a1,a,XX/normal,0\n"
        "a2,a,XX/normal,0\n"
        "b1,b,XY/normal,1\n"
        "c1,c,XY/numerica,2\n"
    )
    train, val = load_fold(1, folds_csv)
    assert val == ["b1"]
    assert Counter(train) == Counter(["a1", "a2", "c1"])

    with pytest.raises(ValueError):
        load_fold(3, folds_csv)
