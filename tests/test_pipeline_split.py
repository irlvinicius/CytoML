from pipeline import make_folds, split_train_val_test


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
