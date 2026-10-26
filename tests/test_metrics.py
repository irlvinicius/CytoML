import numpy as np
import pytest
from metrics import bootstrap_ci, classification_metrics, macro_prf, mean_std, summarize_folds


def test_mean_std_uses_sample_std():
    mean, std = mean_std([0.98, 0.99, 1.00])
    assert mean == pytest.approx(0.99)
    assert std == pytest.approx(0.01)


def test_mean_std_single_value_has_zero_std():
    assert mean_std([0.5]) == (0.5, 0.0)


def test_summarize_folds_only_scalar_metrics():
    folds = [
        {"accuracy": 0.9, "macro_f1": 0.8, "per_class": None},
        {"accuracy": 1.0, "macro_f1": 0.9, "per_class": None},
    ]
    summary = summarize_folds(folds)
    assert set(summary) == {"accuracy", "macro_f1"}
    assert summary["accuracy"][0] == pytest.approx(0.95)


def test_bootstrap_ci_contains_point_estimate_and_is_deterministic():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 3, 300)
    probs = np.eye(3)[np.where(rng.random(300) < 0.9, labels, (labels + 1) % 3)]

    def accuracy(p, y):
        return float((p.argmax(axis=1) == y).mean())

    low, high = bootstrap_ci(probs, labels, accuracy, n_resamples=200)
    assert low <= accuracy(probs, labels) <= high
    assert (low, high) == bootstrap_ci(probs, labels, accuracy, n_resamples=200)


def test_bootstrap_ci_by_group_is_wider_for_correlated_samples():
    # 30 grupos de 20 amostras: cada grupo inteiro acerta ou erra junto
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(30), 20)
    correct = np.repeat(rng.random(30) < 0.8, 20)
    labels = np.zeros(600, dtype=int)
    probs = np.eye(2)[np.where(correct, 0, 1)]

    def accuracy(p, y):
        return float((p.argmax(axis=1) == y).mean())

    low_i, high_i = bootstrap_ci(probs, labels, accuracy, n_resamples=300)
    low_g, high_g = bootstrap_ci(probs, labels, accuracy, n_resamples=300, groups=groups)
    assert high_g - low_g > 2 * (high_i - low_i)


def test_macro_prf_matches_definition():
    labels = np.array([0, 0, 1, 1, 2, 2])
    preds = np.array([0, 1, 1, 1, 2, 0])
    precision, recall, f1 = macro_prf(preds, labels, 3)
    # por classe: P = [1/2, 2/3, 1], R = [1/2, 1, 1/2]
    assert precision == pytest.approx((1 / 2 + 2 / 3 + 1) / 3)
    assert recall == pytest.approx((1 / 2 + 1 + 1 / 2) / 3)
    assert f1 == pytest.approx((1 / 2 + 0.8 + 2 / 3) / 3)


def test_classification_metrics_perfect_predictions():
    pytest.importorskip("sklearn")
    labels = np.array([0, 1, 2, 2, 1, 0])
    probs = np.eye(3)[labels] * 0.8 + 0.2 / 3
    metrics = classification_metrics(probs, labels)

    for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc"):
        assert metrics[name] == pytest.approx(1.0)
    assert metrics["per_class"]["support"].tolist() == [2, 2, 2]
