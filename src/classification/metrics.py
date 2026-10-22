"""Métricas do classificador a partir das probabilidades salvas (`test_probs.npy`).

Não depende de torch: dá pra recalcular tudo a partir dos `.npy` de qualquer run, sem
recarregar o modelo. O scikit-learn é importado só dentro das funções que o usam, para o
módulo continuar importável no CI (que não instala o grupo `classifier`).
"""

import numpy as np

SCALAR_METRICS = (
    "accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc",
)


def classification_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Acurácia, precisão/recall/F1 (macro e por classe) e AUC-ROC one-vs-rest (macro e por classe)."""
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    n_classes = probs.shape[1]
    classes = list(range(n_classes))
    preds = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=classes, zero_division=0
    )
    one_hot = np.eye(n_classes, dtype=int)[labels]
    auc = np.array([roc_auc_score(one_hot[:, c], probs[:, c]) for c in classes])
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "macro_auc": float(auc.mean()),
        "per_class": {
            "precision": precision, "recall": recall, "f1": f1, "auc": auc, "support": support,
        },
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    """Média e desvio padrão amostral (ddof=1) -- o desvio entre folds, não o erro da média."""
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), std


def summarize_folds(fold_metrics: list[dict]) -> dict[str, tuple[float, float]]:
    """{métrica: (média, desvio)} das métricas escalares de cada fold."""
    return {
        name: mean_std([m[name] for m in fold_metrics])
        for name in SCALAR_METRICS
        if all(name in m for m in fold_metrics)
    }


def bootstrap_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    metric,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Intervalo de confiança percentil de `metric(probs, labels)` por reamostragem com reposição."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    scores = [metric(probs[idx], labels[idx]) for idx in (rng.integers(0, n, n) for _ in range(n_resamples))]
    alpha = (1 - confidence) / 2
    low, high = np.quantile(scores, [alpha, 1 - alpha])
    return float(low), float(high)
