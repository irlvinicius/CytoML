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


def macro_prf(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> tuple[float, float, float]:
    """Precisão, recall e F1 macro pela matriz de confusão -- 
    Similar ao do scikit-learn com "zero_division=0", otimizada para rodar no bootstrap
    """
    cm = np.bincount(labels * n_classes + preds, minlength=n_classes**2).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(np.float64)
    pred_total, true_total = cm.sum(axis=0), cm.sum(axis=1)
    precision = np.divide(tp, pred_total, out=np.zeros_like(tp), where=pred_total > 0)
    recall = np.divide(tp, true_total, out=np.zeros_like(tp), where=true_total > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(precision.mean()), float(recall.mean()), float(f1.mean())


def bootstrap_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    metric,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> tuple[float, float]:
    """Intervalo de confiança percentil de `metric(probs, labels)` por reamostragem com reposição.

    Com `groups` (ex.: a metáfase de cada recorte), reamostra grupos inteiros em vez de amostras
    soltas (bootstrap por cluster): recortes da mesma metáfase não são independentes, e reamostrá-los
    um a um daria um intervalo estreito demais.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(labels))
    unique, inverse = np.unique(groups, return_inverse=True)
    members = np.split(np.argsort(inverse, kind="stable"), np.cumsum(np.bincount(inverse))[:-1])

    scores = []
    for _ in range(n_resamples):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(unique), len(unique))])
        scores.append(metric(probs[idx], labels[idx]))
    alpha = (1 - confidence) / 2
    low, high = np.quantile(scores, [alpha, 1 - alpha])
    return float(low), float(high)
