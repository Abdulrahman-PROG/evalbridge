"""
Standalone metrics for binary classification evaluation.

All functions accept Python lists or numpy arrays of equal length.
Raises ValueError on invalid inputs rather than returning silent NaN.
"""

import numpy as np
from scipy.stats import rankdata


def _validate(y_true, y_pred_proba, context=""):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred_proba, dtype=float)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(f"{context}y_true and y_pred must be 1-D arrays")
    if len(y_true) == 0:
        raise ValueError(f"{context}Input arrays must not be empty")
    if len(y_true) != len(y_pred):
        raise ValueError(f"{context}y_true and y_pred must have the same length")
    if np.any(np.isnan(y_pred)):
        raise ValueError(f"{context}y_pred contains NaN values")
    if np.any((y_pred < 0) | (y_pred > 1)):
        raise ValueError(f"{context}y_pred probabilities must be in [0, 1]")
    unique = np.unique(y_true[~np.isnan(y_true)])
    for v in unique:
        if v not in (0, 1):
            raise ValueError(f"{context}y_true must contain only 0 and 1, got {v}")

    return y_true, y_pred


def accuracy(y_true, y_pred_proba, threshold=0.5):
    """Return fraction of correct predictions.

    Example: accuracy([1, 0, 1], [0.8, 0.2, 0.9]) -> 1.0
    """
    y_true, y_pred = _validate(y_true, y_pred_proba, "accuracy: ")
    y_hat = (y_pred >= threshold).astype(float)
    return float(np.mean(y_hat == y_true))


def auc_roc(y_true, y_pred_proba):
    """Return area under the ROC curve via the Wilcoxon-Mann-Whitney statistic.

    Example: auc_roc([1, 0, 1], [0.8, 0.2, 0.9]) -> 1.0
    """
    y_true, y_pred = _validate(y_true, y_pred_proba, "auc_roc: ")
    classes = np.unique(y_true)
    if len(classes) < 2:
        raise ValueError("auc_roc: y_true must contain both classes (0 and 1)")

    pos = y_pred[y_true == 1]
    neg = y_pred[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auc_roc: y_true must contain both classes (0 and 1)")

    combined = np.concatenate([pos, neg])
    ranks = rankdata(combined)
    rank_sum_pos = np.sum(ranks[:n_pos])
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(u_stat / (n_pos * n_neg))


def f1(y_true, y_pred_proba, threshold=0.5):
    """Return F1 score (harmonic mean of precision and recall).

    Example: f1([1, 0, 1], [0.8, 0.2, 0.9]) -> 1.0
    """
    y_true, y_pred = _validate(y_true, y_pred_proba, "f1: ")
    y_hat = (y_pred >= threshold).astype(float)
    tp = np.sum((y_hat == 1) & (y_true == 1))
    fp = np.sum((y_hat == 1) & (y_true == 0))
    fn = np.sum((y_hat == 0) & (y_true == 1))
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float(2 * tp / denom)


def log_loss(y_true, y_pred_proba):
    """Return mean binary cross-entropy loss.

    Example: log_loss([1, 0], [0.9, 0.1]) -> ~0.105
    """
    y_true, y_pred = _validate(y_true, y_pred_proba, "log_loss: ")
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def brier_score(y_true, y_pred_proba):
    """Return mean squared error between probabilities and true labels.

    Example: brier_score([1, 0], [0.9, 0.1]) -> 0.01
    """
    y_true, y_pred = _validate(y_true, y_pred_proba, "brier_score: ")
    return float(np.mean((y_pred - y_true) ** 2))


def compute_all(y_true, y_pred_proba, threshold=0.5):
    """Compute all metrics and return as a dict.

    Example: compute_all([1,0,1], [0.8,0.2,0.9]) -> {'accuracy': 1.0, ...}
    """
    return {
        "accuracy": accuracy(y_true, y_pred_proba, threshold=threshold),
        "auc_roc": auc_roc(y_true, y_pred_proba),
        "f1": f1(y_true, y_pred_proba, threshold=threshold),
        "log_loss": log_loss(y_true, y_pred_proba),
        "brier_score": brier_score(y_true, y_pred_proba),
        "n_samples": len(y_true),
    }
