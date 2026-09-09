"""Selective-prediction metrics: ranking, calibration, and risk-coverage."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """AUROC, or None when one class is absent and the metric is undefined."""
    if len(set(labels.tolist())) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def brier(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean((scores - labels) ** 2))


def ece(labels: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-width confidence bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Include the right edge only in the final bin so 1.0 is not dropped.
        in_bin = (scores > lo) & (scores <= hi) if lo > 0 else (scores >= lo) & (scores <= hi)
        if not in_bin.any():
            continue
        weight = in_bin.mean()
        total += weight * abs(labels[in_bin].mean() - scores[in_bin].mean())
    return float(total)


def risk_coverage(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Selective risk as a function of coverage, accepting highest scores first.

    Returns (coverage, risk) where risk is the error rate among accepted items.
    """
    order = np.argsort(-scores)
    accepted = labels[order]
    n = len(accepted)
    errors = np.cumsum(1 - accepted)
    counts = np.arange(1, n + 1)
    return counts / n, errors / counts


def threshold_at_risk(
    labels: np.ndarray, scores: np.ndarray, target_risk: float
) -> float | None:
    """Most permissive threshold whose selective risk stays within `target_risk`.

    Feasibility is evaluated only at distinct score values, using the same
    `scores >= threshold` rule as `risk_at_threshold`. Candidates sharing a score
    cannot be separated by any threshold, so sweeping ranks instead would report
    a coverage the threshold cannot actually deliver.
    """
    best: float | None = None
    for t in np.unique(scores)[::-1]:  # most selective first
        accepted = scores >= t
        if float(1 - labels[accepted].mean()) <= target_risk:
            best = float(t)  # keep descending to maximize coverage
    return best


def coverage_at_risk(labels: np.ndarray, scores: np.ndarray, target_risk: float) -> float:
    """Coverage achieved at this language's own risk-feasible threshold."""
    thr = threshold_at_risk(labels, scores, target_risk)
    if thr is None:
        return 0.0
    return risk_at_threshold(labels, scores, thr)[1]


def risk_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> tuple[float, float]:
    """(realized risk, coverage) when accepting everything scoring >= threshold."""
    accepted = scores >= threshold
    if not accepted.any():
        return 0.0, 0.0
    return float(1 - labels[accepted].mean()), float(accepted.mean())
