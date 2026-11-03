"""Saliency metrics from Bylinskii et al., computed per image and then averaged."""

from __future__ import annotations

import numpy as np


def _flat(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.clip(pred, 0.0, None)
    target = np.clip(target, 0.0, None)
    return pred, target


def correlation_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = _flat(pred, target)
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = np.sqrt((pred ** 2).sum() * (target ** 2).sum())
    if denom < 1e-8:
        return 0.0
    return float((pred * target).sum() / denom)


def similarity(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = _flat(pred, target)
    if pred.sum() <= 0 or target.sum() <= 0:
        return 0.0
    pred = pred / pred.sum()
    target = target / target.sum()
    return float(np.minimum(pred, target).sum())


def kl_divergence(pred: np.ndarray, target: np.ndarray) -> float:
    eps = 2.2204e-16
    pred, target = _flat(pred, target)
    if pred.sum() <= 0 or target.sum() <= 0:
        return 0.0
    pred = pred / pred.sum()
    target = target / target.sum()
    return float((target * np.log(eps + target / (pred + eps))).sum())


def normalized_scanpath_saliency(pred: np.ndarray, fixation: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    fixation = np.asarray(fixation, dtype=np.float64) > 0
    if fixation.sum() == 0:
        return 0.0
    std = pred.std()
    if std < 1e-8:
        return 0.0
    z = (pred - pred.mean()) / std
    return float(z[fixation].mean())


def auc_judd(pred: np.ndarray, fixation: np.ndarray) -> float:
    """ROC AUC with fixation pixels as positives and other pixels as negatives."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    labels = np.asarray(fixation, dtype=np.float64).ravel() > 0
    pos = pred[labels]
    neg = pred[~labels]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    if neg.size > 10000:
        step = max(neg.size // 10000, 1)
        neg = neg[::step][:10000]
    scores = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(pos.size, dtype=np.uint8), np.zeros(neg.size, dtype=np.uint8)])
    order = np.argsort(scores, kind="mergesort")
    scores = scores[order]
    y = y[order]
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    ranks = np.empty(y.size, dtype=np.float64)
    start = 0
    while start < y.size:
        end = start
        while end + 1 < y.size and scores[end + 1] == scores[start]:
            end += 1
        ranks[start : end + 1] = (start + 1 + end + 1) / 2.0
        start = end + 1
    sum_pos_ranks = ranks[y == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def score_pair(pred: np.ndarray, target: np.ndarray, fixation: np.ndarray | None = None) -> dict[str, float]:
    if fixation is None:
        fixation = target >= np.quantile(target, 0.9) if target.max() > 0 else target
    return {
        "CC": correlation_coefficient(pred, target),
        "NSS": normalized_scanpath_saliency(pred, fixation),
        "AUC-Judd": auc_judd(pred, fixation),
        "SIM": similarity(pred, target),
        "KL-Divergence": kl_divergence(pred, target),
    }


def average_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = ["CC", "NSS", "AUC-Judd", "SIM", "KL-Divergence"]
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
