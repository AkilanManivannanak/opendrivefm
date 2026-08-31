"""Detection metrics for autonomy-model validation.

Pure numpy, no sklearn and no torch, so these are unit-testable on their own and
importable from CI jobs that do not install a deep-learning stack.
"""
from __future__ import annotations

import numpy as np


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U), with correct handling of ties.

    Equivalent to sklearn.metrics.roc_auc_score. 0.5 means the detector carries
    no information; below 0.5 means it is anti-correlated with the fault.
    """
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")

    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=float)
    ranks[order] = np.arange(1, allv.size + 1, dtype=float)

    # Average the ranks inside each tie group.
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1

    n1, n0 = pos.size, neg.size
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(pos: np.ndarray, neg: np.ndarray) -> float:
    """Area under the precision-recall curve (step interpolation)."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    tp = np.cumsum(labels)
    precision = tp / np.arange(1, labels.size + 1)
    return float((precision * labels).sum() / labels.sum())


def tpr_at_fpr(pos: np.ndarray, neg: np.ndarray, fpr: float) -> tuple[float, float]:
    """Detection rate at a fixed false-alarm rate, plus the threshold used.

    This is the number an operator actually cares about: at a false-alarm budget
    of `fpr`, what fraction of genuinely degraded cameras get flagged?
    """
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan"), float("nan")
    thr = float(np.quantile(neg, 1.0 - fpr))
    return float((pos > thr).mean()), thr


def bootstrap_auroc_ci(pos: np.ndarray, neg: np.ndarray, n_boot: int,
                       seed: int) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for AUROC.

    With ~80 validation frames the point estimate alone is not trustworthy, so
    every AUROC below is reported with an interval.
    """
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0 or n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        vals[b] = auroc(rng.choice(pos, pos.size, replace=True),
                        rng.choice(neg, neg.size, replace=True))
    lo, hi = np.percentile(vals[~np.isnan(vals)], [2.5, 97.5])
    return float(lo), float(hi)




def detector_verdict(point: float, lo: float, hi: float,
                     strong: float = 0.8,
                     score_name: str = "the detector score",
                     positive_cond: str = "the condition is present") -> tuple[str, str]:
    """Classify a detector from its AUROC point estimate and 95% CI.

    Returns (code, human-readable message). The four cases are genuinely
    different diagnoses and must not be collapsed:

    inverted  : the whole CI sits below 0.5. The detector is significantly
                ANTI-correlated with the fault, i.e. it grows more confident as
                the input degrades. This is a bug with a sign, not noise, and it
                is more dangerous in deployment than no detector at all.
    chance    : the CI straddles 0.5. The detector carries no information.
    weak      : significantly better than chance but below `strong`.
    working   : significantly better than chance and at or above `strong`.
    """
    if hi < 0.5:
        return "inverted", (
            f"AUROC {point:.3f}, 95% CI [{lo:.3f}, {hi:.3f}] lies entirely BELOW 0.5.\n"
            f"{score_name} moves the WRONG way when {positive_cond}. This is an\n"
            "inverted detector, not an absent one, and it is more dangerous than no\n"
            "detector: acting on it does the opposite of the right thing.")
    if lo <= 0.5 <= hi:
        return "chance", (
            f"AUROC {point:.3f}, 95% CI [{lo:.3f}, {hi:.3f}] includes 0.5.\n"
            "The detector carries no information about camera degradation.")
    if point < strong:
        return "weak", (
            f"AUROC {point:.3f}, 95% CI [{lo:.3f}, {hi:.3f}] is above chance but\n"
            f"below {strong:.2f}. Usable as a soft signal, not as a standalone\n"
            "camera-fault monitor. Report the AUROC, not a detection rate.")
    return "working", (
        f"AUROC {point:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]. The scorer separates\n"
        "clean from faulted cameras. Report AUROC with its CI and the detection\n"
        "rate at a stated false-alarm budget.")
