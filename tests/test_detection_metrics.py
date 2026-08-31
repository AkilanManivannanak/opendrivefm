"""Unit tests for opendrivefm.validation.detection_metrics.

These run without torch, so CI can verify the metric implementations without
installing a deep-learning stack. Expected values are computed by hand from the
definition of each metric, not from a reference library, so the test is a real
check rather than a tautology.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opendrivefm.validation.detection_metrics import (  # noqa: E402
    auroc, average_precision, bootstrap_auroc_ci, detector_verdict, tpr_at_fpr)


# ── auroc ────────────────────────────────────────────────────────────────────

def test_auroc_perfect_separation():
    assert auroc([3.0, 4.0, 5.0], [0.0, 1.0, 2.0]) == 1.0


def test_auroc_perfectly_inverted():
    assert auroc([0.0, 1.0, 2.0], [3.0, 4.0, 5.0]) == 0.0


def test_auroc_all_ties_is_chance():
    """Every score identical: no information, must be exactly 0.5.

    This is the case that a naive implementation gets wrong by assigning
    ordinal ranks to tied values.
    """
    assert auroc([1.0] * 5, [1.0] * 5) == 0.5


def test_auroc_interleaved_hand_computed():
    """pos=[1,2], neg=[0,3]. Concordant pairs: (1>0), (2>0). Discordant: 2.

    AUROC = 2 / 4 = 0.5
    """
    assert auroc([1.0, 2.0], [0.0, 3.0]) == pytest.approx(0.5)


def test_auroc_partial_ties_count_as_half():
    """pos=[1,1], neg=[1,0].

    Pairs: (1 vs 1)=0.5, (1 vs 0)=1, (1 vs 1)=0.5, (1 vs 0)=1  ->  3/4 = 0.75
    """
    assert auroc([1.0, 1.0], [1.0, 0.0]) == pytest.approx(0.75)


def test_auroc_empty_input_is_nan():
    assert math.isnan(auroc([], [1.0, 2.0]))
    assert math.isnan(auroc([1.0], []))


# ── average_precision ────────────────────────────────────────────────────────

def test_average_precision_perfect_ranking():
    assert average_precision([5.0, 6.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_average_precision_hand_computed():
    """Ranked scores: 5(pos) 4(neg) 3(pos). Precision at each hit: 1/1 and 2/3.

    AP = (1.0 + 2/3) / 2 = 0.8333...
    """
    assert average_precision([5.0, 3.0], [4.0]) == pytest.approx((1.0 + 2 / 3) / 2)


# ── tpr_at_fpr ───────────────────────────────────────────────────────────────

def test_tpr_at_fpr_detects_everything_when_separated():
    neg = np.arange(100, dtype=float)
    pos = np.full(50, 200.0)
    det, thr = tpr_at_fpr(pos, neg, 0.05)
    assert det == pytest.approx(1.0)
    assert thr < 200.0


def test_tpr_at_fpr_detects_nothing_when_identical():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    det, _ = tpr_at_fpr(x, x, 0.05)
    # At a 5% false-alarm budget on the same distribution, detection ~= 5%.
    assert det < 0.15


# ── bootstrap_auroc_ci ───────────────────────────────────────────────────────

def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(7)
    pos = rng.normal(2.0, 1.0, 200)
    neg = rng.normal(0.0, 1.0, 200)
    point = auroc(pos, neg)
    lo, hi = bootstrap_auroc_ci(pos, neg, n_boot=200, seed=1)
    assert lo <= point <= hi
    assert lo > 0.5, "well-separated data should exclude chance"


def test_bootstrap_ci_includes_chance_for_noise():
    """The guard that decides the script's verdict: pure noise must not look
    like a working detector."""
    rng = np.random.default_rng(3)
    pos = rng.normal(0.0, 1.0, 150)
    neg = rng.normal(0.0, 1.0, 150)
    lo, hi = bootstrap_auroc_ci(pos, neg, n_boot=300, seed=2)
    assert lo <= 0.5 <= hi


def test_bootstrap_ci_is_deterministic_given_seed():
    rng = np.random.default_rng(11)
    pos, neg = rng.normal(1, 1, 80), rng.normal(0, 1, 80)
    assert bootstrap_auroc_ci(pos, neg, 100, 5) == bootstrap_auroc_ci(pos, neg, 100, 5)


# ── detector_verdict ─────────────────────────────────────────────────────────

def test_verdict_inverted_when_ci_entirely_below_chance():
    """The case that the first version of this script got wrong.

    Real measured values from checkpoints_v11_temporal: AUROC 0.434 with CI
    [0.419, 0.449]. That is NOT 'includes chance', it is significantly inverted.
    """
    code, _ = detector_verdict(0.434, 0.419, 0.449)
    assert code == "inverted"


def test_verdict_chance_when_ci_straddles_half():
    code, _ = detector_verdict(0.51, 0.46, 0.56)
    assert code == "chance"


def test_verdict_weak_when_above_chance_but_below_strong():
    code, _ = detector_verdict(0.68, 0.60, 0.75)
    assert code == "weak"


def test_verdict_working_when_strong_and_significant():
    code, _ = detector_verdict(0.91, 0.86, 0.95)
    assert code == "working"


def test_verdict_boundary_ci_touching_half_is_chance_not_inverted():
    code, _ = detector_verdict(0.47, 0.40, 0.50)
    assert code == "chance"
