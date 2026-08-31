"""A calibrated trust scorer must score a frame the same way in any batch.

THE BUG THESE PIN
-----------------
`_image_stats` centred on `stats.mean(dim=0)`, the mean over the batch. Since the
scorer sees every camera of every frame in one call, a camera's trust depended on
what else shared the forward pass. Measured consequence: the faulted camera's
response fell 36% between batch_size 1 and 8.

Test 1 reproduces it. Tests 2-4 pin the fix. Test 5 pins the property that made
the bug survivable, so that a future change cannot quietly make it worse.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opendrivefm.models.model import CameraTrustScorer  # noqa: E402


def _scorer(grid: int = 1, calibrated: bool = True, seed: int = 0):
    torch.manual_seed(seed)
    s = CameraTrustScorer(grid=grid).eval()
    if calibrated:
        with torch.no_grad():
            # A plausible calibration: the mean of the raw statistics over data.
            s.stat_running_mean.copy_(s._raw_stats(torch.rand(24, 3, 90, 160)).mean(0))
            s.stat_calibrated.fill_(True)
    return s


def _probe_and_filler(seed: int = 1):
    torch.manual_seed(seed)
    probe = torch.rand(1, 3, 90, 160) * 0.5 + 0.25
    filler = torch.rand(11, 3, 90, 160) * 0.5 + 0.25
    return probe, filler


def test_uncalibrated_scorer_is_batch_dependent():
    """The defect itself, asserted where it lives.

    This deliberately probes `_image_stats` rather than the full forward. The
    statistics branch reaches the output through `stats_head` and then `fuse`,
    and at random initialisation those layers can attenuate the difference below
    any sensible tolerance -- the first version of this test compared 0.5216 to
    0.5216 and concluded the bug was gone. A regression test whose sensitivity
    depends on the random seed is not a test. The defect is in the centring, so
    the assertion belongs on the centring.

    A single-image batch is the sharpest case: `stats - stats.mean(0)` is
    exactly zero, so EVERY feature collapses to sigmoid(0) = 0.5 and the frame's
    own statistics are annihilated.
    """
    un, cal = _scorer(calibrated=False), _scorer(calibrated=True)
    probe, filler = _probe_and_filler()

    # Assertion 1 is threshold-free and decisive.
    with torch.no_grad():
        alone = un._image_stats(probe)[0]
    assert torch.allclose(alone, torch.full_like(alone, 0.5), atol=1e-6), (
        "a batch of one must self-cancel to 0.5 under batch-mean centring")

    # Assertion 2: neighbours change the score. The filler must actually DIFFER
    # in image statistics for this to measure anything -- with uniform-random
    # filler the batch mean barely moves and the effect was 7e-04, which says
    # more about the fixture than about the model. A dark, smooth frame is both
    # the realistic case (that is what a faulted camera looks like) and the
    # adversarial one.
    #
    # The bar is the CALIBRATED model under the identical perturbation, not a
    # constant picked to pass. Same probe, same neighbours, same everything but
    # the fix.
    dark = torch.full((11, 3, 90, 160), 0.05)
    with torch.no_grad():
        d_un = (un._image_stats(probe)[0]
                - un._image_stats(torch.cat([probe, dark]))[0]).abs().max()
        d_cal = (cal._image_stats(probe)[0]
                 - cal._image_stats(torch.cat([probe, dark]))[0]).abs().max()
    assert d_cal < 1e-6, f"calibrated model must not move at all, moved {d_cal:.3e}"
    assert d_un > 1000 * max(d_cal, 1e-9), (
        f"expected the uncalibrated path to be batch-dependent: it moved "
        f"{d_un:.3e} where the calibrated path moved {d_cal:.3e}")


def test_calibrated_scorer_survives_a_batch_of_one():
    """The complement: with a fixed reference, a single-image batch keeps its
    statistics instead of collapsing to 0.5."""
    s = _scorer()
    probe, _ = _probe_and_filler()
    with torch.no_grad():
        stats = s._image_stats(probe)[0]
    assert not torch.allclose(stats, torch.full_like(stats, 0.5), atol=1e-3), (
        "calibrated statistics must not self-cancel for a single image")


@pytest.mark.parametrize("grid", [1, 4])
def test_calibrated_scorer_is_batch_invariant(grid):
    """The fix: identical score in any batch, to floating-point exactness."""
    s = _scorer(grid=grid)
    probe, filler = _probe_and_filler()
    with torch.no_grad():
        alone = s(probe)[0]
        for k in (1, 3, 7, 11):
            mixed = s(torch.cat([probe, filler[:k]]))[0]
            assert torch.allclose(alone, mixed, atol=1e-6), (
                f"score moved by {abs(alone - mixed):.3e} when batched with {k} "
                f"other images")


def test_calibrated_score_ignores_neighbour_content():
    """Batch SIZE is not the only channel: batch CONTENT must not leak either.
    Padding with black frames is the adversarial case, since it drags a batch
    mean hard."""
    s = _scorer()
    probe, filler = _probe_and_filler()
    black = torch.zeros(5, 3, 90, 160)
    with torch.no_grad():
        a = s(torch.cat([probe, filler[:5]]))[0]
        b = s(torch.cat([probe, black]))[0]
    assert torch.allclose(a, b, atol=1e-6)


def test_uncalibrated_fallback_is_bit_identical_to_the_old_behaviour():
    """The fallback must reproduce the previous function exactly, so existing
    published numbers stay reproducible from an uncalibrated checkpoint."""
    s = _scorer(calibrated=False)
    x = torch.rand(8, 3, 90, 160)
    with torch.no_grad():
        stats = s._raw_stats(x)
        expected = torch.sigmoid(stats - stats.detach().mean(dim=0))
        assert torch.equal(s._image_stats(x), expected)


def test_calibration_buffers_do_not_change_the_parameter_count():
    """Buffers, not parameters: calibration must not add anything trainable, or
    `finetune_trust_head.py`'s "0.36% of parameters" claim silently changes."""
    a = sum(p.numel() for p in CameraTrustScorer(grid=1).parameters())
    torch.manual_seed(0)
    b = sum(p.numel() for p in _scorer().parameters())
    assert a == b


def test_ranking_is_preserved_by_calibration():
    """Explains why AUROC was never wrong: within one fixed batch, calibration
    shifts every score by a common reference and cannot reorder them."""
    torch.manual_seed(3)
    x = torch.rand(16, 3, 90, 160)
    un, cal = _scorer(calibrated=False), _scorer(calibrated=True)
    with torch.no_grad():
        a, b = un(x), cal(x)
    assert torch.equal(a.argsort(), b.argsort()), (
        "calibration reordered scores within a batch; AUROC comparisons across "
        "the change would not be valid")
