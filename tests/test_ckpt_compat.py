"""Regression tests for the legacy-checkpoint rename that ran a random trunk.

THE BUG THESE ENCODE
--------------------
`CameraTrustScorer.cnn` was renamed to `.trunk`. `load_state_dict(strict=False)`
matches on name, so 18 structurally-identical trunk tensors stopped loading and
the feature extractor silently ran on random init while the loader reported a
90.5% match rate. An AUROC of 0.764 was measured and published from that state.

Test 1 reproduces the silent failure and would have caught it.
Test 2 pins the fix.
Tests 3-5 pin the safety properties, because a remap that renames too eagerly
is a worse bug than the one it fixes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opendrivefm.models.model import CameraTrustScorer  # noqa: E402
from opendrivefm.validation.ckpt_compat import remap_legacy_keys  # noqa: E402

PREFIX = "backbone.trust_scorer."


# The legacy layout kept the trunk AND the head in one Sequential named `cnn`:
# trunk convs at cnn.0-5, and the head's two Linears at cnn.8 and cnn.10.
LEGACY_RENAMES = {"trunk.": "cnn.", "cnn_head.0.": "cnn.8.", "cnn_head.2.": "cnn.10."}


def _n_legacy(scorer: nn.Module) -> int:
    """How many tensors the rename moved, derived from the model itself."""
    return sum(any(k.startswith(new) for new in LEGACY_RENAMES)
               for k in scorer.state_dict())


def _n_trunk(scorer: nn.Module) -> int:
    return sum(k.startswith("trunk.") for k in scorer.state_dict())


def _trained(grid: int = 1, seed: int = 0) -> nn.Module:
    """A scorer whose every tensor is distinguishable from a fresh init.

    A freshly-constructed model is NOT a usable stand-in for a trained one here.
    BatchNorm initialises deterministically -- weight=1, bias=0, running_mean=0,
    running_var=1 -- so two fresh scorers agree exactly on those tensors, and an
    assertion of the form "the values differ, therefore it did not load" is
    vacuously false for them regardless of whether loading worked. Randomising
    every tensor is what makes "did it arrive?" answerable at all.
    """
    torch.manual_seed(seed)
    scorer = CameraTrustScorer(grid=grid)
    with torch.no_grad():
        for name, t in scorer.state_dict().items():
            if t.dtype.is_floating_point:
                t.copy_(torch.randn_like(t) if "running_var" not in name
                        else torch.rand_like(t) + 0.5)   # variance must be > 0
            else:
                t.fill_(7)                               # num_batches_tracked
    return scorer


def _legacy_state(scorer: nn.Module) -> dict[str, torch.Tensor]:
    """The scorer's state_dict as an OLD checkpoint would have stored it."""
    out = {}
    for k, v in scorer.state_dict().items():
        for new, old in LEGACY_RENAMES.items():
            if k.startswith(new):
                k = old + k[len(new):]
                break
        out[PREFIX + k] = v
    return out


def _reference(scorer: nn.Module) -> dict[str, torch.Tensor]:
    return {PREFIX + k: v for k, v in scorer.state_dict().items()}


def test_legacy_names_do_not_load_without_the_remap():
    """The failure itself: strict=False accepts the checkpoint and loads nothing
    of the trunk, which is what made the bug invisible."""
    trained = _trained()
    fresh = CameraTrustScorer(grid=1)
    legacy = {k[len(PREFIX):]: v for k, v in _legacy_state(trained).items()}

    result = fresh.load_state_dict(legacy, strict=False)
    # Derive the expected set from the model instead of hardcoding a count.
    # A hardcoded number here was wrong twice: it was read off a log rather than
    # derived, and it silently encoded a wrong theory of which tensors moved.
    expected = {k for k in fresh.state_dict()
                if any(k.startswith(new) for new in LEGACY_RENAMES)}
    assert expected, "LEGACY_RENAMES no longer matches any tensor in the model"

    # `missing_keys` UNDERSTATES the damage, and this is the second reason the
    # bug stayed invisible. `_NormBase._load_from_state_dict` deliberately
    # tolerates an absent `num_batches_tracked` for backward compatibility with
    # pre-1.0 checkpoints: it fills in a default and does NOT report the key as
    # missing. So 18 tensors failed to load and the loader admitted to 16.
    #
    # The lesson generalises past this bug: the loader's own report is not a
    # completeness check. Assert on the WEIGHTS, not on the report.
    silent = {k for k in expected if k.endswith("num_batches_tracked")}
    assert set(result.missing_keys) & expected == expected - silent, (
        "every renamed tensor except num_batches_tracked must be reported "
        f"missing; these were not: {sorted(expected - silent - set(result.missing_keys))}")

    # The check that cannot be fooled: did the values actually arrive?
    got, want = fresh.state_dict(), trained.state_dict()
    for k in sorted(expected - silent):
        assert not torch.equal(got[k], want[k]), (
            f"{k} loaded despite the rename, so this test no longer reproduces "
            f"the bug it exists to pin")
    assert any(k.startswith("trunk.") for k in expected)
    assert any(k.startswith("cnn_head.") for k in expected)
    # And the damage is real: the whole CNN branch is still random.
    assert not torch.equal(fresh.trunk[0].weight, trained.trunk[0].weight)
    assert not torch.equal(fresh.cnn_head[0].weight, trained.cnn_head[0].weight)


def test_remap_recovers_every_trunk_tensor():
    trained = _trained()
    fresh = CameraTrustScorer(grid=1)
    ref = _reference(fresh)

    remapped, applied, rejected = remap_legacy_keys(_legacy_state(trained), ref)
    n = _n_legacy(fresh)
    assert len(applied) == n, f"expected {n} renames, got {len(applied)}"
    assert rejected == []

    stripped = {k[len(PREFIX):]: v for k, v in remapped.items()}
    result = fresh.load_state_dict(stripped, strict=False)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    for k, v in trained.state_dict().items():
        assert torch.equal(fresh.state_dict()[k], v), f"{k} did not round-trip"


def test_remapped_model_is_numerically_identical():
    """The point of loading weights is the output, so assert on the output."""
    torch.manual_seed(0)
    trained = _trained().eval()
    fresh = CameraTrustScorer(grid=1).eval()
    ref = _reference(fresh)
    remapped, _, _ = remap_legacy_keys(_legacy_state(trained), ref)
    fresh.load_state_dict({k[len(PREFIX):]: v for k, v in remapped.items()},
                          strict=False)

    x = torch.rand(4, 3, 90, 160)
    with torch.no_grad():
        assert torch.allclose(trained(x), fresh(x), atol=0, rtol=0)


def test_remap_refuses_a_shape_mismatch():
    """A rename that does not fit is a coincidence, not a rename. It must be
    reported, not forced, so the caller's shape check can reject it loudly."""
    trained = _trained()
    legacy = _legacy_state(trained)
    ref = _reference(CameraTrustScorer(grid=1))
    ref[PREFIX + "trunk.0.weight"] = torch.zeros(99, 3, 5, 5)   # wrong shape

    remapped, applied, rejected = remap_legacy_keys(legacy, ref)
    assert (PREFIX + "trunk.0.weight", ) not in [(a[1],) for a in applied]
    assert (PREFIX + "cnn.0.weight", PREFIX + "trunk.0.weight") in rejected
    assert PREFIX + "cnn.0.weight" in remapped, "mismatched key must survive as-is"


def test_modern_checkpoint_is_untouched():
    """A checkpoint already using the current names must pass through byte-exact,
    with no renames applied."""
    scorer = _trained()
    modern = _reference(scorer)
    remapped, applied, rejected = remap_legacy_keys(modern, modern)
    assert applied == [] and rejected == []
    assert set(remapped) == set(modern)
    for k in modern:
        assert torch.equal(remapped[k], modern[k])


def test_modern_name_wins_when_both_are_present():
    """A checkpoint carrying both spellings must not have the modern tensor
    clobbered by the legacy one."""
    scorer = _trained()
    ref = _reference(scorer)
    mixed = dict(ref)
    mixed[PREFIX + "cnn.0.weight"] = torch.full_like(ref[PREFIX + "trunk.0.weight"], 7.0)

    remapped, applied, _ = remap_legacy_keys(mixed, ref)
    assert applied == []
    assert torch.equal(remapped[PREFIX + "trunk.0.weight"],
                       ref[PREFIX + "trunk.0.weight"])


def test_grid4_keeps_the_trunk_but_refuses_the_resized_head():
    """Promoting a legacy grid=1 checkpoint to grid=4 must transfer the trunk
    (unchanged) and refuse the head (its input width doubles). Silently forcing
    the head across would corrupt it; silently dropping the trunk would throw
    away the only transferable weights."""
    trained = _trained()
    ref = _reference(CameraTrustScorer(grid=4))
    _, applied, rejected = remap_legacy_keys(_legacy_state(trained), ref)

    # Exactly ONE tensor changes shape with the grid: cnn_head.0.weight, which
    # is (16,64) at grid=1 and (16,128) at grid=4. Its bias stays (16,) and the
    # second Linear stays (1,16), so those transfer too. Being this specific is
    # the point -- "the head is refused" would have been wrong.
    assert rejected == [(PREFIX + "cnn.8.weight", PREFIX + "cnn_head.0.weight")]
    assert len(applied) == _n_legacy(trained) - 1
    assert (sum(a[1].startswith(PREFIX + "trunk.") for a in applied)
            == _n_trunk(trained))


@pytest.mark.parametrize("grid", [1, 2, 4])
def test_trunk_shapes_are_grid_independent(grid):
    """The remap is only valid because the trunk is unaffected by `grid`. If a
    future change makes the trunk depend on the grid, this fails and the
    LEGACY_PREFIX_MAP entry must be revisited."""
    base = {k: tuple(v.shape) for k, v in CameraTrustScorer(grid=1).state_dict().items()
            if k.startswith("trunk.")}
    other = {k: tuple(v.shape) for k, v in CameraTrustScorer(grid=grid).state_dict().items()
             if k.startswith("trunk.")}
    assert base == other
