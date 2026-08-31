"""Rename legacy checkpoint keys onto the current module names.

WHY THIS EXISTS
---------------
`CameraTrustScorer` was rewritten to take a `grid` argument, and in that rewrite
its convolutional trunk was renamed from `self.cnn` to `self.trunk`. The tensors
are structurally identical -- same layers, same shapes, same order -- but
`load_state_dict(..., strict=False)` matches on NAME. So every trust checkpoint
trained before the rewrite stopped loading its 18 trunk tensors and silently ran
the feature extractor on random initialisation, while reporting a 90% match rate
that looked like a rounding detail.

That is the worst failure mode available: not a crash, not a zero, but a
plausible number produced by a model that is half noise. The AUROC of 0.764
reported for `checkpoints_v11_trustfix2/trust_fixed_v2.ckpt` was measured this
way and does not describe the trained scorer.

A rename is a silent, load-bearing API change. This module makes it explicit.

SAFETY
------
`remap_legacy_keys` never renames a tensor onto a name whose shape disagrees.
A rename that does not fit is not a rename, it is a coincidence, and it is left
alone so the caller's shape check can reject it loudly.
"""
from __future__ import annotations

from typing import Mapping

import torch

# old prefix -> new prefix. ORDER MATTERS: the first match wins, so the
# specific entries must precede the general one.
#
# The legacy CameraTrustScorer put the whole CNN branch in ONE Sequential
# called `cnn`:
#
#   cnn.0  Conv2d(3, 32, 5, stride=4)      cnn.6  AdaptiveAvgPool2d(1)
#   cnn.1  BatchNorm2d(32)                 cnn.7  Flatten
#   cnn.2  GELU                            cnn.8  Linear(64, 16)   <- head
#   cnn.3  Conv2d(32, 64, 5, stride=4)     cnn.9  GELU
#   cnn.4  BatchNorm2d(64)                 cnn.10 Linear(16, 1)    <- head
#   cnn.5  GELU                            cnn.11 Sigmoid
#
# The rewrite split it into `trunk` (the convolutions) and `cnn_head` (the two
# Linears), because grid>1 changes the pooling between them and the head's input
# width along with it. So `cnn.` maps to `trunk.` for indices 0-5, but indices
# 8 and 10 are the head and map somewhere else entirely.
#
# At grid>1 the head's first Linear legitimately changes shape (16,64)->(16,128),
# so that rename is refused on shape and the head is retrained from scratch,
# which is correct: a wider input needs new weights. The trunk still transfers.
LEGACY_PREFIX_MAP: tuple[tuple[str, str], ...] = (
    ("backbone.trust_scorer.cnn.8.", "backbone.trust_scorer.cnn_head.0."),
    ("backbone.trust_scorer.cnn.10.", "backbone.trust_scorer.cnn_head.2."),
    ("backbone.trust_scorer.cnn.", "backbone.trust_scorer.trunk."),
)


def remap_legacy_keys(
    state: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[tuple[str, str]], list[tuple[str, str]]]:
    """Rename legacy keys in `state` to match `reference`.

    Returns (new_state, applied, rejected) where `applied` and `rejected` are
    lists of (old_key, new_key). A rename is applied only when the new key
    exists in `reference`, is not already present in `state`, and the shapes
    agree. Everything else is returned untouched.
    """
    out: dict[str, torch.Tensor] = {}
    applied: list[tuple[str, str]] = []
    rejected: list[tuple[str, str]] = []

    for key, tensor in state.items():
        new_key = key
        for old_prefix, new_prefix in LEGACY_PREFIX_MAP:
            if not key.startswith(old_prefix):
                continue
            cand = new_prefix + key[len(old_prefix):]
            if cand in state:
                # The checkpoint already carries the modern name too. Renaming
                # would clobber it; the modern one wins.
                break
            if cand not in reference:
                rejected.append((key, cand))
                break
            if tuple(reference[cand].shape) != tuple(tensor.shape):
                rejected.append((key, cand))
                break
            new_key = cand
            applied.append((key, cand))
            break
        out[new_key] = tensor

    return out, applied, rejected


def report_remap(applied, rejected) -> None:
    """Print what was renamed. Silence when nothing was."""
    if applied:
        print(f"  remapped {len(applied)} legacy key(s) onto current module names "
              f"(e.g. {applied[0][0]} -> {applied[0][1]})")
    if rejected:
        print(f"  WARNING: {len(rejected)} legacy key(s) look renameable but do NOT "
              f"fit the current model and were left as-is:")
        for old, new in rejected[:4]:
            print(f"    {old} -> {new}")
