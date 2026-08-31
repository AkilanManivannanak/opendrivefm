#!/usr/bin/env python3
"""Report which trust_scorer tensors a checkpoint actually contains, and whether
they fit the current model at a given --trust_grid.

`load_state_dict(strict=False)` reports missing and unexpected keys separately,
so a checkpoint that stores the right tensors under slightly wrong NAMES shows
up as both "N missing" and "N unexpected" and loads nothing, while looking like
a partial-match warning rather than a failure. This prints the two lists side by
side so that case is unmistakable.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0, "src")
from opendrivefm.training.lightning_module import LitOpenDriveFM  # noqa: E402
from opendrivefm.validation.ckpt_compat import remap_legacy_keys  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="+")
ap.add_argument("--bev", type=int, default=128)
ap.add_argument("--trust_grid", type=int, default=1)
a = ap.parse_args()

ref = LitOpenDriveFM(bev=a.bev, trust_grid=a.trust_grid).model.state_dict()
want = {k: tuple(v.shape) for k, v in ref.items() if "trust" in k}

for path in a.ckpts:
    c = torch.load(path, map_location="cpu", weights_only=False)
    raw = c["state_dict"]
    state = {k.replace("model.", "", 1): v for k, v in raw.items()}
    state, applied, _ = remap_legacy_keys(state, ref)
    if applied:
        print(f"  (after remapping {len(applied)} legacy key(s))")
    have = {k: tuple(v.shape) for k, v in state.items() if "trust" in k}
    print("=" * 78)
    print(f"{path}   (--trust_grid {a.trust_grid})")
    print(f"  model wants {len(want)} trust tensors; checkpoint has {len(have)}")
    missing = [k for k in want if k not in have]
    extra   = [k for k in have if k not in want]
    mism    = [k for k in want if k in have and have[k] != want[k]]
    print(f"  present & shape-OK : {len(want) - len(missing) - len(mism)}")
    print(f"  shape mismatch     : {len(mism)}")
    for k in mism[:6]:
        print(f"      {k}: ckpt {have[k]} vs model {want[k]}")
    print(f"  missing from ckpt  : {len(missing)}")
    for k in missing[:8]:
        print(f"      {k}")
    print(f"  in ckpt, not model : {len(extra)}")
    for k in extra[:8]:
        print(f"      {k}  {have[k]}")
    if missing and extra and len(missing) == len(extra):
        print("  >> LIKELY A NAMING MISMATCH, not absent weights: the counts match.")
    print(f"  ALL unexpected keys in ckpt (any name): "
          f"{[k for k in state if k not in ref][:10]}")
