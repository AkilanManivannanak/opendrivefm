#!/usr/bin/env python3
"""Freeze the CameraTrustScorer's statistics reference, making trust a pure
function of one frame.

THE PROBLEM
-----------
`_image_stats` centred on `stats.mean(dim=0)`, the mean over the BATCH. The
scorer scores every camera of every frame in one call, so a camera's trust
depended on what else shared the forward pass. The same frame scored differently
in different batches, and `mean_trust_drop` moved 36% between batch sizes 1 and
8. AUROC survived it (all scores in a sweep share one reference); absolute
thresholds and effect sizes did not.

THE FIX
-------
Estimate the centre ONCE over a fixed calibration set and store it in the
`stat_running_mean` buffer, the way BatchNorm stores running statistics. At eval
the buffer is used and the batch is irrelevant.

This script averages the UNCENTRED statistics over the calibration split. It
deliberately does not run the model in train mode, which would also move the
trunk's BatchNorm running statistics and change the trained function.

Calibrate on TRAIN scenes, never on the evaluation scenes: the centre is a fitted
parameter, and fitting it on the frames you then score is the same in-sample bias
this repo measures at 12.7x for scene-level OOD.

USAGE
    python scripts/calibrate_trust_stats.py \
        --ckpt outputs/artifacts/checkpoints_v11_trustfix2/trust_fixed_v2.ckpt \
        --out  outputs/artifacts/checkpoints_v11_trustfix2/trust_fixed_v2_cal.ckpt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "src")

from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView  # noqa: E402
from opendrivefm.training.lightning_module import LitOpenDriveFM  # noqa: E402
from opendrivefm.validation.ckpt_compat import (  # noqa: E402
    remap_legacy_keys, report_remap)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest",
                    default="outputs/artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", default="outputs/artifacts/nuscenes_labels_128")
    ap.add_argument("--trust_grid", type=int, default=1)
    ap.add_argument("--bev", type=int, default=128)
    ap.add_argument("--image_hw", default="90,160")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--val_scenes", default=None)
    args = ap.parse_args()

    H, W = [int(v) for v in args.image_hw.split(",")]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}
    lit = LitOpenDriveFM(bev=args.bev, trust_grid=args.trust_grid)
    ref = lit.model.state_dict()
    state, applied, rejected = remap_legacy_keys(state, ref)
    report_remap(applied, rejected)
    for k in [k for k, v in state.items()
              if k in ref and tuple(v.shape) != tuple(ref[k].shape)]:
        state.pop(k)
    lit.model.load_state_dict(state, strict=False)
    model = lit.model.eval().to(device)
    scorer = model.backbone.trust_scorer

    rows = [json.loads(l) for l in
            Path(args.manifest).read_text().splitlines() if l.strip()]
    if args.val_scenes:
        val = {s.strip() for s in args.val_scenes.split(",") if s.strip()}
    else:
        scenes = sorted({r["scene"] for r in rows})
        rng = random.Random(args.seed); rng.shuffle(scenes)
        val = set(scenes[:max(1, int(round(len(scenes) * args.val_frac)))])
    idx_train = [i for i, r in enumerate(rows) if r["scene"] not in val]
    print(f"calibrating on {len(idx_train)} TRAIN frames "
          f"(held out: {sorted(val)})")

    ds = NuScenesMiniMultiView(
        args.manifest, image_hw=(H, W), frames=1, label_root=args.label_root,
        return_motion=True, return_trel=True, return_calib=True, augment=False)
    loader = DataLoader(Subset(ds, idx_train), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)

    total, n = None, 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            B, V = x.shape[0], x.shape[1]
            imgs = x[:, :, -1].reshape(B * V, *x.shape[3:])
            # Accumulate in float64 on the CPU. MPS has no float64 at all, and
            # the blur statistic is a Laplacian VARIANCE whose magnitude is far
            # from 1, so summing ~2k of them in float32 loses real precision in
            # a value every downstream score is centred on.
            st = scorer._raw_stats(imgs).detach().cpu().double()
            total = st.sum(0) if total is None else total + st.sum(0)
            n += st.shape[0]

    mean = (total / n).float()
    old = scorer.stat_running_mean.detach().cpu().clone()
    scorer.stat_running_mean.copy_(mean.to(scorer.stat_running_mean.device))
    scorer.stat_calibrated.fill_(True)

    print(f"\ncalibrated over {n} camera-images "
          f"({n // max(1, len(idx_train))} cameras x {len(idx_train)} frames)")
    print(f"  buffer before: {[round(v, 5) for v in old.tolist()]}")
    print(f"  buffer after : {[round(v, 5) for v in mean.tolist()]}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": {f"model.{k}": v
                              for k, v in model.state_dict().items()}}
    payload["trust_calibration"] = {
        "source_ckpt": args.ckpt, "n_images": int(n),
        "train_frames": len(idx_train), "held_out_scenes": sorted(val),
        "stat_running_mean": [float(v) for v in mean.tolist()],
        "trust_grid": args.trust_grid,
    }
    torch.save(payload, out)
    print(f"\nwrote {out}")
    print("\nTrust is now a pure function of a single frame. Verify with:\n"
          f"  python scripts/eval/check_batch_contamination.py --ckpt {out} \\\n"
          f"    --trust_grid {args.trust_grid} --fault occlusion --batch_sizes 1,2,4,8")


if __name__ == "__main__":
    main()
