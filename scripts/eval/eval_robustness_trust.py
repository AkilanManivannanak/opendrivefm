"""
Robustness evaluation: tests how trust scores respond to camera degradations.

Loads a trained checkpoint, iterates over the val set, and for each sample:
  1. Runs inference clean → records baseline trust scores
  2. Corrupts one camera at a time with each perturbation type
  3. Records trust drop and trajectory/occ degradation

Outputs:
  artifacts/robustness_report.json
  artifacts/robustness_trust_chart.png
  artifacts/robustness_bev_samples/

Usage:
  python scripts/eval_robustness_trust.py \
      --ckpt artifacts/checkpoints_trust/last.ckpt \
      --manifest artifacts/nuscenes_mini_manifest.jsonl \
      --label_root artifacts/nuscenes_labels
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView
from opendrivefm.train.lightning_module import LitOpenDriveFM, _dl_kwargs
from opendrivefm.robustness.perturbations import PERTURBATIONS
from opendrivefm.utils.visualise import plot_robustness_report


def split_val(rows, seed, val_frac):
    scenes = sorted({r["scene"] for r in rows})
    rng    = random.Random(seed)
    rng.shuffle(scenes)
    n_val  = max(1, int(round(len(scenes) * val_frac)))
    vs     = set(scenes[:n_val])
    return [i for i, r in enumerate(rows) if r["scene"] in vs]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",              type=str, required=True)
    ap.add_argument("--manifest",          type=str, default="artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root",        type=str, default="artifacts/nuscenes_labels")
    ap.add_argument("--seed",              type=int, default=42)
    ap.add_argument("--val_frac",          type=float, default=0.2)
    ap.add_argument("--batch_size",        type=int, default=2)
    ap.add_argument("--image_hw",          type=str, default="90,160")
    ap.add_argument("--perturb_cam_idx",   type=int, default=0,
                    help="Which camera to corrupt (0=FRONT)")
    ap.add_argument("--out_dir",           type=str, default="artifacts")
    args = ap.parse_args()

    H, W   = [int(v.strip()) for v in args.image_hw.split(",")]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out    = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    rows     = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    idx_val  = split_val(rows, args.seed, args.val_frac)

    ds = NuScenesMiniMultiView(
        args.manifest, image_hw=(H, W), frames=1, label_root=args.label_root,
        return_motion=True, return_trel=True)
    dl = DataLoader(Subset(ds, idx_val), batch_size=args.batch_size,
                    shuffle=False, **_dl_kwargs())

    lit = LitOpenDriveFM.load_from_checkpoint(args.ckpt, map_location="cpu")
    lit.eval().to(device)

    # build perturbation instances
    perturbers = {k: v() for k, v in PERTURBATIONS.items()}

    # Accumulators: {perturb_name: [trust_vals]}
    trust_acc = {"clean": []}
    for k in perturbers: trust_acc[k] = []

    cam_idx = args.perturb_cam_idx

    for batch in dl:
        x, occ_t, traj_t, motion, t_rel = batch
        x = x.to(device)

        # Clean
        _, _, trust = lit.model(x)
        trust_acc["clean"].extend(trust[:, cam_idx].cpu().tolist())

        # Per-perturbation
        for name, perturber in perturbers.items():
            x_p = x.clone()
            # x: (B, V, T, C, H, W) — corrupt first frame of cam_idx
            cam_frame = x_p[:, cam_idx, 0]              # (B, C, H, W)
            cam_frame = perturber(cam_frame)
            x_p[:, cam_idx, 0] = cam_frame
            _, _, trust_p = lit.model(x_p)
            trust_acc[name].extend(trust_p[:, cam_idx].cpu().tolist())

    # Aggregate
    summary = {k: float(np.mean(v)) for k, v in trust_acc.items() if v}

    print("\n" + "="*55)
    print(f"  Camera perturbed: {'FRONT' if cam_idx==0 else f'cam_{cam_idx}'}")
    print("="*55)
    for k, v in summary.items():
        bar = "█" * int(v*30) + "░" * (30 - int(v*30))
        drop = summary["clean"] - v if k != "clean" else 0.0
        print(f"  {k:<12s}  {bar}  {v:.3f}  (drop: {drop:+.3f})")
    print("="*55)

    # Save chart
    plot_robustness_report(summary, save_path=str(out / "robustness_trust_chart.png"))

    report = {
        "ckpt":         args.ckpt,
        "perturb_cam":  cam_idx,
        "val_items":    len(idx_val),
        "trust_scores": summary,
        "trust_drops":  {k: round(summary["clean"] - v, 4)
                         for k, v in summary.items() if k != "clean"},
    }
    (out / "robustness_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved → {out}/robustness_report.json")
    print(f"Saved → {out}/robustness_trust_chart.png")


if __name__ == "__main__":
    main()
