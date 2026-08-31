"""
Test-suite prioritisation by failure sensitivity.

THE PROBLEM
-----------
Running the full perturbation battery on every validation frame costs
frames x faults x severities forward passes. That is fine nightly and far too
slow per-commit, so in practice per-commit testing gets skipped and regressions
land unnoticed. The fix is not a faster machine, it is a smaller suite that
retains most of the signal.

THE IDEA
--------
Frames are not equally informative. A frame whose IoU barely moves under any
corruption tells you almost nothing when it passes. A frame whose IoU collapses
under mild corruption is where a regression will show up first. Rank frames by
sensitivity, keep the top k%, and measure what fraction of total observed
degradation that subset still captures.

That last number is the whole justification. "The top 20% of frames capture 71%
of total degradation" is an argument for a smoke suite. "We picked 20% of the
frames" is not.

WHAT IT REPORTS
---------------
  * per-frame sensitivity (mean and max degradation across the battery);
  * the degradation-capture curve: what share of total degradation the top
    k% of frames accounts for, for k in 5..100;
  * a suggested smoke set written to JSON, consumable by CI.

Usage
-----
    python scripts/eval/prioritize_tests.py \
        --ckpt outputs/artifacts/checkpoints_v11_trustfix2/trust_fixed_v2.ckpt \
        --out outputs/artifacts/test_prioritisation.json
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
from opendrivefm.robustness.perturbations import PERTURBATIONS  # noqa: E402
from opendrivefm.training.lightning_module import LitOpenDriveFM  # noqa: E402

THR = 0.6
VAL_SCENES = {"scene-0655", "scene-1077"}


@torch.no_grad()
def per_frame_metrics(model, loader, device, fault=None, cams=()) -> tuple:
    """Per-frame IoU and ADE (not averaged), so frames can be ranked."""
    perts = [(c, PERTURBATIONS[fault]()) for c in cams] if fault else []
    ious, ades = [], []
    for batch in loader:
        x, occ_t, traj = (batch[i].to(device) for i in range(3))
        motion, t_rel = batch[3].to(device), batch[4].to(device)
        if occ_t.ndim == 3:
            occ_t = occ_t.unsqueeze(1)
        vel = motion[:, 1:3]
        if perts:
            x = x.clone()
            for b in range(x.shape[0]):
                for c, p in perts:
                    x[b, c, -1] = p(x[b, c, -1].unsqueeze(0)).squeeze(0)
        occ_logits, traj_res, _, _ = model(x, velocity=vel)
        if occ_logits.ndim == 3:
            occ_logits = occ_logits.unsqueeze(1)
        pred = (torch.sigmoid(occ_logits) > THR).float()
        inter = (pred * occ_t).sum((1, 2, 3))
        union = (pred + occ_t).clamp(0, 1).sum((1, 2, 3))
        ious.append((inter / (union + 1e-6)).cpu().numpy())
        p_traj = t_rel.unsqueeze(-1) * vel.unsqueeze(1) + traj_res
        ades.append(torch.linalg.norm(p_traj - traj, dim=-1).mean(1).cpu().numpy())
    return np.concatenate(ious), np.concatenate(ades)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="outputs/artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", default="outputs/artifacts/nuscenes_labels_128")
    ap.add_argument("--bev", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--image_hw", default="90,160")
    ap.add_argument("--cams", default="0,1,4",
                    help="Cameras to perturb. Default includes FRONT_LEFT and "
                         "BACK_LEFT, which the fuzzer found dominate the tail.")
    ap.add_argument("--smoke_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/artifacts/test_prioritisation.json")
    args = ap.parse_args()

    H, W = [int(v) for v in args.image_hw.split(",")]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}
    lit = LitOpenDriveFM(bev=args.bev)
    res = lit.model.load_state_dict(state, strict=False)
    n_exp = len(lit.model.state_dict())
    print(f"Loaded {n_exp - len(res.missing_keys)}/{n_exp} weights")
    model = lit.model.eval().to(device)

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    idx = [i for i, r in enumerate(rows) if r["scene"] in VAL_SCENES]
    ds = NuScenesMiniMultiView(args.manifest, image_hw=(H, W), frames=1,
                               label_root=args.label_root, return_motion=True,
                               return_trel=True, return_calib=True, augment=False)
    loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                        num_workers=0)
    cams = [int(c) for c in args.cams.split(",")]
    print(f"  frames: {len(idx)}   battery: {len(PERTURBATIONS)} faults on cameras {cams}\n")

    base_iou, base_ade = per_frame_metrics(model, loader, device)
    deg_iou = np.zeros((len(PERTURBATIONS), base_iou.size))
    deg_ade = np.zeros_like(deg_iou)
    for i, fault in enumerate(PERTURBATIONS):
        f_iou, f_ade = per_frame_metrics(model, loader, device, fault=fault, cams=cams)
        deg_iou[i] = base_iou - f_iou      # positive = worse
        deg_ade[i] = f_ade - base_ade
        print(f"  {fault:<10} mean dIoU {deg_iou[i].mean():+.4f}  "
              f"max dIoU {deg_iou[i].max():+.4f}")

    # Sensitivity: how much this frame's metrics move under the battery. Max is
    # the right aggregator, not mean -- a frame that is catastrophic under ONE
    # fault is worth keeping even if the other four leave it untouched.
    sens = deg_iou.max(axis=0)
    order = np.argsort(-sens)
    total = float(np.clip(sens, 0, None).sum())

    print(f"\nDEGRADATION-CAPTURE CURVE (why a smoke suite is defensible)")
    print(f"{'top k%':>8}{'frames':>9}{'captured':>11}{'of total deg':>14}")
    curve = {}
    for k in [5, 10, 20, 30, 50, 75, 100]:
        n = max(1, int(round(len(order) * k / 100)))
        cap = float(np.clip(sens[order[:n]], 0, None).sum())
        frac = cap / total if total > 0 else float("nan")
        curve[str(k)] = {"n_frames": n, "captured_fraction": frac}
        print(f"{k:>7}%{n:>9}{cap:>11.4f}{100*frac:>13.1f}%")

    n_smoke = max(1, int(round(len(order) * args.smoke_frac)))
    smoke = [int(idx[i]) for i in order[:n_smoke]]
    print(f"\nSuggested smoke set: {n_smoke} frames "
          f"({100*args.smoke_frac:.0f}%) capturing "
          f"{100*curve[str(int(args.smoke_frac*100))]['captured_fraction']:.1f}% "
          f"of total degradation" if str(int(args.smoke_frac*100)) in curve else "")

    print(f"\n{'rank':>5}{'frame':>8}{'scene':>14}{'clean IoU':>11}{'max dIoU':>10}"
          f"   worst fault")
    fault_names = list(PERTURBATIONS)
    for r, i in enumerate(order[:10], 1):
        worst = fault_names[int(np.argmax(deg_iou[:, i]))]
        print(f"{r:>5}{idx[i]:>8}{rows[idx[i]]['scene']:>14}{base_iou[i]:>11.4f}"
              f"{sens[i]:>10.4f}   {worst}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "checkpoint": args.ckpt, "cameras_perturbed": cams,
        "n_frames": len(idx), "faults": fault_names,
        "capture_curve": curve,
        "smoke_frac": args.smoke_frac, "smoke_set_manifest_indices": smoke,
        "per_frame": [
            {"manifest_index": int(idx[i]), "scene": rows[idx[i]]["scene"],
             "clean_iou": float(base_iou[i]), "clean_ade": float(base_ade[i]),
             "sensitivity_max_diou": float(sens[i]),
             "sensitivity_mean_diou": float(deg_iou[:, i].mean()),
             "worst_fault": fault_names[int(np.argmax(deg_iou[:, i]))]}
            for i in order],
    }, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
