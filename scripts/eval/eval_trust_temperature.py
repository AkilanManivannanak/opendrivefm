"""
Temperature sweep for TrustWeightedFusion: does detecting a degraded camera
actually help perception, and at what softmax temperature?

THE QUESTION
------------
`eval_ood_detection.py` fixed *detection*: pooled AUROC 0.434 -> 0.764. But
`eval_trust_ablation.py` shows trust-weighted fusion beats uniform fusion by at
most +0.0015 IoU, with the wrong sign on three of five faults. Detection and
mitigation are separate problems and only the first one had been measured.

THE MECHANISM
-------------
    TrustWeightedFusion.forward:  w = softmax(trust, dim=1)

Softmax carries an implicit temperature of 1.0. Trust scores across six cameras
differ by roughly 0.1-0.3, so softmax(trust) lands very close to uniform (1/6 =
0.1667): a degraded camera is down-weighted ~25%, not suppressed. Even a perfect
detector cannot move the fused feature much through that bottleneck.

Sweeping softmax(trust / T) with T < 1 sharpens the distribution toward hard
camera dropout. This script measures, for every temperature:

  * IoU / ADE / FDE, clean and under each fault, trust fusion vs uniform fusion
  * w_faulted: the softmax weight actually assigned to the degraded camera

w_faulted is the diagnostic that makes the result explainable rather than
empirical. If IoU does not improve even when w_faulted -> 0, the limitation is
not the fusion weighting and the README claim should be retired.

Usage
-----
    python scripts/eval/eval_trust_temperature.py \
        --ckpt outputs/artifacts/checkpoints_v11_trustfix2/trust_fixed_v2.ckpt \
        --out  outputs/artifacts/trust_temperature_sweep.json
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


def load_model(ckpt_path: str, bev: int, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}
    lit = LitOpenDriveFM(bev=bev)
    res = lit.model.load_state_dict(state, strict=False)
    n = len(lit.model.state_dict())
    print(f"  {n - len(res.missing_keys)}/{n} weights matched from {ckpt_path}")
    return lit.model.eval().to(device)


@torch.no_grad()
def evaluate(model, loader, device, *, temperature, uniform, fault, cam):
    """Returns (IoU, ADE, FDE, mean softmax weight on the faulted camera)."""
    fuse = model.backbone.trust_fuse
    scorer = model.backbone.trust_scorer
    orig_fuse_forward = fuse.forward
    orig_scorer_forward = scorer.forward
    weights_on_faulted = []

    def temped_fuse(hv, trust):
        w = torch.softmax(trust / temperature, dim=1)
        weights_on_faulted.append(float(w[:, cam].mean()))
        return fuse.mlp((w.unsqueeze(-1) * hv).sum(dim=1))

    def flat_trust(imgs):
        return torch.full((imgs.shape[0],), 0.7, device=imgs.device)

    fuse.forward = temped_fuse
    if uniform:
        scorer.forward = flat_trust

    perturb = PERTURBATIONS[fault]() if fault else None
    ious, ades, fdes = [], [], []
    try:
        for batch in loader:
            x, occ_t, traj = (batch[i].to(device) for i in range(3))
            motion, t_rel = batch[3].to(device), batch[4].to(device)
            if occ_t.ndim == 3:
                occ_t = occ_t.unsqueeze(1)
            vel = motion[:, 1:3]

            if perturb is not None:
                x = x.clone()
                for b in range(x.shape[0]):
                    x[b, cam, -1] = perturb(x[b, cam, -1].unsqueeze(0)).squeeze(0)

            occ_logits, traj_res, _, _ = model(x, velocity=vel)
            if occ_logits.ndim == 3:
                occ_logits = occ_logits.unsqueeze(1)
            if occ_logits.shape[-2:] != occ_t.shape[-2:]:
                raise SystemExit(
                    f"Grid mismatch: model {tuple(occ_logits.shape[-2:])} vs labels "
                    f"{tuple(occ_t.shape[-2:])}. Pass --label_root "
                    f"outputs/artifacts/nuscenes_labels_128.")

            pred = (torch.sigmoid(occ_logits) > THR).float()
            inter = (pred * occ_t).sum((1, 2, 3))
            union = (pred + occ_t).clamp(0, 1).sum((1, 2, 3))
            ious.extend((inter / (union + 1e-6)).cpu().tolist())

            pred_t = t_rel.unsqueeze(-1) * vel.unsqueeze(1) + traj_res
            d = torch.linalg.norm(pred_t - traj, dim=-1)
            ades.extend(d.mean(1).cpu().tolist())
            fdes.extend(d[:, -1].cpu().tolist())
    finally:
        fuse.forward = orig_fuse_forward
        scorer.forward = orig_scorer_forward

    return (float(np.mean(ious)), float(np.mean(ades)), float(np.mean(fdes)),
            float(np.mean(weights_on_faulted)) if weights_on_faulted else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="outputs/artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", default="outputs/artifacts/nuscenes_labels_128")
    ap.add_argument("--bev", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--image_hw", default="90,160")
    ap.add_argument("--cam", type=int, default=0, help="Camera to degrade (0 = CAM_FRONT).")
    ap.add_argument("--temps", default="1.0,0.5,0.25,0.1,0.05,0.02",
                    help="Softmax temperatures. 1.0 is the current behaviour.")
    ap.add_argument("--faults", default="all")
    ap.add_argument("--seeds", default="0",
                    help="Comma-separated seeds. The perturbations are stochastic\n"
                         "(GaussianBlur samples sigma per call), so a single run's\n"
                         "dIoU of +/-0.003 may be inside the noise floor. Multiple\n"
                         "seeds give a mean and a standard deviation, which is what\n"
                         "a trend needs before it can be quoted.")
    ap.add_argument("--out", default="outputs/artifacts/trust_temperature_sweep.json")
    args = ap.parse_args()

    H, W = [int(v) for v in args.image_hw.split(",")]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Loading model")
    model = load_model(args.ckpt, args.bev, device)

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    idx = [i for i, r in enumerate(rows) if r["scene"] in VAL_SCENES]
    ds = NuScenesMiniMultiView(args.manifest, image_hw=(H, W), frames=1,
                               label_root=args.label_root, return_motion=True,
                               return_trel=True, return_calib=True, augment=False)
    loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                        num_workers=0)
    print(f"  val frames: {len(idx)}  degraded camera: {args.cam}  device: {device}")

    temps = [float(t) for t in args.temps.split(",")]
    faults = list(PERTURBATIONS) if args.faults == "all" else args.faults.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]

    results = {}
    print(f"\n{'T':>7}{'fault':>11}{'w_fault':>9}{'dIoU mean':>11}{'dIoU sd':>9}"
          f"{'dIoU %':>9}{'seeds':>7}   verdict")
    print("-" * 78)
    for T in temps:
        for fault in faults:
            d_ious, d_ades, pcts, wfs = [], [], [], []
            for seed in seeds:
                torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
                t_iou, t_ade, _, wf = evaluate(model, loader, device, temperature=T,
                                               uniform=False, fault=fault, cam=args.cam)
                torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
                u_iou, u_ade, _, _ = evaluate(model, loader, device, temperature=T,
                                              uniform=True, fault=fault, cam=args.cam)
                d_ious.append(t_iou - u_iou)
                d_ades.append(u_ade - t_ade)
                pcts.append(100.0 * (t_iou - u_iou) / u_iou if u_iou > 0 else float("nan"))
                wfs.append(wf)

            mean_d, sd_d = float(np.mean(d_ious)), float(np.std(d_ious, ddof=1)) if len(seeds) > 1 else 0.0
            # A trend is only real if the effect is larger than the seed-to-seed
            # spread. |mean| > 2*sd is a crude but honest significance screen.
            verdict = ("noise" if len(seeds) > 1 and abs(mean_d) <= 2 * sd_d
                       else ("helps" if mean_d > 0 else "HURTS"))
            results[f"T{T}_{fault}"] = {
                "temperature": T, "fault": fault,
                "weight_on_faulted_cam": float(np.mean(wfs)),
                "delta_iou_mean": mean_d, "delta_iou_sd": sd_d,
                "delta_iou_per_seed": d_ious,
                "delta_iou_pct_mean": float(np.mean(pcts)),
                "delta_ade_mean": float(np.mean(d_ades)),
                "seeds": seeds, "verdict_vs_noise": verdict,
            }
            print(f"{T:>7.3f}{fault:>11}{np.mean(wfs):>9.4f}{mean_d:>+11.4f}"
                  f"{sd_d:>9.4f}{np.mean(pcts):>+8.1f}%{len(seeds):>7}   {verdict}")

    # Mean delta per temperature: the honest summary, since a single lucky fault
    # at one temperature is not evidence.
    summary = {}
    for T in temps:
        ds_ = [v["delta_iou_mean"] for v in results.values() if v["temperature"] == T]
        ws = [v["weight_on_faulted_cam"] for v in results.values() if v["temperature"] == T]
        summary[str(T)] = {"mean_delta_iou": float(np.mean(ds_)),
                           "min_delta_iou": float(np.min(ds_)),
                           "n_faults_improved": int(sum(d > 0 for d in ds_)),
                           "n_faults": len(ds_),
                           "mean_weight_on_faulted_cam": float(np.mean(ws))}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"checkpoint": args.ckpt, "camera": args.cam, "temperatures": temps,
         "seeds": seeds,
         "per_temperature_summary": summary, "per_run": results}, indent=2))

    print(f"\n{'T':>7}{'w_faulted':>11}{'mean dIoU':>12}{'min dIoU':>11}{'faults improved':>18}")
    print("-" * 60)
    for T in temps:
        s = summary[str(T)]
        print(f"{T:>7.3f}{s['mean_weight_on_faulted_cam']:>11.4f}"
              f"{s['mean_delta_iou']:>+12.4f}{s['min_delta_iou']:>+11.4f}"
              f"{s['n_faults_improved']:>12}/{s['n_faults']:<5}")

    bestT = max(summary.items(), key=lambda kv: kv[1]["mean_delta_iou"])
    print(f"\nBest temperature by mean dIoU: T={bestT[0]} "
          f"(mean {bestT[1]['mean_delta_iou']:+.4f}, "
          f"{bestT[1]['n_faults_improved']}/{bestT[1]['n_faults']} faults improved)")
    if bestT[1]["mean_delta_iou"] <= 0.001:
        print("\nCONCLUSION: trust-weighted fusion does not measurably improve IoU at ANY\n"
              "temperature, even when the degraded camera's weight approaches zero.\n"
              "The bottleneck is not the softmax. Detection works; mitigation via this\n"
              "fusion does not. The '+26.6% IoU under sensor faults' claim should be\n"
              "removed from the README.")
    else:
        print(f"\nCONCLUSION: trust fusion gives a real IoU gain at T={bestT[0]}. Report it as\n"
              f"a temperature-tuned result and state the temperature, since T=1.0 (the\n"
              f"shipped default) does not produce it.")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
