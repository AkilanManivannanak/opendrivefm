"""
Split-conformal prediction intervals for the trajectory head.

WHY A PLANNER NEEDS THIS
------------------------
`ADE = 2.763 m` is an average. A planner cannot act on an average: it needs to
know how much room to leave. Split conformal prediction converts any point
predictor into an interval predictor with a *distribution-free, finite-sample*
coverage guarantee -- no Gaussian assumption, no assumption that the model is
well specified. If you calibrate at alpha = 0.1, at least 90% of future
waypoints fall inside the interval, provided calibration and test data are
exchangeable.

That last clause is the whole point of measuring it here rather than trusting
it. Under sensor degradation the exchangeability assumption breaks, and the
empirical coverage drops below nominal. Quantifying that gap is what makes this
a validation result rather than a textbook exercise.

WHAT IS COMPUTED
----------------
1. Calibrate on a held-out split: nonconformity score = per-waypoint L2 error.
2. q_hat = the ceil((n+1)(1-alpha))/n empirical quantile (the finite-sample
   correction; using the plain (1-alpha) quantile under-covers on small n, and
   n is 40-ish here, so the correction is not cosmetic).
3. Report empirical coverage on the test split, clean and under each fault.
4. Report per-horizon q_hat, because uncertainty grows with prediction horizon
   and a single scalar interval hides that.

INTERPRETATION
--------------
  coverage >= 1-alpha on clean data   -> calibration is sound
  coverage <  1-alpha under fault     -> the interval is optimistic exactly when
                                         the vehicle is degraded, which is the
                                         condition a planner most needs it for.
                                         Report the shortfall; do not hide it.

Usage
-----
    python scripts/eval/eval_conformal_trajectory.py \
        --ckpt outputs/artifacts/checkpoints_v11_trustfix2/trust_fixed_v2.ckpt \
        --alpha 0.1 \
        --out outputs/artifacts/conformal_trajectory.json
"""
from __future__ import annotations

import argparse
import json
import math
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


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected empirical quantile.

    The correction is ceil((n+1)(1-alpha))/n rather than simply (1-alpha).
    With n = 40 and alpha = 0.1 that is the 93rd percentile, not the 90th;
    skipping it silently under-covers, which is the failure mode that makes a
    'guaranteed' interval untrustworthy.
    """
    n = scores.size
    if n == 0:
        return float("nan")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float(np.max(scores))   # n too small for this alpha
    return float(np.sort(scores)[k - 1])


@torch.no_grad()
def waypoint_errors(model, loader, device, fault=None, cam=0) -> np.ndarray:
    """Per-sample per-waypoint L2 error, shape (N, horizon)."""
    perturb = PERTURBATIONS[fault]() if fault else None
    out = []
    for batch in loader:
        x, _, traj = (batch[i].to(device) for i in range(3))
        motion, t_rel = batch[3].to(device), batch[4].to(device)
        vel = motion[:, 1:3]
        if perturb is not None:
            x = x.clone()
            for b in range(x.shape[0]):
                x[b, cam, -1] = perturb(x[b, cam, -1].unsqueeze(0)).squeeze(0)
        _, traj_res, _, _ = model(x, velocity=vel)
        pred = t_rel.unsqueeze(-1) * vel.unsqueeze(1) + traj_res
        out.append(torch.linalg.norm(pred - traj, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="outputs/artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", default="outputs/artifacts/nuscenes_labels_128")
    ap.add_argument("--bev", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--image_hw", default="90,160")
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="Miss rate. 0.1 targets >=90%% coverage.")
    ap.add_argument("--cal_frac", type=float, default=0.5,
                    help="Fraction of the val split used for calibration.")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/artifacts/conformal_trajectory.json")
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
    val_idx = [i for i, r in enumerate(rows) if r["scene"] in {"scene-0655", "scene-1077"}]
    ds = NuScenesMiniMultiView(args.manifest, image_hw=(H, W), frames=1,
                               label_root=args.label_root, return_motion=True,
                               return_trel=True, return_calib=True, augment=False)

    # Split calibration and test by INDEX ORDER, not randomly, so consecutive
    # frames from the same scene do not straddle the split. Neighbouring frames
    # are near-duplicates; letting them leak across inflates coverage.
    n_cal = int(len(val_idx) * args.cal_frac)
    cal_idx, test_idx = val_idx[:n_cal], val_idx[n_cal:]
    mk = lambda idx: DataLoader(Subset(ds, idx), batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
    print(f"  calibration frames: {len(cal_idx)}   test frames: {len(test_idx)}")
    print(f"  alpha = {args.alpha}  -> nominal coverage {100*(1-args.alpha):.0f}%")

    cal_err = waypoint_errors(model, mk(cal_idx), device)          # (n_cal, H)
    horizon = cal_err.shape[1]

    # Per-horizon calibration: uncertainty grows with prediction distance, so a
    # single global radius is far too wide early and too tight late.
    q_per_h = np.array([conformal_quantile(cal_err[:, h], args.alpha)
                        for h in range(horizon)])
    q_global = conformal_quantile(cal_err.reshape(-1), args.alpha)

    print(f"\n  global conformal radius (alpha={args.alpha}): {q_global:.3f} m")
    print(f"  per-horizon radius: t1 {q_per_h[0]:.3f} m ... "
          f"t{horizon} {q_per_h[-1]:.3f} m  "
          f"(grows {q_per_h[-1]/max(q_per_h[0], 1e-9):.1f}x)")

    conditions = [("clean", None)] + [(f, f) for f in PERTURBATIONS]
    report, nominal = {}, 1.0 - args.alpha
    print(f"\n{'condition':<12}{'cov global':>12}{'cov per-h':>12}"
          f"{'mean err':>11}{'p95 err':>10}   status")
    print("-" * 68)
    for name, fault in conditions:
        err = waypoint_errors(model, mk(test_idx), device, fault=fault, cam=args.cam)
        cov_g = float((err <= q_global).mean())
        cov_h = float((err <= q_per_h[None, :]).mean())

        # Coverage is a binomial proportion over n_points Bernoulli trials, so a
        # point estimate below nominal is only meaningful if it survives its own
        # sampling error. With ~500 points the standard error is ~1.3pp; calling
        # a 0.4pp shortfall a violation would be the same mistake as quoting an
        # unreproducible p99.9.
        n_points = int(err.size)
        se = math.sqrt(max(cov_h * (1 - cov_h), 1e-12) / n_points)
        lo95 = cov_h - 1.96 * se
        status = ("OK" if lo95 >= nominal else
                  (f"UNDER by {100*(nominal-cov_h):.1f}pp (significant)" if cov_h + 1.96 * se < nominal
                   else f"-{100*(nominal-cov_h):.1f}pp (within +/-{100*1.96*se:.1f}pp noise)"))
        report[name] = {
            "coverage_global_radius": cov_g,
            "coverage_per_horizon": cov_h,
            "nominal_coverage": nominal,
            "mean_error_m": float(err.mean()),
            "p95_error_m": float(np.percentile(err, 95)),
            "coverage_shortfall_pp": float(max(0.0, nominal - cov_h) * 100),
            "n_points": n_points,
            "coverage_stderr_pp": float(se * 100),
            "significantly_under": bool(cov_h + 1.96 * se < nominal),
        }
        print(f"{name:<12}{cov_g:>11.1%}{cov_h:>12.1%}{err.mean():>11.3f}"
              f"{np.percentile(err, 95):>10.3f}   {status}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "checkpoint": args.ckpt, "alpha": args.alpha,
        "nominal_coverage": nominal, "horizon": int(horizon),
        "n_calibration": len(cal_idx), "n_test": len(test_idx),
        "conformal_radius_global_m": q_global,
        "conformal_radius_per_horizon_m": q_per_h.tolist(),
        "coverage": report,
    }, indent=2))

    worst = max(report.items(), key=lambda kv: kv[1]["coverage_shortfall_pp"])
    print(f"\nWorst condition: '{worst[0]}' at {worst[1]['coverage_per_horizon']:.1%} "
          f"coverage ({worst[1]['coverage_shortfall_pp']:.1f}pp below nominal)")
    if any(v["significantly_under"] for v in report.values()):
        print("Conformal guarantees assume calibration and test data are exchangeable.\n"
              "Sensor degradation breaks that assumption, so the interval is optimistic\n"
              "exactly when a planner most needs it to be honest. The fix is either\n"
              "fault-conditional calibration (a separate radius per detected fault, which\n"
              "the trust scorer can now supply at AUROC 0.76) or inflating the radius by\n"
              "the measured shortfall.")
    else:
        print("No condition is significantly below nominal coverage once binomial\n"
              "sampling error is accounted for. The intervals hold under every\n"
              "corruption tested -- which is itself worth understanding: it means the\n"
              "trajectory head barely responds to camera degradation, consistent with\n"
              "ADE moving <0.15 m under fault in the ablation. Coverage is robust\n"
              "because the predictions are largely driven by the constant-velocity\n"
              "prior rather than by the images.")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
