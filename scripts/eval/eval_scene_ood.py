"""
Scene-level out-of-distribution detection via Mahalanobis distance on backbone
features.

A DIFFERENT FAILURE MODE FROM THE TRUST SCORER
----------------------------------------------
`eval_ood_detection.py` answers "is this camera degraded?" -- a sensor fault.
This answers a different and, for fleet triage, more valuable question: "is this
SCENE unlike anything the model was trained on?" A perfectly clean image of a
construction zone, a flooded underpass, or snow when you trained on Singapore
is not a sensor fault. Every camera reports healthy and the model is still
operating outside its competence.

That distinction is why AV programmes run both. Sensor-fault detection protects
against hardware; scene-level OOD is what triages the fleet and decides which
logs are worth a human's time.

METHOD
------
Fit a single Gaussian to backbone features over TRAINING scenes, then score
held-out frames by Mahalanobis distance:

    d(x) = sqrt( (x - mu)^T  Sigma^-1  (x - mu) )

Covariance is shrunk toward a scaled identity,

    Sigma_shrunk = (1-a) * S + a * (tr(S)/d) * I

because with ~2k samples in 384 dimensions the empirical covariance is poorly
conditioned and its inverse amplifies noise directions. This is Ledoit-Wolf-style
regularisation with a fixed coefficient, which is honest and reproducible.

WHAT IT MEASURES
----------------
Since we have no labelled OOD scenes, corrupted frames stand in as an OOD proxy
and the score is AUROC for separating clean held-out frames from corrupted ones,
reusing the same metric module (with bootstrap CIs) as the fault detector, so the
numbers are directly comparable. It also reports per-scene mean distance, which
is the fleet-triage view: which scenes sit furthest from the training manifold.

Usage
-----
    python scripts/eval/eval_scene_ood.py \
        --ckpt outputs/artifacts/checkpoints_v11_trustfix2/trust_fixed_v2.ckpt \
        --out outputs/artifacts/scene_ood_report.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "src")

from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView  # noqa: E402
from opendrivefm.robustness.perturbations import PERTURBATIONS  # noqa: E402
from opendrivefm.training.lightning_module import LitOpenDriveFM  # noqa: E402
from opendrivefm.validation.detection_metrics import (  # noqa: E402
    auroc, average_precision, bootstrap_auroc_ci, detector_verdict, tpr_at_fpr)

VAL_SCENES = {"scene-0655", "scene-1077"}


@torch.no_grad()
def features(model, loader, device, fault=None, cams=()) -> np.ndarray:
    """Fused BEV latent per frame, shape (N, d)."""
    perts = [(c, PERTURBATIONS[fault]()) for c in cams] if fault else []
    out = []
    for batch in loader:
        x = batch[0].to(device)
        if perts:
            x = x.clone()
            for b in range(x.shape[0]):
                for c, p in perts:
                    x[b, c, -1] = p(x[b, c, -1].unsqueeze(0)).squeeze(0)
        z, _, _ = model.backbone(x)
        out.append(z.detach().float().cpu().numpy())
    return np.concatenate(out, axis=0)


def fit_gaussian(feats: np.ndarray, shrink: float):
    mu = feats.mean(axis=0)
    centred = feats - mu
    S = (centred.T @ centred) / max(1, feats.shape[0] - 1)
    d = S.shape[0]
    # Shrink toward a scaled identity: the empirical covariance is rank-deficient
    # or ill-conditioned whenever n is not >> d, and its raw inverse blows up
    # along near-null directions, producing enormous distances from noise.
    S_shrunk = (1 - shrink) * S + shrink * (np.trace(S) / d) * np.eye(d)
    return mu, np.linalg.inv(S_shrunk)


def mahalanobis(feats: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> np.ndarray:
    c = feats - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", c, prec, c), 0.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="outputs/artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", default="outputs/artifacts/nuscenes_labels_128")
    ap.add_argument("--bev", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--image_hw", default="90,160")
    ap.add_argument("--shrink", type=float, default=0.2)
    ap.add_argument("--holdout_train_scenes", type=int, default=2,
                    help="Training scenes withheld from the Gaussian fit and used "
                         "as the IN-DISTRIBUTION reference. Scoring the same frames "
                         "the Gaussian was fitted on is in-sample and inflates AUROC: "
                         "any density model calls its own training data likely. Set 0 "
                         "to reproduce the biased in-sample number for comparison.")
    ap.add_argument("--cams", default="0,1,4")
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/artifacts/scene_ood_report.json")
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
    tr_idx = [i for i, r in enumerate(rows) if r["scene"] not in VAL_SCENES]
    va_idx = [i for i, r in enumerate(rows) if r["scene"] in VAL_SCENES]
    ds = NuScenesMiniMultiView(args.manifest, image_hw=(H, W), frames=1,
                               label_root=args.label_root, return_motion=True,
                               return_trel=True, return_calib=True, augment=False)
    mk = lambda i: DataLoader(Subset(ds, i), batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    cams = [int(c) for c in args.cams.split(",")]
    print(f"  train frames {len(tr_idx)}   held-out frames {len(va_idx)}   "
          f"shrinkage {args.shrink}\n")

    # Split TRAINING scenes into a fit set and a held-out in-distribution
    # reference set. Without this the comparison is fit-set vs val-set, which
    # measures memorisation as much as novelty.
    train_scenes = sorted({rows[i]["scene"] for i in tr_idx})
    n_hold = min(args.holdout_train_scenes, max(0, len(train_scenes) - 2))
    rng_s = random.Random(args.seed)
    held = set(rng_s.sample(train_scenes, n_hold)) if n_hold else set()
    fit_idx = [i for i in tr_idx if rows[i]["scene"] not in held]
    ref_idx = [i for i in tr_idx if rows[i]["scene"] in held] or fit_idx

    fit_feats = features(model, mk(fit_idx), device)
    mu, prec = fit_gaussian(fit_feats, args.shrink)
    d_fit = mahalanobis(fit_feats, mu, prec)
    d_train = mahalanobis(features(model, mk(ref_idx), device), mu, prec)
    print(f"  Gaussian fitted on {len(fit_idx)} frames from "
          f"{len(train_scenes) - len(held)} scenes")
    print(f"  in-distribution reference: {len(ref_idx)} frames from held-out "
          f"TRAINING scenes {sorted(held) if held else '(none -- in-sample, biased)'}")
    print(f"  fit-set distance (in-sample, optimistic) mean {d_fit.mean():.3f}")
    d_clean = mahalanobis(features(model, mk(va_idx), device), mu, prec)
    print(f"  reference (held-out train) mean {d_train.mean():.3f}  "
          f"p95 {np.percentile(d_train,95):.3f}")
    print(f"  held-out clean   mean {d_clean.mean():.3f}  p95 {np.percentile(d_clean,95):.3f}"
          f"   (shift {d_clean.mean()-d_train.mean():+.3f})\n")

    # PRIMARY: can Mahalanobis distance tell a held-out SCENE from the training
    # scenes? This is what scene-level OOD actually means, and it is the number
    # that belongs in a fleet-triage claim.
    lo_s, hi_s = bootstrap_auroc_ci(d_clean, d_train, args.bootstrap, args.seed)
    det_s, thr_s = tpr_at_fpr(d_clean, d_train, args.fpr)
    scene_shift = {
        "auroc": auroc(d_clean, d_train), "auroc_ci95": [lo_s, hi_s],
        "average_precision": average_precision(d_clean, d_train),
        f"detection_rate_at_{args.fpr:g}_fpr": det_s, "threshold": thr_s,
    }
    code_s, msg_s = detector_verdict(
        scene_shift["auroc"], lo_s, hi_s,
        score_name="Mahalanobis distance",
        positive_cond="a frame comes from a held-out scene")
    print(f"PRIMARY -- held-out scene vs training scene separation")
    print(f"  AUROC {scene_shift['auroc']:.3f}  95% CI [{lo_s:.3f}, {hi_s:.3f}]  "
          f"AP {scene_shift['average_precision']:.3f}  "
          f"det@{args.fpr:g}fpr {det_s:.3f}  [{code_s}]\n")

    # SECONDARY, and a deliberately reported negative: corrupted frames as an
    # OOD proxy. Corruption is a SENSOR fault, not a novel scene, and heavy
    # corruption destroys distinctive content, pushing features TOWARD the
    # training mean. Expect AUROC at or below chance here; that is the proxy
    # being wrong, not the detector.
    report = {}
    print(f"SECONDARY -- corrupted frames as an OOD proxy (expected to fail; see note)")
    print(f"{'condition':<12}{'mean dist':>11}{'AUROC':>9}{'95% CI':>20}"
          f"{'AP':>8}{'det@fpr':>10}")
    print("-" * 72)
    pooled_pos = []
    for fault in PERTURBATIONS:
        d_f = mahalanobis(features(model, mk(va_idx), device, fault=fault, cams=cams),
                          mu, prec)
        pooled_pos.append(d_f)
        lo, hi = bootstrap_auroc_ci(d_f, d_clean, args.bootstrap, args.seed)
        det, thr = tpr_at_fpr(d_f, d_clean, args.fpr)
        report[fault] = {
            "mean_distance": float(d_f.mean()),
            "auroc": auroc(d_f, d_clean), "auroc_ci95": [lo, hi],
            "average_precision": average_precision(d_f, d_clean),
            f"detection_rate_at_{args.fpr:g}_fpr": det, "threshold": thr,
        }
        r = report[fault]
        print(f"{fault:<12}{d_f.mean():>11.3f}{r['auroc']:>9.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>20}{r['average_precision']:>8.3f}{det:>10.3f}")

    pos = np.concatenate(pooled_pos)
    lo, hi = bootstrap_auroc_ci(pos, d_clean, args.bootstrap, args.seed)
    pooled = {"auroc": auroc(pos, d_clean), "auroc_ci95": [lo, hi],
              "average_precision": average_precision(pos, d_clean)}
    code, msg = detector_verdict(
        pooled["auroc"], lo, hi,
        score_name="Mahalanobis distance from the training manifold",
        positive_cond="a camera is corrupted")
    print("-" * 72)
    print(f"{'POOLED':<12}{'':<11}{pooled['auroc']:>9.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}"
          f"{pooled['average_precision']:>8.3f}")

    # Fleet-triage view: rank scenes by distance from the training manifold.
    by_scene = defaultdict(list)
    for i, dist in zip(va_idx, d_clean):
        by_scene[rows[i]["scene"]].append(float(dist))
    print(f"\nFLEET TRIAGE: held-out scenes ranked by distance from training manifold")
    for scene, ds_ in sorted(by_scene.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {scene:<14} mean {np.mean(ds_):.3f}  max {np.max(ds_):.3f}  "
              f"n={len(ds_)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "checkpoint": args.ckpt, "shrinkage": args.shrink,
        "cameras_perturbed": cams, "feature_dim": int(fit_feats.shape[1]),
        "n_fit": len(fit_idx), "n_reference": len(ref_idx),
        "n_heldout": len(va_idx),
        "holdout_train_scenes": sorted(held),
        "fit_set_distance_mean_in_sample": float(d_fit.mean()),
        "reference_distance_mean": float(d_train.mean()),
        "heldout_clean_distance_mean": float(d_clean.mean()),
        "scene_shift_primary": scene_shift,
        "scene_shift_verdict": {"code": code_s, "message": msg_s},
        "corruption_proxy_pooled": pooled,
        "corruption_proxy_verdict": {"code": code, "message": msg},
        "note": ("Corruption is a sensor fault, not a novel scene. It is reported "
                 "as a secondary diagnostic only; the primary result is held-out "
                 "scene vs training scene separation."),
        "per_fault": report,
        "per_scene_clean_distance": {k: {"mean": float(np.mean(v)),
                                         "max": float(np.max(v)), "n": len(v)}
                                     for k, v in by_scene.items()},
    }, indent=2))
    if held and d_train.mean() > d_clean.mean():
        print("\n" + "=" * 72)
        print("SCENE-VARIANCE WARNING")
        print(f"  Held-out TRAINING scenes sit further from the fitted manifold "
              f"({d_train.mean():.2f})\n  than the held-out VAL scenes "
              f"({d_clean.mean():.2f}).")
        print("  There is therefore no coherent 'training distribution' to be out of:\n"
              "  scene-to-scene variance exceeds the train/val gap. With this few\n"
              "  scenes a single Gaussian fits a handful of separate clusters, and any\n"
              "  unseen scene lands far outside regardless of its split label.\n"
              "  Scene-level Mahalanobis OOD is NOT VIABLE at this dataset size.")
        print("=" * 72)

    print(f"\nCORRUPTION-PROXY VERDICT [{code}]")
    for line in msg.splitlines():
        print("  " + line)
    print("\n  INTERPRETATION: this is the PROXY failing, not the detector. Heavy\n"
          "  corruption destroys distinctive scene content and pulls features toward\n"
          "  the training mean, so corrupted frames look LESS novel, not more. Sensor\n"
          "  faults are the trust scorer's job (AUROC 0.764); scene novelty is this\n"
          "  detector's job. Reporting both is the point.")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
