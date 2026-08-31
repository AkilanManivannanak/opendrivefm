"""
OOD / camera-fault detection evaluation for the CameraTrustScorer.

WHY THIS EXISTS
---------------
`robustness/perturbations.py` documents "Detection rate: 100% across all 5 fault
types", and the README reports the same. The only measurement committed to the
repo, `outputs/artifacts/robustness_report.json`, shows mean trust of 0.7351
clean versus 0.7350-0.7352 under every fault, i.e. no separation at all. That
report was produced from `artifacts/checkpoints_trust/last.ckpt`, which no
longer exists in the tree, so neither number can currently be reproduced.

A mean-shift comparison cannot settle this anyway: it says nothing about whether
clean and faulted cameras are *separable*. This script answers the detection
question directly, as a binary classification problem:

    positives = trust scores for a camera WITH a fault injected
    negatives = trust scores for the SAME camera on the SAME frame, clean

and reports AUROC, average precision, and detection rate at a fixed false-alarm
rate, with bootstrap confidence intervals.

TWO DETECTORS ARE SCORED
------------------------
absolute : score = -trust[cam]
    Can a fixed global threshold flag a degraded camera? This is what a
    deployed monitor would need.

relative : score = mean(trust[other cams]) - trust[cam]
    Is the faulted camera ranked below its peers on the same frame? A model can
    be a good *ranker* while being badly calibrated in absolute terms. If
    relative AUROC is high and absolute AUROC is not, the scorer works and the
    calibration is the defect, which is a different and much cheaper fix.

Usage
-----
    python scripts/eval/eval_ood_detection.py \
        --ckpt outputs/artifacts/checkpoints_v9/best_val_ade.ckpt \
        --manifest outputs/artifacts/nuscenes_mini_manifest.jsonl \
        --label_root outputs/artifacts/nuscenes_labels \
        --cams all \
        --out outputs/artifacts/ood_detection_report.json
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
from opendrivefm.validation.ckpt_compat import (  # noqa: E402
    remap_legacy_keys, report_remap)
from opendrivefm.validation.detection_metrics import (  # noqa: E402
    auroc, average_precision, bootstrap_auroc_ci, detector_verdict, tpr_at_fpr)

CAM_NAMES = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
             "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


# ── Model / data loading ─────────────────────────────────────────────────────

def load_model(ckpt_path: str, bev: int, device: str, trust_grid: int = 1,
               allow_untrained_trust: bool = False,
               allow_uncalibrated: bool = True):
    """Load a checkpoint and REPORT how much of it actually matched.

    `load_state_dict(..., strict=False)` silently tolerates a checkpoint that
    does not fit the current model, which would make every number below
    meaningless. We surface the match rate instead of hiding it.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}
    try:
        lit = LitOpenDriveFM(bev=bev, trust_grid=trust_grid)
    except AssertionError as e:
        raise SystemExit(
            f"Model construction failed: {e}\n"
            f"The current src/opendrivefm/models/model.py only supports bev_h=128. "
            f"Re-run with --bev 128, or point --ckpt at a checkpoint trained against "
            f"the architecture in this working tree."
        ) from e
    # Drop ONLY the tensors whose shapes genuinely disagree with the model we
    # just built.
    #
    # This used to be `if trust_grid > 1: drop every backbone.trust_scorer.*`,
    # which was wrong in a way that silently destroyed results. That rule fires
    # on the FLAG, not on the checkpoint. It is correct when promoting a grid=1
    # checkpoint to grid>1 (the head shapes really do change), and catastrophic
    # when the checkpoint is ALREADY grid>1: it threw away the very weights
    # finetune_trust_head.py had just trained and evaluated a random-init
    # scorer, while cheerfully printing "this is intended".
    #
    # Comparing shapes cannot get this wrong: a tensor that fits is kept
    # whatever the flag says, and a tensor that does not fit would have raised
    # a RuntimeError below anyway.
    ref = lit.model.state_dict()
    # Legacy renames first: a tensor that is merely misnamed must be recovered
    # before the shape filter below, or it looks like an absent weight.
    state, applied, rejected = remap_legacy_keys(state, ref)
    report_remap(applied, rejected)
    dropped = {k: (tuple(v.shape), tuple(ref[k].shape))
               for k, v in state.items()
               if k in ref and tuple(v.shape) != tuple(ref[k].shape)}
    for k in dropped:
        state.pop(k)
    if dropped:
        print(f"  dropped {len(dropped)} tensor(s) whose shape does not match the "
              f"current model; they will be randomly initialised:")
        for k, (had, want) in list(dropped.items())[:4]:
            print(f"    {k}: checkpoint {had} vs model {want}")
        if len(dropped) > 4:
            print(f"    ... and {len(dropped) - 4} more")
    try:
        result = lit.model.load_state_dict(state, strict=False)
    except RuntimeError as e:
        # strict=False tolerates missing/extra keys but NOT shape mismatches:
        # those mean the checkpoint was trained against a different architecture
        # than the one in this working tree.
        n = str(e).count("size mismatch")
        raise SystemExit(
            f"INCOMPATIBLE CHECKPOINT: {ckpt_path}\n"
            f"  {n} tensor(s) have shapes that do not match the current model.\n"
            f"  This checkpoint was trained against a different architecture "
            f"revision. Any metric produced from it would be meaningless.\n"
            f"  First mismatch: {str(e).splitlines()[1].strip() if len(str(e).splitlines()) > 1 else e}"
        ) from e

    scorer = lit.model.backbone.trust_scorer
    if not bool(scorer.stat_calibrated):
        msg = ("this checkpoint has NO calibrated statistics reference, so the "
               "trust scorer falls back to centring on the batch mean. The same "
               "frame then scores differently in different batches: measured max "
               "|delta trust| is 1.2e-04 across batch sizes 1 to 8, against 1.2e-07 "
               "(float32 noise) once calibrated. AUROC is rank-based and is "
               "unaffected either way; an absolute THRESHOLD is not transferable "
               "across batch shapes, which includes the C++ runtime's 1 frame x 6 "
               "cameras.\n"
               "  Fix it once, with:\n"
               f"    python scripts/calibrate_trust_stats.py --ckpt {ckpt_path} \\\n"
               f"      --trust_grid {trust_grid} --out <calibrated>.ckpt")
        if not allow_uncalibrated:
            raise SystemExit("REFUSING TO EVALUATE: " + msg)
        print("  WARNING (--allow_uncalibrated_trust): " + msg)

    n_expected = len(lit.model.state_dict())
    n_missing = len(result.missing_keys)
    matched = n_expected - n_missing
    match_rate = matched / max(1, n_expected)

    print(f"  checkpoint: {ckpt_path}")
    print(f"  weights matched: {matched}/{n_expected} ({match_rate:.1%}), "
          f"unexpected keys: {len(result.unexpected_keys)}")
    # `stat_running_mean` / `stat_calibrated` are calibration buffers, not
    # trained weights. Their absence is expected for any checkpoint predating
    # calibration and is handled by the explicit check below, not by this guard.
    CAL_BUFFERS = ("stat_running_mean", "stat_calibrated")
    trust_missing = [k for k in result.missing_keys
                     if "trust" in k and not k.endswith(CAL_BUFFERS)]
    if trust_missing:
        # This is a hard failure, not a warning.
        #
        # It was a warning, and a warning is exactly what let a run with 18
        # randomly-initialised trunk tensors produce AUROC 0.764 and have that
        # number copied into the README. A message that says "results below
        # measure nothing" and then prints the results anyway is a message
        # nobody acts on. If the scorer did not load, there is no experiment.
        msg = (f"REFUSING TO EVALUATE: {len(trust_missing)} trust_scorer weights "
               f"are not in this checkpoint, so the scorer would run partly on "
               f"random initialisation and every number below would be noise.\n"
               f"  first missing: {trust_missing[:4]}\n"
               f"  Diagnose with:\n"
               f"    python scripts/eval/inspect_ckpt_trust_keys.py {ckpt_path} "
               f"--trust_grid {trust_grid}\n"
               f"  A checkpoint trained against an older module layout may just "
               f"need a rename added to src/opendrivefm/validation/ckpt_compat.py.\n"
               f"  To measure an intentionally-untrained scorer anyway, pass "
               f"--allow_untrained_trust.")
        if not allow_untrained_trust:
            raise SystemExit(msg)
        print("  " + msg.replace("REFUSING TO EVALUATE", "WARNING (--allow_untrained_trust)"))
    if match_rate < 0.9:
        print("  WARNING: match rate below 90%. Verify --bev and the checkpoint version.")

    return lit.model.eval().to(device), {
        "path": str(ckpt_path),
        "weights_matched": int(matched),
        "weights_expected": int(n_expected),
        "match_rate": float(match_rate),
        "missing_trust_keys": len(trust_missing),
        "trust_stats_calibrated": bool(scorer.stat_calibrated),
        "unexpected_keys": len(result.unexpected_keys),
    }


def val_indices(rows, seed: int, val_frac: float, val_scenes: str | None):
    if val_scenes:
        wanted = {s.strip() for s in val_scenes.split(",") if s.strip()}
        return [i for i, r in enumerate(rows) if r["scene"] in wanted], sorted(wanted)
    scenes = sorted({r["scene"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(scenes)
    n_val = max(1, int(round(len(scenes) * val_frac)))
    chosen = set(scenes[:n_val])
    return [i for i, r in enumerate(rows) if r["scene"] in chosen], sorted(chosen)


@torch.no_grad()
def trust_scores(model, loader, device, perturb=None, cam_idx: int = 0) -> np.ndarray:
    """Return per-sample per-camera trust, shape (N, V).

    If `perturb` is given it is applied to camera `cam_idx`, frame 0 only, which
    matches how eval_trust_ablation.py and eval_robustness_trust.py inject faults.
    """
    out = []
    for batch in loader:
        x = batch[0].to(device)
        motion = batch[3].to(device)
        vel = motion[:, 1:3]

        if perturb is not None:
            x = x.clone()
            for b in range(x.shape[0]):
                # Frame -1: the backbone scores trust on the newest frame
                # (model.py: `imgs_flat = rearrange(x[:, :, -1], ...)`).
                # With frames=1 this is the same as index 0, but using -1 keeps
                # the eval correct for multi-frame configs.
                img = x[b, cam_idx, -1]                     # (C, H, W)
                x[b, cam_idx, -1] = perturb(img.unsqueeze(0)).squeeze(0)

        # NOTE: OpenDriveFM.forward() accepts (x, velocity, ego_deltas,
        # lidar_depth_maps, **_). K and T_ego_cam are swallowed by **_ and are
        # not used by the current forward path, so we do not pass them.
        _, _, trust, _ = model(x, velocity=vel)
        out.append(trust.detach().float().cpu().numpy())
    return np.concatenate(out, axis=0)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="outputs/artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", default="outputs/artifacts/nuscenes_labels_128")
    ap.add_argument("--trust_grid", type=int, default=1,
                    help="CameraTrustScorer spatial pooling grid. 1 = original "
                         "global-average architecture. >1 adds patchwise mean+min "
                         "pooling so localised faults (occlusion) are visible.")
    ap.add_argument("--bev", type=int, default=128,
                    help="BEV grid size. The current model.py (v11) asserts 128; "
                         "older eval scripts in this repo still pass 64.")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--image_hw", default="90,160")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--val_scenes", default=None,
                    help="Comma-separated scene ids; overrides seed/val_frac split.")
    ap.add_argument("--cams", default="0",
                    help="'all', or comma-separated camera indices to fault.")
    ap.add_argument("--faults", default="all",
                    help="'all', or comma-separated subset of blur,glare,occlusion,rain,noise")
    ap.add_argument("--fpr", type=float, default=0.05,
                    help="False-alarm budget for the detection-rate metric.")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default="outputs/artifacts/ood_detection_report.json")
    ap.add_argument("--require_calibrated_trust", action="store_true",
                    help="Fail instead of warning when the checkpoint has no "
                         "calibrated statistics reference. AUROC is rank-based "
                         "and unaffected by the fallback, so this defaults to a "
                         "warning; require it when reporting ABSOLUTE trust "
                         "scores, thresholds, or effect sizes.")
    ap.add_argument("--allow_untrained_trust", action="store_true",
                    help="Proceed even when trust_scorer weights are missing from "
                         "the checkpoint. Only for deliberately measuring an "
                         "untrained scorer as a control; the numbers are otherwise "
                         "meaningless.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Load the checkpoint, report weight match rate, and exit. "
                         "Use this to find which checkpoint fits the current model.")
    args = ap.parse_args()

    H, W = [int(v) for v in args.image_hw.split(",")]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("Loading model")
    model, ckpt_info = load_model(args.ckpt, args.bev, device, args.trust_grid,
                                  args.allow_untrained_trust,
                                  allow_uncalibrated=not args.require_calibrated_trust)
    print(f"  device: {device}")
    if args.dry_run:
        print("\n--dry_run: checkpoint loaded, exiting before evaluation.")
        return

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    idx_val, scenes = val_indices(rows, args.seed, args.val_frac, args.val_scenes)
    ds = NuScenesMiniMultiView(
        args.manifest, image_hw=(H, W), frames=1, label_root=args.label_root,
        return_motion=True, return_trel=True, return_calib=True, augment=False)
    loader = DataLoader(Subset(ds, idx_val), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    print(f"  val scenes: {scenes}  frames: {len(idx_val)}")

    fault_names = (list(PERTURBATIONS) if args.faults == "all"
                   else [f.strip() for f in args.faults.split(",") if f.strip()])
    cams = (list(range(len(CAM_NAMES))) if args.cams == "all"
            else [int(c) for c in args.cams.split(",")])

    print("\nScoring clean frames")
    clean = trust_scores(model, loader, device)          # (N, V)
    n, v = clean.shape
    print(f"  trust matrix: {n} frames x {v} cameras")

    def rel(t: np.ndarray, cam: int) -> np.ndarray:
        """mean(other cameras) - this camera, per frame."""
        others = [c for c in range(t.shape[1]) if c != cam]
        return t[:, others].mean(axis=1) - t[:, cam]

    results, pooled = {}, {"absolute": {"pos": [], "neg": []},
                           "relative": {"pos": [], "neg": []}}

    for fault in fault_names:
        perturb = PERTURBATIONS[fault]()
        for cam in cams:
            faulted = trust_scores(model, loader, device, perturb=perturb, cam_idx=cam)

            scores = {
                "absolute": (-faulted[:, cam], -clean[:, cam]),
                "relative": (rel(faulted, cam), rel(clean, cam)),
            }
            entry = {
                "fault": fault,
                "camera_index": cam,
                "camera": CAM_NAMES[cam] if cam < len(CAM_NAMES) else f"cam{cam}",
                "mean_trust_clean": float(clean[:, cam].mean()),
                "mean_trust_faulted": float(faulted[:, cam].mean()),
                "mean_trust_drop": float(clean[:, cam].mean() - faulted[:, cam].mean()),
                "n_frames": int(n),
                "detectors": {},
            }
            for name, (pos, neg) in scores.items():
                pooled[name]["pos"].append(pos)
                pooled[name]["neg"].append(neg)
                lo, hi = bootstrap_auroc_ci(pos, neg, args.bootstrap, args.seed)
                det, thr = tpr_at_fpr(pos, neg, args.fpr)
                entry["detectors"][name] = {
                    "auroc": auroc(pos, neg),
                    "auroc_ci95": [lo, hi],
                    "average_precision": average_precision(pos, neg),
                    f"detection_rate_at_{args.fpr:g}_fpr": det,
                    "threshold": thr,
                }
            results[f"{fault}@cam{cam}"] = entry
            a = entry["detectors"]["absolute"]
            r = entry["detectors"]["relative"]
            print(f"  {fault:<10} cam{cam}  drop={entry['mean_trust_drop']:+.4f}  "
                  f"AUROC abs={a['auroc']:.3f} rel={r['auroc']:.3f}")

    overall = {}
    for name in pooled:
        pos = np.concatenate(pooled[name]["pos"])
        neg = np.concatenate(pooled[name]["neg"])
        lo, hi = bootstrap_auroc_ci(pos, neg, args.bootstrap, args.seed)
        det, thr = tpr_at_fpr(pos, neg, args.fpr)
        overall[name] = {
            "auroc": auroc(pos, neg),
            "auroc_ci95": [lo, hi],
            "average_precision": average_precision(pos, neg),
            f"detection_rate_at_{args.fpr:g}_fpr": det,
            "threshold": thr,
            "n_positive": int(pos.size),
            "n_negative": int(neg.size),
        }

    report = {
        "checkpoint": ckpt_info,
        "config": {
            "manifest": args.manifest, "label_root": args.label_root,
            "val_scenes": scenes, "n_frames": int(n), "n_cameras": int(v),
            "faults": fault_names, "cameras_faulted": cams,
            "fpr_budget": args.fpr, "bootstrap": args.bootstrap, "seed": args.seed,
            "image_hw": [H, W], "bev": args.bev, "device": device,
        },
        "overall": overall,
        "per_fault": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    # ── Verdict ──────────────────────────────────────────────────────────────
    label = f"det@{args.fpr:g}fpr"
    print("\n" + "=" * 78)
    print(f"{'detector':<12}{'AUROC':>8}{'95% CI':>20}{'AP':>8}{label:>14}")
    print("-" * 78)
    for name, m in overall.items():
        lo, hi = m["auroc_ci95"]
        print(f"{name:<12}{m['auroc']:>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}"
              f"{m['average_precision']:>8.3f}"
              f"{m[f'detection_rate_at_{args.fpr:g}_fpr']:>14.3f}")
    print("=" * 78)

    best_name, best = max(overall.items(), key=lambda kv: kv[1]["auroc"])
    lo, hi = best["auroc_ci95"]
    code, msg = detector_verdict(best["auroc"], lo, hi)
    report["verdict"] = {"detector": best_name, "code": code, "message": msg}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"VERDICT [{code}] (best detector: {best_name})")
    for line in msg.splitlines():
        print("         " + line)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
