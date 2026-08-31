#!/usr/bin/env python3
"""Measure batch-composition contamination in the CameraTrustScorer's scores.

WHY THIS SCRIPT EXISTS
----------------------
`CameraTrustScorer._image_stats` ends with

    return torch.sigmoid(stats - stats.detach().mean(dim=0))

`dim=0` is the batch axis, and the scorer is called on ALL cameras of ALL frames
in a batch at once: `rearrange(x[:, :, -1], "b v c h w -> (b v) c h w")`. So a
camera's trust score is not a function of that camera's pixels alone. It is a
function of that camera's pixels RELATIVE TO the other B*V-1 images that
happened to share the forward pass.

That is a problem for `eval_ood_detection.py`, which scores clean frames in one
pass and faulted frames in a SECOND, SEPARATE pass. In the faulted pass, one
camera of every frame is damaged, which drags the batch mean, which shifts the
scores of the cameras that were never touched. The clean and faulted score
distributions are therefore measured against two different reference points, and
part of the resulting AUROC is that shift rather than the fault.

WHAT WAS ACTUALLY FOUND
-----------------------
Two things are measured here, and only the second one turned out to matter.

1. CONTAMINATION (the original hypothesis, REFUTED). Look at the cameras that
   were NOT faulted: their pixels are bit-identical between the two passes, so
   any movement is pure artifact. Measured ratio: 0.000. The leak is real in
   principle and negligible in practice, because a shared additive centring
   largely cancels when clean is subtracted from faulted.

2. FRAME PURITY (the property that matters). Score the SAME clean frames at
   several batch sizes and compare directly. Uncalibrated, one frame's trust
   moves by up to 1.2e-04 depending only on what it was batched with; after
   `scripts/calibrate_trust_stats.py`, 1.2e-07, which is float32 noise.

   This is a train/deploy problem, not a tidiness one. The C++ runtime scores
   one frame's 6 cameras per call while this eval used 12 images per call, so
   under batch-relative centring they are different functions and a trust
   THRESHOLD chosen offline does not transfer to the vehicle.

A NOTE ON MEASURING THIS AT ALL
-------------------------------
An earlier version of this script reported the faulted camera's response falling
36% "across the sweep" and that number was an artifact of THIS SCRIPT: the
perturbations draw their size and position from the global RNG, and the seed was
set once before the loop, so each batch size was scored against a different set
of random faults. Fault variance was being reported as a batch-size effect. The
sweep now re-seeds before every pass. If you add a metric here, hold the faults
fixed first and confirm the metric responds to the thing it names.

INTERPRETATION
--------------
  ratio < 0.05   negligible; the AUROCs stand as measured.
  0.05 - 0.20    material; report it as a known bias.
  > 0.20         the absolute detector's AUROC is not trustworthy. Prefer the
                 relative detector, which compares cameras inside a single
                 forward pass so the shared offset largely cancels.

USAGE
    python scripts/eval/check_batch_contamination.py \
        --ckpt artifacts/checkpoints_v11/last.ckpt \
        --fault occlusion --cam 0 --batch_sizes 1,2,4,8 \
        --out outputs/artifacts/batch_contamination.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ood_detection import load_model, val_indices  # noqa: E402

V_DEFAULT = 6


@torch.no_grad()
def trust_matrix(model, loader, device, perturb=None, cam_idx: int = 0) -> np.ndarray:
    """Per-frame per-camera trust, shape (N, V). Mirrors eval_ood_detection."""
    out = []
    for batch in loader:
        x = batch[0].to(device)
        motion = batch[3].to(device)
        vel = motion[:, :2] if motion.ndim == 2 and motion.shape[1] >= 2 else None
        if perturb is not None:
            for b in range(x.shape[0]):
                frame = x[b, cam_idx, -1]
                x[b, cam_idx, -1] = perturb(frame.unsqueeze(0)).squeeze(0)
        _, _, trust, _ = model(x, velocity=vel)
        out.append(trust.detach().float().cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest",
                    default="outputs/artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", default="outputs/artifacts/nuscenes_labels_128")
    ap.add_argument("--trust_grid", type=int, default=1)
    ap.add_argument("--bev", type=int, default=128)
    ap.add_argument("--image_hw", default="90,160")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--val_scenes", default=None)
    ap.add_argument("--cam", type=int, default=0, help="Camera index to fault.")
    ap.add_argument("--fault", default="occlusion",
                    help="Perturbation name from opendrivefm.robustness.perturbations.")
    ap.add_argument("--batch_sizes", default="1,2,4,8",
                    help="Comma-separated batch sizes to sweep.")
    ap.add_argument("--allow_untrained_trust", action="store_true",
                    help="Proceed with a checkpoint whose trust weights did not "
                         "load. The contamination ratio is a property of the "
                         "normalisation, not of the weights, but the magnitude "
                         "is only meaningful for a trained scorer.")
    ap.add_argument("--out", default="outputs/artifacts/batch_contamination.json")
    args = ap.parse_args()

    H, W = [int(v) for v in args.image_hw.split(",")]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.fault not in PERTURBATIONS:
        raise SystemExit(f"unknown fault {args.fault!r}; "
                         f"choose from {sorted(PERTURBATIONS)}")
    # PERTURBATIONS maps name -> nn.Module CLASS, not a function. It must be
    # instantiated, and it expects a batched (1,C,H,W) tensor.
    perturb = PERTURBATIONS[args.fault]()

    print("Loading model")
    model, ckpt_info = load_model(args.ckpt, args.bev, device, args.trust_grid,
                                  args.allow_untrained_trust)
    print(f"  device: {device}   trust_grid: {args.trust_grid}")

    rows = [json.loads(l) for l in
            Path(args.manifest).read_text().splitlines() if l.strip()]
    idx_val, scenes = val_indices(rows, args.seed, args.val_frac, args.val_scenes)
    ds = NuScenesMiniMultiView(
        args.manifest, image_hw=(H, W), frames=1, label_root=args.label_root,
        return_motion=True, return_trel=True, return_calib=True, augment=False)
    print(f"  val scenes: {scenes}   frames: {len(idx_val)}")

    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    results = []
    clean_by_bs: dict[int, "np.ndarray"] = {}

    print(f"\nFault: {args.fault}   faulted camera: {args.cam}")
    print(f"{'batch':>6} {'faulted |dtrust|':>18} {'untouched |dtrust|':>20} "
          f"{'ratio':>8} {'verdict':>12}")
    print("-" * 70)

    for bs in batch_sizes:
        loader = DataLoader(Subset(ds, idx_val), batch_size=bs,
                            shuffle=False, num_workers=0)
        # Re-seed before EACH pass. `OcclusionPatch` (and every other
        # perturbation) draws its size and position from the global RNG, so
        # without this the four batch sizes are compared against four DIFFERENT
        # sets of random faults, and fault variance is reported as a batch-size
        # effect. That is exactly what happened: this script reported the
        # faulted camera's response falling 36% "across the sweep" both before
        # and after the batch-relative centring was fixed, which is the tell --
        # a quantity that does not respond to the fix was never measuring it.
        #
        # Seeding identically for the clean and faulted passes costs nothing
        # (the clean pass draws no randomness) and makes the sweep a controlled
        # comparison in which batch size is the only thing that varies.
        def _seeded():
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

        _seeded()
        clean = trust_matrix(model, loader, device)
        _seeded()
        faulted = trust_matrix(model, loader, device,
                               perturb=perturb, cam_idx=args.cam)
        n, v = clean.shape
        others = [c for c in range(v) if c != args.cam]
        clean_by_bs[bs] = clean

        d_faulted = float(np.abs(faulted[:, args.cam] - clean[:, args.cam]).mean())
        # These pixels are bit-identical across the two passes. Any movement
        # here is caused by the batch mean, nothing else.
        d_untouched = float(np.abs(faulted[:, others] - clean[:, others]).mean())
        ratio = d_untouched / max(d_faulted, 1e-12)
        verdict = ("negligible" if ratio < 0.05
                   else "material" if ratio < 0.20 else "SEVERE")
        print(f"{bs:>6} {d_faulted:>18.6f} {d_untouched:>20.6f} "
              f"{ratio:>8.3f} {verdict:>12}")
        results.append({
            "batch_size": bs, "n_frames": n, "n_cameras": v,
            "mean_abs_delta_faulted_cam": d_faulted,
            "mean_abs_delta_untouched_cams": d_untouched,
            "contamination_ratio": ratio,
            "max_abs_delta_untouched_cams":
                float(np.abs(faulted[:, others] - clean[:, others]).max()),
            "verdict": verdict,
        })

    # THE DIRECT TEST OF FRAME-PURITY.
    #
    # Everything above compares a clean pass to a faulted pass at one batch
    # size, and the shared centring largely cancels in that difference, which is
    # why it stayed near zero even before calibration. The question calibration
    # actually answers is different: does ONE frame get ONE score, whatever it
    # is batched with? That is answered by scoring the same clean frames at
    # different batch sizes and comparing directly.
    #
    # It matters beyond tidiness. The C++ runtime scores a single frame's 6
    # cameras per call; this eval used batch_size 2, i.e. 12 images per call.
    # Under batch-relative centring those are different functions, so a trust
    # THRESHOLD chosen offline does not mean the same thing on the vehicle, and
    # Python/C++ parity on trust is not well defined.
    purity = None
    if len(clean_by_bs) > 1:
        base_bs = batch_sizes[0]
        base = clean_by_bs[base_bs]
        worst = max(((bs, float(np.abs(m - base).max()))
                     for bs, m in clean_by_bs.items() if bs != base_bs),
                    key=lambda t: t[1])
        purity = {"reference_batch_size": base_bs,
                  "worst_batch_size": worst[0],
                  "max_abs_trust_delta": worst[1]}
        print("\n" + "=" * 70)
        print("FRAME-PURITY: same clean frames, different batch size")
        print("=" * 70)
        for bs, m in sorted(clean_by_bs.items()):
            if bs == base_bs:
                print(f"  batch_size {bs:>2}: reference")
            else:
                print(f"  batch_size {bs:>2}: max |Δtrust| vs reference = "
                      f"{np.abs(m - base).max():.3e}")
        if worst[1] < 1e-6:
            print("\n  Trust is a pure function of the frame. A threshold chosen "
                  "offline\n  transfers to any batch shape, including the C++ "
                  "runtime's 1 frame\n  x 6 cameras.")
        else:
            print(f"\n  Trust is NOT a pure function of the frame: the same frame "
                  f"moves by\n  {worst[1]:.3e} between batch sizes. An absolute "
                  f"threshold is not\n  transferable across batch shapes. Fix with "
                  f"scripts/calibrate_trust_stats.py.")

    ratios = [r["contamination_ratio"] for r in results]
    worst_ratio = max(ratios)
    # Do not read a trend out of noise. When every ratio is negligible the
    # comparison is between two numbers that are both ~0, and "shrinks with
    # batch size" would be a confident statement about rounding.
    if worst_ratio < 0.05:
        trend = ("is negligible at every batch size, so no trend is claimed "
                 "(comparing near-zero ratios would be reading noise)")
    elif len(ratios) > 1 and ratios[-1] < ratios[0]:
        trend = ("shrinks with batch size, which confirms the batch-mean "
                 "normalisation is the cause")
    else:
        trend = ("does not shrink with batch size, so the normalisation may not "
                 "be the only cause; investigate further")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    worst = max(results, key=lambda r: r["contamination_ratio"])
    print(f"Worst contamination ratio: {worst['contamination_ratio']:.3f} "
          f"at batch_size={worst['batch_size']}.")
    print(f"Across the sweep the ratio {trend}.")

    # The faulted camera's OWN response is batch-dependent even when the
    # untouched cameras are not, because it too is scored relative to the batch
    # mean and it is 1/(B*V) of that mean. This does not bias AUROC, which is
    # rank-based within a single sweep, but it does mean an effect SIZE
    # (mean_trust_drop) is only comparable across runs at equal batch size.
    sig = [r["mean_abs_delta_faulted_cam"] for r in results]
    if sig and max(sig) > 0:
        spread = (max(sig) - min(sig)) / max(sig)
        print(f"\nEffect size on the faulted camera varies {100*spread:.0f}% across "
              f"the sweep\n({max(sig):.6f} at batch_size={batch_sizes[sig.index(max(sig))]} "
              f"down to {min(sig):.6f} at batch_size={batch_sizes[sig.index(min(sig))]}).")
        if spread > 0.01:
            print("With the faults held identical across the sweep, this residual\n"
                  "spread IS a genuine batch-size dependence. Compare\n"
                  "`mean_trust_drop` only between runs at the same --batch_size.")
        else:
            print("The faults are identical across the sweep, so this confirms the\n"
                  "score is a pure function of one frame: batch size no longer\n"
                  "changes the measured effect.")
    if worst["contamination_ratio"] >= 0.05:
        print(
            "\nCameras whose pixels did not change moved their trust score by\n"
            f"{100*worst['contamination_ratio']:.1f}% of the amount the actually-faulted\n"
            "camera moved. eval_ood_detection.py scores clean and faulted frames\n"
            "in two separate passes, so that shift is inside its reported AUROC\n"
            "for the ABSOLUTE detector (-trust[cam]). The RELATIVE detector\n"
            "(mean(trust[others]) - trust[cam]) compares cameras within one\n"
            "forward pass, where the shared offset largely cancels, and should\n"
            "be treated as the primary number.")
    else:
        print("\nContamination is below 5% of the signal at every batch size "
              "tested;\nthe reported AUROCs stand as measured.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": ckpt_info,
        "fault": args.fault,
        "faulted_camera": args.cam,
        "trust_grid": args.trust_grid,
        "val_scenes": scenes,
        "sweep": results,
        "worst_ratio": worst["contamination_ratio"],
        "frame_purity": purity,
        "faulted_effect_size_spread": (
            (max(sig) - min(sig)) / max(sig) if sig and max(sig) > 0 else None),
        "trend": trend,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
