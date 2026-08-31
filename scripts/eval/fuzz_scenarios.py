"""
Coverage-guided scenario fuzzer with rare-event mining.

THE GAP THIS CLOSES
-------------------
Every evaluation in this repo so far tests ONE fault, at ONE severity, on ONE
camera. The real scenario space is:

    5 fault types x continuous severity x 6 cameras x 1..6 simultaneous cameras

which is effectively unbounded, and we have been sampling a handful of points
from one corner of it. A model can pass every hand-written test and fail on a
combination nobody thought to write down. Finding those combinations is the job.

HOW IT SEARCHES
---------------
--strategy random     Uniform sampling of the scenario space. The baseline.
--strategy coverage   Coverage-guided: bias sampling toward cells of the
                      discretised space that have been visited least. This is
                      the same idea as coverage-guided fuzzing in software
                      testing (AFL and friends): spend budget on unexplored
                      behaviour rather than re-confirming known behaviour.
--strategy adaptive   Coverage-guided, then intensify around the worst scenarios
                      found so far (mutate severity and camera set of an elite).
                      Finds the tail faster; explores less.

SEVERITY IS REAL, NOT NOMINAL
-----------------------------
severity in [0,1] is mapped onto each perturbation's own parameter range, so
severity 0.9 blur genuinely means a larger sigma and severity 0.9 occlusion
genuinely means a bigger patch. Without this every "severity" sweep would be
sampling the same default distribution and reporting noise.

WHAT IT REPORTS
---------------
  * the worst-N scenarios by IoU or ADE degradation (the rare events);
  * a coverage grid over (fault x severity bin x n_cameras) showing which cells
    were exercised and what the worst degradation in each cell was;
  * cells never visited, which is the honest statement of what remains untested.

Usage
-----
    python scripts/eval/fuzz_scenarios.py \
        --ckpt outputs/artifacts/checkpoints_v11_trustfix2/trust_fixed_v2.ckpt \
        --trials 200 --strategy coverage \
        --out outputs/artifacts/fuzz_report.json
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
from opendrivefm.robustness import perturbations as P  # noqa: E402
from opendrivefm.training.lightning_module import LitOpenDriveFM  # noqa: E402

THR = 0.6
VAL_SCENES = {"scene-0655", "scene-1077"}
CAM_NAMES = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
             "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def lerp(lo: float, hi: float, s: float) -> float:
    return lo + (hi - lo) * s


def build_perturbation(fault: str, severity: float):
    """Map severity in [0,1] onto each perturbation's own parameter range.

    A narrow band around the target is used rather than a point value, so
    repeated trials at the same severity are not identical -- the perturbations
    are stochastic by design and we want that variance represented.
    """
    s = float(np.clip(severity, 0.0, 1.0))
    band = 0.05
    lo, hi = max(0.0, s - band), min(1.0, s + band)
    if fault == "blur":
        return P.GaussianBlur(sigma_range=(lerp(0.5, 8.0, lo), lerp(0.5, 8.0, hi)))
    if fault == "glare":
        return P.GlareOverlay(intensity_range=(lerp(0.1, 1.0, lo), lerp(0.1, 1.0, hi)),
                              size_range=(lerp(0.05, 0.6, lo), lerp(0.05, 0.6, hi)))
    if fault == "occlusion":
        return P.OcclusionPatch(patch_frac=(lerp(0.05, 0.8, lo), lerp(0.05, 0.8, hi)))
    if fault == "rain":
        return P.RainStreaks(num_streaks=(int(lerp(5, 200, lo)), int(lerp(5, 200, hi)) + 1),
                             alpha=(lerp(0.05, 0.9, lo), lerp(0.05, 0.9, hi)))
    if fault == "noise":
        return P.SaltPepperNoise(amount_range=(lerp(0.005, 0.30, lo), lerp(0.005, 0.30, hi)))
    raise ValueError(f"unknown fault: {fault}")


def severity_bin(s: float, n_bins: int) -> int:
    return min(int(s * n_bins), n_bins - 1)


@torch.no_grad()
def run_scenario(model, loader, device, scenario) -> dict:
    """Apply a scenario to every batch and return mean IoU / ADE."""
    perts = [(c, build_perturbation(scenario["fault"], scenario["severity"]))
             for c in scenario["cameras"]]
    ious, ades = [], []
    for batch in loader:
        x, occ_t, traj = (batch[i].to(device) for i in range(3))
        motion, t_rel = batch[3].to(device), batch[4].to(device)
        if occ_t.ndim == 3:
            occ_t = occ_t.unsqueeze(1)
        vel = motion[:, 1:3]

        x = x.clone()
        for b in range(x.shape[0]):
            for cam, pert in perts:
                x[b, cam, -1] = pert(x[b, cam, -1].unsqueeze(0)).squeeze(0)

        occ_logits, traj_res, _, _ = model(x, velocity=vel)
        if occ_logits.ndim == 3:
            occ_logits = occ_logits.unsqueeze(1)
        pred = (torch.sigmoid(occ_logits) > THR).float()
        inter = (pred * occ_t).sum((1, 2, 3))
        union = (pred + occ_t).clamp(0, 1).sum((1, 2, 3))
        ious.extend((inter / (union + 1e-6)).cpu().tolist())
        p = t_rel.unsqueeze(-1) * vel.unsqueeze(1) + traj_res
        ades.extend(torch.linalg.norm(p - traj, dim=-1).mean(1).cpu().tolist())
    return {"iou": float(np.mean(ious)), "ade": float(np.mean(ades))}


def sample_scenario(rng, faults, n_cams, sev_bins, visits, strategy, elites):
    if strategy == "adaptive" and elites and rng.random() < 0.4:
        base = rng.choice(elites)
        sev = float(np.clip(base["severity"] + rng.gauss(0, 0.1), 0, 1))
        cams = list(base["cameras"])
        if rng.random() < 0.5 and len(cams) < n_cams:
            cams.append(rng.choice([c for c in range(n_cams) if c not in cams]))
        elif len(cams) > 1 and rng.random() < 0.3:
            cams.pop(rng.randrange(len(cams)))
        return {"fault": base["fault"], "severity": sev, "cameras": sorted(cams)}

    if strategy in ("coverage", "adaptive"):
        # Pick the least-visited cell, breaking ties at random, then sample a
        # concrete scenario inside it.
        cells = [(f, b, k) for f in faults for b in range(sev_bins)
                 for k in range(1, n_cams + 1)]
        rng.shuffle(cells)
        fault, sbin, k = min(cells, key=lambda c: visits[c])
        sev = (sbin + rng.random()) / sev_bins
    else:
        fault = rng.choice(faults)
        sev = rng.random()
        k = rng.randint(1, n_cams)
    cams = sorted(rng.sample(range(n_cams), k))
    return {"fault": fault, "severity": sev, "cameras": cams}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="outputs/artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", default="outputs/artifacts/nuscenes_labels_128")
    ap.add_argument("--bev", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--image_hw", default="90,160")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--strategy", choices=["random", "coverage", "adaptive"],
                    default="coverage")
    ap.add_argument("--objective", choices=["iou", "ade"], default="iou")
    ap.add_argument("--severity_bins", type=int, default=5)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--max_frames", type=int, default=24,
                    help="Frames per trial. Fuzzing wants many scenarios on few "
                         "frames, not few scenarios on many frames.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/artifacts/fuzz_report.json")
    args = ap.parse_args()

    H, W = [int(v) for v in args.image_hw.split(",")]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}
    lit = LitOpenDriveFM(bev=args.bev)
    res = lit.model.load_state_dict(state, strict=False)
    n_exp = len(lit.model.state_dict())
    print(f"Loaded {n_exp - len(res.missing_keys)}/{n_exp} weights")
    model = lit.model.eval().to(device)

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    idx = [i for i, r in enumerate(rows) if r["scene"] in VAL_SCENES][:args.max_frames]
    ds = NuScenesMiniMultiView(args.manifest, image_hw=(H, W), frames=1,
                               label_root=args.label_root, return_motion=True,
                               return_trel=True, return_calib=True, augment=False)
    loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    faults = list(P.PERTURBATIONS)
    n_cams = len(CAM_NAMES)
    print(f"  frames/trial: {len(idx)}   strategy: {args.strategy}   "
          f"trials: {args.trials}")
    print(f"  scenario space: {len(faults)} faults x {args.severity_bins} severity "
          f"bins x {n_cams} camera-counts = "
          f"{len(faults) * args.severity_bins * n_cams} cells\n")

    baseline = run_scenario(model, loader, device,
                            {"fault": "blur", "severity": 0.0, "cameras": []})
    print(f"  clean baseline: IoU {baseline['iou']:.4f}  ADE {baseline['ade']:.3f}\n")

    visits = defaultdict(int)
    worst_in_cell: dict = {}
    trials, elites = [], []

    for t in range(args.trials):
        sc = sample_scenario(rng, faults, n_cams, args.severity_bins, visits,
                             args.strategy, elites)
        m = run_scenario(model, loader, device, sc)
        # Degradation is positive = worse than clean, for both objectives.
        deg_iou = baseline["iou"] - m["iou"]
        deg_ade = m["ade"] - baseline["ade"]
        deg = deg_iou if args.objective == "iou" else deg_ade

        cell = (sc["fault"], severity_bin(sc["severity"], args.severity_bins),
                len(sc["cameras"]))
        visits[cell] += 1
        if cell not in worst_in_cell or deg > worst_in_cell[cell]["degradation"]:
            worst_in_cell[cell] = {"degradation": deg, **sc}

        rec = {**sc, "iou": m["iou"], "ade": m["ade"],
               "degradation_iou": deg_iou, "degradation_ade": deg_ade,
               "degradation": deg, "cell": list(cell)}
        trials.append(rec)
        elites = sorted(trials, key=lambda r: -r["degradation"])[:args.top_k]

        if (t + 1) % 20 == 0:
            print(f"  trial {t+1:>4}/{args.trials}  cells hit "
                  f"{len(visits)}/{len(faults)*args.severity_bins*n_cams}  "
                  f"worst so far {elites[0]['degradation']:+.4f} "
                  f"({elites[0]['fault']} sev {elites[0]['severity']:.2f} "
                  f"x{len(elites[0]['cameras'])} cams)")

    total_cells = len(faults) * args.severity_bins * n_cams
    print(f"\nRARE EVENTS: worst {args.top_k} scenarios by {args.objective} degradation")
    print(f"{'rank':>5}{'fault':>11}{'sev':>7}{'cams':>6}{'IoU':>9}{'dIoU':>9}"
          f"{'ADE':>8}{'dADE':>8}   cameras")
    print("-" * 92)
    for i, r in enumerate(elites, 1):
        names = ",".join(CAM_NAMES[c].replace("CAM_", "") for c in r["cameras"])
        print(f"{i:>5}{r['fault']:>11}{r['severity']:>7.2f}{len(r['cameras']):>6}"
              f"{r['iou']:>9.4f}{r['degradation_iou']:>+9.4f}{r['ade']:>8.3f}"
              f"{r['degradation_ade']:>+8.3f}   {names}")

    print(f"\nCOVERAGE: {len(visits)}/{total_cells} cells exercised "
          f"({100*len(visits)/total_cells:.1f}%)")
    by_fault = defaultdict(lambda: [0, 0.0])
    for (f, _, _), v in visits.items():
        by_fault[f][0] += v
    for cell, w in worst_in_cell.items():
        by_fault[cell[0]][1] = max(by_fault[cell[0]][1], w["degradation"])
    print(f"{'fault':>11}{'trials':>9}{'worst deg':>12}")
    for f in faults:
        print(f"{f:>11}{by_fault[f][0]:>9}{by_fault[f][1]:>+12.4f}")

    unvisited = [(f, b, k) for f in faults for b in range(args.severity_bins)
                 for k in range(1, n_cams + 1) if (f, b, k) not in visits]
    if unvisited:
        print(f"\nUNTESTED: {len(unvisited)} cells never sampled. The honest statement "
              f"of\nwhat this run does NOT cover. Examples: "
              + "; ".join(f"{f} sev-bin{b} x{k}cams" for f, b, k in unvisited[:4]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "checkpoint": args.ckpt, "strategy": args.strategy, "trials": args.trials,
        "objective": args.objective, "seed": args.seed,
        "frames_per_trial": len(idx), "baseline": baseline,
        "scenario_space_cells": total_cells, "cells_exercised": len(visits),
        "coverage_fraction": len(visits) / total_cells,
        "rare_events": elites,
        "worst_per_cell": {f"{k[0]}|sev{k[1]}|x{k[2]}": v
                           for k, v in worst_in_cell.items()},
        "untested_cells": [{"fault": f, "severity_bin": b, "n_cameras": k}
                           for f, b, k in unvisited],
        "all_trials": trials,
    }, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
