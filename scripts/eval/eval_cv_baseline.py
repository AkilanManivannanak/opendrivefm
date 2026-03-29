from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView


def ade_fde(pred: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    d = torch.linalg.norm(pred - gt, dim=-1)
    return float(d.mean()), float(d[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", type=str, default="artifacts/nuscenes_labels")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac_scenes", type=float, default=0.2)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    scenes = sorted({r["scene"] for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(scenes)
    n_val = max(1, int(round(len(scenes) * args.val_frac_scenes)))
    val_scenes = set(scenes[:n_val])

    idx_val = [i for i, r in enumerate(rows) if r["scene"] in val_scenes]

    ds = NuScenesMiniMultiView(args.manifest, label_root=args.label_root, frames=1)

    ade = []
    fde = []
    used = 0

    for i in idx_val:
        tok = rows[i]["sample_token"]
        z = np.load(Path(args.label_root) / f"{tok}.npz")
        gt = ds[i][2]  # traj (H,2)

        t_rel = torch.from_numpy(z["t_rel"]).float()  # (H,)
        vxy = torch.from_numpy(z["vxy_prev"]).float()  # (2,)
        dt_prev = float(z["dt_prev"])

        if dt_prev <= 0.0:
            continue

        pred = t_rel[:, None] * vxy[None, :]  # (H,2)
        a, f = ade_fde(pred, gt)
        ade.append(a); fde.append(f)
        used += 1

    ta, tf = torch.tensor(ade), torch.tensor(fde)
    print("VAL scenes:", sorted(val_scenes))
    print("VAL items:", len(idx_val), "used(nonzero dt_prev):", used)
    print("CV ADE mean/median/p95:", float(ta.mean()), float(ta.median()), float(ta.quantile(0.95)))
    print("CV FDE mean/median/p95:", float(tf.mean()), float(tf.median()), float(tf.quantile(0.95)))


if __name__ == "__main__":
    main()
