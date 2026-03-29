from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView
from opendrivefm.train.lightning_module import LitOpenDriveFM, _dl_kwargs


def split_by_scene(rows, seed: int, val_frac: float):
    scenes = sorted({r["scene"] for r in rows})
    rng = np.random.RandomState(seed)
    rng.shuffle(scenes)
    n_val = max(1, int(round(len(scenes) * val_frac)))
    val_scenes = set(scenes[:n_val])
    idx_val = [i for i, r in enumerate(rows) if r["scene"] in val_scenes]
    idx_train = [i for i, r in enumerate(rows) if r["scene"] not in val_scenes]
    return idx_train, idx_val, sorted(val_scenes)


def ade_fde(pred: torch.Tensor, gt: torch.Tensor):
    d = torch.linalg.norm(pred - gt, dim=-1)  # (B,T)
    return d.mean(dim=1), d[:, -1]


def occ_metrics(pred01: torch.Tensor, gt01: torch.Tensor, eps: float = 1e-6):
    # pred01, gt01: (B,1,H,W) float {0,1}
    pred = pred01.view(pred01.size(0), -1)
    gt = gt01.view(gt01.size(0), -1)

    tp = (pred * gt).sum(dim=1)
    fp = (pred * (1 - gt)).sum(dim=1)
    fn = ((1 - pred) * gt).sum(dim=1)

    inter = tp
    union = tp + fp + fn

    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (2 * inter + fp + fn + eps)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return iou, dice, precision, recall


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--manifest", type=str, default="artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", type=str, default="artifacts/nuscenes_labels")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac_scenes", type=float, default=0.2)
    ap.add_argument("--image_hw", type=str, default="90,160")  # H,W
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--out_metrics", type=str, default="artifacts/nuscenes_eval_metrics_residual.json")
    ap.add_argument("--out_items", type=str, default="artifacts/nuscenes_eval_items_residual.jsonl")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"ckpt not found: {ckpt}")

    H, W = [int(x.strip()) for x in args.image_hw.split(",")]

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    idx_train, idx_val, val_scenes = split_by_scene(rows, seed=args.seed, val_frac=args.val_frac_scenes)

    print(f"VAL_SCENES({len(val_scenes)}): {val_scenes}")
    print(f"VAL_ITEMS: {len(idx_val)}  (TRAIN_ITEMS would be {len(idx_train)})")
    print("DL_KWARGS:", _dl_kwargs())

    ds = NuScenesMiniMultiView(
        args.manifest,
        image_hw=(H, W),
        frames=1,
        label_root=args.label_root,
        return_motion=True,
        return_trel=True,
    )
    dl = DataLoader(Subset(ds, idx_val), batch_size=args.batch_size, shuffle=False, **_dl_kwargs())

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    lit = LitOpenDriveFM.load_from_checkpoint(str(ckpt), map_location="cpu")
    lit.eval()
    lit.to(device)

    # threshold sweep for occupancy
    thrs = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05).tolist()]
    thr_acc = {t: {"iou": [], "dice": [], "precision": [], "recall": []} for t in thrs}

    losses = []
    losses_occ = []
    losses_traj = []

    ade_list = []
    fde_list = []
    ade_cv_list = []
    fde_cv_list = []

    items = []

    for batch in dl:
        x, occ_t, traj_t, motion, t_rel = batch
        x = x.to(device)
        occ_t = occ_t.to(device)
        traj_t = traj_t.to(device)
        motion = motion.to(device)
        t_rel = t_rel.to(device)

        occ_logits, traj_res = lit(x)

        # CV + residual
        cv_traj = lit._make_cv_traj(motion, t_rel)
        traj_p = cv_traj + traj_res

        # losses (same as training)
        if occ_t.ndim == 3:
            occ_t2 = occ_t.unsqueeze(1)
        else:
            occ_t2 = occ_t
        if occ_logits.ndim == 3:
            occ_logits2 = occ_logits.unsqueeze(1)
        else:
            occ_logits2 = occ_logits

        l_occ = lit._occ_loss(occ_logits2, occ_t2)
        l_traj = lit._traj_residual_loss(traj_res, traj_t, cv_traj, t_rel)
        l = l_occ + lit.loss_cfg.traj_w * l_traj

        losses.append(float(l.item()))
        losses_occ.append(float(l_occ.item()))
        losses_traj.append(float(l_traj.item()))

        ade, fde = ade_fde(traj_p, traj_t)
        ade_cv, fde_cv = ade_fde(cv_traj, traj_t)

        ade_list.extend([float(a) for a in ade.cpu().tolist()])
        fde_list.extend([float(f) for f in fde.cpu().tolist()])
        ade_cv_list.extend([float(a) for a in ade_cv.cpu().tolist()])
        fde_cv_list.extend([float(f) for f in fde_cv.cpu().tolist()])

        # occupancy metrics per threshold
        prob = torch.sigmoid(occ_logits2)
        for t in thrs:
            pred01 = (prob > t).float()
            iou, dice, prec, rec = occ_metrics(pred01, occ_t2)
            thr_acc[t]["iou"].extend([float(v) for v in iou.cpu().tolist()])
            thr_acc[t]["dice"].extend([float(v) for v in dice.cpu().tolist()])
            thr_acc[t]["precision"].extend([float(v) for v in prec.cpu().tolist()])
            thr_acc[t]["recall"].extend([float(v) for v in rec.cpu().tolist()])

        # items jsonl
        # we don't have sample_token here (Subset(ds, idx_val) loses it), so recover from rows
        # We'll map by absolute index in idx_val order:
        # store later outside if needed — keep simple: omit token in residual eval items
        pass

    # pick best thr by IoU mean
    best_thr = None
    best_iou = -1.0
    best_pack = None
    for t in thrs:
        iou_m = float(np.mean(thr_acc[t]["iou"])) if thr_acc[t]["iou"] else float("nan")
        if iou_m > best_iou:
            best_iou = iou_m
            best_thr = t
            best_pack = {
                "thr": t,
                "occ_iou_mean": iou_m,
                "occ_dice_mean": float(np.mean(thr_acc[t]["dice"])),
                "occ_precision_mean": float(np.mean(thr_acc[t]["precision"])),
                "occ_recall_mean": float(np.mean(thr_acc[t]["recall"])),
                "occ_iou_median": float(np.median(thr_acc[t]["iou"])),
                "occ_dice_median": float(np.median(thr_acc[t]["dice"])),
            }

    metrics = {
        "ckpt": str(ckpt),
        "device": device,
        "seed": args.seed,
        "val_frac_scenes": args.val_frac_scenes,
        "val_scenes": val_scenes,
        "val_items": len(idx_val),
        "val_loss_mean": float(np.mean(losses)) if losses else None,
        "val_occ_loss_mean": float(np.mean(losses_occ)) if losses_occ else None,
        "val_traj_loss_mean": float(np.mean(losses_traj)) if losses_traj else None,
        "traj_ADE_mean": float(np.mean(ade_list)) if ade_list else None,
        "traj_FDE_mean": float(np.mean(fde_list)) if fde_list else None,
        "traj_ADE_median": float(np.median(ade_list)) if ade_list else None,
        "traj_FDE_median": float(np.median(fde_list)) if fde_list else None,
        "cv_ADE_mean": float(np.mean(ade_cv_list)) if ade_cv_list else None,
        "cv_FDE_mean": float(np.mean(fde_cv_list)) if fde_cv_list else None,
        "best_thr": best_thr,
        "best": best_pack,
    }

    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_metrics).write_text(json.dumps(metrics, indent=2))
    print("WROTE:", args.out_metrics)

    # optional: items jsonl omitted in this minimal residual eval
    Path(args.out_items).write_text("")
    print("WROTE:", args.out_items, "(empty; token-level items not wired in this minimal eval)")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
