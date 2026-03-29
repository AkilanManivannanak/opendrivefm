"""
train_v9_lidar_depth.py — Train OpenDriveFM v9 with LiDAR depth supervision.

Usage (from ~/opendrivefm/):
    python scripts/train_v9_lidar_depth.py \
        --manifest artifacts/nuscenes_mini_manifest.jsonl \
        --label_root artifacts/nuscenes_labels \
        --nusc_root data/nuscenes \
        --max_epochs 120 \
        --out_dir artifacts/checkpoints_v9

Estimated time: ~3-4 hours on MPS (same as v8, LiDAR projection adds ~15% overhead).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from opendrivefm.data.nuscenes_mini_v9 import NuScenesMiniV9
from opendrivefm.train.lightning_module_v9 import LitOpenDriveFMV9, LossCfg

VAL_SCENES = {"scene-0655", "scene-1077"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",    default="artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root",  default="artifacts/nuscenes_labels")
    ap.add_argument("--nusc_root",   default="data/nuscenes")
    ap.add_argument("--out_dir",     default="artifacts/checkpoints_v9")
    ap.add_argument("--max_epochs",  type=int, default=120)
    ap.add_argument("--batch_size",  type=int, default=2)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--depth_w",     type=float, default=0.1,
                    help="Weight on LiDAR depth loss at full strength")
    ap.add_argument("--no_lidar",    action="store_true",
                    help="Disable LiDAR supervision (ablation)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    train_idx = [i for i, r in enumerate(rows) if r["scene"] not in VAL_SCENES]
    val_idx   = [i for i, r in enumerate(rows) if r["scene"] in VAL_SCENES]
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

    use_lidar = not args.no_lidar
    ds = NuScenesMiniV9(
        args.manifest,
        label_root  = args.label_root,
        nusc_root   = args.nusc_root,
        return_lidar= use_lidar,
        return_calib= True,
        return_motion=True,
        return_trel =True,
    )

    kw = {"num_workers": 0, "pin_memory": False}
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=args.batch_size, shuffle=False, **kw)

    loss_cfg = LossCfg(depth_w=args.depth_w, depth_warmup_epochs=20)
    lit = LitOpenDriveFMV9(
        lr=args.lr,
        d=384,
        bev=64,
        horizon=12,
        loss=loss_cfg,
        weight_decay=1e-4,
        enable_trust=True,
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=args.out_dir,
            filename="best_val_ade",
            monitor="val/ADE",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor("epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.backends.mps.is_available() else 32,
        callbacks=callbacks,
        log_every_n_steps=5,
        enable_progress_bar=True,
    )

    print(f"Training v9 — LiDAR depth supervision: {use_lidar}")
    trainer.fit(lit, train_loader, val_loader)
    print(f"\nBest checkpoint: {callbacks[0].best_model_path}")
    print(f"Best val/ADE:    {callbacks[0].best_model_score:.4f}")


if __name__ == "__main__":
    main()
