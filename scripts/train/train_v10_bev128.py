"""
train_v10_bev128.py — Train OpenDriveFM v10 with 128×128 BEV.

Run AFTER regen_labels_128.py has completed.

Usage:
    python scripts/train_v10_bev128.py \
        --manifest artifacts/nuscenes_mini_manifest.jsonl \
        --label_root artifacts/nuscenes_labels_128 \
        --nusc_root data/nuscenes \
        --max_epochs 120 \
        --out_dir artifacts/checkpoints_v10_bev128
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Re-use v9 dataloader (handles lidar + 128×128 labels identically)
from opendrivefm.data.nuscenes_mini_v9 import NuScenesMiniV9

# Import the 128×128 model as the active model
import importlib.util, sys as _sys
_spec = importlib.util.spec_from_file_location(
    "model_bev128",
    Path(__file__).parent.parent / "src/opendrivefm/models/model_v10_bev128.py"
)
# Simpler: just use the v9 lightning module but point it at the 128 model
from opendrivefm.train.lightning_module_v9 import LitOpenDriveFMV9, LossCfg
from opendrivefm.models.model import OpenDriveFM   # will be model_v10 after you rename it

VAL_SCENES = {"scene-0655", "scene-1077"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",   default="artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", default="artifacts/nuscenes_labels_128")
    ap.add_argument("--nusc_root",  default="data/nuscenes")
    ap.add_argument("--out_dir",    default="artifacts/checkpoints_v10_bev128")
    ap.add_argument("--max_epochs", type=int,   default=120)
    ap.add_argument("--batch_size", type=int,   default=2)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--no_lidar",   action="store_true")
    args = ap.parse_args()

    rows      = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    train_idx = [i for i, r in enumerate(rows) if r["scene"] not in VAL_SCENES]
    val_idx   = [i for i, r in enumerate(rows) if r["scene"] in VAL_SCENES]

    ds = NuScenesMiniV9(
        args.manifest,
        label_root   = args.label_root,
        nusc_root    = args.nusc_root,
        return_lidar = not args.no_lidar,
        return_calib = True,
        return_motion= True,
        return_trel  = True,
    )

    kw = {"num_workers": 0, "pin_memory": False}
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=args.batch_size, shuffle=False, **kw)

    loss_cfg = LossCfg(depth_w=0.1, depth_warmup_epochs=20)
    lit = LitOpenDriveFMV9(
        lr=args.lr, d=384, bev=128, horizon=12,
        loss=loss_cfg, weight_decay=1e-4, enable_trust=True,
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(dirpath=args.out_dir, filename="best_val_ade",
                        monitor="val/ADE", mode="min", save_last=True),
        LearningRateMonitor("epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs   = args.max_epochs,
        accelerator  = "mps" if torch.backends.mps.is_available() else "cpu",
        devices      = 1,
        precision    = "16-mixed" if torch.backends.mps.is_available() else 32,
        callbacks    = callbacks,
        log_every_n_steps = 5,
    )

    print("Training v10 — 128×128 BEV + LiDAR depth supervision")
    trainer.fit(lit, train_loader, val_loader)
    print(f"\nBest val/ADE: {callbacks[0].best_model_score:.4f}")
    print(f"Checkpoint:   {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
