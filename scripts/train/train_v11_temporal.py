"""
train_v11_temporal.py — Train OpenDriveFM v11 with T=4 temporal BEV.

Run AFTER v10 training is complete (warm-start from v10 checkpoint optional).

Usage:
    python scripts/train_v11_temporal.py \
        --manifest artifacts/nuscenes_mini_manifest.jsonl \
        --label_root artifacts/nuscenes_labels_128 \
        --nusc_root data/nuscenes \
        --max_epochs 120 \
        --out_dir artifacts/checkpoints_v11_temporal

Optional warm-start from v10 (recommended — saves ~30 epochs):
    python scripts/train_v11_temporal.py ... \
        --warmstart artifacts/checkpoints_v10_bev128/best_val_ade.ckpt
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from opendrivefm.data.nuscenes_mini_temporal import NuScenesMiniTemporal
from opendrivefm.train.lightning_module_v9 import LitOpenDriveFMV9, LossCfg

VAL_SCENES = {"scene-0655", "scene-1077"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",    default="artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root",  default="artifacts/nuscenes_labels_128")
    ap.add_argument("--nusc_root",   default="data/nuscenes")
    ap.add_argument("--out_dir",     default="artifacts/checkpoints_v11_temporal")
    ap.add_argument("--max_epochs",  type=int,   default=120)
    ap.add_argument("--batch_size",  type=int,   default=2)
    ap.add_argument("--lr",          type=float, default=5e-5,
                    help="Lower LR than v9/v10 — new temporal layers need gentle warmup")
    ap.add_argument("--n_frames",    type=int,   default=4)
    ap.add_argument("--warmstart",   type=str,   default=None,
                    help="Path to v10 checkpoint for warm-starting encoder weights")
    args = ap.parse_args()

    rows      = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    train_idx = [i for i, r in enumerate(rows) if r["scene"] not in VAL_SCENES]
    val_idx   = [i for i, r in enumerate(rows) if r["scene"] in VAL_SCENES]
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Frames per sample: {args.n_frames}")

    ds = NuScenesMiniTemporal(
        args.manifest,
        label_root   = args.label_root,
        nusc_root    = args.nusc_root,
        n_frames     = args.n_frames,
        return_lidar = False,    # skip LiDAR for speed; add back if time allows
    )

    kw = {"num_workers": 0, "pin_memory": False}
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=args.batch_size, shuffle=False, **kw)

    loss_cfg = LossCfg(depth_w=0.0, depth_warmup_epochs=0)   # no LiDAR in v11 by default
    lit = LitOpenDriveFMV9(
        lr=args.lr, d=384, bev=128, horizon=12,
        loss=loss_cfg, weight_decay=1e-4, enable_trust=True,
    )

    # Warm-start: load v10 backbone weights (ignore BEV accum layers — those are new)
    if args.warmstart and Path(args.warmstart).exists():
        print(f"Warm-starting from {args.warmstart}")
        ckpt = torch.load(args.warmstart, map_location="cpu")
        sd   = {k: v for k, v in ckpt["state_dict"].items()
                if "bev_accum" not in k}    # skip new temporal layers
        missing, unexpected = lit.load_state_dict(sd, strict=False)
        print(f"  Loaded {len(sd)-len(missing)} layers | "
              f"Missing (new): {len(missing)} | Unexpected: {len(unexpected)}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(dirpath=args.out_dir, filename="best_val_ade",
                        monitor="val/ADE", mode="min", save_last=True),
        LearningRateMonitor("epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs        = args.max_epochs,
        accelerator       = "mps" if torch.backends.mps.is_available() else "cpu",
        devices           = 1,
        precision         = "16-mixed" if torch.backends.mps.is_available() else 32,
        callbacks         = callbacks,
        log_every_n_steps = 5,
    )

    print(f"Training v11 — T={args.n_frames} temporal BEV + 128×128")
    trainer.fit(lit, train_loader, val_loader)
    print(f"\nBest val/ADE: {callbacks[0].best_model_score:.4f}")
    print(f"Checkpoint:   {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
