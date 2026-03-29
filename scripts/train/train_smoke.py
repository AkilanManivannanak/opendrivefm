from __future__ import annotations

import argparse
import os
import shutil

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from opendrivefm.train.lightning_module import LitOpenDriveFM, make_synth_loaders


def pick_precision() -> str:
    # CUDA: AMP helps
    import torch

    if torch.cuda.is_available():
        return "16-mixed"

    # MPS: keep stable default; enable AMP only if you explicitly request it
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "16-mixed" if os.getenv("OPENDRIVEFM_MPS_AMP", "0") == "1" else "32-true"

    return "32-true"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--bev", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--n_train", type=int, default=512)
    ap.add_argument("--n_val", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean_ckpt", action="store_true")
    args = ap.parse_args()

    if args.clean_ckpt:
        shutil.rmtree("artifacts/checkpoints", ignore_errors=True)
    os.makedirs("artifacts/checkpoints", exist_ok=True)

    train_loader, val_loader = make_synth_loaders(
        batch_size=args.batch,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
        views=args.views,
        frames=args.frames,
        bev=args.bev,
        horizon=args.horizon,
    )

    ckpt = ModelCheckpoint(
        dirpath="artifacts/checkpoints",
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        save_last=True,
        filename="{epoch:02d}-{step}",
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision=pick_precision(),
        callbacks=[ckpt],
        log_every_n_steps=5,
    )

    model = LitOpenDriveFM(lr=args.lr, bev=args.bev, horizon=args.horizon)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("BEST CKPT:", ckpt.best_model_path)


if __name__ == "__main__":
    main()
