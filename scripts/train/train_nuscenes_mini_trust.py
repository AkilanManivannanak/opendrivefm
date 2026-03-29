"""
Train OpenDriveFM with Trust-Aware Fusion + contrastive trust signal.

Key addition over previous version:
  --augment flag enables fault-injection during training.
  A random camera gets a perturbation; trust loss now includes
  a contrastive term: degraded cam should score LOW, clean should score HIGH.
  This teaches the scorer to actually differentiate bad cameras.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView
from opendrivefm.train.lightning_module import LitOpenDriveFM, LossCfg, _dl_kwargs
from opendrivefm.robustness.perturbations import PERTURBATIONS


def split_by_scene(rows, seed, val_frac):
    scenes = sorted({r["scene"] for r in rows})
    rng    = random.Random(seed)
    rng.shuffle(scenes)
    n_val      = max(1, int(round(len(scenes) * val_frac)))
    val_scenes = set(scenes[:n_val])
    idx_val    = [i for i, r in enumerate(rows) if r["scene"]     in val_scenes]
    idx_train  = [i for i, r in enumerate(rows) if r["scene"] not in val_scenes]
    return idx_train, idx_val, sorted(val_scenes)


class TrustAwareTrainer(pl.LightningModule):
    """
    Wraps LitOpenDriveFM and adds contrastive trust training:
      - With prob `fault_prob`, randomly corrupt 1 camera
      - Add contrastive loss: trust[degraded] < trust[clean] - margin
      - This gives the trust scorer a direct supervision signal
    """
    def __init__(self, lit: LitOpenDriveFM, fault_prob: float = 0.5,
                 margin: float = 0.2, contrastive_w: float = 1.0):
        super().__init__()
        self.lit           = lit
        self.fault_prob    = fault_prob
        self.margin        = margin
        self.contrastive_w = contrastive_w
        self._perturbers   = {k: v() for k, v in PERTURBATIONS.items()}
        self.save_hyperparameters(ignore=["lit"])

    def forward(self, x):
        return self.lit.model(x)  # inference — no velocity needed

    def _inject_fault(self, x):
        """
        Randomly corrupt one camera. Returns (x_corrupted, cam_idx).
        x: (B, V, T, C, H, W)
        """
        V = x.shape[1]
        cam_idx = random.randint(0, V - 1)
        perturber = random.choice(list(self._perturbers.values()))
        x_out = x.clone()
        # Apply to that camera's first frame across the batch
        cam_imgs = x_out[:, cam_idx, 0]          # (B, C, H, W)
        x_out[:, cam_idx, 0] = perturber(cam_imgs)
        return x_out, cam_idx

    def _step(self, batch, training: bool):
        x, occ_t, traj_t, motion, t_rel = batch[0], batch[1], batch[2], batch[3], batch[4]
        K     = batch[5].to(batch[0].device) if len(batch) > 5 else None
        T_ego = batch[6].to(batch[0].device) if len(batch) > 6 else None
        x = x.to(self.device)
        occ_t  = occ_t.to(self.device)
        traj_t = traj_t.to(self.device)
        motion = motion.to(self.device)
        t_rel  = t_rel.to(self.device)

        # ── Task losses on CLEAN input ────────────────────────────────────
        velocity = motion[:, 1:3]  # (B,2) vx,vy
        occ_logits, traj_res, trust_clean, _ = self.lit.model(x, K=K, T_ego_cam=T_ego, velocity=velocity)
        cv_traj   = self.lit._make_cv_traj(motion, t_rel)
        traj_pred = cv_traj + traj_res

        if occ_t.ndim == 3:    occ_t2 = occ_t.unsqueeze(1)
        else:                  occ_t2 = occ_t
        if occ_logits.ndim==3: occ_logits = occ_logits.unsqueeze(1)

        l_occ  = self.lit._occ_loss(occ_logits, occ_t2)
        l_traj = self.lit._traj_residual_loss(traj_res, traj_t, cv_traj, t_rel)
        l_trust_reg = self.lit._trust_reg_loss(trust_clean)

        # ── Contrastive trust loss (training only) ────────────────────────
        l_contrastive = torch.tensor(0.0, device=self.device)
        degraded_cam  = -1
        if training and random.random() < self.fault_prob:
            x_deg, degraded_cam = self._inject_fault(x)
            _, _, trust_deg, _ = self.lit.model(x_deg, K=K, T_ego_cam=T_ego, velocity=velocity)

            # trust_clean[:, degraded_cam] should be HIGHER than trust_deg[:, degraded_cam]
            # by at least `margin`
            t_clean_cam = trust_clean[:, degraded_cam]   # (B,)
            t_deg_cam   = trust_deg[:, degraded_cam]     # (B,)

            # Hinge loss: max(0, t_deg - t_clean + margin)
            l_contrastive = F.relu(t_deg_cam - t_clean_cam + self.margin).mean()

            # Also: other cameras should stay HIGH on degraded input
            other_mask = torch.ones(trust_clean.shape[1], device=self.device, dtype=torch.bool)
            other_mask[degraded_cam] = False
            if other_mask.any():
                t_others_deg = trust_deg[:, other_mask]
                # Penalise if other cameras drop below 0.6 on degraded input
                l_contrastive = l_contrastive + F.relu(0.6 - t_others_deg).mean() * 0.3

        loss = (l_occ
                + self.lit.loss_cfg.traj_w * l_traj
                + self.lit.loss_cfg.trust_w * l_trust_reg
                + self.contrastive_w * l_contrastive)

        # ADE/FDE
        d = torch.linalg.norm(traj_pred - traj_t, dim=-1)
        ade = d.mean(dim=1).mean()
        fde = d[:, -1].mean()

        return loss, l_occ, l_traj, l_trust_reg, l_contrastive, ade, fde, trust_clean, degraded_cam

    def training_step(self, batch, _):
        loss, l_occ, l_traj, l_trust, l_cont, ade, fde, trust, dcam = self._step(batch, training=True)
        self.log("train/loss",        loss,   on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/occ",         l_occ,  on_step=False, on_epoch=True)
        self.log("train/traj",        l_traj, on_step=False, on_epoch=True)
        self.log("train/trust_reg",   l_trust,on_step=False, on_epoch=True)
        self.log("train/contrastive", l_cont, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/ADE",         ade,    on_step=False, on_epoch=True)
        self.log("train/FDE",         fde,    on_step=False, on_epoch=True)
        self.log("train/trust_mean",  trust.mean(), on_step=False, on_epoch=True)
        self.log("train/trust_min",   trust.min(),  on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        loss, l_occ, l_traj, l_trust, l_cont, ade, fde, trust, _ = self._step(batch, training=False)
        self.log("val/loss",       loss,   on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ADE",        ade,    on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/FDE",        fde,    on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/trust_mean", trust.mean(), on_step=False, on_epoch=True)
        self.log("val/trust_min",  trust.min(),  on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return self.lit.configure_optimizers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",        type=str,   default="artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root",      type=str,   default="artifacts/nuscenes_labels")
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--val_frac_scenes", type=float, default=0.2)
    ap.add_argument("--batch_size",      type=int,   default=2)
    ap.add_argument("--max_epochs",      type=int,   default=30)
    ap.add_argument("--lr",              type=float, default=3e-4)
    ap.add_argument("--image_hw_h",      type=int,   default=90)
    ap.add_argument("--image_hw_w",      type=int,   default=160)
    ap.add_argument("--ckpt_dir",        type=str,   default="artifacts/checkpoints_trust")
    ap.add_argument("--disable_trust",   action="store_true")
    ap.add_argument("--trust_w",         type=float, default=0.05)
    ap.add_argument("--trust_target",    type=float, default=0.75)
    # NEW FLAGS
    ap.add_argument("--augment",         action="store_true",
                    help="Enable fault-injection contrastive trust training")
    ap.add_argument("--fault_prob",      type=float, default=0.5,
                    help="Probability of injecting a fault each step")
    ap.add_argument("--margin",          type=float, default=0.2,
                    help="Contrastive margin: trust(clean) - trust(degraded) > margin")
    ap.add_argument("--contrastive_w",   type=float, default=1.0,
                    help="Weight on contrastive trust loss")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    idx_train, idx_val, val_scenes = split_by_scene(rows, args.seed, args.val_frac_scenes)
    print("VAL_SCENES:", val_scenes)
    print(f"TRAIN: {len(idx_train)}  VAL: {len(idx_val)}")
    print(f"Augment/contrastive trust: {args.augment}")

    ds = NuScenesMiniMultiView(
        args.manifest,
        image_hw=(args.image_hw_h, args.image_hw_w),
        frames=1, label_root=args.label_root,
        return_motion=True, return_trel=True,
        return_calib=True,
    )
    dl_train = DataLoader(Subset(ds, idx_train), batch_size=args.batch_size,
                          shuffle=True,  **_dl_kwargs())
    dl_val   = DataLoader(Subset(ds, idx_val),   batch_size=args.batch_size,
                          shuffle=False, **_dl_kwargs())

    loss_cfg = LossCfg(trust_w=args.trust_w, trust_target=args.trust_target)
    lit = LitOpenDriveFM(lr=args.lr, enable_trust=not args.disable_trust, loss=loss_cfg)

    if args.augment:
        model = TrustAwareTrainer(lit,
                                  fault_prob=args.fault_prob,
                                  margin=args.margin,
                                  contrastive_w=args.contrastive_w)
    else:
        model = lit

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    monitor_key = "val/loss"
    accel = "mps" if torch.backends.mps.is_available() else "cpu"
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="trust_aug_{epoch:03d}-{val/loss:.3f}" if args.augment
                     else "trust_{epoch:03d}-{val/FDE:.3f}",
            monitor=monitor_key, mode="min",
            save_last=True, save_top_k=3,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        accelerator=accel, devices=1,
        max_epochs=args.max_epochs,
        precision="16-mixed" if accel == "mps" else "32-true",
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)
    print("DONE. Last ckpt:", ckpt_dir / "last.ckpt")


if __name__ == "__main__":
    main()
