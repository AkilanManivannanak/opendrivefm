"""
lightning_module_v9.py — adds LiDAR depth supervision loss.

Changes vs v8:
  - Accepts optional lidar_depth_maps in batch (position 7)
  - Calls model with lidar_depth_maps when available → gets depth_pred back
  - Adds depth_loss (scale-invariant L1) weighted by depth_w (default 0.1)
  - depth_w anneals from 0 → 0.1 over first 20 epochs (warmup)
    so the model first learns BEV/traj, then adds depth refinement

Drop this file in as src/opendrivefm/train/lightning_module.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Import from model_v9 (rename it to model.py when you deploy)
from opendrivefm.models.model import OpenDriveFM, lidar_depth_loss
from opendrivefm.data.synth import SyntheticMultiViewVideo


def _dl_kwargs() -> dict[str, Any]:
    return {"num_workers": 0, "pin_memory": False, "persistent_workers": False}


def dice_loss_from_logits(logits, target01, eps=1e-6):
    prob  = torch.sigmoid(logits).view(logits.size(0), -1)
    tgt   = target01.view(target01.size(0), -1)
    inter = (prob * tgt).sum(dim=1)
    denom = prob.sum(dim=1) + tgt.sum(dim=1)
    return 1.0 - ((2.0*inter+eps)/(denom+eps)).mean()


def ade_fde(pred, gt):
    d = torch.linalg.norm(pred - gt, dim=-1)
    return d.mean(dim=1), d[:, -1]


@dataclass
class LossCfg:
    occ_dice_w:        float = 0.7
    pos_weight_cap:    float = 15.0
    traj_beta:         float = 1.0
    traj_w:            float = 1.0
    resid_l2_w:        float = 0.02
    time_weight_power: float = 1.0
    trust_w:           float = 0.05
    trust_target:      float = 0.75
    # ── NEW depth supervision ──────────────────────────────────────────
    depth_w:           float = 0.1      # weight on depth loss at full strength
    depth_warmup_epochs: int = 20       # ramp depth_w linearly over these epochs


class LitOpenDriveFMV9(pl.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        d=384,
        bev=64,
        horizon=12,
        loss=None,
        weight_decay=1e-4,
        grad_clip=1.0,
        enable_trust=True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss"])
        self.model        = OpenDriveFM(d=d, bev_h=bev, bev_w=bev,
                                        horizon=horizon, enable_trust=enable_trust)
        self.lr           = lr
        self.weight_decay = weight_decay
        self.grad_clip    = grad_clip
        self.loss_cfg     = loss or LossCfg()
        self.horizon      = int(horizon)
        self.enable_trust = enable_trust

    # ── Loss helpers ────────────────────────────────────────────────────────

    def _depth_weight(self) -> float:
        """Linear warmup: 0 → depth_w over first depth_warmup_epochs."""
        ep = self.current_epoch
        wu = self.loss_cfg.depth_warmup_epochs
        if wu <= 0:
            return self.loss_cfg.depth_w
        return self.loss_cfg.depth_w * min(1.0, ep / wu)

    def _occ_loss(self, occ_logits, occ_tgt):
        with torch.no_grad():
            pos = occ_tgt.sum()
            neg = float(occ_tgt.numel()) - pos
            pw  = (neg / (pos + 1e-6)).clamp(1.0, self.loss_cfg.pos_weight_cap)
        bce = F.binary_cross_entropy_with_logits(occ_logits, occ_tgt, pos_weight=pw)
        dsc = dice_loss_from_logits(occ_logits, occ_tgt)
        return bce + self.loss_cfg.occ_dice_w * dsc

    def _make_cv_traj(self, motion, t_rel):
        dt_prev = motion[:, 0:1]
        vxy     = motion[:, 1:3] * (dt_prev > 0.0).to(motion.dtype)
        return t_rel.unsqueeze(-1) * vxy.unsqueeze(1)

    def _traj_residual_loss(self, traj_res, traj_t, cv_traj, t_rel):
        target_res = traj_t - cv_traj
        tnorm = t_rel / t_rel[:, -1:].clamp(min=1e-6)
        w     = tnorm.clamp(0).pow(self.loss_cfg.time_weight_power).unsqueeze(-1)
        per   = F.smooth_l1_loss(traj_res, target_res,
                                  beta=self.loss_cfg.traj_beta, reduction="none")
        return (per * w).mean() + self.loss_cfg.resid_l2_w * (traj_res**2).mean()

    def _trust_reg_loss(self, trust):
        mean_dev = (trust.mean() - self.loss_cfg.trust_target)**2
        p        = trust / (trust.sum(dim=1, keepdim=True) + 1e-8)
        entropy  = -(p * (p + 1e-8).log()).sum(dim=1).mean()
        return mean_dev - 0.1 * entropy

    # ── Batch unpacking ─────────────────────────────────────────────────────

    def _unpack_batch(self, batch):
        """
        Accepts 5, 6, 7, or 8-element batches.
        Returns (x, occ_t, traj_t, motion, t_rel, K, T_ego, lidar_maps)
        K, T_ego, lidar_maps may be None.
        """
        n = len(batch)

        if n == 5:
            x, occ_t, traj_t, motion, t_rel = batch
            return x, occ_t, traj_t, motion, t_rel, None, None, None
        elif n == 6:
            x, occ_t, traj_t, motion, t_rel, K = batch
            return x, occ_t, traj_t, motion, t_rel, K, None, None
        elif n == 7:
            x, occ_t, traj_t, motion, t_rel, K, T_ego = batch
            return x, occ_t, traj_t, motion, t_rel, K, T_ego, None
        elif n == 8:
            x, occ_t, traj_t, motion, t_rel, K, T_ego, last = batch
            # last is lidar_maps (ndim==5) or ego_deltas (ndim==3)
            ldm = last if last.ndim == 5 else None
            return x, occ_t, traj_t, motion, t_rel, K, T_ego, ldm
        else:
            raise ValueError(f"Unexpected batch length: {n}")

    # ── Core step ───────────────────────────────────────────────────────────

    def _step(self, batch):
        x, occ_t, traj_t, motion, t_rel, K, T_ego, lidar_maps = \
            self._unpack_batch(batch)

        vel = motion[:, 1:3]

        # Forward — pass lidar_maps only if available
        if lidar_maps is not None:
            occ_logits, traj_res, trust, depth_pred, Hf, Wf = \
                self.model(x, velocity=vel, lidar_depth_maps=lidar_maps)
        else:
            occ_logits, traj_res, trust, _ = self.model(x, velocity=vel)
            depth_pred, Hf, Wf = None, None, None

        if occ_t.ndim == 3:      occ_t      = occ_t.unsqueeze(1)
        if occ_logits.ndim == 3: occ_logits = occ_logits.unsqueeze(1)

        cv_traj = self._make_cv_traj(motion, t_rel)
        traj_p  = cv_traj + traj_res

        l_occ   = self._occ_loss(occ_logits, occ_t)
        l_traj  = self._traj_residual_loss(traj_res, traj_t, cv_traj, t_rel)
        l_trust = self._trust_reg_loss(trust) if self.enable_trust \
                  else torch.tensor(0., device=x.device)

        # Depth loss (only when we have LiDAR supervision)
        if depth_pred is not None and lidar_maps is not None:
            l_depth = lidar_depth_loss(depth_pred, lidar_maps, Hf, Wf)
            dw      = self._depth_weight()
        else:
            l_depth = torch.tensor(0., device=x.device)
            dw      = 0.0

        loss = (l_occ
                + self.loss_cfg.traj_w  * l_traj
                + self.loss_cfg.trust_w * l_trust
                + dw                    * l_depth)

        ade_m,  fde_m  = ade_fde(traj_p,  traj_t)
        ade_cv, fde_cv = ade_fde(cv_traj, traj_t)

        return (loss, l_occ, l_traj, l_trust, l_depth, dw,
                ade_m.mean(), fde_m.mean(), ade_cv.mean(), fde_cv.mean(), trust)

    # ── Lightning hooks ─────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        loss, l_occ, l_traj, l_trust, l_depth, dw, \
            ade_m, fde_m, ade_cv, fde_cv, trust = self._step(batch)

        self.log("train/loss",        loss,          prog_bar=True,  on_step=True, on_epoch=True)
        self.log("train/occ",         l_occ,                         on_step=True, on_epoch=True)
        self.log("train/traj",        l_traj,                        on_step=True, on_epoch=True)
        self.log("train/trust_loss",  l_trust,                       on_step=True, on_epoch=True)
        self.log("train/depth_loss",  l_depth,                       on_step=True, on_epoch=True)
        self.log("train/depth_w",     dw,                            on_step=False, on_epoch=True)
        self.log("train/ADE",         ade_m,         prog_bar=True,  on_step=True, on_epoch=True)
        self.log("train/FDE",         fde_m,         prog_bar=True,  on_step=True, on_epoch=True)
        self.log("train/CV_ADE",      ade_cv,                        on_step=True, on_epoch=True)
        self.log("train/trust_mean",  trust.mean(),                  on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, l_occ, l_traj, l_trust, l_depth, dw, \
            ade_m, fde_m, ade_cv, fde_cv, trust = self._step(batch)

        self.log("val/loss",       loss,         prog_bar=True,  on_step=False, on_epoch=True)
        self.log("val/occ",        l_occ,                        on_step=False, on_epoch=True)
        self.log("val/traj",       l_traj,                       on_step=False, on_epoch=True)
        self.log("val/depth_loss", l_depth,                      on_step=False, on_epoch=True)
        self.log("val/ADE",        ade_m,        prog_bar=True,  on_step=False, on_epoch=True)
        self.log("val/FDE",        fde_m,        prog_bar=True,  on_step=False, on_epoch=True)
        self.log("val/CV_ADE",     ade_cv,                       on_step=False, on_epoch=True)
        self.log("val/trust_mean", trust.mean(),                  on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None,
                                    gradient_clip_algorithm=None):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
