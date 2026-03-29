"""
LitOpenDriveFM — Updated Lightning module.
Changes vs original:
  - BCE + Dice loss with dynamic pos_weight (class imbalance handling)
  - Passes K, T_ego_cam to model for geometry lifting
  - Handles new 4-tuple model output (occ, traj, trust, weights)
  - Contrastive trust loss
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from opendrivefm.models.model import OpenDriveFM


def _dl_kwargs():
    return {"num_workers": 0, "pin_memory": False, "persistent_workers": False}


def dice_loss_from_logits(logits, target, eps=1e-6):
    """Soft Dice loss for class-imbalanced BEV occupancy."""
    prob  = torch.sigmoid(logits).view(logits.size(0), -1)
    tgt   = target.view(target.size(0), -1)
    inter = (prob * tgt).sum(dim=1)
    denom = prob.sum(dim=1) + tgt.sum(dim=1)
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


def focal_loss_from_logits(logits, target, gamma=2.0, alpha=0.25, eps=1e-6):
    """Focal loss — suppresses easy negatives, fixes over-prediction."""
    prob = torch.sigmoid(logits)
    bce  = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t  = prob * target + (1 - prob) * (1 - target)
    a_t  = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = a_t * (1 - p_t).pow(gamma)
    return (focal_weight * bce).mean()


def ade_fde(pred, gt):
    d = torch.linalg.norm(pred - gt, dim=-1)
    return d.mean(dim=1), d[:, -1]


@dataclass
class LossCfg:
    occ_dice_w:        float = 1.5    # higher dice weight for sparse objects
    occ_focal_w:       float = 2.0    # higher focal weight for sparse objects
    focal_gamma:       float = 3.0    # stronger focus for sparse objects
    focal_alpha:       float = 0.85   # high alpha = penalise misses (sparse FG)
    pos_weight_cap:    float = 15.0   # max pos_weight for imbalance
    traj_w:            float = 1.0    # balanced — BEV needs room to learn
    traj_beta:         float = 0.5    # tighter smooth L1
    resid_l2_w:        float = 0.01
    time_weight_power: float = 2.0    # stronger end-point weighting
    trust_w:           float = 0.05
    trust_target:      float = 0.75
    contrastive_w:     float = 0.10
    contrastive_margin: float = 0.20


class LitOpenDriveFM(pl.LightningModule):
    def __init__(self, lr=3e-4, d=384, bev=64, horizon=12,
                 loss=None, weight_decay=1e-2, grad_clip=1.0,
                 enable_trust=True, use_geometry=True):
        super().__init__()
        self.save_hyperparameters(ignore=["loss"])
        self.model = OpenDriveFM(
            d=d, bev_h=bev, bev_w=bev, horizon=horizon,
            enable_trust=enable_trust, use_geometry=use_geometry)
        self.lr           = lr
        self.weight_decay = weight_decay
        self.grad_clip    = grad_clip
        self.loss_cfg     = loss or LossCfg()
        self.horizon      = int(horizon)
        self.enable_trust = enable_trust

    def forward(self, x, K=None, T_ego_cam=None):
        occ, traj, trust, _ = self.model(x, K, T_ego_cam)
        return occ, traj

    # ── Loss functions ────────────────────────────────────────

    def _occ_loss(self, occ_logits, occ_tgt):
        """Focal + Dice: focal suppresses easy negatives, fixes solid-blue over-prediction."""
        focal = focal_loss_from_logits(
            occ_logits, occ_tgt,
            gamma=self.loss_cfg.focal_gamma,
            alpha=self.loss_cfg.focal_alpha)
        dsc = dice_loss_from_logits(occ_logits, occ_tgt)
        return self.loss_cfg.occ_focal_w * focal + self.loss_cfg.occ_dice_w * dsc

    def _make_cv_traj(self, motion, t_rel):
        dt_prev = motion[:, 0:1]
        vxy     = motion[:, 1:3] * (dt_prev > 0.0).to(motion.dtype)
        return t_rel.unsqueeze(-1) * vxy.unsqueeze(1)

    def _traj_loss(self, traj_res, traj_t, cv_traj, t_rel):
        """
        Trajectory loss designed to beat CV baseline:
        - Direct prediction loss on absolute trajectory (not residual only)
        - FDE endpoint loss (2x weight — FDE is key metric)
        - ADE loss (mean over all timesteps)
        - Residual magnitude penalty (prevents ignoring CV prior)
        - Time-weighted: later timesteps matter more
        """
        target_res  = traj_t - cv_traj           # what residual should be
        traj_pred   = cv_traj + traj_res          # absolute predicted trajectory

        # Time weights: ramp up toward end of sequence
        tnorm = t_rel / t_rel[:, -1:].clamp(min=1e-6)
        w     = tnorm.clamp(0).pow(self.loss_cfg.time_weight_power).unsqueeze(-1)

        # ADE loss (SmoothL1 over all timesteps, time-weighted)
        per  = F.smooth_l1_loss(traj_pred, traj_t,
                                beta=self.loss_cfg.traj_beta, reduction="none")
        ade_loss = (per * w).mean()

        # FDE loss (2x weight — endpoint matters most)
        fde_loss = F.smooth_l1_loss(
            traj_pred[:, -1].contiguous(),
            traj_t[:, -1].contiguous(),
            beta=self.loss_cfg.traj_beta)

        # Residual loss: keep residuals small so CV prior is respected
        res_loss = (traj_res**2).mean()

        # CV penalty: if model is WORSE than CV, add extra penalty
        with torch.no_grad():
            cv_ade  = F.l1_loss(cv_traj, traj_t)
            our_ade = F.l1_loss(traj_pred, traj_t)
        worse_than_cv = F.relu(our_ade - cv_ade)  # 0 if beating CV

        return (ade_loss
                + 2.0 * fde_loss
                + self.loss_cfg.resid_l2_w * res_loss
                + 0.5 * worse_than_cv)

    # Aliases for backward compatibility with train_nuscenes_mini_trust.py
    def _trust_reg_loss(self, trust):
        loss, _ = self._trust_loss(trust)
        return loss


    def _traj_residual_loss(self, traj_res, traj_t, cv_traj, t_rel):
        return self._traj_loss(traj_res, traj_t, cv_traj, t_rel)

    def _trust_loss(self, trust, trust_aug=None):
        """Trust regularisation + optional contrastive loss."""
        mean_dev = (trust.mean() - self.loss_cfg.trust_target) ** 2
        p        = trust / (trust.sum(dim=1, keepdim=True) + 1e-8)
        entropy  = -(p * (p + 1e-8).log()).sum(dim=1).mean()
        reg_loss = mean_dev - 0.1 * entropy

        if trust_aug is not None:
            # Contrastive: clean trust should exceed augmented trust by margin
            margin = self.loss_cfg.contrastive_margin
            cont   = F.relu(margin - (trust.mean(1) - trust_aug.mean(1))).mean()
            return reg_loss + self.loss_cfg.contrastive_w * cont, cont
        return reg_loss, torch.tensor(0.0, device=trust.device)

    # ── Batch unpacking ───────────────────────────────────────

    def _unpack_batch(self, batch):
        """Handles (x, occ, traj), (x, occ, traj, motion), or
           (x, occ, traj, motion, t_rel) with optional K, T_ego_cam."""
        K, T_ego = None, None
        if isinstance(batch, dict):
            x      = batch["x"]
            occ_t  = batch["occ_gt"]
            traj_t = batch["traj_gt"]
            motion = batch.get("motion", None)
            t_rel  = batch.get("t_rel",  None)
            K      = batch.get("K",      None)
            T_ego  = batch.get("T_ego_cam", None)
        else:
            items  = list(batch)
            x, occ_t, traj_t = items[0], items[1], items[2]
            motion = items[3] if len(items) > 3 else None
            t_rel  = items[4] if len(items) > 4 else None

        B, T = traj_t.shape[0], traj_t.shape[1]
        if motion is None:
            motion = torch.zeros(B, 3, device=traj_t.device, dtype=traj_t.dtype)
        if t_rel is None:
            t_rel = torch.arange(1, T+1, device=traj_t.device,
                                 dtype=traj_t.dtype)[None].repeat(B, 1) * 0.5
        return x, occ_t, traj_t, motion, t_rel, K, T_ego

    # ── Training/val step ─────────────────────────────────────

    def _step(self, batch, training=False):
        x, occ_t, traj_t, motion, t_rel, K, T_ego = self._unpack_batch(batch)

        # Extract velocity from motion for scene-aware trajectory
        velocity = motion[:, 1:3].to(x.device) if motion is not None else None
        occ_logits, traj_res, trust, weights = self.model(x, K, T_ego, velocity=velocity)

        if occ_t.ndim == 3:      occ_t      = occ_t.unsqueeze(1)
        if occ_logits.ndim == 3: occ_logits = occ_logits.unsqueeze(1)

        cv_traj = self._make_cv_traj(motion.to(x.device), t_rel.to(x.device))
        traj_p  = cv_traj + traj_res

        l_occ  = self._occ_loss(occ_logits, occ_t.float())
        l_traj = self._traj_loss(traj_res, traj_t, cv_traj, t_rel)

        if self.enable_trust:
            l_trust, l_cont = self._trust_loss(trust)
        else:
            l_trust = l_cont = torch.tensor(0.0, device=x.device)

        loss = (l_occ
                + self.loss_cfg.traj_w   * l_traj
                + self.loss_cfg.trust_w  * l_trust)

        ade_m,  fde_m  = ade_fde(traj_p,  traj_t)
        ade_cv, fde_cv = ade_fde(cv_traj, traj_t)

        # Count dropped cameras (trust < threshold)
        n_dropped = (trust < 0.15).float().sum(dim=1).mean()

        return (loss, l_occ, l_traj, l_trust, l_cont,
                ade_m.mean(), fde_m.mean(), ade_cv.mean(), fde_cv.mean(),
                trust, weights, n_dropped)

    def training_step(self, batch, batch_idx):
        out = self._step(batch, training=True)
        loss, l_occ, l_traj, l_trust, l_cont, ade_m, fde_m, ade_cv, fde_cv, \
            trust, _, n_dropped = out
        self.log_dict({
            "train/loss": loss, "train/occ": l_occ, "train/traj": l_traj,
            "train/trust": l_trust, "train/contrastive": l_cont,
            "train/ADE": ade_m, "train/FDE": fde_m,
            "train/trust_mean": trust.mean(), "train/dropped_cams": n_dropped,
        }, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self._step(batch, training=False)
        loss, l_occ, l_traj, l_trust, l_cont, ade_m, fde_m, ade_cv, fde_cv, \
            trust, _, n_dropped = out
        self.log_dict({
            "val/loss": loss, "val/ADE": ade_m, "val/FDE": fde_m,
            "val/trust_mean": trust.mean(), "val/dropped_cams": n_dropped,
        }, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    def configure_gradient_clipping(self, optimizer, **_):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
