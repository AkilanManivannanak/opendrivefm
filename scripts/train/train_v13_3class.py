"""
train_v13_3class.py — v3, uses correct v8 backbone (MultiViewVideoBackbone)
Focused fix: use the working backbone, separate traj from occ training.

Copy to ~/opendrivefm/scripts/train_v13_3class.py
Run:
  python scripts/train_v13_3class.py \
    --manifest artifacts/nuscenes_mini_manifest.jsonl \
    --label_root artifacts/nuscenes_labels_3class \
    --nusc_root data/nuscenes \
    --warmstart artifacts/checkpoints_v9/best_val_ade.ckpt \
    --max_epochs 80 \
    --out_dir artifacts/checkpoints_v13_3class_v3
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

VAL_SCENES = {"scene-0655", "scene-1077"}
CAMS = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT",
        "CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]


class SemanticBEVDataset(Dataset):
    def __init__(self, rows, label_root, image_hw=(90,160), augment=False):
        self.rows       = rows
        self.label_root = Path(label_root)
        self.augment    = augment
        self.tf  = T.Compose([T.Resize(image_hw), T.ToTensor()])
        self.aug = T.ColorJitter(brightness=0.3, contrast=0.3) if augment else None

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r   = self.rows[idx]
        tok = r["sample_token"]
        imgs = []
        for cam in CAMS:
            im = Image.open(Path(r["cams"][cam])).convert("RGB")
            if self.aug: im = self.aug(im)
            imgs.append(self.tf(im))
        x = torch.stack(imgs).unsqueeze(1)              # (6,1,3,H,W)

        z    = np.load(self.label_root / f"{tok}.npz")
        occ  = torch.from_numpy(z["occ"]).float()       # (3,128,128)
        traj = torch.from_numpy(z["traj"]).float()
        t_rel= torch.from_numpy(
            z["t_rel"] if "t_rel" in z.files
            else np.arange(1,13,dtype=np.float32)*0.5).float()
        return x, occ, traj, torch.zeros(3), t_rel


class BEVHead3Class(nn.Module):
    """3-class semantic BEV head with upsampling ConvTranspose layers."""
    def __init__(self, d=384, bev_h=128, bev_w=128, n_classes=3):
        super().__init__()
        self.n_classes = n_classes
        # Project d → seed for ConvTranspose upsampling
        self.seed_proj = nn.Sequential(
            nn.Linear(d, d), nn.GELU(),
            nn.Dropout(0.2),
        )
        # Seed: reshape to (d, 4, 4) then upsample to (128,128)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(d,    256, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(256,  128, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(128,   64, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(64,    32, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(32,    16, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, n_classes, 1),
        )
        # 5 ups: 4→8→16→32→64→128 ✓

    def forward(self, z):
        B = z.size(0)
        seed = self.seed_proj(z).view(B, -1, 1, 1)     # (B, d, 1, 1)
        seed = seed.expand(-1, -1, 4, 4)               # (B, d, 4, 4)
        return self.up(seed)                            # (B, 3, 128, 128)


class OpenDriveFMV13(nn.Module):
    """
    v13: uses v8 backbone + 3-class semantic BEV head + trajectory head.
    Backbone: MultiViewVideoBackbone (correct class name in current model.py)
    """
    def __init__(self, d=384, bev_h=128, bev_w=128, n_classes=3, horizon=12):
        super().__init__()
        from opendrivefm.models.model import MultiViewTemporalBackbone as MultiViewVideoBackbone, TrajHead
        self.backbone  = MultiViewVideoBackbone(d=d, enable_trust=True)
        self.occ_head  = BEVHead3Class(d=d, bev_h=bev_h, bev_w=bev_w, n_classes=n_classes)
        self.traj_head = TrajHead(d=d, horizon=horizon)

    def forward(self, x):
        z, _, trust = self.backbone(x)
        return self.occ_head(z), self.traj_head(z), trust


class LitV13(pl.LightningModule):
    def __init__(self, lr=3e-4):
        super().__init__()
        self.model = OpenDriveFMV13()
        self.lr    = lr

    def _step(self, batch):
        x, occ_gt, traj_gt, _, _ = batch
        occ_pred, traj_pred, _ = self.model(x)

        if occ_gt.shape[-1] != occ_pred.shape[-1]:
            occ_gt = F.interpolate(occ_gt, size=occ_pred.shape[-2:], mode="nearest")

        occ_cls = occ_gt.argmax(dim=1).long()
        # class weights: free=0.05, vehicle=3.0, pedestrian=10.0
        w = torch.tensor([0.05, 3.0, 10.0], device=x.device)
        l_occ = F.cross_entropy(occ_pred, occ_cls, weight=w, label_smoothing=0.05)

        prob   = F.softmax(occ_pred, dim=1)
        l_dice = 0.0
        for c in [1, 2]:
            p = prob[:, c]; g = occ_gt[:, c]
            l_dice += 1 - (2*(p*g).sum()+1)/(p.sum()+g.sum()+1)
        l_dice /= 2.0

        l_traj = F.smooth_l1_loss(traj_pred, traj_gt)
        loss   = l_occ + 0.5*l_dice + 0.3*l_traj

        pred_cls = occ_pred.argmax(dim=1)
        ious = {}
        for c, name in enumerate(["free","veh","ped"]):
            tp = ((pred_cls==c)&(occ_cls==c)).float().sum()
            fp = ((pred_cls==c)&(occ_cls!=c)).float().sum()
            fn = ((pred_cls!=c)&(occ_cls==c)).float().sum()
            ious[name] = (tp/(tp+fp+fn+1e-8)).item()

        ade = (traj_pred - traj_gt).norm(dim=-1).mean()
        fde = (traj_pred[:,-1] - traj_gt[:,-1]).norm(dim=-1).mean()
        return loss, ade, fde, ious

    def training_step(self, batch, _):
        loss, ade, _, ious = self._step(batch)
        self.log("train/loss",    loss,        prog_bar=True)
        self.log("train/ADE",     ade,         prog_bar=True)
        self.log("train/IoU_veh", ious["veh"], prog_bar=False)
        self.log("train/IoU_ped", ious["ped"], prog_bar=False)
        return loss

    def validation_step(self, batch, _):
        loss, ade, fde, ious = self._step(batch)
        self.log("val/loss",    loss,        prog_bar=True)
        self.log("val/ADE",     ade,         prog_bar=True)
        self.log("val/FDE",     fde,         prog_bar=False)
        self.log("val/IoU_veh", ious["veh"], prog_bar=True)
        self.log("val/IoU_ped", ious["ped"], prog_bar=True)

    def configure_optimizers(self):
        new_p, old_p = [], []
        for name, p in self.model.named_parameters():
            if "occ_head" in name:
                new_p.append(p)
            else:
                old_p.append(p)
        opt = torch.optim.AdamW([
            {"params": new_p, "lr": self.lr},
            {"params": old_p, "lr": self.lr * 0.1},
        ], weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80, eta_min=1e-5)
        return [opt], [sch]


def load_warmstart(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw  = ckpt.get("state_dict", ckpt)
    sd   = {}
    for k, v in raw.items():
        if   k.startswith("model."):     sd[k[6:]]  = v
        elif k.startswith("lit.model."): sd[k[10:]] = v
        elif k.startswith("lit."):       sd[k[4:]]  = v
        else:                            sd[k]       = v
    model_sd = model.state_dict()
    loaded = 0
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v; loaded += 1
    model.load_state_dict(model_sd)
    print(f"  Loaded {loaded} layers from warmstart")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",   default="artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", default="artifacts/nuscenes_labels_3class")
    ap.add_argument("--nusc_root",  default="data/nuscenes")
    ap.add_argument("--warmstart",  default="artifacts/checkpoints_v9/best_val_ade.ckpt")
    ap.add_argument("--max_epochs", type=int,   default=80)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--out_dir",    default="artifacts/checkpoints_v13_3class_v3")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    scene_key  = "scene" if "scene" in rows[0] else "scene_name"
    train_rows = [r for r in rows if r.get(scene_key,"") not in VAL_SCENES]
    val_rows   = [r for r in rows if r.get(scene_key,"") in VAL_SCENES]
    print(f"Train: {len(train_rows)} | Val: {len(val_rows)}")

    train_ds = SemanticBEVDataset(train_rows, args.label_root, augment=True)
    val_ds   = SemanticBEVDataset(val_rows,   args.label_root, augment=False)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0)

    lit = LitV13(lr=args.lr)

    if args.warmstart and Path(args.warmstart).exists():
        print(f"Warm-starting from: {args.warmstart}")
        load_warmstart(lit.model, args.warmstart)
    else:
        print("No warmstart found — training from scratch")

    ckpt_cb = ModelCheckpoint(
        dirpath=args.out_dir, filename="best_val_ade",
        monitor="val/ADE", mode="min", save_last=True)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
        devices=1, precision="16-mixed",
        callbacks=[ckpt_cb], log_every_n_steps=5,
    )

    print("Training v13 v3 — 3-class semantic BEV using v8 backbone")
    trainer.fit(lit, train_dl, val_dl)
    print(f"\nBest checkpoint: {ckpt_cb.best_model_path}")
    print(f"Best val/ADE:    {ckpt_cb.best_model_score:.4f}")

if __name__ == "__main__":
    main()
