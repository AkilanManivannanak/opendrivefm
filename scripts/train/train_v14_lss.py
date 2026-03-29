"""
train_v14_lss.py — Step 4 FULLY COMPLIANT training
Uses LSSGeometricBEV for proper 2D→3D back-projection.

Step 4 compliance:
  (i)  Camera intrinsics + extrinsics from nuScenes → pixel rays → frustum points
  (ii) 32 depth bins → depth distribution head → lift features into 3D → splat into BEV
  (iii) Per-camera BEV tensors in ego-centric frame, trust-weighted and summed

Copy to ~/opendrivefm/scripts/train_v14_lss.py
Run:
  python scripts/train_v14_lss.py \
    --manifest artifacts/nuscenes_mini_manifest.jsonl \
    --label_root artifacts/nuscenes_labels \
    --nusc_root data/nuscenes \
    --max_epochs 60 \
    --out_dir artifacts/checkpoints_v14_lss
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
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from opendrivefm.models.model_lss_bev import LSSGeometricBEV, DualOutputCNNStem

VAL_SCENES = {"scene-0655", "scene-1077"}
CAMS = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT",
        "CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]


class NuScenesLSSDataset(Dataset):
    """Returns images + intrinsics + extrinsics for geometric lifting."""
    def __init__(self, rows, label_root, nusc, image_hw=(90,160), augment=False):
        self.rows       = rows
        self.label_root = Path(label_root)
        self.nusc       = nusc
        self.tf = T.Compose([T.Resize(image_hw), T.ToTensor()])
        self.H, self.W  = image_hw

    def __len__(self): return len(self.rows)

    def _get_calib(self, sample, cam_name):
        sd    = self.nusc.get("sample_data", sample["data"][cam_name])
        cs    = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        # Intrinsics — scale to image resolution
        K     = np.array(cs["camera_intrinsic"], dtype=np.float32)
        # Scale K from original image to our resized image
        orig_sd = self.nusc.get("sample_data", sample["data"][cam_name])
        K[0] *= self.W / orig_sd["width"]
        K[1] *= self.H / orig_sd["height"]
        # Extrinsics: cam → ego
        R_c2e = Quaternion(cs["rotation"]).rotation_matrix.astype(np.float32)
        t_c2e = np.array(cs["translation"], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3,:3] = R_c2e; T[:3,3] = t_c2e
        return torch.from_numpy(K), torch.from_numpy(T)

    def __getitem__(self, idx):
        r      = self.rows[idx]
        tok    = r["sample_token"]
        sample = self.nusc.get("sample", tok)

        imgs, Ks, Ts = [], [], []
        for cam in CAMS:
            im = Image.open(Path(r["cams"][cam])).convert("RGB")
            imgs.append(self.tf(im))
            K, T_c2e = self._get_calib(sample, cam)
            Ks.append(K); Ts.append(T_c2e)

        x  = torch.stack(imgs).unsqueeze(1)   # (6,1,3,H,W)
        K  = torch.stack(Ks)                   # (6,3,3)
        Tc = torch.stack(Ts)                   # (6,4,4)

        z    = np.load(self.label_root / f"{tok}.npz")
        occ  = torch.from_numpy(z["occ"]).float()
        traj = torch.from_numpy(z["traj"]).float()
        t_rel= torch.from_numpy(
            z["t_rel"] if "t_rel" in z.files
            else np.arange(1,13,dtype=np.float32)*0.5).float()

        return x, occ, traj, torch.zeros(3), t_rel, K, Tc


class OpenDriveFMV14(nn.Module):
    """
    v14 — Full Step 4 compliance.
    Architecture:
      DualOutputCNNStem → per-camera spatial feature maps
      LSSGeometricBEV   → geometric BEV (frustum → splat) [Step 4 compliant]
      BEV decoder       → occupancy logits
      TrajHead          → ego trajectory
      CameraTrustScorer → per-camera trust (weights the BEV splatting)
    """
    def __init__(self, d=256, feat_ch=64, bev_ch=64,
                 bev_h=64, bev_w=64, n_depth=32, horizon=12):
        super().__init__()
        from opendrivefm.models.model import CameraTrustScorer

        self.stem        = DualOutputCNNStem(feat_ch=feat_ch, d=d)
        self.trust       = CameraTrustScorer()
        self.geo_bev     = LSSGeometricBEV(
            feat_ch=feat_ch, bev_ch=bev_ch,
            bev_h=bev_h, bev_w=bev_w, n_depth=n_depth)

        # BEV decoder: ConvTranspose2d upsampling → occupancy
        self.occ_dec = nn.Sequential(
            nn.Conv2d(bev_ch, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),     nn.ReLU(),
            nn.Conv2d(32,  1, 1),
        )

        # Trajectory head: pool BEV → global descriptor → MLP
        self.bev_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(bev_ch, d), nn.ReLU(),
        )
        self.traj_head = nn.Sequential(
            nn.Linear(d, 2*d), nn.ReLU(),
            nn.Linear(2*d, d), nn.ReLU(),
            nn.Linear(d, horizon*2),
        )
        self.horizon = horizon

    def forward(self, x, K, T_cam2ego):
        B, V, T_f, C, H, W = x.shape
        device = x.device

        # CNN stem → spatial feature maps + pooled vecs
        xt       = x.view(B*V*T_f, C, H, W)
        fm, _    = self.stem(xt)                          # (B*V, feat_ch, Hf, Wf)
        fm       = fm.view(B, V, *fm.shape[1:])           # (B, V, feat_ch, Hf, Wf)

        # Trust scores per camera
        imgs_flat = x[:, :, 0].reshape(B*V, C, H, W)
        trust_flat= self.trust(imgs_flat)
        trust     = trust_flat.view(B, V)                 # (B, V)

        # Scale K to feature map resolution
        Hf, Wf = fm.shape[-2:]
        K_scaled = K.clone().float()
        K_scaled[:, :, 0, :] *= (Wf / W)
        K_scaled[:, :, 1, :] *= (Hf / H)

        # STEP 4: Geometric BEV lifting (LSS)
        # (i)  intrinsics+extrinsics → frustum points in ego frame
        # (ii) depth bins → lift features → splat into BEV
        # (iii) per-camera BEV tensors → trust-weighted sum
        bev = self.geo_bev(fm, K_scaled, T_cam2ego, trust)  # (B, bev_ch, H, W)

        # Decode BEV → occupancy
        occ  = self.occ_dec(bev)                          # (B, 1, bev_h, bev_w)

        # BEV pool → trajectory
        z    = self.bev_pool(bev)                         # (B, d)
        traj = self.traj_head(z).view(B, self.horizon, 2) # (B, 12, 2)

        return occ, traj, trust


class LitV14(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = OpenDriveFMV14()
        self.lr    = lr

    def _step(self, batch):
        x, occ_gt, traj_gt, _, _, K, T_cam = batch
        occ_pred, traj_pred, trust = self.model(x, K, T_cam)

        occ_gt_r = occ_gt
        if occ_gt_r.ndim == 3: occ_gt_r = occ_gt_r.unsqueeze(1)
        occ_gt_r = F.interpolate(occ_gt_r, size=occ_pred.shape[-2:], mode='nearest')

        # Losses
        l_bce  = F.binary_cross_entropy_with_logits(occ_pred, occ_gt_r)
        prob   = torch.sigmoid(occ_pred)
        inter  = (prob * occ_gt_r).sum()
        l_dice = 1 - (2*inter+1)/(prob.sum()+occ_gt_r.sum()+1)
        l_traj = F.smooth_l1_loss(traj_pred, traj_gt)
        loss   = l_bce + 0.5*l_dice + 0.3*l_traj

        # IoU
        pred  = (prob > 0.5).float()
        TP = (pred*occ_gt_r).sum().item()
        FP = (pred*(1-occ_gt_r)).sum().item()
        FN = ((1-pred)*occ_gt_r).sum().item()
        iou = TP/(TP+FP+FN+1e-8)

        ade = (traj_pred - traj_gt).norm(dim=-1).mean()
        fde = (traj_pred[:,-1] - traj_gt[:,-1]).norm(dim=-1).mean()
        return loss, iou, ade, fde

    def training_step(self, batch, _):
        loss, iou, ade, _ = self._step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/IoU",  iou,  prog_bar=False)
        self.log("train/ADE",  ade,  prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, iou, ade, fde = self._step(batch)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/IoU",  iou,  prog_bar=True)
        self.log("val/ADE",  ade,  prog_bar=True)
        self.log("val/FDE",  fde,  prog_bar=False)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
        return [opt], [sch]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",   default="artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", default="artifacts/nuscenes_labels")
    ap.add_argument("--nusc_root",  default="data/nuscenes")
    ap.add_argument("--max_epochs", type=int,   default=60)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--out_dir",    default="artifacts/checkpoints_v14_lss")
    args = ap.parse_args()

    nusc = NuScenes(version="v1.0-mini", dataroot=args.nusc_root, verbose=False)
    rows = [json.loads(l) for l in
            Path(args.manifest).read_text().splitlines() if l.strip()]
    scene_key  = "scene" if "scene" in rows[0] else "scene_name"
    train_rows = [r for r in rows if r.get(scene_key,"") not in VAL_SCENES]
    val_rows   = [r for r in rows if r.get(scene_key,"") in VAL_SCENES]
    print(f"Train: {len(train_rows)} | Val: {len(val_rows)}")

    train_ds = NuScenesLSSDataset(train_rows, args.label_root, nusc, augment=True)
    val_ds   = NuScenesLSSDataset(val_rows,   args.label_root, nusc, augment=False)
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0)

    lit = LitV14(lr=args.lr)

    ckpt_cb = ModelCheckpoint(
        dirpath=args.out_dir, filename="best_val_ade",
        monitor="val/ADE", mode="min", save_last=True)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
        devices=1, precision="16-mixed",
        callbacks=[ckpt_cb], log_every_n_steps=5,
    )

    print("Training v14 — Step 4 compliant LSS Geometric BEV")
    print("  (i)  Camera intrinsics/extrinsics → pixel rays → frustum points")
    print("  (ii) 32 depth bins → lift features → splat into BEV grid")
    print("  (iii) Per-camera BEV tensors in ego-centric frame → trust-weighted sum")
    trainer.fit(lit, train_dl, val_dl)

    print(f"\nBest checkpoint: {ckpt_cb.best_model_path}")
    print(f"Best val/ADE:    {ckpt_cb.best_model_score:.4f}")

if __name__ == "__main__":
    main()
