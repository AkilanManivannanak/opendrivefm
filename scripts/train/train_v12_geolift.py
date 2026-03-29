"""
train_v12_geolift.py — Fine-tune v12 with geometric BEV lifting
Warm-starts from v11 checkpoint (loads all compatible weights).

Copy to ~/opendrivefm/scripts/train_v12_geolift.py
Run:
  python scripts/train_v12_geolift.py \
    --manifest artifacts/nuscenes_mini_manifest.jsonl \
    --label_root artifacts/nuscenes_labels_128 \
    --nusc_root data/nuscenes \
    --warmstart artifacts/checkpoints_v11_temporal/best_val_ade.ckpt \
    --max_epochs 60 \
    --out_dir artifacts/checkpoints_v12_geolift
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
from opendrivefm.models.model_v12_geolift import OpenDriveFMV12, occ_loss, traj_loss

VAL_SCENES = {"scene-0655", "scene-1077"}
CAMS = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT",
        "CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]


class NuScenesV12Dataset(Dataset):
    def __init__(self, rows, label_root, nusc, image_hw=(90,160), augment=False):
        self.rows       = rows
        self.label_root = Path(label_root)
        self.nusc       = nusc
        self.augment    = augment
        self.tf = T.Compose([T.Resize(image_hw), T.ToTensor()])
        self.H, self.W  = image_hw

    def __len__(self): return len(self.rows)

    def _get_cam_calib(self, sample, cam_name):
        sd    = self.nusc.get("sample_data", sample["data"][cam_name])
        cs    = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        K     = torch.tensor(cs["camera_intrinsic"], dtype=torch.float32)
        R_c2e = Quaternion(cs["rotation"]).rotation_matrix
        t_c2e = np.array(cs["translation"])
        T     = np.eye(4)
        T[:3,:3] = R_c2e
        T[:3, 3] = t_c2e
        return K, torch.tensor(T, dtype=torch.float32)

    def __getitem__(self, idx):
        r      = self.rows[idx]
        tok    = r["sample_token"]
        sample = self.nusc.get("sample", tok)

        imgs, Ks, Ts = [], [], []
        for cam in CAMS:
            im = Image.open(Path(r["cams"][cam])).convert("RGB")
            imgs.append(self.tf(im))
            K, T_c2e = self._get_cam_calib(sample, cam)
            Ks.append(K)
            Ts.append(T_c2e)

        x  = torch.stack(imgs).unsqueeze(1)              # (6,1,3,H,W)
        K  = torch.stack(Ks)                              # (6,3,3)
        Tc = torch.stack(Ts)                              # (6,4,4)

        z    = np.load(self.label_root / f"{tok}.npz")
        occ  = torch.from_numpy(z["occ"]).float()
        traj = torch.from_numpy(z["traj"]).float()
        t_rel= torch.from_numpy(
            z["t_rel"] if "t_rel" in z.files else np.arange(1,13)*0.5).float()

        return x, occ, traj, torch.zeros(3), t_rel, K, Tc


class LitV12(pl.LightningModule):
    def __init__(self, lr=3e-4):
        super().__init__()
        self.model = OpenDriveFMV12()
        self.lr    = lr

    def _step(self, batch):
        x, occ_t, traj_t, _, _, K, T_cam = batch
        occ_pred, traj_pred, trust, bev_geo = self.model(x, K, T_cam)

        occ_gt = occ_t
        if occ_gt.ndim == 3: occ_gt = occ_gt.unsqueeze(1)
        occ_gt = F.interpolate(occ_gt, size=occ_pred.shape[-2:], mode="nearest")

        l_occ  = occ_loss(occ_pred, occ_gt)
        l_traj = traj_loss(traj_pred, traj_t)
        prob   = torch.sigmoid(occ_pred)
        inter  = (prob * occ_gt).sum()
        dice   = 1 - (2*inter+1)/(prob.sum()+occ_gt.sum()+1)
        loss   = l_occ + 0.5*dice + 0.3*l_traj

        ade = (traj_pred - traj_t).norm(dim=-1).mean()
        fde = (traj_pred[:,-1] - traj_t[:,-1]).norm(dim=-1).mean()
        return loss, ade, fde

    def training_step(self, batch, _):
        loss, ade, _ = self._step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/ADE",  ade,  prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, ade, fde = self._step(batch)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/ADE",  ade,  prog_bar=True)
        self.log("val/FDE",  fde,  prog_bar=True)

    def configure_optimizers(self):
        new_params, old_params = [], []
        for name, p in self.model.named_parameters():
            if any(k in name for k in
                   ["geo_lifter","gate","bev_pool","alpha","geo_dec","stem"]):
                new_params.append(p)
            else:
                old_params.append(p)
        opt = torch.optim.AdamW([
            {"params": new_params, "lr": self.lr},
            {"params": old_params, "lr": self.lr * 0.1},
        ], weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
        return [opt], [sch]


def load_warmstart(model, ckpt_path):
    """Load v11 weights into v12, skipping size-mismatched layers."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw  = ckpt.get("state_dict", ckpt)

    # Strip prefix
    sd = {}
    for k, v in raw.items():
        if   k.startswith("model."):     sd[k[6:]]  = v
        elif k.startswith("lit.model."): sd[k[10:]] = v
        elif k.startswith("lit."):       sd[k[4:]]  = v
        else:                            sd[k]       = v

    model_sd = model.state_dict()
    filtered = {}
    skipped  = []
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    model_sd.update(filtered)
    model.load_state_dict(model_sd)
    print(f"  Loaded {len(filtered)} layers | Skipped {len(skipped)} (shape mismatch)")
    if skipped:
        print(f"  Skipped keys: {skipped[:5]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",   default="artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", default="artifacts/nuscenes_labels_128")
    ap.add_argument("--nusc_root",  default="data/nuscenes")
    ap.add_argument("--warmstart",  default="artifacts/checkpoints_v11_temporal/best_val_ade.ckpt")
    ap.add_argument("--max_epochs", type=int, default=60)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--out_dir",    default="artifacts/checkpoints_v12_geolift")
    args = ap.parse_args()

    nusc = NuScenes(version="v1.0-mini", dataroot=args.nusc_root, verbose=False)
    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    scene_key  = "scene" if "scene" in rows[0] else "scene_name"
    train_rows = [r for r in rows if r.get(scene_key,"") not in VAL_SCENES]
    val_rows   = [r for r in rows if r.get(scene_key,"") in VAL_SCENES]
    print(f"Train: {len(train_rows)} | Val: {len(val_rows)}")

    train_ds = NuScenesV12Dataset(train_rows, args.label_root, nusc, augment=True)
    val_ds   = NuScenesV12Dataset(val_rows,   args.label_root, nusc, augment=False)
    train_dl = DataLoader(train_ds, batch_size=2, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=0)

    lit = LitV12(lr=args.lr)

    if args.warmstart and Path(args.warmstart).exists():
        print(f"Warm-starting from: {args.warmstart}")
        load_warmstart(lit.model, args.warmstart)

    ckpt_cb = ModelCheckpoint(
        dirpath=args.out_dir,
        filename="best_val_ade",
        monitor="val/ADE", mode="min", save_last=True)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        callbacks=[ckpt_cb],
        log_every_n_steps=10,
    )

    print("Training v12 — Geometric BEV Lifting (Step 4 compliance)")
    trainer.fit(lit, train_dl, val_dl)
    print(f"\nBest checkpoint: {ckpt_cb.best_model_path}")
    print(f"Best val/ADE:    {ckpt_cb.best_model_score:.4f}")

if __name__ == "__main__":
    main()
