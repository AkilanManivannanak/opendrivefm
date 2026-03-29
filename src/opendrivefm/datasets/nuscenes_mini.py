"""
NuScenesMiniMultiView Dataset
Returns 6-camera tensors + labels + optional calibration (K, T_ego_cam).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

CAMS = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT",
        "CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]


class NuScenesMiniMultiView(Dataset):
    """
    Returns:
      x:      (V=6, T=1, C=3, H, W) float32 [0,1]
      occ:    (1, 64, 64) float32 {0,1}
      traj:   (12, 2) float32 meters in ego frame
      motion: (3,) [dt_prev, vx, vy]     if return_motion
      t_rel:  (12,) seconds              if return_trel
      K_v:    (V, 3, 3) float32          if return_calib
      T_v:    (V, 4, 4) float32          if return_calib  (T_ego_cam)
    """

    def __init__(
        self,
        manifest: str,
        image_size: Tuple[int,int] = (160, 90),
        *,
        image_hw: Optional[Tuple[int,int]] = None,
        frames: int = 1,
        label_root: str = "artifacts/nuscenes_labels",
        return_motion: bool = False,
        return_trel: bool = False,
        return_calib: bool = False,
        augment: bool = False,
    ):
        self.manifest = Path(manifest)
        if not self.manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest}")
        self.rows = [json.loads(l) for l in self.manifest.read_text().splitlines() if l.strip()]
        if not self.rows:
            raise ValueError(f"Manifest is empty: {self.manifest}")

        self.frames         = int(frames)
        self.label_root     = Path(label_root)
        self.return_motion  = return_motion
        self.return_trel    = return_trel
        self.return_calib   = return_calib
        self.augment        = augment

        size_hw = image_hw if image_hw is not None else image_size
        self.tf = T.Compose([
            T.Resize(size_hw, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

        # Check whether manifest has calibration
        self._has_calib = "intrinsics" in self.rows[0] if self.rows else False

        if return_calib and not self._has_calib:
            raise ValueError(
                "return_calib=True but manifest has no intrinsics/extrinsics. "
                "Re-run prepare_nuscenes_mini.py to rebuild manifest with calibration."
            )

        if augment:
            from opendrivefm.models.augmentations import MultiCameraAugment
            self._aug = MultiCameraAugment()
        else:
            self._aug = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r   = self.rows[idx]
        tok = r["sample_token"]

        # ── Load images ─────────────────────────────────────────
        imgs: List[torch.Tensor] = []
        for cam in CAMS:
            p = Path(r["cams"][cam])
            if not p.exists():
                raise FileNotFoundError(f"Missing image for {cam}: {p}")
            imgs.append(self.tf(Image.open(p).convert("RGB")))
        x = torch.stack(imgs, dim=0).unsqueeze(1).contiguous()  # (6,1,3,H,W)

        # Augment during training
        if self._aug is not None:
            x = self._aug(x, training=True)

        # ── Load labels ─────────────────────────────────────────
        lab = np.load(self.label_root / f"{tok}.npz")
        occ  = torch.from_numpy(lab["occ"]).float()
        traj = torch.from_numpy(lab["traj"]).float()
        t_rel = (torch.from_numpy(lab["t_rel"]).float()
                 if "t_rel" in lab.files
                 else torch.arange(1, traj.shape[0]+1, dtype=torch.float32) * 0.5)

        dt_prev  = float(lab["dt_prev"])  if "dt_prev"  in lab.files else 0.0
        vxy_prev = lab["vxy_prev"].astype(np.float32) if "vxy_prev" in lab.files else np.zeros(2, np.float32)
        motion   = torch.tensor([dt_prev, float(vxy_prev[0]), float(vxy_prev[1])], dtype=torch.float32)

        # ── Calibration ─────────────────────────────────────────
        if self.return_calib and self._has_calib:
            K_v = torch.tensor(
                [r["intrinsics"][cam] for cam in CAMS], dtype=torch.float32)  # (6,3,3)
            T_v = torch.tensor(
                [r["extrinsics"][cam] for cam in CAMS], dtype=torch.float32)  # (6,4,4)

        # ── Return ───────────────────────────────────────────────
        base = [x, occ, traj]
        if self.return_motion:  base.append(motion)
        if self.return_trel:    base.append(t_rel)
        if self.return_calib and self._has_calib:
            base.extend([K_v, T_v])
        return tuple(base) if len(base) > 1 else base[0]
