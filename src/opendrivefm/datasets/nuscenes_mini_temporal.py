"""
nuscenes_mini_temporal.py — Multi-frame dataloader (T=4).

Returns 4 consecutive frames per sample, with BEV feature warping metadata.
Each sample is anchored at the CURRENT frame (t=0); previous 3 frames are t-1, t-2, t-3.

Returns (T=4 variant):
    x:           (V=6, T=4, C=3, H, W)  — 4 frames, newest last
    occ:         (1, BEV_H, BEV_W)       — occupancy at t=0
    traj:        (12, 2)                 — future trajectory from t=0
    motion:      (3,)                    — [dt, vx, vy] at t=0
    t_rel:       (12,)                   — future time offsets
    K:           (V, 3, 3)              — camera intrinsics
    T_ego_cam:   (V, 4, 4)              — ego-to-cam transforms at t=0
    ego_deltas:  (T-1, 3)               — [dx, dy, dyaw] for warping frames t-1..t-3 into t=0

Drop this in as src/opendrivefm/data/nuscenes_mini_temporal.py
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion
    HAS_NUSCENES = True
except ImportError:
    HAS_NUSCENES = False

CAMS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK",  "CAM_BACK_LEFT",  "CAM_BACK_RIGHT",
]


def _lidar_sd(nusc, sample):
    return nusc.get("sample_data", sample["data"]["LIDAR_TOP"])


def _ego_pose(nusc, sd):
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    R = Quaternion(pose["rotation"]).rotation_matrix.astype(np.float32)
    t = np.array(pose["translation"], dtype=np.float32)
    return R, t


def _yaw_from_R(R: np.ndarray) -> float:
    """Extract yaw angle (rotation around Z) from 3×3 rotation matrix."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def get_ego_delta(nusc, sample_curr, sample_prev) -> np.ndarray:
    """
    Returns [dx, dy, dyaw] that transforms points FROM sample_prev ego frame
    INTO sample_curr ego frame.
    Used to warp BEV features from past frames into current frame.
    """
    sd_curr = _lidar_sd(nusc, sample_curr)
    sd_prev = _lidar_sd(nusc, sample_prev)

    R_curr, t_curr = _ego_pose(nusc, sd_curr)
    R_prev, t_prev = _ego_pose(nusc, sd_prev)

    # prev → global → curr
    # displacement in current ego frame
    disp_global = t_prev - t_curr                  # (3,)
    disp_ego    = R_curr.T @ disp_global            # (3,)

    yaw_curr = _yaw_from_R(R_curr)
    yaw_prev = _yaw_from_R(R_prev)
    dyaw     = float(yaw_prev - yaw_curr)
    # wrap to [-pi, pi]
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi

    return np.array([disp_ego[0], disp_ego[1], dyaw], dtype=np.float32)


class NuScenesMiniTemporal(Dataset):
    """
    Multi-frame (T=4) dataset for temporal BEV fusion.
    
    Samples with fewer than T previous frames still work — missing frames
    are filled by repeating the oldest available frame (zero-motion padding).
    
    Parameters
    ----------
    n_frames : int
        Number of temporal frames to return (default 4).
    label_root : str
        Path to 128×128 label directory (use artifacts/nuscenes_labels_128).
    """

    def __init__(
        self,
        manifest: str,
        *,
        image_hw:    Tuple[int, int] = (90, 160),
        n_frames:    int  = 4,
        label_root:  str  = "artifacts/nuscenes_labels_128",
        nusc_root:   str  = "data/nuscenes",
        nusc_version:str  = "v1.0-mini",
        return_lidar:bool = True,
        augment:     bool = False,
    ):
        if not HAS_NUSCENES:
            raise ImportError("nuscenes-devkit required")

        self.rows        = [json.loads(l)
                            for l in Path(manifest).read_text().splitlines() if l.strip()]
        self.image_hw    = image_hw
        self.n_frames    = n_frames
        self.label_root  = Path(label_root)
        self.return_lidar= return_lidar
        self.augment     = augment

        self._nusc_root    = nusc_root
        self._nusc_version = nusc_version
        self._nusc: Optional["NuScenes"] = None

        self.tf = T.Compose([
            T.Resize(image_hw, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

    @property
    def nusc(self):
        if self._nusc is None:
            self._nusc = NuScenes(version=self._nusc_version,
                                  dataroot=self._nusc_root, verbose=False)
        return self._nusc

    def __len__(self):
        return len(self.rows)

    def _load_images(self, sample: dict) -> torch.Tensor:
        """Returns (V, 1, 3, H, W) for a single sample."""
        imgs = []
        for cam in CAMS:
            sd = self.nusc.get("sample_data", sample["data"][cam])
            p  = Path(self.nusc.dataroot) / sd["filename"]
            im = Image.open(p).convert("RGB")
            imgs.append(self.tf(im))
        return torch.stack(imgs, dim=0).unsqueeze(1)   # (V, 1, 3, H, W)

    def _get_calib(self, sample: dict):
        H, W = self.image_hw
        Ks, Ts = [], []
        for cam in CAMS:
            sd = self.nusc.get("sample_data", sample["data"][cam])
            cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            K  = np.array(cs["camera_intrinsic"], dtype=np.float32)
            K[0] *= W / 1600.0; K[1] *= H / 900.0
            Ks.append(K)
            R  = Quaternion(cs["rotation"]).rotation_matrix.astype(np.float32)
            t  = np.array(cs["translation"], dtype=np.float32)
            T4 = np.eye(4, dtype=np.float32); T4[:3,:3] = R; T4[:3,3] = t
            Ts.append(np.linalg.inv(T4))
        return torch.from_numpy(np.stack(Ks)), torch.from_numpy(np.stack(Ts))

    def __getitem__(self, idx: int):
        r0  = self.rows[idx]
        tok = r0["sample_token"]
        s0  = self.nusc.get("sample", tok)

        # ── Collect T frames (current + T-1 previous) ───────────────────────
        samples = [s0]
        s = s0
        for _ in range(self.n_frames - 1):
            prev = s.get("prev", "")
            if prev:
                s = self.nusc.get("sample", prev)
                samples.append(s)
            else:
                samples.append(samples[-1])   # repeat oldest available

        samples = list(reversed(samples))     # oldest first → [t-(T-1), ..., t=0]

        # ── Load images for all T frames ────────────────────────────────────
        frame_tensors = [self._load_images(s) for s in samples]
        # Stack: (V, T, 3, H, W)
        x = torch.cat(frame_tensors, dim=1)

        # ── Labels at t=0 ───────────────────────────────────────────────────
        z       = np.load(self.label_root / f"{tok}.npz")
        occ     = torch.from_numpy(z["occ"]).float()
        traj    = torch.from_numpy(z["traj"]).float()
        t_rel   = torch.from_numpy(z["t_rel"]).float() if "t_rel" in z.files \
                  else torch.arange(1, traj.shape[0]+1, dtype=torch.float32) * 0.5
        dt_prev = float(z["dt_prev"])  if "dt_prev"  in z.files else 0.0
        vxy     = z["vxy_prev"].astype(np.float32) if "vxy_prev" in z.files \
                  else np.zeros(2, np.float32)
        motion  = torch.tensor([dt_prev, vxy[0], vxy[1]], dtype=torch.float32)

        # ── Camera calibration at t=0 ────────────────────────────────────────
        K, T_ego = self._get_calib(s0)

        # ── Ego-motion deltas for BEV warping ────────────────────────────────
        # For each past frame, compute transform into t=0 ego frame
        # ego_deltas[i] = transform from frame i (oldest first) into t=0
        ego_deltas = []
        for s_past in samples[:-1]:   # all except the last (=current) frame
            delta = get_ego_delta(self.nusc, s0, s_past)  # past→current
            ego_deltas.append(delta)
        # ego_deltas: (T-1, 3) = [dx, dy, dyaw] for each past frame
        ego_deltas_t = torch.from_numpy(np.stack(ego_deltas))

        return x, occ, traj, motion, t_rel, K, T_ego, ego_deltas_t
