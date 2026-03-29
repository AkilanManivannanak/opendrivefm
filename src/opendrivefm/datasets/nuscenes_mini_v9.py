"""
nuscenes_mini_v9.py — extends NuScenesMiniMultiView to return
per-camera sparse LiDAR depth maps for depth supervision.

Usage:
    ds = NuScenesMiniV9(manifest, label_root, nusc_root='data/nuscenes',
                        return_motion=True, return_trel=True, return_calib=True)

    # Returns (x, occ, traj, motion, t_rel, K, T_ego_cam, lidar_depth_maps)
    # lidar_depth_maps: (V, 1, H, W) float32, 0 = invalid pixel

Drop this file into src/opendrivefm/data/
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# nuScenes imports — only needed when lidar=True
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion
    HAS_NUSCENES = True
except ImportError:
    HAS_NUSCENES = False

CAMS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def _project_lidar_to_cam(
    nusc: "NuScenes",
    sample: dict,
    cam_name: str,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Project LiDAR points onto a single camera image plane.
    Returns sparse depth map (H, W) float32. 0 = no point projected here.
    Points behind camera or outside image are discarded.
    """
    # ── Get LiDAR pointcloud in LiDAR sensor frame ──────────────────────────
    lidar_sd   = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_cs   = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

    pc = LidarPointCloud.from_file(str(Path(nusc.dataroot) / lidar_sd["filename"]))
    pts = pc.points[:3, :].copy()       # (3, N)

    # LiDAR sensor → ego
    R_ls = Quaternion(lidar_cs["rotation"]).rotation_matrix
    t_ls = np.array(lidar_cs["translation"])
    pts  = R_ls @ pts + t_ls[:, None]

    # ego → global
    R_eg = Quaternion(lidar_pose["rotation"]).rotation_matrix
    t_eg = np.array(lidar_pose["translation"])
    pts  = R_eg @ pts + t_eg[:, None]

    # ── Get camera calibration ───────────────────────────────────────────────
    cam_sd   = nusc.get("sample_data", sample["data"][cam_name])
    cam_cs   = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    cam_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])

    # global → camera ego
    R_ce = Quaternion(cam_pose["rotation"]).rotation_matrix
    t_ce = np.array(cam_pose["translation"])
    pts  = R_ce.T @ (pts - t_ce[:, None])

    # camera ego → camera sensor frame
    R_cs = Quaternion(cam_cs["rotation"]).rotation_matrix
    t_cs = np.array(cam_cs["translation"])
    pts  = R_cs.T @ (pts - t_cs[:, None])

    # ── Keep points in front of camera ──────────────────────────────────────
    keep = pts[2, :] > 0.5           # depth > 0.5m
    pts  = pts[:, keep]
    if pts.shape[1] == 0:
        return np.zeros((img_h, img_w), dtype=np.float32)

    # ── Project to pixel coordinates ────────────────────────────────────────
    K = np.array(cam_cs["camera_intrinsic"], dtype=np.float32)   # (3,3)

    # nuScenes intrinsics are for the ORIGINAL 1600×900 image
    # Scale K to (img_w, img_h)
    orig_w, orig_h = 1600.0, 900.0
    K_scaled        = K.copy()
    K_scaled[0, :] *= img_w  / orig_w
    K_scaled[1, :] *= img_h  / orig_h

    uvd  = K_scaled @ pts               # (3, N)
    u    = uvd[0] / uvd[2]
    v    = uvd[1] / uvd[2]
    d    = uvd[2]

    # ── Rasterise: keep valid pixels ────────────────────────────────────────
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    valid = (ui >= 0) & (ui < img_w) & (vi >= 0) & (vi < img_h)
    ui, vi, d = ui[valid], vi[valid], d[valid]

    depth_map = np.zeros((img_h, img_w), dtype=np.float32)
    # Where multiple points project to same pixel, keep closest (min depth)
    # Sort by depth descending so closest overwrites
    order = np.argsort(d)[::-1]
    depth_map[vi[order], ui[order]] = d[order]

    return depth_map


class NuScenesMiniV9(Dataset):
    """
    Extended dataset that additionally returns per-camera LiDAR depth maps
    for depth supervision during training.

    Returns (when return_lidar=True):
        x, occ, traj, motion, t_rel, K, T_ego_cam, lidar_depth_maps

    lidar_depth_maps: (V=6, 1, H, W) float32  — 0 means no LiDAR point
    K:               (V=6, 3, 3)    float32  — scaled intrinsics
    T_ego_cam:       (V=6, 4, 4)    float32  — ego-to-cam transforms
    """

    def __init__(
        self,
        manifest: str,
        *,
        image_hw: Tuple[int, int] = (90, 160),
        label_root: str = "artifacts/nuscenes_labels",
        nusc_root: str  = "data/nuscenes",
        nusc_version: str = "v1.0-mini",
        return_motion: bool = True,
        return_trel: bool   = True,
        return_calib: bool  = True,
        return_lidar: bool  = True,
        augment: bool       = False,
    ):
        self.manifest_path = Path(manifest)
        self.rows = [
            json.loads(l)
            for l in self.manifest_path.read_text().splitlines()
            if l.strip()
        ]
        self.image_hw       = image_hw          # (H, W)
        self.label_root     = Path(label_root)
        self.return_motion  = return_motion
        self.return_trel    = return_trel
        self.return_calib   = return_calib
        self.return_lidar   = return_lidar
        self.augment        = augment

        self.tf = T.Compose([
            T.Resize(image_hw, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

        self._nusc: Optional["NuScenes"] = None
        self._nusc_root    = nusc_root
        self._nusc_version = nusc_version

        if return_lidar:
            if not HAS_NUSCENES:
                raise ImportError("nuscenes-devkit not installed. "
                                  "pip install nuscenes-devkit")
            # Validate the data root exists
            if not Path(nusc_root).exists():
                raise FileNotFoundError(
                    f"nuScenes data not found at {nusc_root}. "
                    "Check --nusc_root argument."
                )

    @property
    def nusc(self) -> "NuScenes":
        """Lazy-load NuScenes to avoid per-worker overhead."""
        if self._nusc is None:
            self._nusc = NuScenes(
                version=self._nusc_version,
                dataroot=self._nusc_root,
                verbose=False,
            )
        return self._nusc

    def __len__(self) -> int:
        return len(self.rows)

    # ── Calibration helpers ─────────────────────────────────────────────────

    def _get_calib(
        self, sample: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            Ks:        (V, 3, 3) float32 — scaled camera intrinsics
            T_ego_cam: (V, 4, 4) float32 — ego-to-cam extrinsics
        """
        H, W = self.image_hw
        Ks       = []
        T_ego_cams = []
        for cam in CAMS:
            sd  = self.nusc.get("sample_data", sample["data"][cam])
            cs  = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

            K = np.array(cs["camera_intrinsic"], dtype=np.float32)
            K[0, :] *= W / 1600.0
            K[1, :] *= H / 900.0
            Ks.append(K)

            R = Quaternion(cs["rotation"]).rotation_matrix.astype(np.float32)
            t = np.array(cs["translation"], dtype=np.float32)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3,  3] = t
            T_ego_cams.append(np.linalg.inv(T))   # ego→cam

        return np.stack(Ks), np.stack(T_ego_cams)

    # ── LiDAR depth maps ────────────────────────────────────────────────────

    def _get_lidar_depth_maps(self, sample: dict) -> np.ndarray:
        """Returns (V, H, W) float32 sparse depth maps."""
        H, W = self.image_hw
        maps = []
        for cam in CAMS:
            dm = _project_lidar_to_cam(self.nusc, sample, cam, H, W)
            maps.append(dm)
        return np.stack(maps)   # (V, H, W)

    # ── Main __getitem__ ────────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        r   = self.rows[idx]
        tok = r["sample_token"]

        # Images
        imgs: List[torch.Tensor] = []
        for cam in CAMS:
            p  = Path(r["cams"][cam])
            im = Image.open(p).convert("RGB")
            imgs.append(self.tf(im))
        x = torch.stack(imgs, dim=0).unsqueeze(1)   # (6,1,3,H,W)

        # Labels
        z   = np.load(self.label_root / f"{tok}.npz")
        occ = torch.from_numpy(z["occ"]).float()
        traj= torch.from_numpy(z["traj"]).float()

        t_rel = (
            torch.from_numpy(z["t_rel"]).float()
            if "t_rel" in z.files
            else torch.arange(1, traj.shape[0]+1, dtype=torch.float32) * 0.5
        )
        dt_prev  = float(z["dt_prev"])  if "dt_prev"  in z.files else 0.0
        vxy_prev = z["vxy_prev"].astype(np.float32) if "vxy_prev" in z.files else np.zeros(2, np.float32)
        motion   = torch.tensor([dt_prev, vxy_prev[0], vxy_prev[1]], dtype=torch.float32)

        if not (self.return_calib or self.return_lidar):
            return x, occ, traj, motion, t_rel

        sample = self.nusc.get("sample", tok)
        K_np, T_np = self._get_calib(sample)
        K       = torch.from_numpy(K_np)
        T_ego   = torch.from_numpy(T_np)

        if not self.return_lidar:
            return x, occ, traj, motion, t_rel, K, T_ego

        # Sparse LiDAR depth maps
        ldm_np = self._get_lidar_depth_maps(sample)   # (V, H, W)
        ldm    = torch.from_numpy(ldm_np).unsqueeze(1) # (V, 1, H, W)

        return x, occ, traj, motion, t_rel, K, T_ego, ldm
