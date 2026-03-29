"""
Geometry Module: Back-Projection / BEV Builder  (Slide Step 4)
==============================================================
Takes per-camera feature maps F_v and camera intrinsics/extrinsics,
lifts 2D features into 3D frustum (depth bins / ray casting),
then splats/pools them into a BEV grid aligned to the ego frame.

Input:
  F_v   : (B, V, C, H, W)  per-camera feature maps
  K_v   : (B, V, 3, 3)     camera intrinsics
  T_ego_cam_v : (B, V, 4, 4)  extrinsics (cam→ego transform)

Output:
  BEV_v : (B, C, bev_h, bev_w)  fused BEV feature tensor
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrustumGrid(nn.Module):
    """
    Pre-computes a frustum of 3D sample points for each camera,
    then uses grid_sample to lift image features into 3D voxels.
    Depth bins are learnable via a small depth head.
    """

    def __init__(
        self,
        d_feat: int = 64,
        n_depth: int = 16,
        bev_h: int = 64,
        bev_w: int = 64,
        extent_m: float = 20.0,
        z_min: float = -1.0,
        z_max: float = 3.0,
    ):
        super().__init__()
        self.n_depth  = n_depth
        self.bev_h    = bev_h
        self.bev_w    = bev_w
        self.extent_m = extent_m
        self.z_min    = z_min
        self.z_max    = z_max
        self.d_feat   = d_feat

        # Learnable depth distribution head (predicts soft depth bins)
        self.depth_head = nn.Sequential(
            nn.Conv2d(d_feat, d_feat, 3, padding=1),
            nn.BatchNorm2d(d_feat),
            nn.GELU(),
            nn.Conv2d(d_feat, n_depth, 1),
        )

        # BEV pooling: reduce depth dimension to BEV feature
        self.bev_pool = nn.Sequential(
            nn.Conv2d(d_feat, d_feat, 3, padding=1),
            nn.BatchNorm2d(d_feat),
            nn.GELU(),
        )

    def forward(
        self,
        feat_v: torch.Tensor,         # (B, V, C, Hf, Wf)
        K_v: torch.Tensor,             # (B, V, 3, 3)
        T_ego_cam_v: torch.Tensor,     # (B, V, 4, 4)
    ) -> torch.Tensor:                 # (B, C, bev_h, bev_w)

        B, V, C, Hf, Wf = feat_v.shape
        device = feat_v.device

        # Build BEV grid of 3D points in ego frame: (bev_h, bev_w, 3)
        xs = torch.linspace(-self.extent_m, self.extent_m, self.bev_w, device=device)
        ys = torch.linspace(-self.extent_m, self.extent_m, self.bev_h, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (bev_h, bev_w)
        z_levels = torch.linspace(self.z_min, self.z_max, self.n_depth, device=device)

        # (n_depth, bev_h, bev_w, 3) in ego frame
        pts_ego = torch.stack([
            grid_x.unsqueeze(0).expand(self.n_depth, -1, -1),
            grid_y.unsqueeze(0).expand(self.n_depth, -1, -1),
            z_levels[:, None, None].expand(-1, self.bev_h, self.bev_w),
        ], dim=-1)  # (D, H, W, 3)

        pts_ego_h = torch.cat([
            pts_ego,
            torch.ones(*pts_ego.shape[:-1], 1, device=device)
        ], dim=-1)  # (D, H, W, 4)

        pts_flat = pts_ego_h.view(-1, 4).T  # (4, D*H*W)

        bev_acc = torch.zeros(B, C, self.bev_h, self.bev_w, device=device)
        count   = torch.zeros(B, 1, self.bev_h, self.bev_w, device=device)

        for v in range(V):
            # Transform ego 3D points → camera frame
            T_cam_ego = torch.linalg.inv(T_ego_cam_v[:, v])  # (B, 4, 4)
            K          = K_v[:, v]                             # (B, 3, 3)

            # (B, 4, N) → camera coords
            pts_cam = torch.bmm(T_cam_ego, pts_flat.unsqueeze(0).expand(B, -1, -1))  # (B,4,N)
            pts_cam = pts_cam[:, :3]  # (B, 3, N)

            # Project to pixel coords
            pts_img = torch.bmm(K, pts_cam)  # (B, 3, N)
            depth   = pts_img[:, 2:3].clamp(min=1e-4)
            uv      = pts_img[:, :2] / depth  # (B, 2, N)

            # Normalise to [-1, 1] for grid_sample
            # uv in pixel space, image is (Hf, Wf)
            uv_norm = torch.stack([
                uv[:, 0] / (Wf - 1) * 2 - 1,   # x
                uv[:, 1] / (Hf - 1) * 2 - 1,   # y
            ], dim=-1)  # (B, N, 2)

            N = uv_norm.shape[1]
            uv_norm = uv_norm.view(B, self.n_depth, self.bev_h * self.bev_w, 2)

            # Sample features from this camera's feature map for each depth level
            fv = feat_v[:, v]  # (B, C, Hf, Wf)

            # Per-depth sampling: for each depth level, sample the feature map
            sampled_list = []
            for d in range(self.n_depth):
                grid_d = uv_norm[:, d:d+1]  # (B, 1, bev_h*bev_w, 2)
                grid_d = grid_d.view(B, 1, self.bev_h * self.bev_w, 2)
                s = F.grid_sample(fv, grid_d, align_corners=True,
                                  mode="bilinear", padding_mode="zeros")  # (B, C, 1, N)
                sampled_list.append(s.squeeze(2))  # (B, C, N)

            # Stack: (B, C, D, N) then mean over depth
            sampled = torch.stack(sampled_list, dim=2)  # (B, C, D, N)

            # Compute depth weights from learned depth head
            depth_logits = self.depth_head(fv)  # (B, n_depth, Hf, Wf)
            depth_w_flat = uv_norm[:, :, :, :1] * 0  # placeholder
            depth_w = torch.softmax(depth_logits.mean(dim=[2, 3]), dim=1)  # (B, D)
            depth_w = depth_w[:, :, None]  # (B, D, 1)

            # Weighted sum over depth: (B, C, N)
            lifted = (sampled * depth_w.unsqueeze(1)).sum(dim=2)  # (B, C, N)
            lifted = lifted.view(B, C, self.bev_h, self.bev_w)

            # Mask: only accumulate points with valid projection
            valid_u = (uv[:, 0] >= 0) & (uv[:, 0] <= Wf - 1)
            valid_v = (uv[:, 1] >= 0) & (uv[:, 1] <= Hf - 1)
            valid_z = pts_cam[:, 2] > 0.5
            valid_mask = (valid_u & valid_v & valid_z)  # (B, N)
            valid_bev  = valid_mask.view(B, 1, self.bev_h, self.bev_w).float()

            bev_acc += lifted * valid_bev
            count   += valid_bev

        # Average over contributing cameras
        bev = bev_acc / (count + 1e-6)
        return self.bev_pool(bev)  # (B, C, bev_h, bev_w)


class BEVDecoder(nn.Module):
    """
    BEV CNN Decoder + Segmentation Head  (Slide Step 6 / Func 5)
    Input:  BEV_fused (B, C, bev_h, bev_w)
    Output: occ_pred  (B, 1, bev_h, bev_w)  occupancy logits
    """
    def __init__(self, in_ch: int = 64, bev_h: int = 64, bev_w: int = 64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),   nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 64,  3, padding=1),   nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64,  32,  3, padding=1),   nn.BatchNorm2d(32),  nn.GELU(),
            nn.Conv2d(32,  1,   1),  # segmentation head → logits
        )

    def forward(self, bev_fused: torch.Tensor) -> torch.Tensor:
        return self.decoder(bev_fused)


class GeometryAwareBackbone(nn.Module):
    """
    Full geometry-aware pipeline:
      images → CNN encoder → back-projection → BEV grid → occupancy logits
    This replaces the MLP-based BEVOccupancyHead for geometry-aware lifting.

    Note: requires calibration data (K_v, T_ego_cam_v) to be passed in.
    Use OpenDriveFMGeo (below) as the top-level model when using this.
    """
    def __init__(self, d_feat: int = 64, n_depth: int = 16,
                 bev_h: int = 64, bev_w: int = 64, extent_m: float = 20.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, d_feat // 2, 7, stride=2, padding=3),
            nn.BatchNorm2d(d_feat // 2), nn.GELU(),
            nn.Conv2d(d_feat // 2, d_feat, 3, stride=2, padding=1),
            nn.BatchNorm2d(d_feat), nn.GELU(),
        )  # output: (B*V, d_feat, H/4, W/4)
        self.frustum = FrustumGrid(d_feat=d_feat, n_depth=n_depth,
                                   bev_h=bev_h, bev_w=bev_w, extent_m=extent_m)
        self.bev_decoder = BEVDecoder(in_ch=d_feat, bev_h=bev_h, bev_w=bev_w)

    def forward(
        self,
        x: torch.Tensor,               # (B, V, 1, 3, H, W)
        K_v: torch.Tensor,             # (B, V, 3, 3)
        T_ego_cam_v: torch.Tensor,     # (B, V, 4, 4)
    ) -> torch.Tensor:                 # (B, 1, bev_h, bev_w) occupancy logits

        B, V, T, C, H, W = x.shape
        imgs = x[:, :, 0]  # first frame: (B, V, 3, H, W)

        # Encode each camera independently
        imgs_flat = imgs.view(B * V, C, H, W)
        feat_flat = self.encoder(imgs_flat)                  # (B*V, d_feat, Hf, Wf)
        _, d_feat, Hf, Wf = feat_flat.shape
        feat_v    = feat_flat.view(B, V, d_feat, Hf, Wf)    # (B, V, d_feat, Hf, Wf)

        # Back-project to BEV
        bev_fused = self.frustum(feat_v, K_v, T_ego_cam_v)  # (B, d_feat, bev_h, bev_w)

        # Decode to occupancy
        occ_logits = self.bev_decoder(bev_fused)             # (B, 1, bev_h, bev_w)
        return occ_logits, bev_fused
