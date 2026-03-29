"""
model_lss_bev.py — Step 4 compliant Lift-Splat-Shoot BEV module
Implements ALL three Step 4 requirements:
  (i)  Camera intrinsics/extrinsics + ego pose → pixel rays → frustum points
  (ii) Depth bins (D=32) to lift 2D features into 3D, then splat/pool into BEV grid
  (iii) Per-camera BEV feature tensors aligned to ego-centric coordinate system

Copy to ~/opendrivefm/src/opendrivefm/models/model_lss_bev.py
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LSSGeometricBEV(nn.Module):
    """
    Lift-Splat-Shoot (LSS) style geometric BEV construction.

    Step 4 Requirements:
      (i)  Uses camera intrinsics K and extrinsics T_cam2ego to map
           image pixels → pixel rays → 3D frustum points in ego frame
      (ii) D=32 discrete depth bins partition each ray. A depth distribution
           head predicts per-pixel depth bin probabilities. Features are
           lifted into 3D voxels and splatted (pillar pooling) into BEV grid
      (iii) Each camera produces a (bev_ch, bev_h, bev_w) BEV tensor in the
            ego-centric coordinate system. All 6 cameras are summed with
            trust weighting into a single unified BEV representation.

    Architecture:
      Input:  per-camera feature maps (B, V, C, Hf, Wf)
      Output: BEV feature map (B, bev_ch, bev_h, bev_w)
    """

    def __init__(
        self,
        feat_ch: int = 64,
        bev_ch: int  = 64,
        bev_h: int   = 64,
        bev_w: int   = 64,
        d_min: float = 1.0,
        d_max: float = 50.0,
        n_depth: int = 32,
        extent_m: float = 20.0,
    ):
        super().__init__()
        self.bev_h    = bev_h
        self.bev_w    = bev_w
        self.n_depth  = n_depth
        self.d_min    = d_min
        self.d_max    = d_max
        self.extent_m = extent_m
        self.feat_ch  = feat_ch
        self.bev_ch   = bev_ch

        # (ii) Depth distribution head — predicts per-pixel depth bin logits
        self.depth_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, n_depth, 1),   # (B*V, D, Hf, Wf)
        )

        # Feature channel reducer before splatting
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_ch, bev_ch, 1),
            nn.BatchNorm2d(bev_ch),
            nn.ReLU(inplace=True),
        )

        # BEV refinement after aggregation across all cameras
        self.bev_refine = nn.Sequential(
            nn.Conv2d(bev_ch, bev_ch, 3, padding=1),
            nn.BatchNorm2d(bev_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_ch, bev_ch, 3, padding=1),
            nn.BatchNorm2d(bev_ch),
            nn.ReLU(inplace=True),
        )

        # Pre-compute discrete depth bin centres (D,)
        depths = torch.linspace(d_min, d_max, n_depth)
        self.register_buffer("depths", depths)

    # ── Step 4(i): pixel rays → frustum points ────────────────────────────────

    def _make_frustum(self, Hf: int, Wf: int, K_inv: torch.Tensor,
                      device: torch.device) -> torch.Tensor:
        """
        Create 3D frustum points in camera frame for every pixel × depth bin.

        Uses camera intrinsics (via K_inv) to back-project pixel coordinates
        into normalised ray directions, then scales each ray by D depth bins.

        Args:
            Hf, Wf : feature map height/width
            K_inv  : (3,3) inverse intrinsic matrix (scaled to feature map)
        Returns:
            pts_cam: (Hf*Wf*D, 3) 3D points in camera coordinate frame
        """
        # (i) Build pixel coordinate grid
        ys = torch.arange(Hf, dtype=torch.float32, device=device)
        xs = torch.arange(Wf, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')       # (Hf, Wf)
        ones   = torch.ones_like(xx)
        # Homogeneous pixel coords: (3, N)  where N = Hf*Wf
        pix_h  = torch.stack([xx.flatten(), yy.flatten(),
                               ones.flatten()], dim=0)        # (3, N)

        # (i) Ray direction in camera frame via inverse intrinsic
        rays = K_inv @ pix_h                                  # (3, N)

        # (ii) Scale each ray by D depth bin centres → (3, N, D)
        rays = rays.unsqueeze(2) * self.depths.view(1, 1, -1)

        # Reshape to (N*D, 3) — one 3D point per pixel per depth bin
        pts_cam = rays.permute(1, 2, 0).reshape(-1, 3)        # (N*D, 3)
        return pts_cam

    # ── Step 4(i): camera frame → ego frame ──────────────────────────────────

    @staticmethod
    def _cam_to_ego(pts_cam: torch.Tensor, R: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        """
        Transform frustum points from camera frame to ego-centric frame.

        Uses extrinsic rotation R (3×3) and translation t (3,):
            pts_ego = R @ pts_cam + t

        Args:
            pts_cam: (N, 3) points in camera frame
            R       : (3,3) rotation matrix  (cam → ego)
            t       : (3,)  translation       (cam → ego)
        Returns:
            pts_ego: (N, 3) points in ego-centric coordinate system  [Step 4(iii)]
        """
        return (R @ pts_cam.T).T + t.unsqueeze(0)            # (N, 3)

    # ── Step 4(ii): splat features onto BEV grid ──────────────────────────────

    def _splat(self, pts_ego: torch.Tensor,
               feat_vox: torch.Tensor,
               trust_w: float) -> torch.Tensor:
        """
        Splat voxel features onto BEV grid via pillar pooling (scatter-add).

        Each 3D point in ego frame is projected onto the BEV plane (X-Y).
        Features are accumulated (summed) into the corresponding BEV cell,
        weighted by the camera's trust score.

        Args:
            pts_ego  : (N, 3) — 3D points in ego frame
            feat_vox : (bev_ch, N) — feature for each frustum point
            trust_w  : scalar camera trust weight
        Returns:
            bev: (bev_ch, bev_h, bev_w) — per-camera BEV feature tensor [Step 4(iii)]
        """
        bev = torch.zeros(self.bev_ch, self.bev_h, self.bev_w,
                          device=feat_vox.device)

        # Map ego X (forward) → BEV row, ego Y (left) → BEV col
        x_ego = pts_ego[:, 0]
        y_ego = pts_ego[:, 1]

        s = (2.0 * self.extent_m)
        col = ((x_ego + self.extent_m) / s * (self.bev_w - 1)).long()
        row = ((y_ego + self.extent_m) / s * (self.bev_h - 1)).long()

        valid = (col >= 0) & (col < self.bev_w) & \
                (row >= 0) & (row < self.bev_h)
        col = col[valid]
        row = row[valid]
        fv  = feat_vox[:, valid]                              # (bev_ch, N_valid)

        # Scatter-add into BEV grid
        idx      = row * self.bev_w + col                     # (N_valid,)
        bev_flat = bev.view(self.bev_ch, -1)
        bev_flat.scatter_add_(1,
            idx.unsqueeze(0).expand(self.bev_ch, -1),
            fv * trust_w)
        return bev

    # ── Forward: full geometric BEV pipeline ──────────────────────────────────

    def forward(
        self,
        feat_maps:  torch.Tensor,   # (B, V, feat_ch, Hf, Wf)
        K_list:     torch.Tensor,   # (B, V, 3, 3) — camera intrinsics
        T_list:     torch.Tensor,   # (B, V, 4, 4) — cam→ego extrinsics
        trust:      torch.Tensor,   # (B, V)        — trust scores
    ) -> torch.Tensor:
        """
        Full Step 4 geometric BEV construction:

        For each batch and each camera:
          1. (i)  Compute pixel ray directions using K_inv
          2. (ii) Predict depth distribution with depth_head
          3. (ii) Multiply depth probs × projected features → per-voxel features
          4. (i)  Transform frustum points cam→ego using extrinsics T
          5. (ii) Splat (scatter-add) features onto BEV grid
          6. (iii) Sum all per-camera BEV tensors (trust-weighted) → unified BEV

        Returns:
            bev: (B, bev_ch, bev_h, bev_w) — ego-centric BEV feature map
        """
        B, V, C, Hf, Wf = feat_maps.shape
        device = feat_maps.device

        # Project features and predict depth distributions
        fm_flat     = feat_maps.view(B*V, C, Hf, Wf)
        depth_logits = self.depth_head(fm_flat)               # (B*V, D, Hf, Wf)
        depth_probs  = depth_logits.softmax(dim=1)            # (B*V, D, Hf, Wf)
        feat_proj    = self.feat_proj(fm_flat)                # (B*V, bev_ch, Hf, Wf)

        # (ii) Outer product: depth_probs × feat_proj → voxel features
        # Shape: (B*V, bev_ch, D, Hf, Wf)
        feat_vox = feat_proj.unsqueeze(2) * depth_probs.unsqueeze(1)

        # Accumulate per-camera BEV tensors
        bev_out = torch.zeros(B, self.bev_ch, self.bev_h, self.bev_w,
                              device=device)

        for b in range(B):
            bev_b = torch.zeros(self.bev_ch, self.bev_h, self.bev_w,
                                device=device)
            for v in range(V):
                K   = K_list[b, v].float()                    # (3,3)
                T   = T_list[b, v].float()                    # (4,4)
                R   = T[:3, :3]                               # cam→ego rotation
                t   = T[:3,  3]                               # cam→ego translation
                K_inv = torch.linalg.inv(K)

                # Scale K_inv to feature map resolution
                K_inv_scaled = K_inv.clone()
                K_inv_scaled[0] *= Wf  # scale x
                K_inv_scaled[1] *= Hf  # scale y

                # (i) Frustum points in camera frame
                pts_cam = self._make_frustum(Hf, Wf, K_inv_scaled, device)

                # (i) Transform to ego frame using extrinsics [Step 4(iii)]
                pts_ego = self._cam_to_ego(pts_cam, R, t)

                # (ii) Flatten voxel features for this camera: (bev_ch, Hf*Wf*D)
                fv = feat_vox[b*V + v]                        # (bev_ch, D, Hf, Wf)
                fv = fv.permute(0, 2, 3, 1).reshape(self.bev_ch, -1)

                # (ii) Splat onto per-camera BEV, weighted by trust [Step 4(iii)]
                tw   = trust[b, v].item()
                bev_b = bev_b + self._splat(pts_ego, fv, tw)

            bev_out[b] = bev_b

        # Refine aggregated BEV
        return self.bev_refine(bev_out)                       # (B, bev_ch, H, W)


# ── Lightweight CNN stem that returns BOTH spatial maps and pooled vectors ─────

class DualOutputCNNStem(nn.Module):
    """
    Shared CNN backbone that returns:
      - feat_maps: (B*V, feat_ch, Hf, Wf) — spatial feature maps for LSSGeometricBEV
      - vec:       (B*V, d) — pooled vectors for Transformer path
    """
    def __init__(self, feat_ch: int = 64, d: int = 384):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, feat_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(feat_ch), nn.ReLU(inplace=True),
        )
        self.pool_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_ch, d),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        fm  = self.spatial(x)      # (B*V, feat_ch, Hf, Wf)
        vec = self.pool_proj(fm)   # (B*V, d)
        return fm, vec
