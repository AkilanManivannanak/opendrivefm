"""
OpenDriveFM v12 — Geometric BEV Lifting (Step 4 compliance)
Adds proper camera-intrinsic frustum lifting alongside existing implicit path.
The two paths are fused via a learned gate.

Step 4 requirements met:
  (i)  Camera intrinsics/extrinsics used to map features into 3D frustum points
  (ii) Depth bins used to lift 2D features into 3D, splat/pool into BEV grid
  (iii) Per-camera BEV feature tensors aligned to ego-centric coordinate system
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─── Existing modules (unchanged) ─────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    def __init__(self, d=384, nheads=6, nlayers=4, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nheads, dim_feedforward=4*d,
            dropout=dropout, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

    def forward(self, x):
        return self.enc(x)


class CameraTrustScorer(nn.Module):
    def __init__(self, in_ch=3, hidden=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 5, stride=4, padding=2),
            nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden*2, 5, stride=4, padding=2),
            nn.BatchNorm2d(hidden*2), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(hidden*2, 16), nn.GELU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )
        self.stats_head = nn.Sequential(
            nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())
        self.fuse = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
        lap = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]).view(1,1,3,3)
        sx  = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3)
        self.register_buffer("_lap", lap)
        self.register_buffer("_sx",  sx)

    def _image_stats(self, x):
        gray = x.mean(dim=1, keepdim=True)
        blur = F.conv2d(gray, self._lap, padding=1).var(dim=[1,2,3])
        lum  = gray.mean(dim=[1,2,3])
        ex   = F.conv2d(gray, self._sx, padding=1)
        ey   = F.conv2d(gray, self._sx.transpose(-1,-2), padding=1)
        edge = (ex**2 + ey**2).sqrt().mean(dim=[1,2,3])
        stats = torch.stack([blur, lum, edge], dim=1)
        return torch.sigmoid(stats - stats.detach().mean(dim=0))

    def forward(self, x):
        cnn_s  = self.cnn(x)
        stat_s = self.stats_head(self._image_stats(x))
        return self.fuse(torch.cat([cnn_s, stat_s], dim=1)).squeeze(1)


class TrustWeightedFusion(nn.Module):
    def __init__(self, d=384):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, hv, trust):
        w = torch.softmax(trust, dim=1).unsqueeze(-1)
        z = (w * hv).sum(dim=1)
        return self.mlp(z)


# ─── NEW: Geometric BEV Lifter (Step 4) ───────────────────────────────────────

class GeometricBEVLifter(nn.Module):
    """
    Lifts per-camera 2D features into a BEV grid using camera intrinsics/extrinsics.

    Step 4 implementation:
      (i)  Uses K (intrinsics) + T_cam2ego (extrinsics) to project pixel rays into 3D
      (ii) D discrete depth bins partition the frustum; features are splatted per bin
      (iii) Each camera produces a (C, BEV_H, BEV_W) BEV tensor in ego-centric frame
           All cameras are summed with trust weighting into a single BEV feature map

    Args:
        feat_ch:   input feature channels from CNN stem
        bev_ch:    output BEV feature channels
        bev_h/w:   BEV grid spatial size
        d_min/max: depth range in metres
        n_depth:   number of discrete depth bins
        extent_m:  BEV covers [-extent_m, extent_m] in X and Y
    """
    def __init__(self, feat_ch=64, bev_ch=128,
                 bev_h=128, bev_w=128,
                 d_min=1.0, d_max=50.0, n_depth=32,
                 extent_m=20.0):
        super().__init__()
        self.bev_h    = bev_h
        self.bev_w    = bev_w
        self.n_depth  = n_depth
        self.d_min    = d_min
        self.d_max    = d_max
        self.extent_m = extent_m

        # Depth distribution head: predicts per-pixel depth bin logits
        self.depth_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
            nn.BatchNorm2d(feat_ch), nn.GELU(),
            nn.Conv2d(feat_ch, n_depth, 1),   # (B*V, n_depth, Hf, Wf)
        )

        # Feature projection: reduce channels before splatting
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_ch, bev_ch, 1),
            nn.BatchNorm2d(bev_ch), nn.GELU(),
        )

        # BEV refinement after splatting
        self.bev_refine = nn.Sequential(
            nn.Conv2d(bev_ch, bev_ch, 3, padding=1),
            nn.BatchNorm2d(bev_ch), nn.GELU(),
            nn.Conv2d(bev_ch, bev_ch, 3, padding=1),
            nn.BatchNorm2d(bev_ch), nn.GELU(),
        )

        # Pre-compute depth bin centres
        depths = torch.linspace(d_min, d_max, n_depth)
        self.register_buffer("depths", depths)

    def _make_frustum_points(self, Hf, Wf, K_inv, device):
        """
        Create (Hf*Wf*n_depth, 3) 3D points in camera frame for each pixel+depth.
        K_inv: (3,3) inverse intrinsic (normalised to feature map scale)
        """
        # Pixel grid in feature map space
        ys = torch.arange(Hf, device=device).float()
        xs = torch.arange(Wf, device=device).float()
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')   # (Hf, Wf)
        ones   = torch.ones_like(xx)
        # Homogeneous pixel coords: (3, Hf*Wf)
        pix_h  = torch.stack([xx.flatten(), yy.flatten(), ones.flatten()], dim=0)
        # Ray direction in camera frame: (3, Hf*Wf)
        rays   = K_inv @ pix_h                            # (3, N)
        # Expand over depth bins: (3, N, D)
        rays   = rays.unsqueeze(2) * self.depths.view(1, 1, -1)
        # → (N*D, 3)
        pts_cam = rays.permute(1, 2, 0).reshape(-1, 3)
        return pts_cam                                     # (Hf*Wf*n_depth, 3)

    def _cam_to_ego(self, pts_cam, R, t):
        """
        Transform points from camera frame to ego frame.
        pts_cam: (N, 3)
        R: (3,3)  t: (3,)
        Returns: (N, 3)
        """
        return (R @ pts_cam.T).T + t.unsqueeze(0)        # (N, 3)

    def _splat_to_bev(self, pts_ego, feat_voxel, trust_w):
        """
        Splat voxel features onto BEV grid via bilinear splatting.

        pts_ego:    (N, 3) — 3D points in ego frame (x=forward, y=left)
        feat_voxel: (bev_ch, N) — feature per voxel point
        trust_w:    scalar — camera trust weight

        Returns: (bev_ch, bev_h, bev_w)
        """
        bev = torch.zeros(feat_voxel.shape[0], self.bev_h, self.bev_w,
                          device=feat_voxel.device)

        # Map ego X,Y → BEV pixel indices
        # ego X (forward) → BEV row (0=front), ego Y (left) → BEV col (0=left)
        x_ego = pts_ego[:, 0]   # forward
        y_ego = pts_ego[:, 1]   # left

        # Normalise to [0, 1]
        u = (x_ego + self.extent_m) / (2 * self.extent_m)
        v = (y_ego + self.extent_m) / (2 * self.extent_m)

        # Scale to grid indices
        col = (u * (self.bev_w - 1)).long()
        row = (v * (self.bev_h - 1)).long()

        # Mask in-bounds points
        valid = (col >= 0) & (col < self.bev_w) & \
                (row >= 0) & (row < self.bev_h)
        col = col[valid]
        row = row[valid]
        fv  = feat_voxel[:, valid]                        # (bev_ch, N_valid)

        # Scatter add (pillar pooling — sum features per BEV cell)
        idx = row * self.bev_w + col                      # (N_valid,)
        bev_flat = bev.view(feat_voxel.shape[0], -1)      # (bev_ch, H*W)
        bev_flat.scatter_add_(1, idx.unsqueeze(0).expand_as(fv), fv * trust_w)

        return bev                                         # (bev_ch, bev_h, bev_w)

    def forward(self, feat_maps, K_list, T_list, trust):
        """
        Args:
            feat_maps: (B, V, feat_ch, Hf, Wf) — per-camera feature maps
            K_list:    (B, V, 3, 3)             — camera intrinsics (scaled to Hf,Wf)
            T_list:    (B, V, 4, 4)             — cam→ego extrinsics
            trust:     (B, V)                   — per-camera trust scores

        Returns:
            bev_geo:   (B, bev_ch, bev_h, bev_w) — geometric BEV feature map
        """
        B, V, C, Hf, Wf = feat_maps.shape
        device = feat_maps.device

        # Predict depth distributions and project features
        fm_flat = feat_maps.view(B*V, C, Hf, Wf)
        depth_logits = self.depth_head(fm_flat)            # (B*V, D, Hf, Wf)
        depth_probs  = depth_logits.softmax(dim=1)         # (B*V, D, Hf, Wf)
        feat_proj    = self.feat_proj(fm_flat)             # (B*V, bev_ch, Hf, Wf)

        # Weighted voxel features: (B*V, bev_ch, D, Hf, Wf)
        feat_vox = feat_proj.unsqueeze(2) * depth_probs.unsqueeze(1)

        bev_out = torch.zeros(B, self.bev_ch if hasattr(self, 'bev_ch') else feat_proj.shape[1],
                              self.bev_h, self.bev_w, device=device)
        bev_ch_dim = feat_proj.shape[1]
        bev_out = torch.zeros(B, bev_ch_dim, self.bev_h, self.bev_w, device=device)

        for b in range(B):
            bev_b = torch.zeros(bev_ch_dim, self.bev_h, self.bev_w, device=device)
            for v in range(V):
                K   = K_list[b, v]                        # (3,3)
                T   = T_list[b, v]                        # (4,4)
                R   = T[:3, :3]
                t   = T[:3,  3]
                K_inv = torch.linalg.inv(K.float())

                # (i) Create frustum points in camera frame
                pts_cam = self._make_frustum_points(Hf, Wf, K_inv, device)
                # (ii) Transform to ego frame
                pts_ego = self._cam_to_ego(pts_cam, R, t)

                # Flatten voxel features for this camera: (bev_ch, Hf*Wf*D)
                fv = feat_vox[b*V + v]                    # (bev_ch, D, Hf, Wf)
                fv = fv.permute(0, 2, 3, 1).reshape(bev_ch_dim, -1)

                # (iii) Splat onto BEV grid with trust weighting
                tw = trust[b, v].item()
                bev_b = bev_b + self._splat_to_bev(pts_ego, fv, tw)

            bev_out[b] = bev_b

        return self.bev_refine(bev_out)                   # (B, bev_ch, H, W)


# ─── NEW: Shallow CNN stem that also returns feature maps ──────────────────────

class MultiViewCNNStem(nn.Module):
    """
    Shared CNN stem that returns BOTH:
      - pooled vectors for the Transformer path (implicit BEV)
      - spatial feature maps for GeometricBEVLifter (explicit BEV)
    """
    def __init__(self, d=384, feat_ch=64):
        super().__init__()
        self.feat_ch = feat_ch
        # Spatial feature extractor (returns feature maps)
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, feat_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(feat_ch), nn.GELU(),
        )
        # Pool + project for Transformer path
        self.pool_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_ch, d),
        )

    def forward(self, x):
        """x: (B*V*T, 3, H, W) → feat_maps: (B*V*T, feat_ch, Hf, Wf), vec: (B*V*T, d)"""
        feat_maps = self.spatial(x)
        vec       = self.pool_proj(feat_maps)
        return feat_maps, vec


# ─── Updated backbone with dual BEV path ──────────────────────────────────────

class MultiViewVideoBackboneV12(nn.Module):
    """
    V12 backbone: implicit Transformer BEV + explicit Geometric BEV lifting.
    Both paths are fused via a learned scalar gate.
    """
    def __init__(self, d=384, feat_ch=64, bev_ch=128,
                 bev_h=128, bev_w=128, enable_trust=True):
        super().__init__()
        self.enable_trust = enable_trust
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Shared CNN stem (returns both feature maps and pooled vectors)
        self.stem = MultiViewCNNStem(d=d, feat_ch=feat_ch)

        # Existing implicit path
        self.temporal    = TemporalTransformer(d=d, nheads=6, nlayers=4)
        self.trust_scorer= CameraTrustScorer()
        self.trust_fuse  = TrustWeightedFusion(d=d)

        # NEW: Geometric BEV lifting path (Step 4)
        self.geo_lifter  = GeometricBEVLifter(
            feat_ch=feat_ch, bev_ch=bev_ch,
            bev_h=bev_h, bev_w=bev_w)

        # BEV feature → global descriptor for trajectory head
        self.bev_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bev_ch, d),
            nn.GELU(),
        )

        # Learned gate: mix implicit z and geometric bev_desc
        self.gate = nn.Sequential(
            nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, 1), nn.Sigmoid())
        self.proj_fused = nn.Linear(d, d)

    def forward(self, x, K=None, T_cam2ego=None):
        """
        x:         (B, V, T, 3, H, W)
        K:         (B, V, 3, 3) camera intrinsics — optional
        T_cam2ego: (B, V, 4, 4) extrinsics — optional
        """
        B, V, T_frames, C, H, W = x.shape
        xt = rearrange(x, "b v t c h w -> (b v t) c h w")

        # CNN stem → both spatial features and pooled vectors
        feat_maps_flat, vec_flat = self.stem(xt)          # (B*V*T, feat_ch, Hf, Wf), (B*V*T, d)

        # ── Implicit Transformer path (unchanged from v8-v11) ─────────────────
        ft = rearrange(vec_flat, "(b v t) d -> b v t d", b=B, v=V, t=T_frames)
        ft2 = rearrange(ft, "b v t d -> (b v) t d")
        ht  = self.temporal(ft2)
        hv  = ht.mean(dim=1)
        hv  = rearrange(hv, "(b v) d -> b v d", b=B, v=V)

        # Trust scores
        cam_imgs   = x[:, :, 0]
        imgs_flat  = rearrange(cam_imgs, "b v c h w -> (b v) c h w")
        trust_flat = self.trust_scorer(imgs_flat)
        trust      = rearrange(trust_flat, "(b v) -> b v", b=B, v=V)

        z_implicit = self.trust_fuse(hv, trust)           # (B, d)

        # ── Geometric BEV lifting path (NEW — Step 4) ─────────────────────────
        if K is not None and T_cam2ego is not None:
            # Use first temporal frame only for BEV lifting
            feat_maps = rearrange(
                feat_maps_flat[::(T_frames), ...] if T_frames > 1 else feat_maps_flat,
                "(b v) c hf wf -> b v c hf wf", b=B, v=V)

            # Scale intrinsics to feature map resolution
            Hf, Wf = feat_maps.shape[-2:]
            K_scaled = K.clone().float()
            K_scaled[:, :, 0, :] *= (Wf / W)
            K_scaled[:, :, 1, :] *= (Hf / H)

            bev_geo    = self.geo_lifter(feat_maps, K_scaled, T_cam2ego, trust)
            bev_desc   = self.bev_pool(bev_geo)           # (B, d)

            # Learned gate to fuse implicit and geometric paths
            gate_in    = torch.cat([z_implicit, bev_desc], dim=-1)
            alpha      = self.gate(gate_in)               # (B, 1)
            z_fused    = self.proj_fused(alpha * z_implicit + (1-alpha) * bev_desc)
        else:
            # Fall back to implicit-only if no calibration data
            bev_geo  = None
            z_fused  = z_implicit

        return z_fused, feat_maps_flat, trust, bev_geo


# ─── BEV Occupancy Head (unchanged) ───────────────────────────────────────────

class BEVOccupancyHeadV12(nn.Module):
    """
    Dual-input occupancy head:
    - If bev_geo provided: use it directly (geometric path preferred)
    - Else: fall back to implicit MLP decoder
    """
    def __init__(self, d=384, bev_ch=128, bev_h=128, bev_w=128):
        super().__init__()
        self.bev_h, self.bev_w = bev_h, bev_w

        # Implicit decoder (fallback)
        self.mlp = nn.Sequential(
            nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, bev_h*bev_w))

        # Geometric decoder (preferred)
        self.geo_dec = nn.Sequential(
            nn.Conv2d(bev_ch, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

        # Fusion gate
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, z, bev_geo=None):
        implicit = self.mlp(z).view(z.size(0), 1, self.bev_h, self.bev_w)
        if bev_geo is not None:
            geo = self.geo_dec(bev_geo)                   # (B, 1, H, W)
            a   = torch.sigmoid(self.alpha)
            return a * geo + (1-a) * implicit
        return implicit


# ─── Trajectory head (unchanged) ──────────────────────────────────────────────

class TrajHead(nn.Module):
    def __init__(self, d=384, horizon=12):
        super().__init__()
        self.horizon = horizon
        self.mlp = nn.Sequential(
            nn.Linear(d, 2*d), nn.GELU(),
            nn.Linear(2*d, d), nn.GELU(),
            nn.Linear(d, horizon*2))

    def forward(self, z):
        return self.mlp(z).view(z.size(0), self.horizon, 2)


# ─── Full model v12 ───────────────────────────────────────────────────────────

class OpenDriveFMV12(nn.Module):
    """
    OpenDriveFM v12 — adds geometric BEV lifting (Step 4 compliance).

    Returns: (occ_logits, traj_xy, trust, bev_geo)
      occ_logits: (B, 1, 128, 128)
      traj_xy:    (B, 12, 2)
      trust:      (B, 6)
      bev_geo:    (B, 128, 128, 128) or None
    """
    def __init__(self, d=384, feat_ch=64, bev_ch=128,
                 bev_h=128, bev_w=128, horizon=12, enable_trust=True):
        super().__init__()
        self.backbone = MultiViewVideoBackboneV12(
            d=d, feat_ch=feat_ch, bev_ch=bev_ch,
            bev_h=bev_h, bev_w=bev_w, enable_trust=enable_trust)
        self.occ  = BEVOccupancyHeadV12(d=d, bev_ch=bev_ch, bev_h=bev_h, bev_w=bev_w)
        self.traj = TrajHead(d=d, horizon=horizon)

    def forward(self, x, K=None, T_cam2ego=None):
        z, _, trust, bev_geo = self.backbone(x, K, T_cam2ego)
        return self.occ(z, bev_geo), self.traj(z), trust, bev_geo


def occ_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)

def traj_loss(pred, target):
    return F.smooth_l1_loss(pred, target)
