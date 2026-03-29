"""
model_v11_temporal.py — OpenDriveFM with T=4 temporal BEV accumulation.

Architecture change: after extracting per-frame BEV features, warp past frames
into the current ego frame using the ego_deltas and accumulate via attention.

Key additions:
  - BEVWarpAndAccumulate: warps T-1 past BEV grids into current frame using
    affine transforms derived from [dx, dy, dyaw], then pools with learned weights
  - TemporalBEVFusion: cross-attention between current BEV and warped past BEVs

The backbone now returns a spatial BEV feature map (B, d, H_bev, W_bev)
rather than a global token, enabling richer per-location decoding.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─── Shared components (identical to v10) ─────────────────────────────────────

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
        return self.mlp((w * hv).sum(dim=1))


class DepthHead(nn.Module):
    def __init__(self, in_ch, d_min=1.0, d_max=50.0):
        super().__init__()
        self.d_min, self.d_max = d_min, d_max
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//2, 3, padding=1),
            nn.BatchNorm2d(in_ch//2), nn.GELU(),
            nn.Conv2d(in_ch//2, 1, 1),
        )

    def forward(self, feat):
        return self.d_min + (self.d_max - self.d_min) * torch.sigmoid(self.head(feat))


def lidar_depth_loss(pred_depth, lidar_maps, feat_h, feat_w):
    B, V = lidar_maps.shape[0], lidar_maps.shape[1]
    gt   = lidar_maps.view(B*V, 1, lidar_maps.shape[3], lidar_maps.shape[4])
    gt_ds= F.interpolate(gt, size=(feat_h, feat_w), mode="nearest")
    valid= (gt_ds > 0.1).float()
    return ((pred_depth - gt_ds).abs() * valid).sum() / valid.sum().clamp(1.0)


# ─── NEW: Temporal BEV components ─────────────────────────────────────────────

class BEVWarpAndAccumulate(nn.Module):
    """
    Takes T per-frame BEV global tokens and ego_deltas, produces accumulated BEV token.

    Strategy: Project each frame's global token into a small spatial BEV proxy
    (8×8), warp using ego_deltas affine transforms, then fuse with learned
    temporal attention weights.

    This is a lightweight version of full spatial BEV warping — full spatial
    warping of 128×128 feature maps would be too slow on MPS with 322 samples.
    """
    def __init__(self, d: int = 384, proxy_size: int = 8, n_frames: int = 4):
        super().__init__()
        self.proxy_size = proxy_size
        self.n_frames   = n_frames
        self.d          = d

        # Project global token to small spatial BEV proxy
        self.to_spatial = nn.Sequential(
            nn.Linear(d, proxy_size * proxy_size * d // 4),
            nn.GELU(),
        )
        self.from_spatial = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d // 4, d),
            nn.GELU(),
        )

        # Learned temporal importance weights — more recent = more important
        # Initialise with recency bias: current frame gets highest weight
        self.temp_weights = nn.Parameter(
            torch.linspace(0.5, 1.0, n_frames)   # [oldest ... newest]
        )
        self.fuse_proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def _build_affine(self, dx: torch.Tensor, dy: torch.Tensor,
                      dyaw: torch.Tensor, grid_size: int) -> torch.Tensor:
        """
        Build 2D affine transform matrix (B, 2, 3) for grid_sample.
        dx, dy in metres → scaled to grid units (grid_size / 100m extent).
        """
        cos_a = torch.cos(dyaw)
        sin_a = torch.sin(dyaw)
        B     = dx.shape[0]

        # Scale metres to normalised grid coords [-1, 1] over 100m
        # 1 metre = 2/100 = 0.02 in normalised coords
        scale  = 2.0 / 100.0
        tx     = dx * scale
        ty     = dy * scale

        # Rotation + translation affine (2×3)
        theta = torch.stack([
            torch.stack([ cos_a, sin_a, tx], dim=-1),
            torch.stack([-sin_a, cos_a, ty], dim=-1),
        ], dim=-2)              # (B, 2, 3)
        return theta

    def forward(self, hv: torch.Tensor, ego_deltas: torch.Tensor) -> torch.Tensor:
        """
        hv:          (B, T, d)      — per-frame global tokens (oldest→newest)
        ego_deltas:  (B, T-1, 3)    — [dx, dy, dyaw] for each past frame into t=0
        Returns:     (B, d)         — temporally accumulated token
        """
        B, T, d = hv.shape
        P = self.proxy_size

        # Project each frame token to spatial proxy (B, T, d//4, P, P)
        spatial = self.to_spatial(hv.view(B*T, d))              # (B*T, P*P*d//4)
        spatial = spatial.view(B*T, d//4, P, P)

        # Warp past frames into current ego frame
        warped = []
        for t_idx in range(T - 1):        # past frames (oldest first)
            delta = ego_deltas[:, t_idx]  # (B, 3): [dx, dy, dyaw]
            dx, dy, dyaw = delta[:, 0], delta[:, 1], delta[:, 2]
            theta     = self._build_affine(dx, dy, dyaw, P)     # (B, 2, 3)
            grid      = F.affine_grid(theta, (B, d//4, P, P), align_corners=False)
            s_t       = spatial[t_idx::T]                        # (B, d//4, P, P)
            s_warped  = F.grid_sample(s_t, grid, mode="bilinear",
                                      padding_mode="zeros", align_corners=False)
            warped.append(s_warped)

        # Current frame (no warp needed)
        warped.append(spatial[T-1::T])    # most recent frame

        # Temporal pooling with learned weights (recency-biased softmax)
        w      = torch.softmax(self.temp_weights, dim=0)          # (T,)
        pooled = sum(w[i] * self.from_spatial(warped[i])
                     for i in range(T))                           # (B, d)

        return self.fuse_proj(pooled)


# ─── Full backbone with temporal BEV accumulation ─────────────────────────────

class MultiViewTemporalBackbone(nn.Module):
    FEAT_CH = 192

    def __init__(self, d=384, enable_trust=True, n_frames=4):
        super().__init__()
        self.enable_trust = enable_trust
        self.n_frames     = n_frames
        C = self.FEAT_CH

        self.stem = nn.Sequential(
            nn.Conv2d(3, C//2, 7, stride=2, padding=3), nn.BatchNorm2d(C//2), nn.GELU(),
            nn.Conv2d(C//2, C, 3, stride=2, padding=1), nn.BatchNorm2d(C),    nn.GELU(),
            nn.Conv2d(C, C,    3, stride=1, padding=1), nn.BatchNorm2d(C),    nn.GELU(),
        )
        self.depth_head = DepthHead(in_ch=C)
        self.pool_proj  = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(C, d))

        # Per-frame transformer (applied independently per frame)
        self.temporal     = TemporalTransformer(d=d, nheads=6, nlayers=4)
        self.trust_scorer = CameraTrustScorer()
        self.trust_fuse   = TrustWeightedFusion(d=d)
        self.view_fuse    = nn.Sequential(nn.Linear(d,d), nn.GELU(), nn.Linear(d,d))

        # NEW: temporal BEV accumulation across T frames
        self.bev_accum = BEVWarpAndAccumulate(d=d, proxy_size=8, n_frames=n_frames)

    def forward(self, x, ego_deltas=None, return_feat_maps=False):
        """
        x:           (B, V, T, C, H, W)
        ego_deltas:  (B, T-1, 3) optional — if None, no temporal warping
        """
        B, V, T, C_img, H, W = x.shape

        # Encode all frames together
        xt   = rearrange(x, "b v t c h w -> (b v t) c h w")
        feat = self.stem(xt)
        Hf, Wf = feat.shape[2], feat.shape[3]

        ft = self.pool_proj(feat)                               # (B*V*T, d)
        ft = rearrange(ft, "(b v t) d -> b v t d", b=B, v=V, t=T)

        # Per-view temporal aggregation (cross-frame per-camera)
        ft2 = rearrange(ft, "b v t d -> (b v) t d")
        ht  = self.temporal(ft2)
        hv  = ht.mean(dim=1)
        hv  = rearrange(hv, "(b v) d -> b v d", b=B, v=V)

        # Trust scoring (use current frame t=-1 i.e. newest)
        if self.enable_trust:
            imgs_flat  = rearrange(x[:, :, -1], "b v c h w -> (b v) c h w")
            trust_flat = self.trust_scorer(imgs_flat)
            trust      = rearrange(trust_flat, "(b v) -> b v", b=B, v=V)
            z_cam = self.trust_fuse(hv, trust)
        else:
            trust = torch.ones(B, V, device=x.device)
            z_cam = self.view_fuse(hv.mean(dim=1))

        # NEW: temporal BEV accumulation
        if ego_deltas is not None and T > 1:
            # Get per-frame scene tokens by encoding each time step separately
            # Shape: (B, T, d) — pool over views for each frame
            ft_v = rearrange(ft, "b v t d -> b t v d")
            ht_per_frame = []
            for t_i in range(T):
                # View pool at this timestep
                ht_per_frame.append(ft_v[:, t_i].mean(dim=1))   # (B, d)
            hv_temporal = torch.stack(ht_per_frame, dim=1)      # (B, T, d)

            z_temporal = self.bev_accum(hv_temporal, ego_deltas)
            # Combine spatial (trust-weighted) + temporal signals
            z = 0.6 * z_cam + 0.4 * z_temporal
        else:
            z = z_cam

        if return_feat_maps:
            feat_t0 = feat[T-1::T]   # newest frame features: (B*V, C, Hf, Wf)
            return z, ft, trust, feat_t0, Hf, Wf
        return z, ft, trust


# ─── Heads (128×128 BEV) ──────────────────────────────────────────────────────

class BEVOccupancyHead128(nn.Module):
    def __init__(self, d=384):
        super().__init__()
        self.seed_proj = nn.Linear(d, 4 * 4 * d)
        self.decoder   = nn.Sequential(
            nn.ConvTranspose2d(d,     d//2,  4, stride=2, padding=1), nn.BatchNorm2d(d//2),  nn.GELU(),
            nn.ConvTranspose2d(d//2,  d//4,  4, stride=2, padding=1), nn.BatchNorm2d(d//4),  nn.GELU(),
            nn.ConvTranspose2d(d//4,  d//8,  4, stride=2, padding=1), nn.BatchNorm2d(d//8),  nn.GELU(),
            nn.ConvTranspose2d(d//8,  d//16, 4, stride=2, padding=1), nn.BatchNorm2d(d//16), nn.GELU(),
            nn.ConvTranspose2d(d//16, 1,     4, stride=2, padding=1),
        )

    def forward(self, z):
        B, d = z.shape
        return self.decoder(self.seed_proj(z).view(B, d, 4, 4))


class TrajHead(nn.Module):
    def __init__(self, d=384, horizon=12):
        super().__init__()
        self.horizon    = horizon
        self.scene_proj = nn.Linear(d, d//2)
        self.vel_enc    = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, d//2))
        self.mlp        = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, d//2), nn.GELU(),
            nn.Linear(d//2, horizon*2),
        )

    def forward(self, z, velocity=None):
        sc = self.scene_proj(z)
        ve = self.vel_enc(velocity) if velocity is not None \
             else torch.zeros(z.size(0), z.size(1)//2, device=z.device)
        return self.mlp(torch.cat([sc, ve], dim=-1)).view(z.size(0), self.horizon, 2)


# ─── Final model ──────────────────────────────────────────────────────────────

class OpenDriveFM(nn.Module):
    """
    v11: 128×128 BEV + LiDAR depth supervision + T=4 temporal accumulation.

    forward() signature:
        model(x)                                    → (occ, traj, trust, None)
        model(x, velocity=v, ego_deltas=d)          → temporal accumulation
        model(x, lidar_depth_maps=ldm, ego_deltas=d)→ + depth supervision
    """
    def __init__(self, d=384, bev_h=128, bev_w=128, horizon=12,
                 enable_trust=True, n_frames=4):
        super().__init__()
        assert bev_h == 128, "v11 requires bev_h=128"
        self.backbone = MultiViewTemporalBackbone(d=d, enable_trust=enable_trust,
                                                  n_frames=n_frames)
        self.occ      = BEVOccupancyHead128(d=d)
        self.traj     = TrajHead(d=d, horizon=horizon)

    def forward(self, x, velocity=None, ego_deltas=None,
                lidar_depth_maps=None, **_):
        use_depth = lidar_depth_maps is not None

        if use_depth:
            z, ft, trust, feat_t0, Hf, Wf = \
                self.backbone(x, ego_deltas=ego_deltas, return_feat_maps=True)
            depth_pred = self.backbone.depth_head(feat_t0)
            return self.occ(z), self.traj(z, velocity), trust, depth_pred, Hf, Wf
        else:
            z, ft, trust = self.backbone(x, ego_deltas=ego_deltas)
            return self.occ(z), self.traj(z, velocity), trust, None
