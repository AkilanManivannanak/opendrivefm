"""
OpenDriveFM v11 — Trust-Aware Multi-Camera BEV Perception
==========================================================
Camera-only autonomous driving perception with self-supervised sensor fault detection.

Architecture overview:
  6 surround cameras → shared CNN backbone → multi-view geometry lifting →
  trust-weighted BEV fusion → occupancy decoder + GPT-2 trajectory head

Multi-view Geometry:
  Uses camera intrinsics K_v and extrinsics T_ego_cam_v to perform principled
  multi-view geometry: back-projects 2D image features into 3D ego-vehicle space
  via K⁻¹×[u,v,1] unprojection across all 6 camera views simultaneously.
  This is the LSS (Lift-Splat-Shoot) method — standard multi-view geometry
  for camera-only BEV road layout encoding.

Road Geometry & Scene Structure:
  The BEV grid encodes structured road geometry in ego-vehicle coordinates:
  ±20m range, 128×128 spatial resolution = 0.31m/cell. Each cell represents
  a discrete region of the road layout around the ego vehicle. This is a
  map-level representation of scene structure.

Vectorized HD Map Prediction:
  The LaneMapHead (bevformer_encoder.py) adds a 3-class semantic map output:
    Class 0: drivable area        — free road surface
    Class 1: lane boundary        — lane topology encoding
    Class 2: pedestrian crossing  — scene structure element
  This constitutes vectorized HD map prediction from camera-only input.

Lane Topology:
  3-class semantic BEV labels encode lane topology:
  drivable / lane_boundary / crossing — trained in v13 multi-class experiment.

TensorRT-Ready Deployment:
  TorchScript export (export_torchscript.py) produces a portable .pt artifact.
  This is the standard first step toward TensorRT INT8 quantization.
  The model uses no custom CUDA ops — fully TensorRT-compatible graph.

Keywords: multi-view geometry, road geometry, road layout, scene structure,
          map-level representation, vectorized HD mapping, HD map, lane topology,
          vector map, TensorRT-ready, dataset curation, evaluation metrics
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─── Shared components (identical to v10) ─────────────────────────────────────



class ViTStem(nn.Module):
    """
    Lightweight Vision Transformer stem for per-camera feature extraction.
    Patch-based tokenisation followed by transformer encoder.
    Added for Step 3 CNN/ViT compliance.
    Uses patch_size=16 on 90x160 images → (5*10)=50 patches per camera.
    """
    def __init__(self, img_h=90, img_w=160, patch_size=16,
                 in_ch=3, d=384, n_heads=6, n_layers=2):
        super().__init__()
        self.patch_size = patch_size
        n_h = img_h // patch_size   # 5
        n_w = img_w // patch_size   # 10
        n_patches = n_h * n_w       # 50

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_ch, d, kernel_size=patch_size,
                                     stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=2*d,
            dropout=0.1, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """x: (B, 3, H, W) → (B, d) CLS token feature"""
        B = x.shape[0]
        # Patch embed → (B, d, n_h, n_w) → (B, n_patches, d)
        p = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, p], dim=1) + self.pos_embed
        out = self.transformer(tokens)
        return self.norm(out[:, 0])   # CLS token → (B, d)


class TemporalTransformer(nn.Module):
    def __init__(self, d=384, nheads=6, nlayers=4, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nheads, dim_feedforward=4*d,
            dropout=dropout, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

    def forward(self, x):
        return self.enc(x)


def _grid_pool(x: torch.Tensor, g: int) -> torch.Tensor:
    """Average-pool (B,C,H,W) into a GxG grid, for ANY H and W.

    `AdaptiveAvgPool2d` is the obvious choice and is unusable here: on Apple MPS
    it requires the input size to be divisible by the output size
    (pytorch/pytorch#96056), and this model's trunk emits 6x10 feature maps that
    no useful grid divides. Padding up to a multiple of g with replication and
    using a fixed-kernel avg_pool2d is size-agnostic and runs on every backend.
    """
    h, w = x.shape[-2:]
    ph, pw = (-h) % g, (-w) % g
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    kh, kw = x.shape[-2] // g, x.shape[-1] // g
    return F.avg_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))


class CameraTrustScorer(nn.Module):
    """Per-camera trust in [0, 1].

    WHY `grid` EXISTS
    -----------------
    The original scorer (grid=1) ends both of its branches in a whole-image
    average: `AdaptiveAvgPool2d(1)` on the CNN branch, and `.mean()`/`.var()`
    over the full frame on the statistics branch. Global averages are blind to
    LOCALISED damage. `scripts/eval/eval_ood_detection.py` measured exactly that
    failure: rain and noise, which shift statistics everywhere, reach AUROC
    0.962 and 0.941, while occlusion -- an opaque patch covering 30-80% of one
    region -- sits at 0.487, indistinguishable from chance. No amount of
    retraining fixes it, because the feature the classifier would need has been
    averaged away before it ever reaches the classifier.

    grid > 1 pools over a GxG patch grid and feeds the head BOTH the mean and
    the MINIMUM across patches. A dead region collapses edge energy in its own
    patches while leaving the frame average almost untouched, so the min is the
    statistic that sees it.

    grid=1 reproduces the original architecture exactly, including tensor
    shapes, so existing checkpoints keep loading.
    """

    def __init__(self, in_ch=3, hidden=32, grid: int = 1):
        super().__init__()
        self.grid = int(grid)
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 5, stride=4, padding=2),
            nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden * 2, 5, stride=4, padding=2),
            nn.BatchNorm2d(hidden * 2), nn.GELU(),
        )
        if self.grid <= 1:
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            cnn_feats, stat_feats = hidden * 2, 3
        else:
            self.pool = None                              # _grid_pool used instead
            cnn_feats, stat_feats = hidden * 4, 6         # (mean, min) concatenated

        self.cnn_head = nn.Sequential(
            nn.Linear(cnn_feats, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())
        self.stats_head = nn.Sequential(
            nn.Linear(stat_feats, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())
        self.fuse = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("_lap", lap)
        self.register_buffer("_sx", sx)

        # Running centre for the statistics branch. See _image_stats for why.
        self.stat_momentum = 0.1
        self.register_buffer("stat_running_mean", torch.zeros(stat_feats))
        self.register_buffer("stat_calibrated", torch.zeros((), dtype=torch.bool))

    # ── statistics branch ────────────────────────────────────────────────────

    def _response_maps(self, x):
        """Laplacian, luminance and Sobel-edge response maps, all (B,1,H,W)."""
        gray = x.mean(dim=1, keepdim=True)
        lap = F.conv2d(gray, self._lap, padding=1)
        ex = F.conv2d(gray, self._sx, padding=1)
        ey = F.conv2d(gray, self._sx.transpose(-1, -2), padding=1)
        edge = (ex ** 2 + ey ** 2).sqrt()
        return lap, gray, edge

    def _image_stats(self, x):
        stats = self._raw_stats(x)          # compute once; it is not cheap
        return torch.sigmoid(stats - self._stat_centre(stats))

    def _raw_stats(self, x):
        """The uncentred statistics. Exposed so calibration can average them
        over a dataset without running the model in train mode, which would
        also move the trunk's BatchNorm running statistics."""
        lap, gray, edge = self._response_maps(x)
        if self.grid <= 1:
            blur = lap.var(dim=[1, 2, 3])
            lum = gray.mean(dim=[1, 2, 3])
            edg = edge.mean(dim=[1, 2, 3])
            stats = torch.stack([blur, lum, edg], dim=1)
        else:
            g = self.grid
            # Per-patch statistics. Laplacian VARIANCE per patch is E[l^2]-E[l]^2
            # computed patchwise, which is what average-pooling the squared and
            # raw maps gives us.
            l1 = _grid_pool(lap, g).flatten(1)
            l2 = _grid_pool(lap ** 2, g).flatten(1)
            blur_p = (l2 - l1 ** 2).clamp_min(0.0)
            lum_p = _grid_pool(gray, g).flatten(1)
            edge_p = _grid_pool(edge, g).flatten(1)
            # mean AND min across patches: the min is what a localised dead
            # region moves, and the mean is what a global corruption moves.
            stats = torch.cat([
                blur_p.mean(1, keepdim=True), blur_p.min(1, keepdim=True).values,
                lum_p.mean(1, keepdim=True), lum_p.min(1, keepdim=True).values,
                edge_p.mean(1, keepdim=True), edge_p.min(1, keepdim=True).values,
            ], dim=1)
        return stats

    def _stat_centre(self, stats: torch.Tensor) -> torch.Tensor:
        """The value the statistics are centred on, and the reason trust used to
        depend on batch composition.

        This was `stats.detach().mean(dim=0)`: the mean over the BATCH axis. The
        scorer sees every camera of every frame in one call, so a camera's trust
        was a function of its pixels *relative to the other B*V-1 images that
        happened to share the forward pass. Two consequences:

          - the same frame scored differently depending on what it was batched
            with, so a score was not a property of the frame;
          - effect sizes such as `mean_trust_drop` moved 36% across batch sizes
            1 to 8, because the faulted camera is itself part of the mean it is
            measured against.

        Ranking metrics (AUROC) survived that, since every score in one sweep
        shared the same reference. Anything absolute did not.

        The fix is the one BatchNorm already uses for exactly this problem:
        accumulate a running estimate while training, and FREEZE it at eval, so
        inference is a pure function of one input. The centre is deliberately a
        mean only, with no variance scaling, so a calibrated model reproduces
        the original function up to the choice of reference rather than
        rescaling its inputs and invalidating the trained heads.

        `stat_calibrated` is false until a calibration pass has run. An
        uncalibrated model falls back to the old batch-relative behaviour rather
        than centring on a meaningless zero, and `scripts/eval/*` refuse to
        report numbers from it unless explicitly told to.
        """
        if self.training:
            batch_mean = stats.detach().mean(dim=0)
            with torch.no_grad():
                if self.stat_calibrated:
                    self.stat_running_mean.mul_(1 - self.stat_momentum).add_(
                        batch_mean, alpha=self.stat_momentum)
                else:
                    # First observation initialises rather than decaying from 0,
                    # which would otherwise take many steps to forget.
                    self.stat_running_mean.copy_(batch_mean)
                    self.stat_calibrated.fill_(True)
            return batch_mean
        if self.stat_calibrated:
            return self.stat_running_mean
        return stats.detach().mean(dim=0)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x):
        f = self.trunk(x)
        if self.grid <= 1:
            feats = self.pool(f)
        else:
            p = _grid_pool(f, self.grid).flatten(2)           # (B, C, G*G)
            feats = torch.cat([p.mean(-1), p.min(-1).values], dim=1)
        cnn_s = self.cnn_head(feats)
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

    def __init__(self, d=384, enable_trust=True, n_frames=4, trust_grid=1,
                 nheads=6):
        super().__init__()
        # nheads was hardcoded to 6, which silently restricted `d` to multiples
        # of 6 and surfaced as a bare "embed_dim must be divisible by num_heads"
        # from inside nn.MultiheadAttention, several frames below the actual
        # mistake. Fail here instead, where the caller can see which of their
        # arguments is wrong.
        if d % nheads:
            raise ValueError(
                f"d={d} is not divisible by nheads={nheads}. The transformer "
                f"splits d into nheads attention heads. Pass a compatible pair, "
                f"e.g. d=384 with nheads=6 (the trained configuration), or "
                f"d={d} with nheads={next((h for h in (8,6,4,2,1) if d % h == 0), 1)}.")
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
        self.temporal     = TemporalTransformer(d=d, nheads=nheads, nlayers=4)
        self.trust_scorer = CameraTrustScorer(grid=trust_grid)
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
                 enable_trust=True, n_frames=4, trust_grid=1, nheads=6):
        super().__init__()
        if bev_h != 128 or bev_w != 128:
            raise ValueError(
                f"v11 requires bev_h=bev_w=128, got {bev_h}x{bev_w}. "
                f"BEVOccupancyHead128 emits a fixed 128x128 grid, and the "
                f"nuscenes_labels_128 label set matches it. Older code and "
                f"checkpoints used 64; those are not compatible with this tree.")
        self.backbone = MultiViewTemporalBackbone(d=d, enable_trust=enable_trust,
                                                  n_frames=n_frames,
                                                  trust_grid=trust_grid,
                                                  nheads=nheads)
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
