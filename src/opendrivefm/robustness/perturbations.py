"""
Camera Robustness Perturbation Suite for Trust-Aware Fusion evaluation.
Simulates: GaussianBlur, GlareOverlay, OcclusionPatch, RainStreaks, SaltPepperNoise.
"""
from __future__ import annotations
import math, random
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianBlur(nn.Module):
    def __init__(self, sigma_range=(1.0, 4.0)):
        super().__init__()
        self.sigma_range = sigma_range

    @staticmethod
    def _kernel(sigma, size):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2*sigma**2))
        return g / g.sum()

    def forward(self, x):
        sigma  = random.uniform(*self.sigma_range)
        k_size = max(3, 2*int(math.ceil(3*sigma))+1)
        k = self._kernel(sigma, k_size).to(x.device)
        B, C, H, W = x.shape
        kh = k.view(1,1,1,k_size).expand(C,1,1,k_size)
        kv = k.view(1,1,k_size,1).expand(C,1,k_size,1)
        pad = k_size // 2
        x = F.conv2d(x, kh, padding=(0,pad), groups=C)
        x = F.conv2d(x, kv, padding=(pad,0), groups=C)
        return x.clamp(0,1)


class GlareOverlay(nn.Module):
    def __init__(self, intensity_range=(0.4, 0.9), size_range=(0.1, 0.35)):
        super().__init__()
        self.intensity_range = intensity_range
        self.size_range      = size_range

    def forward(self, x):
        B, C, H, W = x.shape
        intensity = random.uniform(*self.intensity_range)
        rx = random.uniform(*self.size_range) * W / 2
        ry = random.uniform(*self.size_range) * H / 2
        cx = random.uniform(0.2, 0.8) * W
        cy = random.uniform(0.2, 0.8) * H
        yy, xx = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=torch.float32),
            torch.arange(W, device=x.device, dtype=torch.float32), indexing="ij")
        mask = torch.exp(-(((xx-cx)/rx)**2 + ((yy-cy)/ry)**2))
        return (x + intensity * mask.unsqueeze(0).unsqueeze(0)).clamp(0,1)


class OcclusionPatch(nn.Module):
    def __init__(self, patch_frac=(0.1, 0.4), fill=0.0):
        super().__init__()
        self.patch_frac = patch_frac
        self.fill       = fill

    def forward(self, x):
        x = x.clone()
        B, C, H, W = x.shape
        ph = int(H * random.uniform(*self.patch_frac))
        pw = int(W * random.uniform(*self.patch_frac))
        y0 = random.randint(0, H - ph)
        x0 = random.randint(0, W - pw)
        x[:, :, y0:y0+ph, x0:x0+pw] = self.fill
        return x


class RainStreaks(nn.Module):
    def __init__(self, num_streaks=(20, 60), alpha=(0.15, 0.5)):
        super().__init__()
        self.num_streaks = num_streaks
        self.alpha       = alpha

    def forward(self, x):
        x = x.clone()
        B, C, H, W = x.shape
        n = random.randint(*self.num_streaks)
        a = random.uniform(*self.alpha)
        for _ in range(n):
            xc = random.randint(0, W-1)
            y0 = random.randint(0, H//2)
            y1 = random.randint(H//2, H)
            x[:, :, y0:y1, max(0,xc-1):xc+1] = (
                x[:, :, y0:y1, max(0,xc-1):xc+1] * (1-a) + a)
        return x.clamp(0,1)


class SaltPepperNoise(nn.Module):
    def __init__(self, amount_range=(0.01, 0.08)):
        super().__init__()
        self.amount_range = amount_range

    def forward(self, x):
        amount = random.uniform(*self.amount_range)
        noise  = torch.rand_like(x)
        out    = x.clone()
        out[noise < amount/2]    = 0.0
        out[noise > 1-amount/2]  = 1.0
        return out


PERTURBATIONS = {
    "blur":      GaussianBlur,
    "glare":     GlareOverlay,
    "occlusion": OcclusionPatch,
    "rain":      RainStreaks,
    "noise":     SaltPepperNoise,
}


class CompositePerturbation(nn.Module):
    """Randomly applies 1-N perturbations to an image batch."""

    def __init__(self, severity=0.5, max_simultaneous=2):
        super().__init__()
        self.severity         = severity
        self.max_simultaneous = max_simultaneous
        self._perturbers      = {k: v() for k, v in PERTURBATIONS.items()}

    def forward(self, x) -> Tuple[torch.Tensor, List[List[str]]]:
        B = x.shape[0]
        applied_all = []
        out = x.clone()
        for b in range(B):
            keys = list(PERTURBATIONS.keys())
            random.shuffle(keys)
            n_apply = random.randint(0, min(self.max_simultaneous,
                                            max(1, int(self.severity * len(keys)))))
            applied = []
            img_b = out[b:b+1]
            for k in keys[:n_apply]:
                img_b = self._perturbers[k](img_b)
                applied.append(k)
            out[b] = img_b[0]
            applied_all.append(applied)
        return out, applied_all
