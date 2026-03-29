"""
Multi-Camera Augmentation Suite  (Slide Step 3)
===============================================
Applied during training to improve invariance:
  - Color jitter (brightness, contrast, saturation, hue)
  - Random blur (Gaussian)
  - Random occlusion patches
  - Consistent normalisation across all cameras

These are applied PER-CAMERA independently during training only.
"""
from __future__ import annotations
import random
import math
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ImageNet-style normalisation (standard for pretrained CNNs)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class CameraAugment:
    """
    Applied independently to each camera image during training.
    Input/output: (C, H, W) float tensor in [0, 1]
    """

    def __init__(
        self,
        color_jitter_p:    float = 0.8,
        blur_p:            float = 0.3,
        occlusion_p:       float = 0.2,
        occlusion_frac:    Tuple[float, float] = (0.05, 0.2),
        brightness:        float = 0.3,
        contrast:          float = 0.3,
        saturation:        float = 0.2,
        hue:               float = 0.05,
        blur_sigma:        Tuple[float, float] = (0.5, 2.0),
    ):
        self.color_jitter_p = color_jitter_p
        self.blur_p         = blur_p
        self.occlusion_p    = occlusion_p
        self.occlusion_frac = occlusion_frac
        self.blur_sigma     = blur_sigma
        self._jitter = T.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """img: (C, H, W) float32 in [0,1]"""
        # Color jitter
        if random.random() < self.color_jitter_p:
            img = self._jitter(img)

        # Gaussian blur
        if random.random() < self.blur_p:
            sigma = random.uniform(*self.blur_sigma)
            k = max(3, 2 * int(math.ceil(3 * sigma)) + 1)
            img = TF.gaussian_blur(img, kernel_size=[k, k], sigma=sigma)

        # Random occlusion patch (simulates object blocking camera)
        if random.random() < self.occlusion_p:
            C, H, W = img.shape
            ph = int(H * random.uniform(*self.occlusion_frac))
            pw = int(W * random.uniform(*self.occlusion_frac))
            y0 = random.randint(0, H - ph)
            x0 = random.randint(0, W - pw)
            img = img.clone()
            img[:, y0:y0+ph, x0:x0+pw] = 0.0

        return img.clamp(0.0, 1.0)


class MultiCameraAugment:
    """
    Applies CameraAugment independently to each of V cameras.
    Input:  x (V, T, C, H, W) float32 in [0, 1]
    Output: x_aug (V, T, C, H, W) float32 in [0, 1]

    Augmentation is only applied during training (pass training=True).
    """

    def __init__(self, **kwargs):
        self.cam_aug = CameraAugment(**kwargs)

    def __call__(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        if not training:
            return x
        V, T, C, H, W = x.shape
        out = x.clone()
        for v in range(V):
            # Only augment first frame (T=0) per camera
            out[v, 0] = self.cam_aug(out[v, 0])
        return out


class ConsistentNormalise:
    """
    Applies ImageNet normalisation consistently across all cameras.
    Input:  (V, T, C, H, W) float32 in [0, 1]
    Output: (V, T, C, H, W) float32 normalised
    """

    def __init__(
        self,
        mean: List[float] = IMAGENET_MEAN,
        std:  List[float] = IMAGENET_STD,
    ):
        mean_t = torch.tensor(mean).view(1, 1, 3, 1, 1)
        std_t  = torch.tensor(std).view(1, 1, 3, 1, 1)
        self.register_mean = mean_t
        self.register_std  = std_t

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.register_mean.to(x.device)
        std  = self.register_std.to(x.device)
        return (x - mean) / std

    def denormalise(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.register_mean.to(x.device)
        std  = self.register_std.to(x.device)
        return (x * std + mean).clamp(0, 1)
