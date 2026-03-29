from __future__ import annotations
import torch
from torch.utils.data import Dataset

class SyntheticMultiViewVideo(Dataset):
    def __init__(self, n: int = 512, views: int = 3, frames: int = 6, h: int = 224, w: int = 224, bev: int = 64, horizon: int = 12):
        self.n = n
        self.views = views
        self.frames = frames
        self.h = h
        self.w = w
        self.bev = bev
        self.horizon = horizon

    def __len__(self): return self.n

    def __getitem__(self, idx: int):
        x = torch.rand(self.views, self.frames, 3, self.h, self.w)  # [V,T,C,H,W]
        occ = (torch.rand(1, self.bev, self.bev) > 0.7).float()     # sparse occupancy
        traj = torch.cumsum(torch.randn(self.horizon, 2) * 0.05, dim=0)
        return x, occ, traj
