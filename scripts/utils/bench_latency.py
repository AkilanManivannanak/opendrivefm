from __future__ import annotations
import argparse, time
import numpy as np
import torch
from opendrivefm.models.model import OpenDriveFM

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--h", type=int, default=224)
    ap.add_argument("--w", type=int, default=224)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    model = OpenDriveFM().to(device).eval()

    x = torch.rand(1, args.views, args.frames, 3, args.h, args.w, device=device)

    # warmup
    for _ in range(args.warmup):
        _ = model(x)

    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        _ = model(x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    arr = np.array(times)
    print(f"device={device}")
    print(f"p50(ms)={np.percentile(arr,50):.3f}  p95(ms)={np.percentile(arr,95):.3f}  mean(ms)={arr.mean():.3f}")
    print(f"FPS(p50)={1000.0/np.percentile(arr,50):.2f}")

if __name__ == "__main__":
    main()
