#!/usr/bin/env python3
"""Cityscapes SegFormer -> road assets -> ego-frame BEV points.
Classes: pole(5) traffic_light(6) traffic_sign(7) fence(4) sidewalk(1) road(0)
Curbs are derived as the road/sidewalk mask boundary."""
import json, sys, argparse
import numpy as np, cv2, torch
from pathlib import Path
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
import os
MODEL = os.environ.get("SEGFORMER",
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
ASSETS = {5: "pole", 6: "traffic_light", 7: "traffic_sign", 4: "fence_barrier"}
ROAD, SIDEWALK = 0, 1

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0, help="0 = all samples")
ap.add_argument("--out", default="outputs/artifacts/road_assets.json")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {dev}")
proc = SegformerImageProcessor.from_pretrained(MODEL)
net = SegformerForSemanticSegmentation.from_pretrained(MODEL).to(dev).eval()

rows = [json.loads(l) for l in open(ROOT/"outputs/artifacts/nuscenes_mini_manifest.jsonl")]
if args.limit: rows = rows[:args.limit]

ASSET_Z = {"traffic_light": 4.5, "traffic_sign": 2.2, "pole": 0.0, "fence_barrier": 0.0}

def lift(u, v, K_inv, R, t, z=0.0):
    """Ray through pixel -> intersect plane at height z. Returns (x_fwd, y_left)."""
    d = R @ (K_inv @ np.array([u, v, 1.0]))
    if abs(d[2]) < 1e-6:
        return None
    s = (z - t[2]) / d[2]
    if s <= 0 or s > 60:
        return None
    p = t + s * d
    return float(p[0]), float(p[1])

out = {}
for n, row in enumerate(rows):
    tok = row["sample_token"]
    assets, surf = [], {"road": [], "sidewalk": [], "curb": []}

    for cam, rel in row["cams"].items():
        f = ROOT / rel
        if not f.exists():
            continue
        K = np.array(row["intrinsics"][cam], dtype=np.float64)
        M = np.array(row["extrinsics"][cam], dtype=np.float64)
        K_inv, R, t = np.linalg.inv(K), M[:3, :3], M[:3, 3]

        img = Image.open(f).convert("RGB")
        W, H = img.size
        with torch.no_grad():
            lo = net(**{k: v.to(dev) for k, v in proc(img, return_tensors="pt").items()}).logits
        seg = torch.nn.functional.interpolate(lo, size=(H, W), mode="bilinear",
                                              align_corners=False).argmax(1)[0].cpu().numpy()

        # upright assets: base pixel of each connected component
        for cid, name in ASSETS.items():
            m = (seg == cid).astype(np.uint8)
            if m.sum() < (60 if name in ("traffic_light", "traffic_sign") else 200):
                continue
            k, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
            for i in range(1, k):
                if st[i, cv2.CC_STAT_AREA] < (60 if name in ("traffic_light","traffic_sign") else 200):
                    continue
                z = ASSET_Z.get(name, 0.0)
                u = int(cen[i][0])
                v = (int(cen[i][1]) if z > 0
                     else int(st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] - 1))
                p = lift(u, v, K_inv, R, t, z=z)
                if p:
                    assets.append({"cls": name, "x": round(p[0], 2), "y": round(p[1], 2),
                                   "cam": cam, "px_h": int(st[i, cv2.CC_STAT_HEIGHT])})

        # surfaces + curb = road/sidewalk boundary
        rm = (seg == ROAD).astype(np.uint8)
        sm = (seg == SIDEWALK).astype(np.uint8)
        cm = cv2.dilate(rm, np.ones((9, 9), np.uint8)) & cv2.dilate(sm, np.ones((9, 9), np.uint8))
        for name, mask, step in (("road", rm, 40), ("sidewalk", sm, 40), ("curb", cm, 12)):
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            for i in range(0, len(xs), step):
                p = lift(int(xs[i]), int(ys[i]), K_inv, R, t, z=0.0)
                if p:
                    surf[name].append([round(p[0], 2), round(p[1], 2)])

    out[tok] = {"assets": assets, "surfaces": surf}
    print(f"[{n+1}/{len(rows)}] {tok[:8]} assets={len(assets)} curb={len(surf['curb'])}", flush=True)

op = ROOT / args.out
op.parent.mkdir(parents=True, exist_ok=True)
op.write_text(json.dumps(out))
print(f"wrote {op}  samples={len(out)}")
