#!/usr/bin/env python3
"""Accumulate per-frame detections into a scene-level vector map.
ego frame -> global map frame -> dedupe -> GeoJSON + CSV."""
import json, csv, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from pyquaternion import Quaternion

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "outputs/artifacts/road_assets.json"
L = ROOT / "outputs/artifacts/lidar_manifest.json"
OUT = ROOT / "outputs/artifacts/vector_map"

ap = argparse.ArgumentParser()
ap.add_argument("--cluster", type=float, default=3.0, help="asset merge radius (m)")
ap.add_argument("--min-obs", type=int, default=3)
ap.add_argument("--min-px", type=float, default=45.0, help="min mean pixel height")
ap.add_argument("--max-range", type=float, default=30.0, help="max ego-frame lift range (m)")
args = ap.parse_args()

assets_by_tok = json.loads(A.read_text())
scene_of = {}
for line in open(ROOT/"outputs/artifacts/nuscenes_mini_manifest.jsonl"):
    r = json.loads(line); scene_of[r["sample_token"]] = r["scene"]
lidar = json.loads(L.read_text())
OUT.mkdir(parents=True, exist_ok=True)

def to_global(x, y, m):
    R = Quaternion(m["ego_rot"]).rotation_matrix
    t = np.array(m["ego_trans"])
    p = R @ np.array([x, y, 0.0]) + t
    return float(p[0]), float(p[1])

# ---- accumulate ------------------------------------------------------
raw_assets, curb_pts = [], []
for tok, e in assets_by_tok.items():
    m = lidar.get(tok)
    if m is None:
        continue
    sc_name = scene_of.get(tok, "?")
    for a in e.get("assets", []):
        if np.hypot(a["x"], a["y"]) > args.max_range:
            continue
        if a["cls"] not in ("traffic_light", "traffic_sign") and a.get("px_h", 0) < args.min_px:
            continue
        gx, gy = to_global(a["x"], a["y"], m)
        raw_assets.append((sc_name, a["cls"], gx, gy, a.get("px_h", 0)))
    for x, y in e.get("surfaces", {}).get("curb", []):
        if np.hypot(x, y) > args.max_range:
            continue
        curb_pts.append((sc_name,) + to_global(x, y, m))

print(f"raw: {len(raw_assets)} assets, {len(curb_pts)} curb pts")

# ---- dedupe assets on a grid ----------------------------------------
g = args.cluster
buckets = defaultdict(list)
for sc_name, cls, x, y, h in raw_assets:
    buckets[(sc_name, cls, round(x/g), round(y/g))].append((x, y, h))

merged = []
for (sc_name, cls, _, _), pts in buckets.items():
    arr = np.array(pts)
    merged.append({"scene": sc_name, "cls": cls,
                   "x": float(arr[:, 0].mean()),
                   "y": float(arr[:, 1].mean()),
                   "n_obs": len(pts),
                   "px_h": float(arr[:, 2].mean())})
merged = [m for m in merged if m["n_obs"] >= args.min_obs]   # seen at least twice
print(f"merged: {len(merged)} unique assets (>={args.min_obs} obs, >={args.min_px}px, <={args.max_range}m)")

# ---- curbs -> polylines ---------------------------------------------
cg = 1.0
cb = defaultdict(list)
for sc_name, x, y in curb_pts:
    cb[(sc_name, round(x/cg), round(y/cg))].append((x, y))
nodes = [(float(np.mean([p[0] for p in v])), float(np.mean([p[1] for p in v])))
         for k, v in cb.items() if len(v) >= 8]

lines, used = [], set()
for i, n0 in enumerate(nodes):
    if i in used:
        continue
    chain, cur, ci = [n0], n0, i
    used.add(i)
    while True:
        best, bi, bd = None, None, 4.0
        for j, nj in enumerate(nodes):
            if j in used:
                continue
            d = np.hypot(nj[0]-cur[0], nj[1]-cur[1])
            if d < bd:
                best, bi, bd = nj, j, d
        if best is None:
            break
        chain.append(best); used.add(bi); cur = best
    if len(chain) >= 12:
        lines.append(chain)
print(f"curb polylines: {len(lines)}")

# ---- write GeoJSON ---------------------------------------------------
feats = [{"type": "Feature",
          "geometry": {"type": "Point", "coordinates": [a["x"], a["y"]]},
          "properties": {"class": a["cls"], "observations": a["n_obs"]}}
         for a in merged]
feats += [{"type": "Feature",
           "geometry": {"type": "LineString", "coordinates": [list(p) for p in ln]},
           "properties": {"class": "curb", "n_points": len(ln)}}
          for ln in lines]

(OUT/"vector_map.geojson").write_text(json.dumps(
    {"type": "FeatureCollection",
     "crs": {"type": "name", "properties": {"name": "nuScenes map-local metres"}},
     "features": feats}, indent=1))

with open(OUT/"assets.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scene", "class", "x_m", "y_m", "observations", "mean_pixel_height"])
    for a in sorted(merged, key=lambda z: (z["scene"], z["cls"])):
        w.writerow([a["scene"], a["cls"], round(a["x"], 2), round(a["y"], 2),
                    a["n_obs"], round(a["px_h"], 1)])

counts = defaultdict(int)
for a in merged:
    counts[a["cls"]] += 1
print("\nasset inventory:")
for k, v in sorted(counts.items()):
    print(f"  {k:15s} {v}")
print(f"\nwrote {OUT}/vector_map.geojson and assets.csv")
