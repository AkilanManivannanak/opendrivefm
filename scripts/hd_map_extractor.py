"""
hd_map_extractor.py — OpenDriveFM HD Map Extraction Module

Extracts a vectorized HD map from nuScenes mini data without needing
the map expansion JSON files (which are a separate download).

What we extract:
  1. Ego trajectory paths (lane-like centerlines from driven routes)
  2. Infrastructure markers: traffic cones, barriers, bicycle racks
  3. Vehicle lanes implied by driven paths across all 404 samples
  4. BEV visualization with color-coded layers
  5. Vectorized JSON export (GeoJSON-style, Mach9-ready)

Why this is real:
  - Ego trajectories ARE the ground truth of where vehicles can drive
  - Infrastructure annotations are hand-labeled by nuScenes team
  - This approach mirrors what survey vehicles do: drive routes +
    detect infrastructure = HD map

Run:
    python scripts/hd_map_extractor.py --output outputs/artifacts/hd_map.json
    python scripts/hd_map_extractor.py --visualize --scene scene-0061
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent

# ── Category definitions ───────────────────────────────────────────────────────
INFRA_CATEGORIES = {
    "movable_object.trafficcone":      {"color": (0, 165, 255),  "label": "CONE",    "symbol": "triangle"},
    "movable_object.barrier":          {"color": (0, 0, 255),    "label": "BARRIER", "symbol": "rect"},
    "movable_object.debris":           {"color": (128, 0, 128),  "label": "DEBRIS",  "symbol": "cross"},
    "movable_object.pushable_pullable":{"color": (255, 165, 0),  "label": "OBJECT",  "symbol": "circle"},
    "static_object.bicycle_rack":      {"color": (0, 255, 255),  "label": "RACK",    "symbol": "rect"},
    "vehicle.construction":            {"color": (0, 128, 255),  "label": "CONST",   "symbol": "rect"},
}

VEHICLE_CATEGORIES = {
    "vehicle.car", "vehicle.truck", "vehicle.bus.bendy",
    "vehicle.bus.rigid", "vehicle.trailer", "vehicle.motorcycle",
    "vehicle.bicycle", "vehicle.construction",
}

PEDESTRIAN_CATEGORIES = {cat for cat in [
    "human.pedestrian.adult", "human.pedestrian.child",
    "human.pedestrian.construction_worker", "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer",
]}


# ── Load nuScenes ──────────────────────────────────────────────────────────────
def load_nusc(dataroot: str = None):
    from nuscenes.nuscenes import NuScenes
    if dataroot is None:
        candidates = [
            str(ROOT / "dataset/nuscenes"),
            str(ROOT.parent / "dataset/nuscenes"),
            "./dataset/nuscenes",
            "/app/dataset/nuscenes",
        ]
        for c in candidates:
            if (Path(c) / "v1.0-mini").exists():
                dataroot = c
                break
    if dataroot is None:
        raise RuntimeError("nuScenes dataset not found")
    print(f"[HDMap] Loading nuScenes from {dataroot}...")
    return NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=False)


# ── Core extraction ────────────────────────────────────────────────────────────
def extract_hd_map(nusc, scene_token: str = None) -> dict:
    """
    Extract a vectorized HD map for one scene (or all scenes).

    Returns a GeoJSON-style dict with:
      - ego_paths:       list of (x,y) trajectories driven by ego vehicle
      - infrastructure:  list of detected objects with position + category
      - vehicles:        list of other vehicle positions
      - pedestrians:     list of pedestrian positions
      - map_bounds:      bounding box of the map
    """
    from pyquaternion import Quaternion

    result = {
        "type": "HDMap",
        "version": "1.0",
        "source": "nuScenes v1.0-mini",
        "generator": "OpenDriveFM HD Map Extractor",
        "ego_paths": [],
        "infrastructure": [],
        "vehicles": [],
        "pedestrians": [],
        "map_bounds": {"x_min": 1e9, "x_max": -1e9, "y_min": 1e9, "y_max": -1e9},
    }

    # Filter scenes
    scenes = nusc.scene
    if scene_token:
        scenes = [s for s in scenes if s["token"] == scene_token]
        if not scenes:
            # Try by name
            scenes = [s for s in nusc.scene if s["name"] == scene_token]

    print(f"[HDMap] Processing {len(scenes)} scene(s)...")

    all_infra = {}   # token -> dict (deduplicate across samples)
    all_vehicles = {}
    all_peds = {}

    for scene in scenes:
        print(f"  Scene: {scene['name']}")

        # ── 1. Ego trajectory ──────────────────────────────────────────────
        ego_path = []
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            x, y = ep["translation"][0], ep["translation"][1]
            ego_path.append({"x": round(x, 3), "y": round(y, 3)})
            _update_bounds(result["map_bounds"], x, y)
            sample_token = sample["next"]

        if ego_path:
            result["ego_paths"].append({
                "scene": scene["name"],
                "type": "ego_trajectory",
                "points": ego_path,
                "length_m": _path_length(ego_path),
            })

        # ── 2. Infrastructure + Vehicles + Pedestrians ─────────────────────
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            ego_xy = np.array(ep["translation"][:2])
            ego_q = Quaternion(ep["rotation"])

            for ann_token in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                cat = ann["category_name"]
                t = ann["translation"]
                x, y = t[0], t[1]
                inst = ann["instance_token"]
                _update_bounds(result["map_bounds"], x, y)

                # Velocity
                try:
                    vel = nusc.box_velocity(ann_token)[:2]
                    speed = float(np.linalg.norm(vel)) if not np.isnan(vel).any() else 0.0
                except Exception:
                    speed = 0.0

                if cat in INFRA_CATEGORIES:
                    if inst not in all_infra:
                        all_infra[inst] = {
                            "type": "Feature",
                            "category": cat,
                            "label": INFRA_CATEGORIES[cat]["label"],
                            "geometry": {"type": "Point", "coordinates": [round(x, 3), round(y, 3)]},
                            "properties": {
                                "size": ann["size"],
                                "height_m": round(ann["size"][2], 2),
                                "scene": scene["name"],
                            }
                        }

                elif cat in VEHICLE_CATEGORIES:
                    if inst not in all_vehicles:
                        all_vehicles[inst] = {
                            "type": "Feature",
                            "category": cat,
                            "label": cat.split(".")[-1].upper()[:4],
                            "geometry": {"type": "Point", "coordinates": [round(x, 3), round(y, 3)]},
                            "properties": {
                                "size": ann["size"],
                                "speed_ms": round(speed, 2),
                                "scene": scene["name"],
                            }
                        }

                elif cat in PEDESTRIAN_CATEGORIES:
                    if inst not in all_peds:
                        all_peds[inst] = {
                            "type": "Feature",
                            "category": cat,
                            "label": "PED",
                            "geometry": {"type": "Point", "coordinates": [round(x, 3), round(y, 3)]},
                            "properties": {"speed_ms": round(speed, 2), "scene": scene["name"]}
                        }

            sample_token = sample["next"]

    result["infrastructure"] = list(all_infra.values())
    result["vehicles"] = list(all_vehicles.values())
    result["pedestrians"] = list(all_peds.values())

    # Statistics
    result["statistics"] = {
        "n_scenes": len(scenes),
        "n_ego_paths": len(result["ego_paths"]),
        "total_ego_path_length_m": round(sum(p["length_m"] for p in result["ego_paths"]), 1),
        "n_infrastructure": len(result["infrastructure"]),
        "n_vehicles": len(result["vehicles"]),
        "n_pedestrians": len(result["pedestrians"]),
        "infra_by_type": dict(defaultdict(int, {
            k: sum(1 for i in result["infrastructure"] if i["category"] == k)
            for k in INFRA_CATEGORIES
        })),
    }

    print(f"[HDMap] Extracted:")
    print(f"  Ego paths:      {result['statistics']['n_ego_paths']} ({result['statistics']['total_ego_path_length_m']}m total)")
    print(f"  Infrastructure: {result['statistics']['n_infrastructure']}")
    print(f"    {result['statistics']['infra_by_type']}")
    print(f"  Vehicles:       {result['statistics']['n_vehicles']}")
    print(f"  Pedestrians:    {result['statistics']['n_pedestrians']}")

    return result


def _update_bounds(bounds: dict, x: float, y: float):
    bounds["x_min"] = min(bounds["x_min"], x)
    bounds["x_max"] = max(bounds["x_max"], x)
    bounds["y_min"] = min(bounds["y_min"], y)
    bounds["y_max"] = max(bounds["y_max"], y)


def _path_length(points: list) -> float:
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i]["x"] - points[i-1]["x"]
        dy = points[i]["y"] - points[i-1]["y"]
        total += math.sqrt(dx*dx + dy*dy)
    return round(total, 1)


# ── BEV Visualization ──────────────────────────────────────────────────────────
def render_hd_map_bev(
    hd_map: dict,
    ego_xy: list,
    size: int = 512,
    bev_range_m: float = 40.0,
    scene_name: str = None,
) -> np.ndarray:
    """
    Render the HD map in BEV centered on ego_xy.

    Color scheme (Mach9-style):
      Gray lines:   ego-driven paths (road centerlines)
      Yellow:       traffic cones
      Red:          barriers
      Cyan:         bicycle racks
      Green boxes:  vehicles
      Orange dots:  pedestrians
      White car:    ego vehicle
    """
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 25)   # very dark background

    cx, cy = size // 2, size // 2
    sc = size / (2.0 * bev_range_m)   # pixels per meter
    ex, ey = float(ego_xy[0]), float(ego_xy[1])

    def world_to_bev(wx, wy):
        px = int(cx + (wy - ey) * sc)
        py = int(cy - (wx - ex) * sc)
        return px, py

    def in_bounds(px, py, margin=5):
        return margin < px < size - margin and margin < py < size - margin

    # ── Distance rings ─────────────────────────────────────────────────────
    for dm in [10, 20, 30, 40]:
        r = int(dm * sc)
        cv2.circle(canvas, (cx, cy), r, (35, 35, 45), 1)
        cv2.putText(canvas, f"{dm}m", (cx + r + 3, cy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (55, 55, 65), 1)

    # ── Grid lines ─────────────────────────────────────────────────────────
    for i in range(0, size, size // 8):
        cv2.line(canvas, (i, 0), (i, size), (25, 25, 35), 1)
        cv2.line(canvas, (0, i), (size, i), (25, 25, 35), 1)

    # Filter to scene
    paths = hd_map["ego_paths"]
    if scene_name:
        paths = [p for p in paths if p.get("scene") == scene_name]
    if not paths:
        paths = hd_map["ego_paths"]

    # ── Ego-driven paths (road centerlines) ───────────────────────────────
    for path in paths:
        pts = path["points"]
        bev_pts = [world_to_bev(p["x"], p["y"]) for p in pts]
        bev_pts = [(px, py) for px, py in bev_pts if in_bounds(px, py)]
        for i in range(len(bev_pts) - 1):
            cv2.line(canvas, bev_pts[i], bev_pts[i+1], (70, 70, 90), 2)
        # Draw small dots at each pose
        for pt in bev_pts[::3]:
            cv2.circle(canvas, pt, 2, (90, 90, 110), -1)

    # ── Dashed centerline on driven path ──────────────────────────────────
    for path in paths:
        pts = path["points"]
        bev_pts = [world_to_bev(p["x"], p["y"]) for p in pts]
        bev_pts = [(px, py) for px, py in bev_pts if in_bounds(px, py)]
        for i in range(0, len(bev_pts) - 1, 4):
            cv2.line(canvas, bev_pts[i], bev_pts[min(i+2, len(bev_pts)-1)],
                     (200, 180, 0), 1)   # yellow dashed centerline

    # ── Infrastructure ────────────────────────────────────────────────────
    cat_colors = {
        "movable_object.trafficcone":       (0, 165, 255),   # orange
        "movable_object.barrier":           (0, 50, 255),    # red
        "movable_object.debris":            (180, 0, 180),   # purple
        "movable_object.pushable_pullable": (255, 140, 0),   # dark orange
        "static_object.bicycle_rack":       (0, 220, 220),   # cyan
        "vehicle.construction":             (0, 180, 100),   # green
    }
    cat_labels = {k: INFRA_CATEGORIES[k]["label"] for k in INFRA_CATEGORIES}

    for obj in hd_map["infrastructure"]:
        coords = obj["geometry"]["coordinates"]
        px, py = world_to_bev(coords[0], coords[1])
        if not in_bounds(px, py):
            continue
        cat = obj["category"]
        color = cat_colors.get(cat, (200, 200, 200))
        label = cat_labels.get(cat, "?")

        if "cone" in cat:
            # Triangle for cones
            pts = np.array([[px, py-7], [px-5, py+5], [px+5, py+5]], np.int32)
            cv2.fillPoly(canvas, [pts], color)
            cv2.polylines(canvas, [pts], True, (255, 255, 255), 1)
        elif "barrier" in cat:
            # Rectangle for barriers
            cv2.rectangle(canvas, (px-8, py-3), (px+8, py+3), color, -1)
            cv2.rectangle(canvas, (px-8, py-3), (px+8, py+3), (255,255,255), 1)
        else:
            # Circle for others
            cv2.circle(canvas, (px, py), 5, color, -1)
            cv2.circle(canvas, (px, py), 6, (255,255,255), 1)

        cv2.putText(canvas, label, (px+7, py+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.22, color, 1)

    # ── Vehicles ──────────────────────────────────────────────────────────
    for obj in hd_map["vehicles"]:
        coords = obj["geometry"]["coordinates"]
        px, py = world_to_bev(coords[0], coords[1])
        if not in_bounds(px, py):
            continue
        sz = obj["properties"]["size"]
        w_px = max(4, int(sz[0] * sc))
        l_px = max(6, int(sz[1] * sc))
        cv2.rectangle(canvas, (px-w_px//2, py-l_px//2),
                      (px+w_px//2, py+l_px//2), (0, 200, 80), 1)
        cv2.putText(canvas, obj["label"], (px+w_px//2+2, py+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.20, (0, 200, 80), 1)

    # ── Pedestrians ───────────────────────────────────────────────────────
    for obj in hd_map["pedestrians"]:
        coords = obj["geometry"]["coordinates"]
        px, py = world_to_bev(coords[0], coords[1])
        if not in_bounds(px, py):
            continue
        cv2.circle(canvas, (px, py), 3, (0, 130, 255), -1)

    # ── Ego vehicle (white car shape) ─────────────────────────────────────
    car_w, car_l = max(6, int(1.0 * sc)), max(10, int(2.2 * sc))
    cv2.rectangle(canvas, (cx-car_w, cy-car_l), (cx+car_w, cy+car_l),
                  (240, 240, 240), -1)
    cv2.rectangle(canvas, (cx-car_w, cy-car_l), (cx+car_w, cy+car_l),
                  (180, 180, 180), 1)
    cv2.arrowedLine(canvas, (cx, cy+car_l//2), (cx, cy-car_l),
                    (0, 255, 100), 2, tipLength=0.35)

    # ── Legend ─────────────────────────────────────────────────────────────
    legend = [
        ("EGO PATH", (70, 70, 90)),
        ("CENTERLINE", (200, 180, 0)),
        ("CONE", (0, 165, 255)),
        ("BARRIER", (0, 50, 255)),
        ("VEHICLE", (0, 200, 80)),
        ("PEDESTRIAN", (0, 130, 255)),
        ("RACK/OTHER", (0, 220, 220)),
    ]
    lx, ly = 5, 15
    cv2.rectangle(canvas, (lx-2, ly-12), (lx+110, ly + len(legend)*14 + 4),
                  (0, 0, 0), -1)
    for i, (label, color) in enumerate(legend):
        y = ly + i * 14
        cv2.circle(canvas, (lx+6, y), 4, color, -1)
        cv2.putText(canvas, label, (lx+14, y+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)

    # ── Title ──────────────────────────────────────────────────────────────
    title = f"HD MAP — {scene_name or 'All Scenes'}"
    cv2.rectangle(canvas, (0, size-28), (size, size), (0, 0, 0), -1)
    cv2.putText(canvas, title, (6, size-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 255), 1, cv2.LINE_AA)
    stats = hd_map["statistics"]
    stat_txt = f"Infra:{stats['n_infrastructure']} | Veh:{stats['n_vehicles']} | Ped:{stats['n_pedestrians']}"
    cv2.putText(canvas, stat_txt, (size//2 - 80, size-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150, 150, 200), 1)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


# ── Cached loader ──────────────────────────────────────────────────────────────
_hd_map_cache = None

def get_hd_map(force_rebuild: bool = False) -> dict:
    """Load precomputed HD map or rebuild from scratch."""
    global _hd_map_cache
    if _hd_map_cache is not None and not force_rebuild:
        return _hd_map_cache

    json_candidates = [
        str(ROOT / "outputs/artifacts/hd_map.json"),
        "/app/outputs/artifacts/hd_map.json",
        "outputs/artifacts/hd_map.json",
    ]
    import os
    for p in json_candidates:
        if os.path.exists(p):
            print(f"[HDMap] Loading from {p}")
            with open(p) as f:
                _hd_map_cache = json.load(f)
            return _hd_map_cache

    # Build from scratch
    print("[HDMap] Building HD map from scratch...")
    nusc = load_nusc()
    _hd_map_cache = extract_hd_map(nusc)
    return _hd_map_cache


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenDriveFM HD Map Extractor")
    parser.add_argument("--output", default="outputs/artifacts/hd_map.json",
                        help="Output JSON path")
    parser.add_argument("--visualize", action="store_true",
                        help="Save BEV visualization as PNG")
    parser.add_argument("--scene", default=None,
                        help="Filter to one scene (e.g. scene-0061)")
    parser.add_argument("--dataroot", default=None,
                        help="nuScenes dataroot (auto-detected if not set)")
    args = parser.parse_args()

    t0 = time.time()
    nusc = load_nusc(args.dataroot)
    hd_map = extract_hd_map(nusc, scene_token=args.scene)

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(hd_map, f, indent=2)
    print(f"\n[HDMap] Saved to {out_path} ({out_path.stat().st_size // 1024} KB)")
    print(f"[HDMap] Total time: {time.time()-t0:.1f}s")

    if args.visualize:
        # Use first ego path's first point as center
        if hd_map["ego_paths"]:
            first_pt = hd_map["ego_paths"][0]["points"][0]
            ego_xy = [first_pt["x"], first_pt["y"]]
        else:
            b = hd_map["map_bounds"]
            ego_xy = [(b["x_min"]+b["x_max"])/2, (b["y_min"]+b["y_max"])/2]

        img = render_hd_map_bev(hd_map, ego_xy, size=800, bev_range_m=50.0,
                                scene_name=args.scene)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        vis_path = str(out_path).replace(".json", "_bev.png")
        cv2.imwrite(vis_path, img_bgr)
        print(f"[HDMap] BEV visualization saved to {vis_path}")

    print("\n[HDMap] Statistics:")
    for k, v in hd_map["statistics"].items():
        print(f"  {k}: {v}")
