#!/usr/bin/env python3
"""LIDAR_TOP keyframe + prior sweeps with full pose chain.
Requires nuscenes-devkit; run locally, commit the JSON."""
import json
from pathlib import Path
from nuscenes.nuscenes import NuScenes

ROOT    = Path(__file__).resolve().parents[1]
OUT     = ROOT / "outputs/artifacts/lidar_manifest.json"
N_SWEEP = 10

nusc = NuScenes(version="v1.0-mini",
                dataroot=str(ROOT / "data/nuscenes"), verbose=False)

def rec(nusc, sd):
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])
    return {"path": f"data/nuscenes/{sd['filename']}",
            "cal_rot": cs['rotation'], "cal_trans": cs['translation'],
            "ego_rot": ep['rotation'], "ego_trans": ep['translation']}

out = {}
for sample in nusc.sample:
    sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ref = rec(nusc, sd)
    sweeps, cur = [], sd
    for _ in range(N_SWEEP - 1):
        if not cur['prev']:
            break
        cur = nusc.get('sample_data', cur['prev'])
        sweeps.append(rec(nusc, cur))
    ref["sweeps"] = sweeps
    out[sample['token']] = ref

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out))
print(f"wrote {OUT}  samples={len(out)}  sweeps/sample~{len(sweeps)+1}")
