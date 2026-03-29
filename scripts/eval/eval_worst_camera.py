"""
eval_worst_camera.py — per-camera worst-case failure ranking
Copy to ~/opendrivefm/scripts/eval_worst_camera.py
Run: python scripts/eval_worst_camera.py
"""
import sys, json, tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

VAL_SCENES = {"scene-0655", "scene-1077"}
CAMS = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT",
        "CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]

def load_model_safe(ckpt_path, device):
    from opendrivefm.models.model import OpenDriveFM
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw  = ckpt.get("state_dict", ckpt)
    sd   = {}
    for k, v in raw.items():
        if   k.startswith("model."): sd[k[6:]]  = v
        elif k.startswith("lit.model."): sd[k[10:]] = v
        elif k.startswith("lit."): sd[k[4:]]  = v
        else: sd[k] = v
    model = OpenDriveFM()
    model_sd = model.state_dict()
    loaded = 0
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v; loaded += 1
    model.load_state_dict(model_sd)
    print(f"  Loaded {loaded} layers")
    return model.eval().to(device)

def iou(pred_logits, occ_gt, thr=0.5):
    pred = (torch.sigmoid(pred_logits) > thr).float().cpu()
    gt   = occ_gt.float().cpu()
    if gt.ndim == 3: gt = gt.unsqueeze(1)
    gt = F.interpolate(gt, size=pred.shape[-2:], mode='nearest')
    TP = (pred*gt).sum().item()
    FP = (pred*(1-gt)).sum().item()
    FN = ((1-pred)*gt).sum().item()
    return TP/(TP+FP+FN+1e-8)

def ade(pred, gt):
    return (pred-gt).norm(dim=-1).mean().item()

def main():
    from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView
    from torch.utils.data import DataLoader

    rows = [json.loads(l) for l in
            Path("artifacts/nuscenes_mini_manifest.jsonl").read_text().splitlines() if l.strip()]
    scene_key = "scene" if "scene" in rows[0] else "scene_name"
    val_rows  = [r for r in rows if r.get(scene_key,"") in VAL_SCENES]

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    for r in val_rows: tmp.write(json.dumps(r)+"\n")
    tmp.close()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Loading model...")
    model = load_model_safe("artifacts/checkpoints_v9/best_val_ade.ckpt", device)

    ds = NuScenesMiniMultiView(tmp.name, label_root="artifacts/nuscenes_labels",
                               return_motion=True, return_trel=True)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    # Baseline
    print("Baseline (no fault)...")
    b_ious, b_ades = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device); occ_gt = batch[1].to(device); traj_gt = batch[2].to(device)
            out = model(x)
            b_ious.append(iou(out[0], occ_gt))
            b_ades.append(ade(out[1].cpu(), traj_gt.cpu()))
    baseline_iou = float(np.mean(b_ious))
    baseline_ade = float(np.mean(b_ades))
    print(f"Baseline: IoU={baseline_iou:.4f}  ADE={baseline_ade:.4f}")

    all_results = []
    for fault_type in ["blackout", "blur"]:
        print(f"\n--- {fault_type} ---")
        cam_results = []
        for cam_idx in range(6):
            ious_f, ades_f, trusts_f = [], [], []
            with torch.no_grad():
                for batch in loader:
                    x = batch[0].to(device)
                    occ_gt = batch[1].to(device)
                    traj_gt = batch[2].to(device)
                    x_mod = x.clone()
                    if fault_type == "blackout":
                        x_mod[:, cam_idx] = 0.0
                    else:
                        # Simple blur via avg pool
                        B,V,T,C,H,W = x.shape
                        flat = x_mod[:, cam_idx, 0]  # (B,3,H,W)
                        blurred = F.avg_pool2d(flat, 9, stride=1, padding=4)
                        x_mod[:, cam_idx, 0] = blurred
                    out = model(x_mod)
                    ious_f.append(iou(out[0], occ_gt))
                    ades_f.append(ade(out[1].cpu(), traj_gt.cpu()))
                    if out[2] is not None:
                        trusts_f.append(out[2].cpu()[:, cam_idx].mean().item())

            r = dict(
                cam_idx=cam_idx,
                cam_name=CAMS[cam_idx],
                fault_type=fault_type,
                IoU=float(np.mean(ious_f)),
                ADE=float(np.mean(ades_f)),
                IoU_drop=baseline_iou - float(np.mean(ious_f)),
                ADE_increase=float(np.mean(ades_f)) - baseline_ade,
                trust_faulted=float(np.mean(trusts_f)) if trusts_f else float("nan"),
            )
            cam_results.append(r)
            print(f"  {CAMS[cam_idx]:25s} IoU={r['IoU']:.4f} drop={r['IoU_drop']:+.4f} "
                  f"ADE={r['ADE']:.3f} trust={r['trust_faulted']:.3f}")

        cam_results.sort(key=lambda x: -x["IoU_drop"])
        print(f"\n  Worst→Best camera ranking ({fault_type} fault):")
        print(f"  {'Rank':<5} {'Camera':<25} {'IoU Drop':>10} {'ADE Rise':>10}")
        print(f"  {'─'*55}")
        for rank, r in enumerate(cam_results, 1):
            print(f"  #{rank:<4} {r['cam_name']:<25} {r['IoU_drop']:>+10.4f} {r['ADE_increase']:>+10.4f}")
        all_results.extend(cam_results)

    out_path = Path("artifacts/per_camera_fault_ranking.json")
    out_path.write_text(json.dumps({
        "baseline_iou": baseline_iou,
        "baseline_ade": baseline_ade,
        "results": all_results
    }, indent=2))
    print(f"\nSaved → {out_path}")
    Path(tmp.name).unlink()

if __name__ == "__main__":
    main()
