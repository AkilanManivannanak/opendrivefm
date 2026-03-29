"""
eval_camera_dropout.py — final version, strips 'model.' prefix
Copy to ~/opendrivefm/scripts/eval_camera_dropout.py
Run: python scripts/eval_camera_dropout.py --ckpt artifacts/checkpoints_v11_temporal/best_val_ade.ckpt --label_root artifacts/nuscenes_labels_128
"""
import sys, json, argparse, tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from opendrivefm.models.model import OpenDriveFM
from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView
from torch.utils.data import DataLoader

VAL_SCENES = {"scene-0655", "scene-1077"}

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw  = ckpt.get("state_dict", ckpt)
    sd = {}
    for k, v in raw.items():
        if k.startswith("model."):       sd[k[6:]] = v
        elif k.startswith("lit.model."): sd[k[10:]] = v
        elif k.startswith("lit."):       sd[k[4:]] = v
        else:                            sd[k] = v
    model = OpenDriveFM()
    miss, unex = model.load_state_dict(sd, strict=False)
    print(f"  Loaded — missing={len(miss)} unexpected={len(unex)}")
    model.eval()
    return model.to(device)

def compute_iou(pred_logits, occ_gt, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float().cpu()
    gt   = occ_gt.float().cpu()
    if gt.ndim == 3: gt = gt.unsqueeze(1)
    gt = F.interpolate(gt, size=pred.shape[-2:], mode='nearest')
    TP = (pred * gt).sum().item()
    FP = (pred * (1-gt)).sum().item()
    FN = ((1-pred) * gt).sum().item()
    return TP / (TP + FP + FN + 1e-8)

def make_val_manifest(manifest_path):
    rows = [json.loads(l) for l in Path(manifest_path).read_text().splitlines() if l.strip()]
    scene_key = "scene" if "scene" in rows[0] else "scene_name"
    filtered = [r for r in rows if r.get(scene_key,"") in VAL_SCENES]
    print(f"Val samples: {len(filtered)}")
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    for r in filtered: tmp.write(json.dumps(r) + "\n")
    tmp.close()
    return tmp.name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       default="artifacts/checkpoints_v11_temporal/best_val_ade.ckpt")
    ap.add_argument("--manifest",   default="artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", default="artifacts/nuscenes_labels_128")
    ap.add_argument("--threshold",  type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading: {args.ckpt}")
    model = load_model(args.ckpt, device)

    tmp_manifest = make_val_manifest(args.manifest)
    ds     = NuScenesMiniMultiView(tmp_manifest, label_root=args.label_root,
                                   return_motion=True, return_trel=True)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    CAMERA_ORDER = [0, 1, 2, 3, 4, 5]
    results = []

    for n_drop in range(4):
        ious, ades, t_drop, t_clean = [], [], [], []
        fault_mask = torch.zeros(6, dtype=torch.bool)
        for ci in CAMERA_ORDER[:n_drop]:
            fault_mask[ci] = True

        with torch.no_grad():
            for batch in loader:
                x       = batch[0].to(device)
                occ_gt  = batch[1].to(device)
                traj_gt = batch[2].to(device)
                x_mod   = x.clone()
                for ci in CAMERA_ORDER[:n_drop]:
                    x_mod[:, ci] = 0.0
                out       = model(x_mod)
                occ_pred  = out[0]
                traj_pred = out[1]
                trust     = out[2]
                ious.append(compute_iou(occ_pred, occ_gt, args.threshold))
                ades.append((traj_pred.cpu() - traj_gt.cpu()).norm(dim=-1).mean().item())
                if trust is not None:
                    t = trust.cpu()
                    if n_drop > 0:
                        t_drop.append(t[:, fault_mask].mean().item())
                    if n_drop < 6:
                        t_clean.append(t[:, ~fault_mask].mean().item())

        r = dict(
            n_dropout    = n_drop,
            IoU          = float(np.mean(ious)),
            ADE          = float(np.mean(ades)),
            trust_dropout= float(np.mean(t_drop))  if t_drop  else float("nan"),
            trust_clean  = float(np.mean(t_clean)) if t_clean else float("nan"),
        )
        results.append(r)
        print(f"n_dropout={n_drop}  IoU={r['IoU']:.4f}  ADE={r['ADE']:.4f}  "
              f"trust_drop={r['trust_dropout']:.3f}  trust_clean={r['trust_clean']:.3f}")

    print("\n" + "="*65)
    print(f"{'N Dropped':>10} {'IoU':>8} {'ADE(m)':>8} {'Trust(drop)':>12} {'Trust(clean)':>13}")
    print("="*65)
    for r in results:
        td = f"{r['trust_dropout']:.3f}" if not np.isnan(r['trust_dropout']) else "   n/a"
        tc = f"{r['trust_clean']:.3f}"   if not np.isnan(r['trust_clean'])   else "   n/a"
        print(f"{r['n_dropout']:>10}   {r['IoU']:>6.4f}   {r['ADE']:>6.4f}   {td:>11}   {tc:>12}")
    print("="*65)

    out_path = Path("artifacts/camera_dropout_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved → {out_path}")
    Path(tmp_manifest).unlink()

if __name__ == "__main__":
    main()
