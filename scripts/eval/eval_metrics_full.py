"""
eval_metrics_full.py — final version, strips 'model.' prefix
Copy to ~/opendrivefm/scripts/eval_metrics_full.py
Run: python scripts/eval_metrics_full.py --ckpt artifacts/checkpoints_v11_temporal/best_val_ade.ckpt --label_root artifacts/nuscenes_labels_128
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
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    raw   = ckpt.get("state_dict", ckpt)
    # Strip 'model.' prefix (v11 lightning saves as model.backbone.*)
    sd = {}
    for k, v in raw.items():
        if k.startswith("model."):
            sd[k[len("model."):]] = v
        elif k.startswith("lit.model."):
            sd[k[len("lit.model."):]] = v
        elif k.startswith("lit."):
            sd[k[len("lit."):]] = v
        else:
            sd[k] = v
    model = OpenDriveFM()
    miss, unex = model.load_state_dict(sd, strict=False)
    print(f"  Loaded OK — missing={len(miss)} unexpected={len(unex)}")
    if miss: print(f"  Sample missing: {miss[:2]}")
    model.eval()
    return model.to(device)

def compute_metrics(pred_logits, occ_gt, threshold=0.5):
    prob = torch.sigmoid(pred_logits).cpu()
    pred = (prob > threshold).float()
    gt   = occ_gt.float().cpu()
    if gt.ndim == 3: gt = gt.unsqueeze(1)
    gt = F.interpolate(gt, size=pred.shape[-2:], mode='nearest')
    TP = (pred * gt).sum().item()
    FP = (pred * (1-gt)).sum().item()
    FN = ((1-pred) * gt).sum().item()
    TN = ((1-pred) * (1-gt)).sum().item()
    return dict(
        IoU      = TP / (TP + FP + FN + 1e-8),
        Dice     = 2*TP / (2*TP + FP + FN + 1e-8),
        Precision= TP / (TP + FP + 1e-8),
        Recall   = TP / (TP + FN + 1e-8),
        Accuracy = (TP+TN) / (TP+TN+FP+FN + 1e-8),
    )

def make_filtered_manifest(manifest_path, split):
    rows = [json.loads(l) for l in Path(manifest_path).read_text().splitlines() if l.strip()]
    scene_key = "scene" if "scene" in rows[0] else "scene_name"
    if split == "val":
        filtered = [r for r in rows if r.get(scene_key,"") in VAL_SCENES]
    elif split == "train":
        filtered = [r for r in rows if r.get(scene_key,"") not in VAL_SCENES]
    else:
        filtered = rows
    print(f"Filtered: {len(filtered)} rows for split='{split}'")
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
    ap.add_argument("--split",      default="val", choices=["val","train","all"])
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading: {args.ckpt}")
    model = load_model(args.ckpt, device)
    print(f"Device: {device}")

    tmp_manifest = make_filtered_manifest(args.manifest, args.split)
    ds     = NuScenesMiniMultiView(tmp_manifest, label_root=args.label_root,
                                   return_motion=True, return_trel=True)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    print(f"Dataset: {len(ds)} samples")

    all_m = []
    with torch.no_grad():
        for batch in loader:
            x      = batch[0].to(device)
            occ_gt = batch[1].to(device)
            out    = model(x)
            occ_pred = out[0]
            all_m.append(compute_metrics(occ_pred, occ_gt, args.threshold))

    keys = ["IoU","Dice","Precision","Recall","Accuracy"]
    print("\n" + "="*52)
    print(f"RESULTS — {args.split} | threshold={args.threshold} | n_batches={len(all_m)}")
    print("="*52)
    for k in keys:
        vals = [m[k] for m in all_m]
        print(f"  {k:12s}: {np.mean(vals):.4f}  (std={np.std(vals):.4f})")
    print("="*52)

    agg = {k: float(np.mean([m[k] for m in all_m])) for k in keys}
    agg.update({"threshold": args.threshold, "n_batches": len(all_m)})
    out_path = Path(f"artifacts/metrics_{args.split}.json")
    out_path.write_text(json.dumps(agg, indent=2))
    print(f"Saved → {out_path}")
    Path(tmp_manifest).unlink()

if __name__ == "__main__":
    main()
