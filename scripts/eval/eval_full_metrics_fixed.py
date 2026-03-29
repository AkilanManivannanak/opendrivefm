"""
eval_full_metrics_fixed.py — correct Dice/Precision/Recall
Copy to ~/opendrivefm/scripts/eval_full_metrics_fixed.py
Run: python scripts/eval_full_metrics_fixed.py
"""
import sys, json, tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

VAL_SCENES = {"scene-0655", "scene-1077"}

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
    loaded, skipped = 0, []
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v; loaded += 1
        else:
            skipped.append(f"{k}: ckpt={v.shape} model={model_sd.get(k,'MISSING')}")
    model.load_state_dict(model_sd)
    print(f"  Loaded {loaded} layers | Skipped {len(skipped)}")
    if skipped[:2]: print(f"  Skipped: {skipped[:2]}")
    return model.eval().to(device)

def compute_metrics(pred_logits, occ_gt, threshold=0.5):
    prob = torch.sigmoid(pred_logits).cpu()
    pred = (prob > threshold).float()
    gt   = occ_gt.float().cpu()
    if gt.ndim == 3: gt = gt.unsqueeze(1)
    gt = F.interpolate(gt, size=pred.shape[-2:], mode='nearest')
    TP = (pred*gt).sum().item()
    FP = (pred*(1-gt)).sum().item()
    FN = ((1-pred)*gt).sum().item()
    TN = ((1-pred)*(1-gt)).sum().item()
    return dict(
        IoU      = TP/(TP+FP+FN+1e-8),
        Dice     = 2*TP/(2*TP+FP+FN+1e-8),
        Precision= TP/(TP+FP+1e-8),
        Recall   = TP/(TP+FN+1e-8),
        Accuracy = (TP+TN)/(TP+TN+FP+FN+1e-8),
    )

def main():
    from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView
    from torch.utils.data import DataLoader

    rows = [json.loads(l) for l in
            Path("artifacts/nuscenes_mini_manifest.jsonl").read_text().splitlines() if l.strip()]
    scene_key = "scene" if "scene" in rows[0] else "scene_name"
    val_rows  = [r for r in rows if r.get(scene_key,"") in VAL_SCENES]
    print(f"Val samples: {len(val_rows)}")

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    for r in val_rows: tmp.write(json.dumps(r)+"\n")
    tmp.close()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    for version, ckpt_path, label_root in [
        ("v8", "artifacts/checkpoints_v8/last_fixed.ckpt",  "artifacts/nuscenes_labels"),
        ("v9", "artifacts/checkpoints_v9/best_val_ade.ckpt","artifacts/nuscenes_labels"),
    ]:
        if not Path(ckpt_path).exists():
            print(f"Skipping {version}"); continue

        print(f"\n{'='*50}\nEvaluating {version}")
        model = load_model_safe(ckpt_path, device)

        ds = NuScenesMiniMultiView(tmp.name, label_root=label_root,
                                   return_motion=True, return_trel=True)
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

        all_m = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                x = batch[0].to(device); occ_gt = batch[1].to(device)
                out = model(x)
                occ_pred = out[0]
                if i == 0:
                    p = torch.sigmoid(occ_pred).cpu()
                    print(f"  Pred stats: min={p.min():.3f} max={p.max():.3f} mean={p.mean():.4f}")
                    print(f"  GT   stats: mean={occ_gt.float().cpu().mean():.4f}")
                all_m.append(compute_metrics(occ_pred, occ_gt))

        keys = ["IoU","Dice","Precision","Recall","Accuracy"]
        print(f"\n  RESULTS — {version} | n_batches={len(all_m)}")
        print(f"  {'─'*40}")
        for k in keys:
            vals = [m[k] for m in all_m]
            print(f"  {k:12s}: {np.mean(vals):.4f}  (std={np.std(vals):.4f})")

        agg = {k: float(np.mean([m[k] for m in all_m])) for k in keys}
        agg["version"] = version
        Path(f"artifacts/metrics_{version}_corrected.json").write_text(json.dumps(agg, indent=2))
        print(f"  Saved → artifacts/metrics_{version}_corrected.json")

    Path(tmp.name).unlink()

if __name__ == "__main__":
    main()
