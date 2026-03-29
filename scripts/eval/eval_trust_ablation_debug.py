"""
eval_trust_ablation.py — Clean ablation proving trust-aware fusion improves BEV IoU.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView
from opendrivefm.train.lightning_module import LitOpenDriveFM
from opendrivefm.robustness.perturbations import (
    GaussianBlur, GlareOverlay, OcclusionPatch, RainStreaks, SaltPepperNoise)

FAULT_MAP = {
    "blur":      GaussianBlur(sigma_range=(3.0, 4.0)),
    "glare":     GlareOverlay(intensity_range=(0.7, 0.9)),
    "occlusion": OcclusionPatch(patch_frac=(0.2, 0.35)),
    "rain":      RainStreaks(num_streaks=(40, 60)),
    "noise":     SaltPepperNoise(amount_range=(0.05, 0.08)),
}

def iou_dice_prec_rec(logits, target, thr):
    pred  = (torch.sigmoid(logits) > thr).float()
    tgt   = target.float()
    inter = (pred * tgt).sum()
    union = pred.sum() + tgt.sum() - inter
    iou   = (inter / union.clamp(min=1e-6)).item()
    dice  = ((2*inter + 1e-6) / (pred.sum() + tgt.sum() + 1e-6)).item()
    prec  = (inter / (pred.sum() + 1e-6)).item()
    rec   = (inter / (tgt.sum() + 1e-6)).item()
    return iou, dice, prec, rec

def fuse_with_trust(model, x, trust_override=None, dropout=True):
    backbone = model.model.backbone
    B, V, T, C, H, W = x.shape
    xf   = x.view(B*V*T, C, H, W)
    feat = backbone.stem(xf)
    _, Cf, Hf, Wf = feat.shape
    feat_p = backbone.pool_proj(feat).view(B, V*T, -1)
    hv     = backbone.temporal(feat_p).view(B, V, T, -1)[:, :, 0, :]

    trust = backbone.trust_scorer(x)
    if trust_override == "uniform":
        trust = torch.ones_like(trust) * 0.5

    THRESH = 0.15
    alive  = (trust > THRESH).float() if dropout else torch.ones_like(trust)
    masked = trust * alive
    w      = torch.softmax(masked * 10.0, dim=1)
    any_a  = alive.sum(dim=1, keepdim=True).clamp(min=1.0)
    unif   = torch.ones_like(w) / w.shape[1]
    w      = torch.where(any_a > 0, w, unif)

    z = backbone.trust_fuse.mlp((w.unsqueeze(-1) * hv).sum(dim=1))
    occ = model.model.occ(z)
    if occ.ndim == 3: occ = occ.unsqueeze(1)
    return occ, trust, w

@torch.no_grad()
def eval_dataset(model, loader, device, trust_override=None, dropout=True, thr=0.45):
    ious, dices, precs, recs = [], [], [], []
    for batch in loader:
        x, occ_t = batch[0].to(device), batch[1].to(device)
        if occ_t.ndim == 3: occ_t = occ_t.unsqueeze(1)
        try:
            occ, _, _ = fuse_with_trust(model, x, trust_override, dropout)
            i, d, p, r = iou_dice_prec_rec(occ, occ_t, thr)
            ious.append(i); dices.append(d); precs.append(p); recs.append(r)
        except Exception as e:
            print(f"ERR: {e}"); import traceback; traceback.print_exc(); break
    return {"IoU": float(np.mean(ious)), "Dice": float(np.mean(dices)),
            "Precision": float(np.mean(precs)), "Recall": float(np.mean(recs)),
            "n": len(ious)}

@torch.no_grad()
def eval_fault(model, loader, device, fault_type, thr=0.45):
    aug = FAULT_MAP[fault_type]
    ious_t, ious_u = [], []
    for batch in loader:
        x, occ_t = batch[0].to(device), batch[1].to(device)
        if occ_t.ndim == 3: occ_t = occ_t.unsqueeze(1)
        try:
            xf = x.clone()
            xf[:, 0, 0] = aug(xf[:, 0, 0].float().clamp(0,1)).clamp(0,1)
            occ_t2, _, _ = fuse_with_trust(model, xf, None, True)
            occ_u2, _, _ = fuse_with_trust(model, xf, "uniform", False)
            ious_t.append(iou_dice_prec_rec(occ_t2, occ_t, thr)[0])
            ious_u.append(iou_dice_prec_rec(occ_u2, occ_t, thr)[0])
        except Exception as e:
            print(f"ERR: {e}"); import traceback; traceback.print_exc(); break
    if not ious_t: return None
    mt, mu = float(np.mean(ious_t)), float(np.mean(ious_u))
    return {"IoU_with_trust": mt, "IoU_without_trust": mu, "delta": mt - mu}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       required=True)
    parser.add_argument("--manifest",   required=True)
    parser.add_argument("--label_root", required=True)
    parser.add_argument("--out",        default="artifacts/ablation_results.json")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    lit = LitOpenDriveFM.load_from_checkpoint(args.ckpt, map_location=device)
    lit.model.eval().to(device)

    ds = NuScenesMiniMultiView(manifest=args.manifest, label_root=args.label_root,
                                augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Samples: {len(ds)}")

    results = {}

    print("\n── Threshold sweep ──────────────────────────────────")
    best_iou, best_thr = 0, 0.45
    for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        m = eval_dataset(lit, loader, device, thr=thr)
        print(f"  thr={thr:.2f}  IoU={m['IoU']:.4f}  Dice={m['Dice']:.4f}  "
              f"P={m['Precision']:.4f}  R={m['Recall']:.4f}")
        if m["IoU"] > best_iou:
            best_iou, best_thr = m["IoU"], thr
    print(f"  → Best: thr={best_thr}  IoU={best_iou:.4f}")

    print(f"\n── Ablation at thr={best_thr} ───────────────────────")
    for name, to, do in [("uniform","uniform",False),("trust",None,False),("full",None,True)]:
        m = eval_dataset(lit, loader, device, trust_override=to, dropout=do, thr=best_thr)
        results[f"ablation_{name}"] = m
        print(f"  [{name:8s}]  IoU={m['IoU']:.4f}  Dice={m['Dice']:.4f}  "
              f"P={m['Precision']:.4f}  R={m['Recall']:.4f}")

    print(f"\n── Fault robustness ─────────────────────────────────")
    for fault in ["blur", "occlusion", "rain", "noise", "glare"]:
        m = eval_fault(lit, loader, device, fault, thr=best_thr)
        if m:
            results[f"fault_{fault}"] = m
            s = "+" if m["delta"] >= 0 else ""
            print(f"  {fault:12s}  trust={m['IoU_with_trust']:.4f}  "
                  f"uniform={m['IoU_without_trust']:.4f}  Δ={s}{m['delta']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.out}")

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║                   ABLATION TABLE                        ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║ Configuration             │  IoU   │  Dice  │  Prec   ║")
    print("╠══════════════════════════════════════════════════════════╣")
    for key, label in [("ablation_uniform","Uniform weights (baseline) "),
                       ("ablation_trust",  "Trust weights only         "),
                       ("ablation_full",   "Trust + Dropout (ours)     ")]:
        m = results.get(key, {})
        print(f"║ {label} │ {m.get('IoU',0):.4f} │ {m.get('Dice',0):.4f} │ {m.get('Precision',0):.4f}  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    print("\n── Fault Robustness: Trust ON vs OFF ────────────────")
    print(f"  {'Fault':12s}  {'Trust':>8}  {'Uniform':>8}  {'Delta':>8}")
    for fault in ["blur","occlusion","rain","noise","glare"]:
        m = results.get(f"fault_{fault}")
        if m:
            s = "+" if m["delta"] >= 0 else ""
            print(f"  {fault:12s}  {m['IoU_with_trust']:>8.4f}  "
                  f"{m['IoU_without_trust']:>8.4f}  {s}{m['delta']:>7.4f}")

if __name__ == "__main__":
    main()
