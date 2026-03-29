"""
Ablation: Trust-weighted vs Uniform fusion under clean + fault conditions.
Produces a table showing real IoU/ADE deltas.

Usage:
    python scripts/eval_trust_ablation.py \
        --ckpt artifacts/checkpoints_v8/last_fixed.ckpt \
        --manifest artifacts/nuscenes_mini_manifest.jsonl \
        --label_root artifacts/nuscenes_labels \
        --out artifacts/ablation_results.json
"""
import argparse, json, sys
import torch, numpy as np
sys.path.insert(0, 'src')

from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView
from opendrivefm.train.lightning_module import LitOpenDriveFM
from opendrivefm.robustness.perturbations import (
    GaussianBlur, GlareOverlay, OcclusionPatch, RainStreaks, SaltPepperNoise)
from torch.utils.data import DataLoader


FAULTS = {
    'blur':      GaussianBlur(sigma_range=(3.0, 4.0)),
    'occlusion': OcclusionPatch(patch_frac=(0.3, 0.4)),
    'rain':      RainStreaks(num_streaks=(50, 60), alpha=(0.4, 0.5)),
    'noise':     SaltPepperNoise(amount_range=(0.06, 0.08)),
    'glare':     GlareOverlay(intensity_range=(0.7, 0.9)),
}
THR = 0.6
FAULT_CAM = 0  # inject fault on CAM_FRONT


def iou_ade(model, loader, device, trust_mode, fault_name=None):
    """
    trust_mode: 'trust' | 'uniform'
    fault_name: None = clean, else inject fault on FAULT_CAM
    """
    ious, ades, fdes = [], [], []
    perturb = FAULTS[fault_name] if fault_name else None

    with torch.no_grad():
        for batch in loader:
            x, occ_t, traj, motion, t_rel = [batch[i].to(device) for i in range(5)]
            K     = batch[5].to(device) if len(batch) > 5 else None
            T_ego = batch[6].to(device) if len(batch) > 6 else None
            if occ_t.ndim == 3: occ_t = occ_t.unsqueeze(1)
            vel = motion[:, 1:3]

            # Apply fault to one camera
            if perturb is not None:
                B, V, T, C, H, W = x.shape
                x_fault = x.clone()
                for b in range(B):
                    img = x_fault[b, FAULT_CAM, 0]  # (C,H,W)
                    # denorm, perturb, renorm
                    img_t = perturb(img.unsqueeze(0)).squeeze(0)
                    x_fault[b, FAULT_CAM, 0] = img_t
                x = x_fault

            # Override trust with uniform if requested
            if trust_mode == 'uniform':
                # Monkey-patch trust scorer to return uniform scores
                original_forward = model.backbone.trust_scorer.forward
                def uniform_trust(imgs):
                    B2 = imgs.shape[0]
                    return torch.ones(B2, device=imgs.device) * 0.7
                model.backbone.trust_scorer.forward = uniform_trust

            occ_logits, traj_res, trust, _ = model(x, K=K, T_ego_cam=T_ego, velocity=vel)

            if trust_mode == 'uniform':
                model.backbone.trust_scorer.forward = original_forward

            pred  = (torch.sigmoid(occ_logits) > THR).float()
            inter = (pred * occ_t).sum((1,2,3))
            union = (pred + occ_t).clamp(0,1).sum((1,2,3))
            ious.extend((inter/(union+1e-6)).cpu().tolist())

            cv     = t_rel.unsqueeze(-1) * vel.unsqueeze(1)
            pred_t = cv + traj_res
            ades.extend(torch.linalg.norm(pred_t-traj,dim=-1).mean(1).cpu().tolist())
            fdes.extend(torch.linalg.norm(pred_t-traj,dim=-1)[:,-1].cpu().tolist())

    return np.mean(ious), np.mean(ades), np.mean(fdes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',     required=True)
    p.add_argument('--manifest', default='artifacts/nuscenes_mini_manifest.jsonl')
    p.add_argument('--label_root', default='artifacts/nuscenes_labels')
    p.add_argument('--out',      default='artifacts/ablation_results.json')
    args = p.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    lit = LitOpenDriveFM.load_from_checkpoint(args.ckpt, map_location=device)
    model = lit.model.eval().to(device)

    ds = NuScenesMiniMultiView(
        args.manifest, label_root=args.label_root,
        return_motion=True, return_trel=True, return_calib=True, augment=False)

    # Val split only
    rows = [json.loads(l) for l in open(args.manifest)]
    VAL_SCENES = {'scene-0655', 'scene-1077'}
    val_idx = [i for i,r in enumerate(rows) if r['scene'] in VAL_SCENES]
    from torch.utils.data import Subset
    loader = DataLoader(Subset(ds, val_idx), batch_size=4,
                        shuffle=False, num_workers=0)
    print(f"Val samples: {len(val_idx)}")

    results = {}

    # ── Clean ablation ────────────────────────────────────────────────
    print("\n── Clean: Trust vs Uniform ──────────────────────────────")
    for mode in ['trust', 'uniform']:
        iou, ade, fde = iou_ade(model, loader, device, mode, fault_name=None)
        results[f'clean_{mode}'] = {'IoU': iou, 'ADE': ade, 'FDE': fde}
        print(f"  [{mode:8s}]  IoU={iou:.4f}  ADE={ade:.3f}  FDE={fde:.3f}")

    d_iou = results['clean_trust']['IoU'] - results['clean_uniform']['IoU']
    d_ade = results['clean_uniform']['ADE'] - results['clean_trust']['ADE']
    print(f"  Trust advantage (clean): ΔIoU={d_iou:+.4f}  ΔADE={d_ade:+.3f}")

    # ── Fault robustness ─────────────────────────────────────────────
    print("\n── Fault Robustness: Trust ON vs OFF ────────────────────")
    print(f"  {'Fault':<12} {'Trust IoU':>10} {'Uniform IoU':>12} {'ΔIoU':>8} "
          f"{'Trust ADE':>10} {'Uniform ADE':>12} {'ΔADE':>8}")
    print("  " + "─"*76)

    for fault_name in FAULTS:
        t_iou, t_ade, t_fde = iou_ade(model, loader, device, 'trust',   fault_name)
        u_iou, u_ade, u_fde = iou_ade(model, loader, device, 'uniform', fault_name)
        results[f'fault_{fault_name}_trust']   = {'IoU': t_iou, 'ADE': t_ade, 'FDE': t_fde}
        results[f'fault_{fault_name}_uniform'] = {'IoU': u_iou, 'ADE': u_ade, 'FDE': u_fde}
        d_iou = t_iou - u_iou
        d_ade = u_ade - t_ade
        print(f"  {fault_name:<12} {t_iou:>10.4f} {u_iou:>12.4f} {d_iou:>+8.4f} "
              f"{t_ade:>10.3f} {u_ade:>12.3f} {d_ade:>+8.3f}")

    # Save
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.out}")

    # ── Print final table ─────────────────────────────────────────────
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    ABLATION SUMMARY                             ║
╠══════════════════════════════════════════════════════════════════╣""")
    print(f"║  {'Configuration':<28} {'IoU':>8} {'ADE':>8} {'FDE':>8}         ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    for label, key in [('Trust fusion (ours)', 'clean_trust'),
                        ('Uniform fusion (ablation)', 'clean_uniform')]:
        r = results[key]
        print(f"║  {label:<28} {r['IoU']:>8.4f} {r['ADE']:>8.3f} {r['FDE']:>8.3f}         ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  Fault Robustness (CAM_FRONT degraded):                         ║")
    for fault_name in FAULTS:
        t = results[f'fault_{fault_name}_trust']
        u = results[f'fault_{fault_name}_uniform']
        d = t['IoU'] - u['IoU']
        print(f"║    {fault_name:<10} trust={t['IoU']:.4f}  uniform={u['IoU']:.4f}  Δ={d:+.4f}       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")


if __name__ == '__main__':
    main()
