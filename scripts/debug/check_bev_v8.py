"""
BEV visual check for v8+ checkpoints (with geometry/calibration).
Usage:
    python scripts/check_bev_v8.py \
        --ckpt artifacts/checkpoints_v8/last_fixed.ckpt \
        --out  artifacts/bev_v8_check.png
"""
import argparse, sys, torch, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')
from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView
from opendrivefm.train.lightning_module import LitOpenDriveFM
from torch.utils.data import DataLoader


def fix_ckpt(path):
    ckpt = torch.load(path, map_location='cpu')
    sd   = {k[4:] if k.startswith('lit.') else k: v
            for k, v in ckpt['state_dict'].items()}
    ckpt['state_dict'] = sd
    fixed = path.replace('.ckpt', '_fixed.ckpt')
    torch.save(ckpt, fixed)
    return fixed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',     default='artifacts/checkpoints_v8/last_fixed.ckpt')
    p.add_argument('--manifest', default='artifacts/nuscenes_mini_manifest.jsonl')
    p.add_argument('--labels',   default='artifacts/nuscenes_labels')
    p.add_argument('--out',      default='artifacts/bev_v8_check.png')
    p.add_argument('--n',        type=int, default=4)
    p.add_argument('--thr',      type=float, default=None)
    args = p.parse_args()

    if not args.ckpt.endswith('_fixed.ckpt'):
        args.ckpt = fix_ckpt(args.ckpt)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    lit = LitOpenDriveFM.load_from_checkpoint(args.ckpt, map_location=device)
    lit.model.eval().to(device)

    ds = NuScenesMiniMultiView(
        args.manifest, label_root=args.labels,
        return_motion=True, return_trel=True, return_calib=True, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Find best threshold
    if args.thr is None:
        print("Finding best threshold...")
        loader_thr = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        best_iou, best_thr = 0, 0.5
        for thr in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
            ious = []
            with torch.no_grad():
                for batch in loader_thr:
                    x     = batch[0].to(device)
                    occ_t = batch[1].to(device)
                    K     = batch[5].to(device)
                    T_ego = batch[6].to(device)
                    motion = batch[3].to(device)
                    if occ_t.ndim == 3: occ_t = occ_t.unsqueeze(1)
                    occ_logits, _, _, _ = lit.model(x, K=K, T_ego_cam=T_ego,
                                                    velocity=motion[:,1:3])
                    pred  = (torch.sigmoid(occ_logits) > thr).float()
                    inter = (pred * occ_t).sum((1,2,3))
                    union = (pred + occ_t).clamp(0,1).sum((1,2,3))
                    ious.extend((inter/(union+1e-6)).cpu().tolist())
            iou = np.mean(ious)
            print(f"  thr={thr:.2f}  IoU={iou:.4f}")
            if iou > best_iou:
                best_iou, best_thr = iou, thr
        print(f"  → Best: thr={best_thr}  IoU={best_iou:.4f}")
        args.thr = best_thr

    # Generate visuals
    fig, axes = plt.subplots(args.n, 3, figsize=(12, 4*args.n))
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.n: break
            x     = batch[0].to(device)
            occ_t = batch[1].to(device)
            K     = batch[5].to(device)
            T_ego = batch[6].to(device)
            motion = batch[3].to(device)
            if occ_t.ndim == 3: occ_t = occ_t.unsqueeze(1)
            occ_logits, _, _, _ = lit.model(x, K=K, T_ego_cam=T_ego,
                                            velocity=motion[:,1:3])
            prob = torch.sigmoid(occ_logits)[0, 0].cpu().numpy()
            pred = (prob > args.thr).astype(float)
            gt   = occ_t[0, 0].cpu().numpy()

            # Compute per-sample IoU
            inter = (pred * gt).sum()
            union = np.clip(pred + gt, 0, 1).sum()
            iou   = inter / (union + 1e-6)

            axes[i, 0].imshow(gt,   cmap='Greens', vmin=0, vmax=1)
            axes[i, 0].set_title(f'GT #{i}  ({gt.sum():.0f} px)')
            axes[i, 1].imshow(prob, cmap='hot',    vmin=0, vmax=1)
            axes[i, 1].set_title(f'Probability #{i}')
            axes[i, 2].imshow(pred, cmap='Blues',  vmin=0, vmax=1)
            axes[i, 2].set_title(f'Pred #{i}  IoU={iou:.3f}')
            for ax in axes[i]: ax.axis('off')

    plt.suptitle(f'BEV Object Detection — geometry lifting (thr={args.thr})', fontsize=13)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches='tight')
    print(f"Saved {args.out}")


if __name__ == '__main__':
    main()
