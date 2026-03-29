"""
Step 7: Robustness Testing + Evaluation + BEV Visualisation

Handles both checkpoint types:
  - LitOpenDriveFM (plain training)
  - TrustAwareTrainer (augmented contrastive training, wraps lit.model)

Outputs:
  artifacts/bev_overlays/         GT vs Pred BEV map images
  artifacts/trust_dashboard/      6-camera trust weight visualisation
  artifacts/robustness_curves.png clean→corrupted degradation curves
  artifacts/eval_full.json        IoU/Dice/Precision/Recall per condition
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView
from opendrivefm.train.lightning_module import LitOpenDriveFM, _dl_kwargs
from opendrivefm.robustness.perturbations import PERTURBATIONS


CAMS      = ["FRONT","FRONT-L","FRONT-R","BACK","BACK-L","BACK-R"]
TRUST_CMAP = plt.cm.RdYlGn


# ── Checkpoint loader ──────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str):
    """
    Loads checkpoint regardless of whether it was saved as
    LitOpenDriveFM or TrustAwareTrainer. Returns the inner OpenDriveFM model.
    """
    raw = torch.load(ckpt_path, map_location="cpu")
    state = raw.get("state_dict", {})

    # Detect checkpoint type by key prefix
    if any(k.startswith("lit.") for k in state):
        # TrustAwareTrainer — remap keys: lit.X -> X
        new_state = {k[len("lit."):]: v for k, v in state.items()}
        lit = LitOpenDriveFM.__new__(LitOpenDriveFM)
        # init with defaults matching saved hparams
        hp = raw.get("hyper_parameters", {})
        LitOpenDriveFM.__init__(lit,
            lr=hp.get("lr", 3e-4),
            enable_trust=hp.get("enable_trust", True),
        )
        lit.load_state_dict(new_state, strict=False)
    else:
        # Plain LitOpenDriveFM
        lit = LitOpenDriveFM.load_from_checkpoint(ckpt_path, map_location="cpu")

    lit.eval().to(device)
    return lit


# ── Metrics ────────────────────────────────────────────────────────────────────

def occ_metrics(pred_logits, gt, thr=0.35, eps=1e-6):
    pred = (torch.sigmoid(pred_logits) > thr).float()
    pred = pred.view(pred.size(0), -1)
    gt   = gt.view(gt.size(0), -1)
    tp   = (pred * gt).sum(1)
    fp   = (pred * (1-gt)).sum(1)
    fn   = ((1-pred) * gt).sum(1)
    iou  = (tp+eps)/(tp+fp+fn+eps)
    dice = (2*tp+eps)/(2*tp+fp+fn+eps)
    prec = (tp+eps)/(tp+fp+eps)
    rec  = (tp+eps)/(tp+fn+eps)
    return iou.mean().item(), dice.mean().item(), prec.mean().item(), rec.mean().item()


def split_val(rows, seed=42, val_frac=0.2):
    scenes = sorted({r["scene"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(scenes)
    n_val = max(1, int(round(len(scenes) * val_frac)))
    vs = set(scenes[:n_val])
    return [i for i, r in enumerate(rows) if r["scene"] in vs]


# ── Visualisation helpers ──────────────────────────────────────────────────────

def save_bev_overlay(pred_logits, occ_gt, save_path, thr=0.35):
    pred = (torch.sigmoid(pred_logits[0, 0]) > thr).cpu().numpy().astype(np.uint8)
    gt   = occ_gt[0, 0].cpu().numpy().astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0a0a10")
    for ax in axes:
        ax.set_facecolor("#0a0a10")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_color("#2a2a38")

    axes[0].imshow(gt,   cmap="Greens",  vmin=0, vmax=1)
    axes[0].set_title("Ground Truth BEV",   color="white", fontsize=12)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].imshow(pred, cmap="Blues",   vmin=0, vmax=1)
    axes[1].set_title("Predicted BEV",      color="white", fontsize=12)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # Overlay: TP=green, FP=red, FN=orange
    overlay = np.zeros((*gt.shape, 3), dtype=np.float32)
    overlay[(gt==1)&(pred==1)] = [0.2, 0.9, 0.2]   # TP green
    overlay[(gt==0)&(pred==1)] = [0.9, 0.2, 0.2]   # FP red
    overlay[(gt==1)&(pred==0)] = [1.0, 0.6, 0.0]   # FN orange
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (TP=green FP=red FN=orange)", color="white", fontsize=12)
    axes[2].set_xticks([]); axes[2].set_yticks([])

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=120, bbox_inches="tight", facecolor="#0a0a10")
    plt.close(fig)


def save_trust_dashboard(x_batch, trust_scores, save_path, perturbation=None, perturbed_cam=0):
    trust = trust_scores[0].cpu().tolist()
    imgs  = [x_batch[0, v, 0].cpu().permute(1,2,0).numpy().clip(0,1) for v in range(len(trust))]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor="#0a0a10")
    axes = axes.flatten()

    for i, (img, score) in enumerate(zip(imgs, trust)):
        ax  = axes[i]
        col = TRUST_CMAP(score)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(col); sp.set_linewidth(6)

        warn = " ⚠ DEGRADED" if (perturbation and i == perturbed_cam) else ""
        ax.set_title(f"{CAMS[i]}{warn}\nTrust: {score:.3f}",
                     color="white" if not warn else "#ff6b6b", fontsize=10, pad=4)

        # Trust bar underneath
        bw = img.shape[1]
        fill = int(score * bw)
        bar  = np.zeros((6, bw, 3))
        bar[:, :fill]  = col[:3]
        bar[:, fill:]  = (0.15, 0.15, 0.2)
        ax_b = ax.inset_axes([0, -0.07, 1, 0.06])
        ax_b.imshow(bar); ax_b.set_xticks([]); ax_b.set_yticks([])

    title = "Camera Trust Score Dashboard"
    if perturbation:
        title += f"  |  FRONT: {perturbation}"
    fig.suptitle(title, color="#e8e8f8", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=120, bbox_inches="tight", facecolor="#0a0a10")
    plt.close(fig)


def save_robustness_curves(summary, save_path):
    conditions = list(summary.keys())
    ious   = [summary[c]["iou"]   for c in conditions]
    dices  = [summary[c]["dice"]  for c in conditions]
    trusts = [summary[c]["trust"] for c in conditions]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0a0a10")
    for ax in axes:
        ax.set_facecolor("#12121e")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_color("#2a2a38")

    x = range(len(conditions))

    axes[0].plot(x, ious,  "o-", color="#4a9eff", lw=2.5, ms=8, label="IoU")
    axes[0].plot(x, dices, "s--",color="#ff9f4a", lw=2.0, ms=7, label="Dice")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conditions, rotation=20, ha="right", color="white")
    axes[0].set_ylabel("Score", color="white")
    axes[0].set_title("BEV Metrics: Clean → Corrupted", color="#e8e8f8", fontsize=12, pad=8)
    axes[0].legend(facecolor="#1a1a26", labelcolor="white")
    axes[0].set_ylim(0, 1.1)

    colors = [TRUST_CMAP(t) for t in trusts]
    bars = axes[1].bar(conditions, trusts, color=colors, edgecolor="#2a2a38", lw=1.5)
    for bar, v in zip(bars, trusts):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                     f"{v:.3f}", ha="center", va="bottom", color="white", fontsize=10)
    axes[1].set_ylabel("Mean Trust Score (FRONT cam)", color="white")
    axes[1].set_title("Trust Score Under Degradation", color="#e8e8f8", fontsize=12, pad=8)
    axes[1].tick_params(axis="x", colors="white", rotation=20)
    axes[1].set_ylim(0, 1.15)
    clean_t = summary.get("clean", {}).get("trust", None)
    if clean_t:
        axes[1].axhline(clean_t, color="#5588ff", linestyle="--", lw=1.5,
                        label=f"Clean: {clean_t:.3f}")
        axes[1].legend(facecolor="#1a1a26", labelcolor="white")

    fig.suptitle("Robustness: BEV Performance Under Camera Degradations",
                 color="#e8e8f8", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0a10")
    plt.close(fig)
    print(f"  Saved robustness curves → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",          type=str, required=True)
    ap.add_argument("--manifest",      type=str, default="artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root",    type=str, default="artifacts/nuscenes_labels")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--val_frac",      type=float, default=0.2)
    ap.add_argument("--batch_size",    type=int, default=2)
    ap.add_argument("--image_hw",      type=str, default="90,160")
    ap.add_argument("--n_bev_samples", type=int, default=8)
    ap.add_argument("--out_dir",       type=str, default="artifacts")
    args = ap.parse_args()

    H, W   = [int(v) for v in args.image_hw.split(",")]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out    = Path(args.out_dir)
    bev_dir   = out / "bev_overlays";    bev_dir.mkdir(exist_ok=True)
    trust_dir = out / "trust_dashboard"; trust_dir.mkdir(exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt}")
    lit = load_model(args.ckpt, device)
    print(f"  enable_trust = {lit.model.backbone.enable_trust}")

    rows    = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    idx_val = split_val(rows, args.seed, args.val_frac)

    ds = NuScenesMiniMultiView(
        args.manifest, image_hw=(H, W), frames=1,
        label_root=args.label_root, return_motion=True, return_trel=True)
    dl = DataLoader(Subset(ds, idx_val), batch_size=args.batch_size,
                    shuffle=False, **_dl_kwargs())

    perturbers = {k: v() for k, v in PERTURBATIONS.items()}
    conditions = ["clean"] + list(PERTURBATIONS.keys())
    results    = {c: {"iou":[], "dice":[], "prec":[], "rec":[], "trust":[]}
                  for c in conditions}

    bev_saved = 0
    for bi, batch in enumerate(dl):
        x, occ_t, traj_t, motion, t_rel = batch
        x = x.to(device); occ_t = occ_t.to(device)
        if occ_t.ndim == 3: occ_t = occ_t.unsqueeze(1)

        for cond in conditions:
            x_in = x.clone()
            if cond != "clean":
                x_in[:, 0, 0] = perturbers[cond](x_in[:, 0, 0])  # FRONT cam

            occ_logits, _, trust = lit.model(x_in)
            if occ_logits.ndim == 3: occ_logits = occ_logits.unsqueeze(1)

            iou, dice, prec, rec = occ_metrics(occ_logits, occ_t)
            results[cond]["iou"].append(iou)
            results[cond]["dice"].append(dice)
            results[cond]["prec"].append(prec)
            results[cond]["rec"].append(rec)
            results[cond]["trust"].append(trust[:, 0].mean().item())  # FRONT cam trust

            if cond == "clean" and bev_saved < args.n_bev_samples:
                save_bev_overlay(occ_logits, occ_t,
                                 bev_dir / f"bev_{bi:03d}.png")
                save_trust_dashboard(x_in, trust,
                                     trust_dir / f"trust_clean_{bi:03d}.png")
                bev_saved += 1

            # Also save one degraded trust dashboard per condition for first batch
            if bi == 0 and cond != "clean":
                save_trust_dashboard(x_in, trust,
                                     trust_dir / f"trust_{cond}_000.png",
                                     perturbation=cond, perturbed_cam=0)

    # Aggregate
    summary = {c: {k: float(np.mean(v)) for k, v in m.items()}
               for c, m in results.items()}

    # Print table
    print("\n" + "="*70)
    print(f"  {'Condition':<12}  {'IoU':>6}  {'Dice':>6}  {'Prec':>6}  {'Rec':>6}  {'Trust(FRONT)':>12}")
    print("="*70)
    clean_trust = summary["clean"]["trust"]
    for cond, m in summary.items():
        drop = m["trust"] - clean_trust
        drop_str = f"({drop:+.3f})" if cond != "clean" else "  (base)"
        print(f"  {cond:<12}  {m['iou']:>6.3f}  {m['dice']:>6.3f}  {m['prec']:>6.3f}  {m['rec']:>6.3f}  {m['trust']:>6.3f} {drop_str}")
    print("="*70)

    save_robustness_curves(summary, str(out / "robustness_curves.png"))

    report = {"ckpt": args.ckpt, "val_items": len(idx_val), "results": summary}
    (out / "eval_full.json").write_text(json.dumps(report, indent=2))

    print(f"\n  Saved → {out}/eval_full.json")
    print(f"  Saved → {out}/bev_overlays/     ({bev_saved} clean images)")
    print(f"  Saved → {out}/trust_dashboard/  ({bev_saved + len(PERTURBATIONS)} images)")
    print(f"  Saved → {out}/robustness_curves.png")


if __name__ == "__main__":
    main()
