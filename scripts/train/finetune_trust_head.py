"""
Fine-tune ONLY the CameraTrustScorer with a contrastive fault objective.

CONTEXT
-------
`scripts/eval/eval_ood_detection.py` measured the v11 trust scorer at
AUROC 0.434, 95% CI [0.419, 0.449] -- significantly INVERTED, i.e. trust rises
when a camera is degraded. Root cause: v11 was trained by LitOpenDriveFMV9,
whose only trust supervision is

    reg = (trust.mean() - 0.75)**2 - 0.1 * entropy

which pulls trust to a constant and *rewards uniformity across cameras*. No
contrastive term ever reached the v10-v14 checkpoints, so the scorer's response
to corruption is whatever its hand-crafted image-statistics branch happens to
do: glare, rain and noise all RAISE luminance / edge energy, hence the inversion.

WHY ONLY THE TRUST HEAD
-----------------------
The occupancy and trajectory heads are not the defect. Freezing them and
training only `backbone.trust_scorer` means:
  * the before/after AUROC difference is attributable to one change,
  * IoU and ADE cannot be silently traded away for detection,
  * training is seconds-to-minutes, not hours, because the objective never
    touches the BEV decoder or the trajectory head.

Trust still feeds `TrustWeightedFusion` via softmax over cameras, so IoU and ADE
CAN move. Re-measure them after; do not assume they are unchanged.

OBJECTIVE
---------
    hinge  = relu(margin - (trust_clean[cam] - trust_faulted[cam])).mean()
    anchor = (trust_clean.mean() - trust_target)**2
    loss   = hinge + anchor_w * anchor

The hinge is applied to the specific camera that was corrupted. Averaging over
all 6 cameras dilutes a single-camera fault ~6x and gives almost no gradient.
The anchor keeps trust in a usable range so the fusion softmax does not collapse.
No entropy reward: that term is what trained the scorer flat in the first place.

Usage
-----
    python scripts/train/finetune_trust_head.py \
        --ckpt outputs/artifacts/checkpoints_v11_temporal/best_val_ade.ckpt \
        --out  outputs/artifacts/checkpoints_v11_trustfix/trust_fixed.ckpt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "src")

from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView  # noqa: E402
from opendrivefm.robustness.perturbations import PERTURBATIONS  # noqa: E402
from opendrivefm.validation.ckpt_compat import (  # noqa: E402
    remap_legacy_keys, report_remap)
from opendrivefm.training.lightning_module import LitOpenDriveFM  # noqa: E402


def split_scenes(rows, seed, val_frac, val_scenes):
    if val_scenes:
        val = {s.strip() for s in val_scenes.split(",") if s.strip()}
    else:
        scenes = sorted({r["scene"] for r in rows})
        rng = random.Random(seed)
        rng.shuffle(scenes)
        val = set(scenes[:max(1, int(round(len(scenes) * val_frac)))])
    train_idx = [i for i, r in enumerate(rows) if r["scene"] not in val]
    val_idx = [i for i, r in enumerate(rows) if r["scene"] in val]
    return train_idx, val_idx, sorted(val)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/artifacts/checkpoints_v11_trustfix/trust_fixed.ckpt")
    ap.add_argument("--manifest", default="outputs/artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", default="outputs/artifacts/nuscenes_labels_128")
    ap.add_argument("--bev", type=int, default=128)
    ap.add_argument("--trust_grid", type=int, default=1,
                    help="CameraTrustScorer spatial pooling grid (see model.py). "
                         ">1 changes the trust-scorer shapes, so those weights are "
                         "trained from scratch; the rest of the checkpoint still loads.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.30)
    ap.add_argument("--fault_sampling", choices=["random", "stratified"],
                    default="stratified",
                    help="random: one random fault per sample (pooled hinge). This lets the "
                         "optimiser trade one fault away to win another -- it produced "
                         "rain/noise AUROC 0.99 and blur 0.25. stratified: EVERY fault is "
                         "scored on EVERY batch, so no fault can be silently sacrificed.")
    ap.add_argument("--worst_case_w", type=float, default=0.5,
                    help="Blend between mean-over-faults (0.0) and worst-fault (1.0). "
                         "Hazard-aligned validation optimises the fault you are worst at, "
                         "not the average fault.")
    ap.add_argument("--anchor_w", type=float, default=1.0)
    ap.add_argument("--trust_target", type=float, default=0.75)
    ap.add_argument("--image_hw", default="90,160")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--val_scenes", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    H, W = [int(v) for v in args.image_hw.split(",")]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}
    lit = LitOpenDriveFM(bev=args.bev, trust_grid=args.trust_grid)
    # Drop only the tensors that genuinely do not fit, by comparing shapes.
    # Keying this off `args.trust_grid > 1` instead would discard a scorer that
    # was ALREADY trained at this grid (e.g. resuming this script's own output),
    # silently restarting it from random init. See the same fix in
    # scripts/eval/eval_ood_detection.py.
    ref = lit.model.state_dict()
    state, applied, rejected = remap_legacy_keys(state, ref)
    report_remap(applied, rejected)
    dropped = {k: (tuple(v.shape), tuple(ref[k].shape))
               for k, v in state.items()
               if k in ref and tuple(v.shape) != tuple(ref[k].shape)}
    for k in dropped:
        state.pop(k)
    if dropped:
        print(f"  dropped {len(dropped)} tensor(s) whose shape does not match the "
              f"current model; randomly initialised instead:")
        for k, (had, want) in list(dropped.items())[:4]:
            print(f"    {k}: checkpoint {had} vs model {want}")
        if len(dropped) > 4:
            print(f"    ... and {len(dropped) - 4} more")
    res = lit.model.load_state_dict(state, strict=False)
    n_exp = len(lit.model.state_dict())
    print(f"Loaded {n_exp - len(res.missing_keys)}/{n_exp} weights from {args.ckpt}")

    model = lit.model.to(device)

    # ── Freeze everything except the trust scorer ────────────────────────────
    for p in model.parameters():
        p.requires_grad_(False)
    scorer = model.backbone.trust_scorer
    for p in scorer.parameters():
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in scorer.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Training {n_train:,} of {n_total:,} parameters "
          f"({n_train / n_total:.2%}) -- trust scorer only")

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    tr_idx, va_idx, val_scenes = split_scenes(rows, args.seed, args.val_frac, args.val_scenes)
    ds = NuScenesMiniMultiView(args.manifest, image_hw=(H, W), frames=1,
                               label_root=args.label_root, return_motion=True,
                               return_trel=True, return_calib=True, augment=False)
    dl_tr = DataLoader(Subset(ds, tr_idx), batch_size=args.batch_size, shuffle=True,
                       num_workers=0)
    dl_va = DataLoader(Subset(ds, va_idx), batch_size=args.batch_size, shuffle=False,
                       num_workers=0)
    print(f"train frames: {len(tr_idx)}  val frames: {len(va_idx)}  "
          f"(val scenes held out: {val_scenes})")

    perturbers = [cls() for cls in PERTURBATIONS.values()]
    opt = torch.optim.Adam([p for p in scorer.parameters() if p.requires_grad], lr=args.lr)

    fault_names = list(PERTURBATIONS)

    def corrupt(imgs, perturb, cam):
        """Apply one perturbation to camera `cam` of every sample in the batch."""
        out = imgs.clone()
        for b in range(imgs.shape[0]):
            c = int(cam[b])
            out[b, c] = perturb(imgs[b, c].unsqueeze(0)).squeeze(0)
        return out

    def run_epoch(loader, train: bool):
        """Returns (loss, mean_hinge, anchor, {fault: separation_rate})."""
        tot = anc = hin = n = 0.0
        sep = {f: 0.0 for f in fault_names}
        gapsum = {f: 0.0 for f in fault_names}
        scorer.train(train)
        for batch in loader:
            imgs = batch[0].to(device)[:, :, -1].detach()
            B, V = imgs.shape[0], imgs.shape[1]
            # One camera per sample, held constant across faults so the per-fault
            # comparison is paired rather than confounded by camera identity.
            cam = torch.randint(0, V, (B,), device=imgs.device)
            idx = cam.view(-1, 1)

            with torch.set_grad_enabled(train):
                t_clean = scorer(imgs.reshape(B * V, *imgs.shape[2:])).reshape(B, V)
                clean_c = t_clean.gather(1, idx).squeeze(1)

                chosen = (fault_names if args.fault_sampling == "stratified"
                          else [fault_names[random.randrange(len(fault_names))]])
                hinges = []
                for fname in chosen:
                    bad = corrupt(imgs, perturbers[fault_names.index(fname)], cam)
                    t_bad = scorer(bad.reshape(B * V, *bad.shape[2:])).reshape(B, V)
                    gap = clean_c - t_bad.gather(1, idx).squeeze(1)
                    hinges.append(F.relu(args.margin - gap).mean())
                    sep[fname] += float((gap > args.margin).float().sum().detach())
                    gapsum[fname] += float(gap.sum().detach())

                stacked = torch.stack(hinges)
                # Worst-case blend: a pure mean lets the optimiser sacrifice the
                # hardest fault (blur) to win the easiest (rain/noise).
                hinge = ((1.0 - args.worst_case_w) * stacked.mean()
                         + args.worst_case_w * stacked.max())
                anchor = (t_clean.mean() - args.trust_target) ** 2
                loss = hinge + args.anchor_w * anchor

                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

            tot += loss.detach().item() * B
            hin += hinge.detach().item() * B
            anc += anchor.detach().item() * B
            n += B
        # Report the MEAN GAP (clean trust minus faulted trust), not the rate of
        # samples clearing `--margin`.
        #
        # The rate is a thresholded statistic and it was actively misleading
        # here: it printed 0.00 for every fault at every epoch while the loss
        # fell, because a gap of 0.29 and a gap of 0.00 both score zero against
        # a margin of 0.30. The downstream metric is AUROC, which depends only
        # on ORDERING, so a mean gap that is small but reliably positive is a
        # working detector even at 0.00 separation rate. The mean gap shows
        # that; the rate hides it.
        #
        # sep[] is kept because a rising rate is still the cleanest evidence
        # that the hinge term (not the anchor) is doing the work.
        return (tot / n, hin / n, anc / n,
                {f: gapsum[f] / n for f in fault_names},
                {f: sep[f] / n for f in fault_names})

    hdr = "".join(f"{f[:5]:>8}" for f in fault_names)
    print(f"\n{'epoch':>6}{'train':>9}{'val':>9}   val MEAN GAP (clean - faulted) by fault: {hdr}")
    print(f"{'':>6}{'':>9}{'':>9}   (positive = the fault lowers trust, which is what AUROC needs)")
    best = float("inf")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(dl_tr, True)
        with torch.no_grad():
            va = run_epoch(dl_va, False)
        # 4 decimals: these gaps live in sigmoid space and a reliably positive
        # 0.02 is a usable detector. 2 decimals rounded them all to 0.00.
        bars = "".join(f"{va[3][f]:>8.4f}" for f in fault_names)
        print(f"{ep:>6}{tr[0]:>9.4f}{va[0]:>9.4f}   {'':<41}{bars}")
        if va[0] < best:
            best = va[0]
            torch.save({"state_dict": {f"model.{k}": v
                                       for k, v in model.state_dict().items()},
                        "finetune": {"source_ckpt": args.ckpt, "epoch": ep,
                                     "val_loss": va[0], "margin": args.margin,
                                     "trained_params": int(n_train)}},
                       args.out)
    print(f"\nBest val loss {best:.4f}. Wrote {args.out}")
    print(f"\nFinal-epoch val separation rate (fraction of samples clearing "
          f"margin={args.margin}):")
    print("  " + "  ".join(f"{f}={va[4][f]:.3f}" for f in fault_names))
    print("  A rate of 0.000 alongside a positive mean gap means the ordering is\n"
          "  right but the gap is smaller than the margin. AUROC only needs the\n"
          "  ordering, so check the eval below before concluding it failed.")
    print("\nNow re-measure with the SAME harness:\n"
          f"  python scripts/eval/eval_ood_detection.py --ckpt {args.out} \\\n"
          f"    --trust_grid {args.trust_grid} --cams all --faults all \\\n"
          f"    --out outputs/artifacts/ood_detection_report_grid{args.trust_grid}.json")


if __name__ == "__main__":
    main()
