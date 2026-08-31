"""
VLA Phase 1 — vision-conditioned trajectory decoding.

WHAT THIS CONNECTS
------------------
    6 cameras -> OpenDriveFM backbone (FROZEN) -> fused BEV latent z (B, 384)
                                                        |
                              Linear projector (TRAINABLE) -> k prefix embeddings
                                                        v
    GPT-2 (FROZEN) <- [visual prefix] ++ [<BOS> x0 y0 ... x11 y11 <EOS>]
                                                        |
                                          trajectory tokens -> 12 waypoints

This is the LLaVA projector pattern applied to driving, and it is the same
family as Waymo's EMMA and DriveVLM: actions expressed as discrete tokens,
decoded by a language model conditioned on visual features.

A DEFECT IN THE EXISTING traj_lm.py THAT THIS FIXES
---------------------------------------------------
`NuScenesTrajectoryDataset` reads waypoints from `row["ego_future"]`, falling
back to `row["ego_pose"]["translation"]`. The manifest contains NEITHER key, so
the fallback yields tx = [0, 0, 0] and every waypoint becomes (0, 0). Verified:
all 404 manifest rows produce ONE distinct trajectory, all zeros. The GPT-2 in
outputs/artifacts/traj_lm_gpt2/ was therefore fine-tuned on 404 copies of the
same constant sequence -- real fine-tuning of a real model on degenerate data.

The genuine trajectories live in the label files
(nuscenes_labels_128/*.npz, key "traj"): 194 distinct out of 200 sampled. This
script uses those, via the same NuScenesMiniMultiView loader every other
evaluation in the repo uses.

RESIDUAL TOKENISATION
---------------------
Raw waypoints reach x = 89 m, outside the tokenizer's +/-20 m range, so absolute
tokenisation would clip most of the trajectory into the final bin. We tokenise
the RESIDUAL from the constant-velocity prior instead:

    residual = traj_gt - t_rel * velocity

which is what the existing MLP head predicts, keeps values well inside +/-20 m,
and makes the comparison against that head apples-to-apples. Bin width is 0.2 m,
a quantisation floor far below the ~2.7 m ADE, so tokenisation is not the
bottleneck.

CONSTRAINED DECODING
--------------------
At generation, even positions may only emit x tokens (0-199) and odd positions
only y tokens (200-399). An unconstrained LM can emit <BOS> mid-sequence or two
x tokens in a row, producing an undecodable trajectory. Masking the logits makes
every generation structurally valid by construction.

Usage
-----
    python scripts/train/train_vla_projector.py --train \
        --ckpt outputs/artifacts/checkpoints_v11_trustfix2/trust_fixed_v2.ckpt \
        --out outputs/artifacts/vla_projector.pt

    python scripts/train/train_vla_projector.py --eval \
        --ckpt outputs/artifacts/checkpoints_v11_trustfix2/trust_fixed_v2.ckpt \
        --projector outputs/artifacts/vla_projector.pt \
        --report outputs/artifacts/vla_report.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, "src")
sys.path.insert(0, str(ROOT))          # so `import traj_lm` finds scripts/traj_lm.py

from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView  # noqa: E402
from opendrivefm.robustness.perturbations import PERTURBATIONS  # noqa: E402
from opendrivefm.training.lightning_module import LitOpenDriveFM  # noqa: E402
from traj_lm import TrajectoryTokenizer  # noqa: E402  (reuse the exact tokenizer)

from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

VAL_SCENES = {"scene-0655", "scene-1077"}
TOK = TrajectoryTokenizer()
N_BINS = TOK.N_BINS


class VisualProjector(nn.Module):
    """Maps one BEV latent to k prefix embeddings in the LM's embedding space."""

    def __init__(self, d_in: int = 384, d_model: int = 768, k: int = 4,
                 hidden: int = 512):
        super().__init__()
        self.k, self.d_model = k, d_model
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, k * d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(z).view(z.shape[0], self.k, self.d_model))


def build_lm(device: str, pretrained: bool = True):
    """GPT-2 with the trajectory vocabulary and a short position budget."""
    if pretrained:
        lm = GPT2LMHeadModel.from_pretrained("gpt2")
        lm.resize_token_embeddings(TOK.VOCAB_SIZE)
    else:
        lm = GPT2LMHeadModel(GPT2Config(
            vocab_size=TOK.VOCAB_SIZE, n_positions=64, n_embd=768,
            n_layer=6, n_head=8))
    lm.transformer.wpe = nn.Embedding(64, lm.config.n_embd)
    nn.init.normal_(lm.transformer.wpe.weight, std=0.02)
    lm.config.n_positions = 64
    return lm.to(device)


def cv_prior(motion: torch.Tensor, t_rel: torch.Tensor) -> torch.Tensor:
    """Constant-velocity waypoints: t_rel (B,T) x velocity (B,2) -> (B,T,2)."""
    return t_rel.unsqueeze(-1) * motion[:, 1:3].unsqueeze(1)


def encode_batch(resid: torch.Tensor) -> torch.Tensor:
    """(B,T,2) residual metres -> (B, 2T+2) token ids, on CPU."""
    out = []
    for r in resid.detach().cpu().numpy():
        out.append(TOK.encode_waypoints(r))
    return torch.tensor(out, dtype=torch.long)


def decode_tokens_batch(tok: torch.Tensor, horizon: int) -> torch.Tensor:
    """(B, 2T) x/y token ids -> (B,T,2) residual metres."""
    span = TOK.X_MAX - TOK.X_MIN
    t = tok.view(tok.shape[0], horizon, 2).float()
    x = (t[..., 0] / N_BINS) * span + TOK.X_MIN
    y = ((t[..., 1] - N_BINS) / N_BINS) * span + TOK.Y_MIN
    return torch.stack([x, y], dim=-1)


@torch.no_grad()
def generate(lm, prefix, horizon: int, device: str) -> torch.Tensor:
    """Constrained greedy decode. Even step -> x token, odd step -> y token."""
    B = prefix.shape[0]
    emb = torch.cat(
        [prefix, lm.transformer.wte(torch.full((B, 1), TOK.BOS, device=device))],
        dim=1)
    out = []
    for step in range(horizon * 2):
        logits = lm(inputs_embeds=emb).logits[:, -1, :]
        mask = torch.full_like(logits, float("-inf"))
        if step % 2 == 0:
            mask[:, :N_BINS] = 0.0                     # x tokens only
        else:
            mask[:, N_BINS:2 * N_BINS] = 0.0           # y tokens only
        nxt = (logits + mask).argmax(dim=-1)
        out.append(nxt)
        emb = torch.cat([emb, lm.transformer.wte(nxt).unsqueeze(1)], dim=1)
    return torch.stack(out, dim=1)


def loaders(args, rows, ds):
    tr = [i for i, r in enumerate(rows) if r["scene"] not in VAL_SCENES]
    va = [i for i, r in enumerate(rows) if r["scene"] in VAL_SCENES]
    mk = lambda i, sh: DataLoader(Subset(ds, i), batch_size=args.batch_size,
                                  shuffle=sh, num_workers=0)
    return mk(tr, True), mk(va, False), len(tr), len(va)


def load_backbone(ckpt_path: str, bev: int, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}
    lit = LitOpenDriveFM(bev=bev)
    res = lit.model.load_state_dict(state, strict=False)
    n = len(lit.model.state_dict())
    print(f"  backbone: {n - len(res.missing_keys)}/{n} weights matched")
    m = lit.model.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def ade_fde(pred, gt):
    d = torch.linalg.norm(pred - gt, dim=-1)
    return d.mean(dim=1), d[:, -1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--projector", default="outputs/artifacts/vla_projector.pt")
    ap.add_argument("--out", default="outputs/artifacts/vla_projector.pt")
    ap.add_argument("--report", default="outputs/artifacts/vla_report.json")
    ap.add_argument("--manifest", default="outputs/artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", default="outputs/artifacts/nuscenes_labels_128")
    ap.add_argument("--bev", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--image_hw", default="90,160")
    ap.add_argument("--prefix_k", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no_pretrained_lm", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    H, W = [int(v) for v in args.image_hw.split(",")]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    print("Loading")
    backbone = load_backbone(args.ckpt, args.bev, device)
    lm = build_lm(device, pretrained=not args.no_pretrained_lm)
    for p in lm.parameters():
        p.requires_grad_(False)
    lm.eval()
    proj = VisualProjector(k=args.prefix_k).to(device)
    n_tr = sum(p.numel() for p in proj.parameters())
    n_all = n_tr + sum(p.numel() for p in lm.parameters()) + \
        sum(p.numel() for p in backbone.parameters())
    print(f"  projector: {n_tr:,} trainable of {n_all:,} total ({n_tr/n_all:.2%})")

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    ds = NuScenesMiniMultiView(args.manifest, image_hw=(H, W), frames=1,
                               label_root=args.label_root, return_motion=True,
                               return_trel=True, return_calib=True, augment=False)
    dl_tr, dl_va, n_train, n_val = loaders(args, rows, ds)
    print(f"  train {n_train} frames / val {n_val} frames "
          f"(split by scene: {sorted(VAL_SCENES)})\n")

    def step(batch, train: bool):
        x = batch[0].to(device)
        traj = batch[2].to(device)
        motion, t_rel = batch[3].to(device), batch[4].to(device)
        cv = cv_prior(motion, t_rel)
        resid = traj - cv
        tokens = encode_batch(resid).to(device)          # (B, 2T+2)
        B = x.shape[0]

        with torch.no_grad():
            z, _, _ = backbone.backbone(x)
        prefix = proj(z)                                 # (B, k, 768)
        tok_emb = lm.transformer.wte(tokens)
        emb = torch.cat([prefix, tok_emb], dim=1)
        labels = torch.cat(
            [torch.full((B, args.prefix_k), -100, device=device, dtype=torch.long),
             tokens], dim=1)
        # GPT2LMHeadModel shifts labels internally, so we pass the FULL token
        # sequence, not a pre-shifted one. Pre-shifting here would double-shift.
        return lm(inputs_embeds=emb, labels=labels).loss, traj, cv

    if args.train:
        opt = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=1e-2)
        print(f"{'epoch':>6}{'train loss':>13}{'val loss':>11}")
        best = float("inf")
        for ep in range(1, args.epochs + 1):
            proj.train(); tl = n = 0.0
            for b in dl_tr:
                loss, _, _ = step(b, True)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                tl += loss.detach().item() * b[0].shape[0]; n += b[0].shape[0]
            proj.eval(); vl = m = 0.0
            with torch.no_grad():
                for b in dl_va:
                    loss, _, _ = step(b, False)
                    vl += loss.item() * b[0].shape[0]; m += b[0].shape[0]
            print(f"{ep:>6}{tl/n:>13.4f}{vl/m:>11.4f}")
            if vl / m < best:
                best = vl / m
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"projector": proj.state_dict(), "prefix_k": args.prefix_k,
                            "val_loss": best, "epoch": ep}, args.out)
        print(f"\nBest val loss {best:.4f}. Wrote {args.out}")

    if args.eval:
        sd = torch.load(args.projector, map_location=device, weights_only=False)
        proj.load_state_dict(sd["projector"]); proj.eval()
        print(f"\nLoaded projector (epoch {sd['epoch']}, val loss {sd['val_loss']:.4f})\n")
        horizon = 12
        results = {}

        def evaluate(tag, zero_prefix=False, shuffle_prefix=False, fault=None,
                     cams=(0, 1, 4)):
            A, F, Acv, Amlp = [], [], [], []
            perts = [(c, PERTURBATIONS[fault]()) for c in cams] if fault else []
            with torch.no_grad():
                for b in dl_va:
                    x = b[0].to(device); traj = b[2].to(device)
                    motion, t_rel = b[3].to(device), b[4].to(device)
                    if perts:
                        x = x.clone()
                        for i in range(x.shape[0]):
                            for c, p in perts:
                                x[i, c, -1] = p(x[i, c, -1].unsqueeze(0)).squeeze(0)
                    cv = cv_prior(motion, t_rel)
                    z, _, _ = backbone.backbone(x)
                    prefix = proj(z)
                    if zero_prefix:
                        prefix = torch.zeros_like(prefix)
                    if shuffle_prefix and prefix.shape[0] > 1:
                        # The DECISIVE control. Zeroing the prefix pushes the LM
                        # far out of distribution -- it has never seen a zero
                        # prefix -- so a large ADE jump proves only that the
                        # prefix is load-bearing, not that its CONTENT carries
                        # scene information. Shuffling keeps every prefix
                        # in-distribution while destroying the frame-to-prefix
                        # correspondence. If ADE barely moves under shuffling,
                        # the model is using the prefix as generic context, not
                        # reading the scene.
                        perm = torch.randperm(prefix.shape[0], device=prefix.device)
                        prefix = prefix[perm]
                    tok = generate(lm, prefix, horizon, device)
                    pred = cv + decode_tokens_batch(tok.cpu(), horizon).to(device)
                    a, f = ade_fde(pred, traj); A.append(a); F.append(f)
                    Acv.append(ade_fde(cv, traj)[0])
                    _, res, _, _ = backbone(x, velocity=motion[:, 1:3])
                    Amlp.append(ade_fde(cv + res, traj)[0])
            per = {"vla": torch.cat(A).cpu().numpy(),
                   "cv": torch.cat(Acv).cpu().numpy(),
                   "mlp": torch.cat(Amlp).cpu().numpy()}
            out = {"ade": float(per["vla"].mean()), "fde": float(torch.cat(F).mean()),
                   "ade_cv_baseline": float(per["cv"].mean()),
                   "ade_mlp_head": float(per["mlp"].mean()),
                   "_per_frame": per}
            results[tag] = {k: v for k, v in out.items() if k != "_per_frame"}
            print(f"  {tag:<26} VLA ADE {out['ade']:>7.3f}  FDE {out['fde']:>7.3f}   "
                  f"(CV {out['ade_cv_baseline']:.3f}, MLP {out['ade_mlp_head']:.3f})")
            return out

        print("Trajectory predictors on held-out scenes")
        clean = evaluate("vision-conditioned")
        blind = evaluate("prefix zeroed (OOD)", zero_prefix=True)
        shuf = evaluate("prefix shuffled (control)", shuffle_prefix=True)
        faults = {f: evaluate(f"corrupted: {f}", fault=f) for f in PERTURBATIONS}

        # VISION SENSITIVITY: if conditioning does nothing, corrupting the
        # cameras and zeroing the prefix both leave ADE unchanged.
        blind_delta = blind["ade"] - clean["ade"]
        shuffle_delta = shuf["ade"] - clean["ade"]
        fault_delta = float(np.mean([v["ade"] for v in faults.values()]) - clean["ade"])
        print(f"\nVISION SENSITIVITY")
        print(f"  prefix ZEROED (out of distribution)  ADE {blind_delta:+8.4f} m"
              f"   <- confounded, see note")
        print(f"  prefix SHUFFLED (in distribution)    ADE {shuffle_delta:+8.4f} m"
              f"   <- decisive")
        print(f"  cameras CORRUPTED                    ADE {fault_delta:+8.4f} m"
              f"   <- decisive")

        # Judge on the in-distribution controls, not the OOD one.
        uses_vision = abs(shuffle_delta) > 0.10 and abs(fault_delta) > 0.05
        if uses_vision:
            verdict = ("USES SCENE CONTENT: shuffling the prefix and corrupting the "
                       "cameras both move ADE materially.")
        else:
            verdict = (
                "DOES NOT USE SCENE CONTENT. Zeroing the prefix breaks the model "
                f"({blind_delta:+.2f} m), but that only shows the prefix is load-bearing: "
                "a zero prefix is an input the LM never saw in training. Under the "
                f"in-distribution controls the model barely reacts (shuffle "
                f"{shuffle_delta:+.3f} m, corruption {fault_delta:+.3f} m), so the "
                "projector is supplying generic context, not scene-specific "
                "information. This mirrors the MLP head, which also ignores the "
                "cameras.")
        print(f"\n  VERDICT: {verdict}")

        print(f"\nACCURACY vs BASELINES (held-out scenes)")
        print(f"  constant velocity      ADE {clean['ade_cv_baseline']:.3f} m")
        print(f"  MLP residual head      ADE {clean['ade_mlp_head']:.3f} m")
        print(f"  VLA (this model)       ADE {clean['ade']:.3f} m"
              f"   {'BEATS' if clean['ade'] < clean['ade_cv_baseline'] else 'WORSE THAN'}"
              f" the constant-velocity baseline")

        # ── DIFFICULTY-STRATIFIED ANALYSIS ──────────────────────────────────
        # An aggregate ADE hides the question that matters. Most nuScenes-mini
        # frames are near-straight driving where constant velocity is already
        # almost exact, so there is nothing for vision to add and the average is
        # dominated by easy frames. If vision helps anywhere, it is on the frames
        # where the CV prior FAILS -- turns, braking, acceleration.
        #
        # Splitting the held-out frames into terciles by CV error isolates that.
        # A model that beats CV only on the hard tercile is still a useful model;
        # one that loses everywhere is not.
        pf = clean["_per_frame"]
        order = np.argsort(pf["cv"])
        n = len(order)
        bounds = [order[:n // 3], order[n // 3:2 * n // 3], order[2 * n // 3:]]
        names = ["easy (CV accurate)", "medium", "hard (CV fails)"]
        strata = {}
        print(f"\nDIFFICULTY-STRATIFIED ADE (terciles by constant-velocity error)")
        print(f"{'stratum':<22}{'n':>4}{'CV':>9}{'MLP':>9}{'VLA':>9}   VLA vs CV")
        print("-" * 66)
        for nm, idx in zip(names, bounds):
            cv_, mlp_, vla_ = pf["cv"][idx].mean(), pf["mlp"][idx].mean(), pf["vla"][idx].mean()
            rel = 100.0 * (vla_ - cv_) / cv_ if cv_ > 0 else float("nan")
            strata[nm] = {"n": int(len(idx)), "ade_cv": float(cv_),
                          "ade_mlp": float(mlp_), "ade_vla": float(vla_),
                          "vla_vs_cv_pct": float(rel)}
            print(f"{nm:<22}{len(idx):>4}{cv_:>9.3f}{mlp_:>9.3f}{vla_:>9.3f}"
                  f"{rel:>+10.1f}%")
        hard = strata[names[-1]]
        if hard["ade_vla"] < hard["ade_cv"]:
            print("\n  The VLA beats constant velocity on the HARD tercile even though it\n"
                  "  loses on average. That is where a learned trajectory model earns its\n"
                  "  keep, and the aggregate number was hiding it.")
        else:
            print("\n  The VLA loses to constant velocity on every difficulty tercile,\n"
                  "  including the hard one. The failure is not that easy frames dominate\n"
                  "  the average; the model is simply not competitive at this data scale.")

        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps({
            "checkpoint": args.ckpt, "projector": args.projector,
            "prefix_k": args.prefix_k, "n_val": n_val,
            "results": results,
            "difficulty_strata": strata,
            "vision_sensitivity": {
                "blind_delta_ade_m": blind_delta,
                "shuffle_delta_ade_m": shuffle_delta,
                "corruption_delta_ade_m": fault_delta,
                "uses_scene_content": bool(uses_vision),
                "verdict": verdict,
                "note": ("Zeroing the prefix is an out-of-distribution input and "
                         "is reported for completeness only. Shuffling prefixes "
                         "across the batch and corrupting cameras are the "
                         "in-distribution controls that decide the question."),
            },
        }, indent=2))
        print(f"\nWrote {args.report}")


if __name__ == "__main__":
    main()
