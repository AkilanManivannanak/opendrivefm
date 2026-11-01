#!/usr/bin/env python3
"""Figure 6. What changes when the occluder comes from the model.

Measured, not argued. On MISSED objects observability is unchanged by the
substitution; on DETECTED objects it collapses, because predicted occupancy
is far denser than the ground-truth volume and shadows regions the rig
genuinely sees. The separation between seen and unseen shrinks and the
ranking degrades.

  (a) O from the ground-truth geometry field
  (b) O from the model's own predicted occupancy, same colour scale
  (c) their difference, so the over-occlusion is visible directly
  (d) the mechanism and its cost: mean observability for missed and
      detected objects under both fields, with object-level AUROC
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from egomarker import ego_car

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RNG, FREE = 40.0, 17
C_MISS, C_DET, INK = "#b3261e", "#1268a8", "#1a1a1a"


def main():
    tk = json.load(open(os.path.join(ROOT, "outputs/artifacts/figure_tokens.json")))
    tok = tk["fig5_token"]
    a55 = json.load(open(os.path.join(ROOT,
                         "outputs/artifacts/a55_pred_occluder.json")))

    gc = np.load(glob.glob(os.path.join(
        ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts", "*", tok,
        "labels.npz"))[0])["semantics"]
    pc = np.load(os.path.join(ROOT, "data/preds/preds_voxel", tok + ".npz"))["cls"]
    ogt = np.load(os.path.join(ROOT, "data/pack/obs_max",
                               tok + ".npy")).astype(np.float32) / 255.
    opr = np.load(os.path.join(ROOT, "data/pack/obs_pred_s800",
                               tok + ".npy")).astype(np.float32) / 255.
    fgt, fpr = float((gc != FREE).mean()), float((pc != FREE).mean())

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8.2, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.30),
                           gridspec_kw={"width_ratios": [1, 1, 1, 1.42]})
    fig.subplots_adjust(left=0.045, right=0.985, top=0.855, bottom=0.235,
                        wspace=0.46)
    ext = [RNG, -RNG, -RNG, RNG]

    def bev(a, img, cmap, vmin, vmax, title):
        h = a.imshow(img, origin="lower", extent=ext, cmap=cmap,
                     vmin=vmin, vmax=vmax, interpolation="nearest")
        ego_car(a, 0, 0, length=4.6, edge=INK, lw=0.4, zorder=6)
        a.set_xticks([-40, 0, 40]); a.set_yticks([-40, 0, 40])
        a.tick_params(labelsize=7, length=2, pad=1)
        a.set_title(title, fontsize=8.4, loc="left", pad=3)
        for s in a.spines.values():
            s.set_linewidth(0.6)
        return h

    A, B, C, D = ax
    h1 = bev(A, ogt.max(2).T, "viridis", 0, 1,
             "(a)  $O$ from $\\mathcal{G}_{GT}$")
    h2 = bev(B, opr.max(2).T, "viridis", 0, 1,
             "(b)  $O$ from $\\mathcal{G}_{pred}$")
    for h, a in ((h1, A), (h2, B)):
        c_ = fig.colorbar(h, ax=a, fraction=0.046, pad=0.012)
        c_.set_ticks([0, 1]); c_.ax.tick_params(labelsize=7, length=2)
        c_.outline.set_linewidth(0.5)

    d = (opr.max(2) - ogt.max(2)).T
    h3 = bev(C, d, "RdBu", -1, 1, "(c)  $O_{pred} - O_{GT}$")
    cb3 = fig.colorbar(h3, ax=C, fraction=0.046, pad=0.012)
    cb3.set_ticks([-1, 0, 1]); cb3.ax.tick_params(labelsize=7, length=2)
    cb3.outline.set_linewidth(0.5)
    A.set_ylabel("x (m)", fontsize=7, labelpad=1)
    for a in (A, B, C):
        a.set_xlabel("y (m)", fontsize=7, labelpad=0.5)
    C.text(0.5, -0.30, f"occupied: GT {fgt:.1%}, predicted {fpr:.1%}",
           transform=C.transAxes, fontsize=7, color=INK, ha="center",
           va="top")

    # ---- (d) the mechanism, from the paired per-object measurement
    m = a55["mechanism"] if "mechanism" in a55 else None
    if m is None:
        r = np.load(os.path.join(ROOT, "outputs/artifacts/a55_rows.npz"))
        miss = ~r["det"].astype(bool)
        m = {"missed": [float(r["ogt"][miss].mean()), float(r["opr"][miss].mean())],
             "detected": [float(r["ogt"][~miss].mean()), float(r["opr"][~miss].mean())],
             "n_missed": int(miss.sum()), "n_detected": int((~miss).sum())}
    for key, col in (("detected", C_DET), ("missed", C_MISS)):
        y0, y1 = m[key]
        D.plot([0, 1], [y0, y1], "-o", color=col, lw=1.7, ms=5,
               mfc="white", mew=1.5, zorder=4)
        D.annotate(f"{key}  {y1-y0:+.3f}", xy=(0.5, (y0 + y1) / 2),
                   xytext=(0, -11 if key == "detected" else 7),
                   textcoords="offset points", fontsize=7, color=col,
                   va="center", ha="center")
    sep0 = m["detected"][0] - m["missed"][0]
    sep1 = m["detected"][1] - m["missed"][1]
    D.annotate("", xy=(0.045, m["detected"][0]), xytext=(0.045, m["missed"][0]),
               arrowprops=dict(arrowstyle="<->", lw=0.8, color="#6b6b6b"))
    D.annotate("", xy=(0.955, m["detected"][1]), xytext=(0.955, m["missed"][1]),
               arrowprops=dict(arrowstyle="<->", lw=0.8, color="#6b6b6b"))
    D.text(0.075, (m["detected"][0] + m["missed"][0]) / 2, f"{sep0:.3f}",
           fontsize=7, color="#6b6b6b", va="center")
    D.text(0.925, (m["detected"][1] + m["missed"][1]) / 2, f"{sep1:.3f}",
           fontsize=7, color="#6b6b6b", va="center", ha="right")
    D.set_xlim(-0.16, 1.16); D.set_ylim(0.16, 0.60); D.set_xticks([0, 1])
    D.set_xticklabels(["$\\mathcal{G}_{GT}$", "$\\mathcal{G}_{pred}$"],
                      fontsize=8)
    D.set_ylabel("mean observability", fontsize=7.6, labelpad=2)
    D.tick_params(labelsize=7, length=2)
    D.set_title("(d)  separation shrinks "
                f"{100*(1-sep1/sep0):.0f}%", fontsize=8.4, loc="left", pad=3)
    for s in ("top", "right"):
        D.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        D.spines[s].set_linewidth(0.6)
    os.makedirs(os.path.join(ROOT, "outputs/figures"), exist_ok=True)
    out = os.path.join(ROOT, "outputs/figures/fig5_deployability.pdf")
    fig.savefig(out, dpi=600); fig.savefig(out.replace(".pdf", ".png"), dpi=600)
    print("token", tok, "| occupied GT %.4f pred %.4f" % (fgt, fpr))
    print("mechanism", {k: [round(x, 4) for x in v] for k, v in m.items()
                        if isinstance(v, list)})
    print("wrote", out)


if __name__ == "__main__":
    main()
