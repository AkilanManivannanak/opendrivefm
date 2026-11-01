#!/usr/bin/env python3
"""Figure 4. Why a binary camera mask is not an observability measure.

Four panels, all measured. No evaluative marks in the artwork; the reading
is in the caption.
  (a) mask_camera, BEV, for one sealed-test frame: binary.
  (b) O(v;G), BEV, same frame: continuous, occlusion-resolved.
  (c) the value distribution of both over 60 sealed-test frames.
  (d) per-voxel error rate by observability decile, restricted to
      mask_camera = 1, with paired-frame bootstrap intervals.
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
INK, C_GEOM, C_MASK = "#1a1a1a", "#2d6a9f", "#c05a3e"
RNG_M = 40.0


def main():
    tk = json.load(open(os.path.join(ROOT, "outputs/artifacts/figure_tokens.json")))
    tok = tk["fig4_token"]
    st = json.load(open(os.path.join(ROOT, "outputs/artifacts/maskvsobs.json")))

    z = np.load(glob.glob(os.path.join(
        ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts", "*", tok,
        "labels.npz"))[0])
    mk = z["mask_camera"].astype(bool)
    obs = np.load(os.path.join(ROOT, "data/pack/obs_max",
                               tok + ".npy")).astype(np.float32) / 255.

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8.2, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.15),
                           gridspec_kw={"height_ratios": [1.18, 1.0]})
    fig.subplots_adjust(left=0.075, right=0.975, top=0.955, bottom=0.085,
                        wspace=0.20, hspace=0.30)
    ext = [RNG_M, -RNG_M, -RNG_M, RNG_M]

    # (a) mask_camera, BEV
    A = ax[0, 0]
    A.imshow(mk.any(2).T, origin="lower", extent=ext, cmap="gray",
             vmin=0, vmax=1, interpolation="nearest")
    A.set_title("(a)  mask_camera: binary, no occlusion",
                fontsize=8.4, loc="left", pad=3)

    # (b) observability, BEV, same frame
    B = ax[0, 1]
    im = B.imshow(obs.max(2).T, origin="lower", extent=ext, cmap="viridis",
                  vmin=0, vmax=1, interpolation="nearest")
    B.set_title("(b)  $O(v;\\mathcal{G})$: continuous, occlusion-aware",
                fontsize=8.4, loc="left", pad=3)
    cb = fig.colorbar(im, ax=B, fraction=0.046, pad=0.02)
    cb.set_ticks([0, 0.5, 1.0]); cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.5)

    for a in (A, B):
        ego_car(a, 0, 0, length=4.6, edge=INK, lw=0.4, zorder=6)
        a.set_xticks([-40, 0, 40]); a.set_yticks([-40, 0, 40])
        a.tick_params(labelsize=7, length=2)
        a.set_xlabel("y (m)", fontsize=7, labelpad=1)
        a.set_ylabel("x (m)", fontsize=7, labelpad=1)
        for s in a.spines.values():
            s.set_linewidth(0.6)

    # (c) support, measured over the sampled test frames
    C = ax[1, 0]
    p1 = st["mask_frac_one"]
    C.bar([0.0, 1.0], [1 - p1, p1], width=0.035, color=C_MASK,
          label="mask_camera", zorder=3)
    h = np.asarray(st["obs_hist_counts"], float)
    cen = np.linspace(0, 1, 51)[:-1] + 0.01
    C.fill_between(cen, 0, h / h.sum(), color=C_GEOM, alpha=0.30, zorder=2)
    C.plot(cen, h / h.sum(), color=C_GEOM, lw=1.3, zorder=4,
           label="$O(v;\\mathcal{G})$")
    C.set_xlim(-0.04, 1.04); C.set_yscale("log"); C.set_ylim(2e-4, 1.6)
    C.set_xlabel("value", fontsize=7.8, labelpad=1.5)
    C.set_ylabel("fraction of voxels (log)", fontsize=7.8, labelpad=2)
    C.legend(frameon=False, fontsize=7, loc="upper right",
             handlelength=1.4, borderaxespad=0.2)
    C.set_title(f"(c)  support, {st['frames']} sealed-test frames",
                fontsize=8.4, loc="left", pad=3)

    # (d) measured error rate by observability decile, inside mask_camera = 1
    D = ax[1, 1]
    x = np.asarray(st["decile_centres"]); r = np.asarray(st["error_rate"])
    lo = np.asarray(st["ci_lo"]); hi = np.asarray(st["ci_hi"])
    D.errorbar(x, r, yerr=[r - lo, hi - r], fmt="o", ms=3.6, lw=0,
               elinewidth=1.0, capsize=2.0, color=C_GEOM, zorder=4)
    D.plot(x, r, color=C_GEOM, lw=1.0, alpha=0.55, zorder=3)
    revi = [k + 1 for k in range(9) if r[k + 1] > r[k]]
    if revi:
        D.plot(x[revi], r[revi], "o", ms=7.5, mfc="none", mec=C_MASK,
               mew=1.3, zorder=5)
    D.set_xlim(0, 1); D.set_xlabel("$O(v;\\mathcal{G})$ decile",
                                   fontsize=7.8, labelpad=1.5)
    D.set_ylabel("per-voxel error rate", fontsize=7.8, labelpad=2)
    D.set_title("(d)  inside mask_camera = 1, error still varies with $O$",
                fontsize=8.4, loc="left", pad=3)
    D.text(0.97, 0.94, f"{st['reversals']} reversals",
           transform=D.transAxes, ha="right", va="top", fontsize=7,
           color=C_MASK)
    for a in (C, D):
        a.tick_params(labelsize=7, length=2)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_linewidth(0.6)

    os.makedirs(os.path.join(ROOT, "outputs/figures"), exist_ok=True)
    out = os.path.join(ROOT, "outputs/figures/fig3_whynotmask.pdf")
    fig.savefig(out, dpi=600)
    fig.savefig(out.replace(".pdf", ".png"), dpi=600)
    print("token", tok, "| frames", st["frames"], "| reversals", st["reversals"])
    print("wrote", out)


if __name__ == "__main__":
    main()
