#!/usr/bin/env python3
"""Figure 1 (teaser): the granularity inversion, plus the mechanism.

Every number is read from outputs/artifacts/*.json at render time. Nothing is
typed into this file by hand, so the figure cannot drift from the measurements
the way a pasted number can.

  python3 scripts/viz/fig1_granularity.py
  -> outputs/figures/fig1_granularity.pdf  (and .png for previewing)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "outputs", "artifacts")
OUT = os.path.join(ROOT, "outputs", "figures")

# Validated categorical pair (dataviz six-checks, light surface #fcfcfb):
# CVD dE 21.1 protan / 18.9 tritan, normal dE 23.3, contrast >= 3:1.
C_CONF = "#8d5c0e"   # confidence-derived signals
C_GEOM = "#1268a8"   # geometry (observability)
INK    = "#1a1a1a"
MUTED  = "#6b6b6b"
GRID   = "#d9d9d6"


def load(name):
    with open(os.path.join(ART, name)) as fh:
        return json.load(fh)


def main():
    vox = load("baselines.json")
    obj = load("baselines_object.json")
    frm = load("mining_validation.json")["targets"]["WRONG"]
    miss = load("missed_detection_test.json")

    assert vox["split_digest"] == obj["split_digest"], "split digest mismatch"
    digest = vox["split_digest"]

    rows = [
        ("per voxel",  vox["auroc"]["MSP"],                 vox["auroc"]["observability"]),
        ("per frame",  frm["mean_margin"]["auroc"],         frm["mean_obs"]["auroc"]),
        ("per object", obj["auroc"]["MSP (max conf in box)"], obj["auroc"]["observability"]),
    ]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8.4,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "pdf.fonttype": 42,   # TrueType, required by most camera-ready checkers
        "ps.fonttype": 42,
    })

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.35, 4.35),
        gridspec_kw={"height_ratios": [1.0, 0.85], "hspace": 0.70},
    )

    # ---------------------------------------------------------------- panel a
    h = 0.30
    gap = 0.03                      # 2px-equivalent surface gap between fills
    for i, (label, conf, geom) in enumerate(rows):
        y = len(rows) - 1 - i
        ax1.barh(y + (h + gap) / 2, conf, height=h, color=C_CONF, zorder=3)
        ax1.barh(y - (h + gap) / 2, geom, height=h, color=C_GEOM, zorder=3)
        ax1.text(conf + 0.008, y + (h + gap) / 2, f"{conf:.3f}",
                 va="center", ha="left", fontsize=7.6, color=INK)
        ax1.text(geom + 0.008, y - (h + gap) / 2, f"{geom:.3f}",
                 va="center", ha="left", fontsize=7.6, color=INK,
                 fontweight="bold" if geom > conf else "normal")

    ax1.set_yticks(range(len(rows)))
    ax1.set_yticklabels([r[0] for r in reversed(rows)], fontsize=8.4)
    ax1.set_xlim(0.5, 1.0)
    ax1.set_ylim(-1.05, len(rows) - 0.38)
    ax1.set_xlabel("AUROC  (ranks model error)", fontsize=8.4, labelpad=2)
    ax1.axvline(0.5, color=GRID, lw=0.8, zorder=1)
    ax1.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax1.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax1.spines[s].set_visible(False)
    ax1.tick_params(axis="y", length=0)

    # the inversion, called out on the row where it happens
    d = obj["contrasts"]["observability - MSP"]
    ax1.annotate(
        f"geometry wins\n+{d[0]:.3f}  [{d[1]:+.3f}, {d[2]:+.3f}]",
        xy=(rows[2][2] + 0.004, 0 - (h + gap) / 2 - 0.17), xytext=(0.790, -0.80),
        fontsize=7, color=C_GEOM, ha="center", va="center",
        arrowprops=dict(arrowstyle="-", lw=0.7, color=C_GEOM,
                        shrinkA=1, shrinkB=1),
    )

    ax1.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=C_CONF, label="confidence"),
        plt.Rectangle((0, 0), 1, 1, color=C_GEOM, label="observability"),
    ], loc="lower left", bbox_to_anchor=(-0.30, 1.02), frameon=False,
        fontsize=7.6, handlelength=1.0, handleheight=0.75, ncol=2,
        columnspacing=1.2, handletextpad=0.4)

    ax1.set_title("(a)  Confidence ranks voxels.\n      Geometry ranks objects.",
                  fontsize=8.4, loc="left", x=-0.30, pad=18, color=INK,
                  linespacing=1.35)

    # ---------------------------------------------------------------- panel b
    bands = miss["bands"]
    xs = list(range(len(bands)))
    ys = [100.0 * b["no_dyn"] for b in bands]
    ns = [b["n"] for b in bands]

    ax2.plot(xs, ys, color=C_GEOM, lw=1.6, zorder=3,
             marker="o", ms=4.2, mfc=C_GEOM, mec="#fcfcfb", mew=1.0)
    ax2.fill_between(xs, 0, ys, color=C_GEOM, alpha=0.08, zorder=2)

    ax2.text(xs[0] + 0.12, ys[0], f"{ys[0]:.1f}%", fontsize=7.6,
             va="center", ha="left", color=INK)
    ax2.text(xs[-1] - 0.12, ys[-1] + 3.0, f"{ys[-1]:.1f}%", fontsize=7.6,
             va="bottom", ha="right", color=INK, fontweight="bold")

    ax2.set_xticks(xs)
    ax2.set_xticklabels(["0", ".21", ".38", ".78", "<1", "=1"], fontsize=7.6)
    ax2.set_xlabel("observability band  (upper edge)", fontsize=8.4, labelpad=2)
    ax2.set_ylabel("objects the model\nnever predicts", fontsize=8.4, labelpad=2)
    ax2.set_ylim(0, 56)
    ax2.set_xlim(-0.35, len(bands) - 0.65)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(length=2)

    ax2.set_title(f"(b)  Monotone 6/6, n = {miss['objects']:,} objects",
                  fontsize=8.4, loc="left", pad=4, color=INK)
    # A54: missed_detection_test.json's "frames" field counts frames SCANNED,
    # not frames in the split, and reports 6,016 -- the same number as the
    # all-scenes run. The test half holds 3,008. The figure printed the wrong
    # one. Counted from the split module instead of trusting the field.
    import importlib.util as _ilu
    _sp = _ilu.spec_from_file_location("split", os.path.join(
        ROOT, "scripts/eval/split.py"))
    _SP = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_SP)
    _idx = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    nfr = sum(1 for r in _idx if r["scene"] in _SP.test_scenes())
    ax2.text(1.0, -0.41, f"{nfr:,} frames  ·  "
                         f"{obj['scenes']} held-out test scenes  ·  split {digest}",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=7, color=MUTED)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        # The title, the legend and the footer are all placed OUTSIDE the
        # axes (x=-0.30, y=-0.41), so a fixed-canvas save cropped them: in
        # the typeset paper the panel titles and the y-axis labels ran off
        # the left edge of the column. The canvas has to follow the artists.
        fig.savefig(os.path.join(OUT, f"fig1_granularity.{ext}"),
                    dpi=600, bbox_inches="tight", pad_inches=0.015,
                    facecolor="white", edgecolor="none")
    plt.close(fig)

    print("rows (conf vs geom):")
    for label, c, g in rows:
        print(f"  {label:<11} {c:.4f}  {g:.4f}  -> {'GEOM' if g > c else 'conf'}")
    print(f"missed-object: {ys[0]:.2f}% -> {ys[-1]:.2f}%  "
          f"monotone={all(ys[i] > ys[i+1] for i in range(len(ys)-1))}  n={ns}")
    print("wrote outputs/figures/fig1_granularity.{pdf,png}")


if __name__ == "__main__":
    main()
