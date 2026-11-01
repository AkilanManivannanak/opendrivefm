#!/usr/bin/env python3
"""The confidence-blindness figure, measured rather than drawn.

This figure used to be a hand-made schematic with illustrative softmax
numbers printed on it, and a reviewer is right to discount an argument whose
only evidence is the author's own drawing. The argument is unchanged; the
numbers on it are now the sealed test split.

Panel (a) is the one thing that has to stay drawn, because it is a statement
about what the representation can and cannot encode, not about any
measurement: a wrong prediction occupies a voxel with the wrong label and
leaves a confidence trace there, while a missing prediction occupies it with
`free` and leaves a trace that is identical to correctly observed empty
road. Everything printed inside it is measured.

Panels (b) and (c) are the empirical content. The same 40,863 dynamic
objects from the 75 sealed test scenes are split three ways by what the
prediction put inside the ground-truth box: the right dynamic class
somewhere inside it (correct), some dynamic class but never the right one
(wrong class), or no dynamic class at all (missed, the paper's M^dyn). The
max-softmax distribution of the missed group sits almost on top of the
correct group, which is the structural claim; the observability
distribution does not.

Source: outputs/artifacts/missed_cache_conf.jsonl, the same cache that
produces baselines_object.json, restricted to the test scenes in
val_dev_test.json. Scoring the two columns on this selection reproduces the
paper's headline 0.7623 and 0.7193 exactly.

  python3 scripts/viz/fig_whyconf.py
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C_GEOM, C_CONF, C_HOT = "#1268a8", "#8d5c0e", "#b3261e"
C_OK = "#4a7a3f"
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d6"
FREE_FILL, CAR_FILL, WRONG_FILL = "#f2f2ef", "#1268a8", "#8d5c0e"

# AUROCs for the missed-vs-detected task on exactly this selection. They are
# recomputed below and asserted against these, so the figure cannot silently
# disagree with the paper.
AUROC_OBS, AUROC_MSP = 0.7623, 0.7193


def load():
    sp = json.load(open(os.path.join(ROOT, "outputs/artifacts/val_dev_test.json")))
    test = set(sp["test"])
    rows = []
    with open(os.path.join(ROOT, "outputs/artifacts/missed_cache_conf.jsonl")) as fh:
        for ln in fh:
            r = json.loads(ln)
            if r.get("scene") in test:
                rows.append(r)
    cmax = np.array([r["cmax"] for r in rows], np.float64)
    obs = np.array([r["obs"] for r in rows], np.float64)
    dyn = np.array([r["dyn"] for r in rows], bool)
    exact = np.array([r["exact"] for r in rows], bool)
    return cmax, obs, dyn, exact


def auroc(score, y):
    """Rank AUROC with ties as half credit; y = 1 is the positive class."""
    o = np.argsort(score, kind="mergesort")
    s = score[o]
    rk = np.empty(len(s), np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        rk[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty(len(s), np.float64)
    r[o] = rk
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def ecdf(ax, v, color, label, lw=1.5, ls="-"):
    x = np.sort(v)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.plot(np.r_[x[0], x], np.r_[0.0, y], color=color, lw=lw, ls=ls,
            label=label, solid_joinstyle="round", clip_on=True)


def voxel_strip(ax, x0, y0, fills, w=0.074, h=0.125, gap=0.009):
    for k, f in enumerate(fills):
        ax.add_patch(Rectangle((x0 + k * (w + gap), y0), w, h, facecolor=f,
                               edgecolor="#9a9a95", lw=0.5, zorder=3))
    return x0 + len(fills) * (w + gap) - gap


def panel_a(ax, med_missed, med_correct):
    """The one drawn panel: what the representation is able to encode.

    Everything printed on it is measured; the drawing carries the mechanism,
    which is a statement about the label space and not about any run.
    """
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    X0 = 0.015

    def row(y_title, y_strip, title, title_col, sub, fills, note, note_col,
            verdict, verdict_col):
        ax.text(X0, y_title, title, fontsize=8.5, color=title_col,
                fontweight="bold", va="baseline")
        ax.text(X0, y_title - 0.072, sub, fontsize=6.9, color=MUTED,
                style="italic", va="baseline")
        xr = voxel_strip(ax, X0, y_strip, fills)
        ax.annotate("", xy=(xr + 0.105, y_strip + 0.062),
                    xytext=(xr + 0.016, y_strip + 0.062),
                    arrowprops=dict(arrowstyle="-|>", color=note_col, lw=0.9,
                                    shrinkA=0, shrinkB=0))
        ax.text(xr + 0.125, y_strip + 0.062, note, fontsize=6.9,
                color=note_col, va="center", ha="left", linespacing=1.35)
        ax.text(X0, y_strip - 0.085, verdict, fontsize=7.1,
                color=verdict_col, va="baseline")

    row(0.955, 0.705,
        "wrong prediction", INK,
        "the voxel is occupied, the label on it is wrong",
        [CAR_FILL, CAR_FILL, WRONG_FILL, WRONG_FILL, FREE_FILL, FREE_FILL],
        "softmax has to\nspread over classes", C_CONF,
        "visible to confidence", C_OK)

    row(0.455, 0.205,
        "missing prediction", C_HOT,
        "no dynamic class anywhere inside the box",
        [FREE_FILL] * 6,
        "softmax votes\n\u2018free\u2019 at %.3f" % med_missed, C_HOT,
        "identical to observed empty road, which votes %.3f" % med_correct,
        C_HOT)

    ax.plot([X0, 0.985], [0.545, 0.545], color=GRID, lw=0.7, zorder=0)

    # key for the voxel colours, so the strips are readable on their own
    for k, (col, lab) in enumerate([(CAR_FILL, "ground-truth class"),
                                    (WRONG_FILL, "wrong class"),
                                    (FREE_FILL, "free")]):
        xk = X0 + [0.0, 0.405, 0.735][k]
        ax.add_patch(Rectangle((xk, 0.012), 0.040, 0.050, facecolor=col,
                               edgecolor="#9a9a95", lw=0.5))
        ax.text(xk + 0.052, 0.037, lab, fontsize=6.4, color=MUTED,
                va="center", ha="left")
    ax.set_title("(a)  what the representation is able to encode",
                 fontsize=8.4, loc="left", color=INK, pad=6)


def main():
    cmax, obs, dyn, exact = load()
    missed = ~dyn
    wrong = dyn & ~exact
    correct = exact
    n = len(cmax)
    assert missed.sum() + wrong.sum() + correct.sum() == n

    y = missed.astype(int)
    a_obs, a_msp = auroc(-obs, y), auroc(-cmax, y)
    assert abs(a_obs - AUROC_OBS) < 5e-5, a_obs
    assert abs(a_msp - AUROC_MSP) < 5e-5, a_msp

    med_m = float(np.median(cmax[missed]))
    med_c = float(np.median(cmax[correct]))
    sat = float((cmax[missed] >= 0.999).mean())

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8.4, "text.color": INK,
        "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.7,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.42),
                            gridspec_kw={"width_ratios": [1.16, 1, 1]})
    fig.subplots_adjust(wspace=0.30, left=0.012, right=0.988,
                        top=0.855, bottom=0.185)
    ax, bx, cx = axs

    panel_a(ax, med_m, med_c)

    groups = [(correct, C_OK, "correct", "-"),
              (wrong, C_CONF, "wrong class", (0, (4, 1.6))),
              (missed, C_HOT, "missed", "-")]

    for m, col, lab, ls in groups:
        ecdf(bx, cmax[m], col, f"{lab}  (n = {int(m.sum()):,})",
             lw=1.9 if lab == "missed" else 1.3, ls=ls)
    bx.set_xlim(0.5, 1.001)
    bx.set_ylim(0, 1.005)
    bx.set_xlabel("max softmax probability in box", fontsize=7.8)
    bx.set_ylabel("cumulative fraction of objects", fontsize=7.8)
    bx.set_title(f"(b)  confidence: AUROC {a_msp:.4f}", fontsize=8.4,
                 loc="left", color=C_CONF, pad=6)
    bx.grid(True, color=GRID, lw=0.5, zorder=0)
    bx.set_axisbelow(True)
    bx.legend(fontsize=6.6, loc="upper left", frameon=False,
              handlelength=1.5, labelspacing=0.3, borderpad=0.1)
    bx.annotate(f"medians {med_m:.3f} vs {med_c:.3f}\nthe curves nearly coincide",
                xy=(0.985, 0.50), xytext=(0.60, 0.42), fontsize=6.9,
                color=MUTED, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                                shrinkA=1, shrinkB=1))

    for m, col, lab, ls in groups:
        ecdf(cx, obs[m], col, lab,
             lw=1.9 if lab == "missed" else 1.3, ls=ls)
    cx.set_xlim(-0.004, 1.004)
    cx.set_ylim(0, 1.005)
    cx.set_xlabel("observability $O$ in box", fontsize=7.8)
    cx.set_title(f"(c)  observability: AUROC {a_obs:.4f}", fontsize=8.4,
                 loc="left", color=C_GEOM, pad=6)
    cx.grid(True, color=GRID, lw=0.5, zorder=0)
    cx.set_axisbelow(True)
    z_m = float((obs[missed] <= 1e-6).mean())
    z_c = float((obs[correct] <= 1e-6).mean())
    cx.text(0.315, 0.275,
            "no observability at all:\n"
            f"{100*z_m:.0f}% of missed, {100*z_c:.0f}% of correct",
            fontsize=6.9, color=MUTED, ha="left", va="top")

    out = os.path.join(ROOT, "outputs/figures")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "fig_whyconf.pdf"))
    fig.savefig(os.path.join(out, "fig_whyconf.png"), dpi=600)
    plt.close(fig)

    rec = dict(n=n, missed=int(missed.sum()), wrong=int(wrong.sum()),
               correct=int(correct.sum()), auroc_obs=a_obs, auroc_msp=a_msp,
               msp_median_missed=med_m, msp_median_correct=med_c,
               msp_saturated_missed=sat, obs_zero_missed=z_m,
               obs_zero_correct=z_c,
               source="outputs/artifacts/missed_cache_conf.jsonl",
               scope="test scenes of outputs/artifacts/val_dev_test.json")
    with open(os.path.join(ROOT, "outputs/artifacts/fig_whyconf.json"), "w") as fh:
        json.dump(rec, fh, indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
