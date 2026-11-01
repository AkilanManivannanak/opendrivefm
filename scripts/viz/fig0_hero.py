#!/usr/bin/env python3
"""SUPERSEDED. The paper's Figure 1 is scripts/viz/fig1_pipeline_fill.py.

This is the earlier hero and is kept only because its panels are reused in
the README. It is not part of the paper build.

Figure 1 (hero): the phenomenon, then the number, in one row.

The reviewer's note was that the teaser led with statistics before the
reader knew what failure was being counted. This puts the failure first --
real camera frame, real prediction, real observability field, same object
circled in all three -- and ends on the inversion.

Everything is measured. Panels (a)-(c) are one test-split token; panel (d)
reads the artifacts.

  python3 scripts/viz/fig0_hero.py
"""
from __future__ import annotations
import glob, importlib.util, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import ListedColormap

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from egomarker import ego_car

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOKEN = "dc03fe4e0a374592af7701c66d5276c4"
RES, RNG, FREE = 0.4, 40.0, 17
C_GEOM, C_CONF, C_HOT = "#1268a8", "#8d5c0e", "#b3261e"
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d6"
DYNC = [2, 3, 4, 5, 6, 7, 9, 10]
STATC = [11, 12, 13, 14, 15, 16]


def tri(cls):
    t = np.zeros(cls.shape[:2], np.int16)
    t[np.isin(cls, STATC).any(2)] = 1
    t[np.isin(cls, [1, 8]).any(2)] = 2
    t[np.isin(cls, DYNC).any(2)] = 3
    return t


def main():
    sp = importlib.util.spec_from_file_location(
        "md", os.path.join(ROOT, "scripts/eval/missed_detection.py"))
    md = importlib.util.module_from_spec(sp); sp.loader.exec_module(md)

    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    row = next(r for r in index if r["token"] == TOKEN); fi = index.index(row)
    gcls = np.load(glob.glob(os.path.join(
        ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts", "*", TOKEN,
        "labels.npz"))[0])["semantics"].astype(np.int16)
    pk = np.load(os.path.join(ROOT, "data/preds/preds_voxel", TOKEN + ".npz"))
    pcls = pk["cls"].astype(np.int16); pconf = pk["conf"].astype(np.float32) / 255.
    obs = np.load(os.path.join(ROOT, "data/pack/obs_max",
                               TOKEN + ".npy")).astype(np.float32) / 255.

    ann = np.load(os.path.join(ROOT, "data/pack/annotations_val.npz"),
                  allow_pickle=True)
    names = ann["category_names"]; fr = ann["frame"]
    order = np.argsort(fr, kind="stable")
    bounds = np.searchsorted(fr[order], np.arange(len(index) + 1))
    cam0 = row["cams"]["CAM_FRONT"]
    Re = md.qrot(np.asarray(cam0["ego2global_rotation"], np.float64))
    te = np.asarray(cam0["ego2global_translation"], np.float64)
    best = None
    for k in order[bounds[fi]:bounds[fi + 1]]:
        cid = md.CAT2OCC.get(str(names[ann["category"][k]]))
        if cid is None or cid not in md.DYN:
            continue
        c = Re.T @ (ann["translation"][k].astype(np.float64) - te)
        R = Re.T @ md.qrot(np.asarray(ann["rotation"][k], np.float64))
        vx = md.box_voxels(c, ann["size"][k].astype(np.float64), R, 0.0)
        if vx is None:
            continue
        gm = gcls[vx] == cid
        if gm.sum() < 6 or np.isin(pcls[vx], list(md.DYN)).any():
            continue
        sc = float((pcls[vx] == FREE).mean()) * gm.sum()
        if best is None or sc > best["s"]:
            best = dict(s=sc, c=c, R=R, size=ann["size"][k].astype(np.float64),
                        obs=float(obs[vx][gm].max()),
                        conf=float(pconf[vx].max()),
                        rng=float(np.hypot(c[0], c[1])))
    cx, cy = best["c"][0], best["c"][1]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8, "text.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.95),
                           gridspec_kw={"width_ratios": [1, 1, 1, 1.45]})
    fig.subplots_adjust(wspace=0.26, left=0.004, right=0.996,
                        top=0.80, bottom=0.08)

    # (a) the camera that should see it
    bearing = np.degrees(np.arctan2(cy, cx))
    CB = {"CAM_FRONT": 0., "CAM_FRONT_LEFT": 55., "CAM_FRONT_RIGHT": -55.,
          "CAM_BACK_LEFT": 110., "CAM_BACK_RIGHT": -110., "CAM_BACK": 180.}
    use = min(CB, key=lambda c: abs((bearing - CB[c] + 180) % 360 - 180))
    dc = row["cams"][use]
    fn = dc["filename"]
    jp = fn if os.path.exists(fn) else os.path.join(
        ROOT, "dataset/nuscenes", fn.split("nuscenes/")[-1])
    ax[0].imshow(plt.imread(jp), aspect="auto")
    K = np.asarray(dc["intrinsic"], np.float64)
    Rs = md.qrot(np.asarray(dc["sensor2ego_rotation"], np.float64))
    ts = np.asarray(dc["sensor2ego_translation"], np.float64)
    hw, hl, hh = best["size"] / 2
    corn = np.array([[sx * hl, sy * hw, sz * hh] for sx in (-1, 1)
                     for sy in (-1, 1) for sz in (-1, 1)])
    Cm = ((best["R"] @ corn.T).T + best["c"] - ts) @ Rs
    if (Cm[:, 2] > 0.1).all():
        uv = (K @ Cm.T).T; uv = uv[:, :2] / uv[:, 2:3]
        u0, v0, u1, v1 = uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()
        ax[0].add_patch(Rectangle((u0, v0), u1 - u0, v1 - v0, fill=False,
                                  lw=1.2, ec=C_HOT, zorder=5))
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title("(a)  it is there", fontsize=8, loc="left", pad=3)
    ax[0].text(0.03, 0.05, "pedestrian, 18 m,\nbehind a lamp post",
               transform=ax[0].transAxes, fontsize=6.4, color="#f2f2f2",
               va="bottom", fontweight="bold")

    def bev(a, img, cmap, vmin, vmax, norm=None):
        a.imshow(img, origin="lower", extent=[RNG, -RNG, -RNG, RNG], cmap=cmap,
                 norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest",
                 aspect="auto")
        a.set_xlim(22, -22); a.set_ylim(-6, 32)
        a.set_xticks([]); a.set_yticks([])
        for s_ in a.spines.values():
            s_.set_color(GRID); s_.set_linewidth(0.6)
        a.add_patch(Circle((cy, cx), 2.7, fill=False, lw=1.2, ec=C_HOT, zorder=8))
        ego_car(a, 0, 0, length=4.6, body=INK, glass="#f2f2f4",
                edge="#f2f2f4", lw=0.4, zorder=6)

    bev(ax[1], tri(pcls), ListedColormap(
        ["#ffffff", "#e3e3df", "#b9b9b2", "#d1495b"]), None, None,
        norm=matplotlib.colors.BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4))
    ax[1].set_title("(b)  the model says free", fontsize=8, loc="left", pad=3)
    ax[1].text(0.03, 0.05, f"confidence {best['conf']:.2f}",
               transform=ax[1].transAxes, fontsize=7, color=C_HOT,
               va="bottom", fontweight="bold")

    bev(ax[2], obs.max(2), "magma", 0, 1)
    ax[2].set_title("(c)  the geometry knew", fontsize=8, loc="left", pad=3)
    ax[2].text(0.03, 0.05, f"observability {best['obs']:.2f}",
               transform=ax[2].transAxes, fontsize=7, color="#ffffff",
               va="bottom", fontweight="bold")

    # (d) the inversion
    vox = json.load(open(os.path.join(ROOT, "outputs/artifacts/baselines.json")))
    obj = json.load(open(os.path.join(ROOT, "outputs/artifacts/baselines_object.json")))
    frm = json.load(open(os.path.join(ROOT, "outputs/artifacts/mining_validation.json")))["targets"]["WRONG"]
    rows = [("voxel", vox["auroc"]["MSP"], vox["auroc"]["observability"]),
            ("frame", frm["mean_margin"]["auroc"], frm["mean_obs"]["auroc"]),
            ("object", obj["auroc"]["MSP (max conf in box)"], obj["auroc"]["observability"])]
    b = ax[3]
    xs = np.arange(3)
    b.plot(xs, [r[1] for r in rows], "-o", color=C_CONF, lw=1.8, ms=5,
           mec="#ffffff", mew=1.0, zorder=4, label="confidence")
    b.plot(xs, [r[2] for r in rows], "-o", color=C_GEOM, lw=1.8, ms=5,
           mec="#ffffff", mew=1.0, zorder=4, label="observability")
    # label the higher series above and the lower below, or they collide at
    # the crossover, which is the one place the figure must stay readable
    for i, (_, a_, g_) in enumerate(rows):
        da, dg = (0.014, -0.026) if a_ > g_ else (-0.026, 0.014)
        b.text(i, a_ + da, f"{a_:.3f}", ha="center", fontsize=6.4, color=C_CONF)
        b.text(i, g_ + dg, f"{g_:.3f}", ha="center", fontsize=6.4,
               color=C_GEOM, fontweight="bold" if g_ > a_ else "normal")
    b.axvspan(1.62, 2.38, color=C_GEOM, alpha=0.08, zorder=0)
    b.annotate("geometry wins", xy=(2.02, 0.742), xytext=(1.22, 0.632),
               fontsize=7, color=C_GEOM, fontweight="bold",
               arrowprops=dict(arrowstyle="->", lw=0.9, color=C_GEOM))
    b.set_xticks(xs); b.set_xticklabels([r[0] for r in rows], fontsize=8)
    b.set_xlim(-0.35, 2.5); b.set_ylim(0.58, 0.94)
    b.set_ylabel("AUROC", fontsize=7.6, labelpad=4)
    b.tick_params(labelsize=6.8, length=2)
    b.yaxis.grid(True, color=GRID, lw=0.5, zorder=0); b.set_axisbelow(True)
    for s_ in ("top", "right"):
        b.spines[s_].set_visible(False)
    b.legend(loc="upper right", bbox_to_anchor=(1.02, 1.03), frameon=False,
             fontsize=6.6, handlelength=1.3, labelspacing=0.2)
    b.set_title("(d)  and the ranking inverts", fontsize=8, loc="left", pad=3)

    out = os.path.join(ROOT, "outputs", "figures")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig0_hero.{ext}"), bbox_inches="tight",
                    dpi=400, facecolor="white")
    plt.close(fig)
    print(f"camera {use}  conf {best['conf']:.2f}  obs {best['obs']:.2f}")
    print("wrote outputs/figures/fig0_hero.{pdf,png}")


if __name__ == "__main__":
    main()
