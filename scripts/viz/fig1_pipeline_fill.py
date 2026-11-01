#!/usr/bin/env python3
"""Fill the Figure 1 pipeline shell with REAL renders.

The shell (assets/fig1_shell.png) carries only schematic artwork: blocks,
arrows, formulas, the coverage and first-hit line drawings. Every slot in it
is an empty grey rectangle. This script detects those rectangles, renders the
real content for each from data/pack and outputs/artifacts, and composites.

Nothing in the output is drawn by hand or invented.

This writes fig1_pipeline, which is the paper's Figure 1. It used to write
fig0_hero, the same name as the superseded teaser in fig0_hero.py, so
whichever script ran last decided what the paper showed. The two now have
separate names and fig0_hero.py is no longer part of the build.
"""
import glob, io, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from egomarker import ego_car

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHELL = os.path.join(ROOT, "assets/fig1_shell.png")
OUT = os.path.join(ROOT, "outputs/figures")
SS = 2                      # supersample factor for each rendered slot
INK = "#1a1a1a"
CAM_ORDER = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
             "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def slots(img):
    """Locate the shell's empty placeholder rectangles by colour."""
    a = np.array(img.convert("RGB")).astype(np.int16)
    m = np.abs(a - 232).max(2) <= 6
    lab, n = ndimage.label(m)
    sizes = ndimage.sum_labels(m, lab, np.arange(1, n + 1))
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None or sizes[i] < 3000 * (img.width / 2172.0) ** 2:
            continue
        boxes.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
    thr = 200 * img.width / 2172.0
    cams = sorted([b for b in boxes if b[2] - b[0] < thr], key=lambda b: (b[1], b[0]))
    rest = sorted([b for b in boxes if b[2] - b[0] >= thr], key=lambda b: b[0])
    return dict(A=cams, B=rest[0], C=rest[1], D=rest[2], E=rest[3])


def fig_to_img(fig, w, h):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((w, h), Image.LANCZOS)


def autocrop(im, pad=2):
    a = np.array(im.convert("RGB")).astype(int)
    nz = np.where(np.abs(a - 255).max(2) > 10)
    if len(nz[0]) == 0:
        return im
    y0, y1 = max(0, nz[0].min() - pad), min(a.shape[0], nz[0].max() + pad)
    x0, x1 = max(0, nz[1].min() - pad), min(a.shape[1], nz[1].max() + pad)
    return im.crop((x0, y0, x1, y1))


def fit_center(im, w, h):
    """Scale to fit inside w x h without cropping, centre on white."""
    s = min(w / im.width, h / im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                   Image.LANCZOS)
    bg = Image.new("RGB", (w, h), "white")
    bg.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return bg


def blank(w, h):
    fig = plt.figure(figsize=(w * SS / 100, h * SS / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    return fig, ax


def crop_cover(im, w, h):
    iw, ih = im.size
    s = max(w / iw, h / ih)
    im = im.resize((max(1, int(iw * s)), max(1, int(ih * s))), Image.LANCZOS)
    iw, ih = im.size
    return im.crop(((iw - w) // 2, (ih - h) // 2, (iw - w) // 2 + w, (ih - h) // 2 + h))


def bars(w, h, labels, vals, hi):
    fig = plt.figure(figsize=(w * SS / 100, h * SS / 100), dpi=100)
    ax = fig.add_axes([0.40, 0.19, 0.42, 0.72])
    y = np.arange(len(vals))[::-1]
    cols = ["#2d6a9f" if l == hi else "#c3cfd9" for l in labels]
    ax.barh(y, vals, color=cols, height=0.58, edgecolor="none")
    for yi, v in zip(y, vals):
        ax.text(v + 0.015, yi, f"{v:.3f}", va="center",
                fontsize=6.6 * SS, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.0 * SS)
    ax.set_xlim(0.45, 1.0); ax.set_xticks([0.5, 0.75, 1.0])
    ax.set_xticklabels(["0.50", "0.75", "1.00"], fontsize=6.4 * SS)
    ax.tick_params(length=2.2 * SS, width=0.7 * SS, pad=1.6 * SS)
    ax.set_xlabel("AUROC", fontsize=7.0 * SS, labelpad=1.6 * SS)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.7 * SS)
    ax.axvline(0.5, color="#9a9a9a", lw=0.7 * SS, ls=(0, (3, 3)), zorder=0)
    return fig_to_img(fig, w, h)


def main():
    tok = json.load(open(os.path.join(ROOT, "outputs/artifacts/figure_tokens.json")))["fig1_token"]
    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    row = next(r for r in index if r["token"] == tok)
    vox = json.load(open(os.path.join(ROOT, "outputs/artifacts/baselines.json")))["auroc"]
    obj = json.load(open(os.path.join(ROOT, "outputs/artifacts/baselines_object.json")))["auroc"]

    shell = Image.open(SHELL).convert("RGB")
    if shell.width < 4000:            # upscale before compositing so the
        shell = shell.resize((shell.width * 2, shell.height * 2),
                             Image.LANCZOS)   # slot renders print sharp
    S = slots(shell)

    # ---- A1..A6 : the six real rig images, in the shell's label order
    for (x0, y0, x1, y1), cam in zip(S["A"], CAM_ORDER):
        fn = row["cams"][cam]["filename"]
        jp = fn if os.path.exists(fn) else os.path.join(
            ROOT, "dataset/nuscenes", fn.split("nuscenes/")[-1])
        shell.paste(crop_cover(Image.open(jp).convert("RGB"), x1 - x0, y1 - y0), (x0, y0))

    # ---- B : the ground-truth occupancy volume this frame's G is built from
    g = np.load(glob.glob(os.path.join(
        ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts", "*", tok,
        "labels.npz"))[0])["semantics"].astype(np.int16)
    occ = g != 17
    x0, y0, x1, y1 = S["B"]; w, h = x1 - x0, y1 - y0
    fig = plt.figure(figsize=(w * SS / 100, h * SS / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(0, 0, 1, 1)
    st = 4
    o = occ[::st, ::st, :]
    xs, ys, zs = np.nonzero(o)
    keep = (xs > 4) & (xs < o.shape[0] - 4) & (ys > 4) & (ys < o.shape[1] - 4)
    xs, ys, zs = xs[keep], ys[keep], zs[keep]
    ax.scatter(xs, ys, zs, s=5.5 * SS, c=zs, cmap="bone_r",
               vmin=-6, vmax=o.shape[2] + 1, marker="s",
               linewidths=0, depthshade=False)
    ax.set_xlim(xs.min(), xs.max()); ax.set_ylim(ys.min(), ys.max())
    ax.set_zlim(0, max(6, zs.max()))
    ax.set_box_aspect((1, 1, 0.34)); ax.view_init(elev=30, azim=-58)
    ax.set_axis_off(); ax.set_proj_type("ortho")
    shell.paste(fit_center(autocrop(fig_to_img(fig, w, h)), w, h), (x0, y0))

    # ---- C : the observability volume, BEV, max over z
    obs = np.load(os.path.join(ROOT, "data/pack/obs_max", tok + ".npy")).astype(np.float32) / 255.
    x0, y0, x1, y1 = S["C"]; w, h = x1 - x0, y1 - y0
    fig, ax = blank(w, h)
    ZSL = 4   # 1.6 m above the volume floor
    ax.imshow(obs[:, :, ZSL].T, origin="lower", cmap="viridis", vmin=0, vmax=1,
              interpolation="nearest", aspect="auto")
    n = obs.shape[0]
    # grid cells here, not metres: a 4.6 m car is 11.5 cells at 0.4 m
    ego_car(ax, n / 2, n / 2, length=11.5, edge=INK, lw=0.4 * SS, zorder=5)
    ax.set_xlim(0, n); ax.set_ylim(0, n)
    shell.paste(fig_to_img(fig, w, h), (x0, y0))

    # ---- D : voxel-level ranking, measured
    x0, y0, x1, y1 = S["D"]
    shell.paste(bars(x1 - x0, y1 - y0,
                     ["model conf.", "observability", "mask_camera"],
                     [vox["MSP"], vox["observability"], vox["mask_camera"]],
                     "model conf."), (x0, y0))

    # ---- E : object-level missed-detection ranking, measured
    x0, y0, x1, y1 = S["E"]
    shell.paste(bars(x1 - x0, y1 - y0,
                     ["observability", "model conf."],
                     [obj["observability"], obj["MSP (max conf in box)"]],
                     "observability"), (x0, y0))

    # The shell artwork sets the strapline as "--- learning-free, one
    # parameter ---". The paper sets no em dash anywhere else, and a dash
    # that appears only inside a raster figure is the kind of inconsistency
    # a copy editor catches and an author cannot explain. The shell is a
    # flat PNG, so the two glyphs are painted out rather than re-typeset;
    # the phrase reads the same without them. The assertion keeps this from
    # silently erasing the wrong pixels if the shell is ever redrawn.
    px = shell.load()
    CREAM = (254, 246, 229)
    for xa, xb in ((1790, 1842), (2497, 2549)):
        dark = sum(1 for x in range(xa, xb) for y in range(1271, 1317)
                   if sum(px[x, y][:3]) < 420)
        assert 40 < dark < 900, (xa, dark)
        for x in range(xa, xb):
            for y in range(1271, 1317):
                px[x, y] = CREAM

    os.makedirs(OUT, exist_ok=True)
    shell.save(os.path.join(OUT, "fig1_pipeline.png"), dpi=(600, 600))
    shell.save(os.path.join(OUT, "fig1_pipeline.pdf"), "PDF",
               resolution=shell.width / 7.0)
    print("token", tok)
    print("voxel ", {k: round(vox[k], 4) for k in ("MSP", "observability", "mask_camera")})
    print("object", {k: round(v, 4) for k, v in obj.items()
                     if k in ("observability", "MSP (max conf in box)")})
    print("wrote outputs/figures/fig1_pipeline.{png,pdf}")


if __name__ == "__main__":
    main()
