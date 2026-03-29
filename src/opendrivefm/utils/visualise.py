"""
Visualisation utilities for OpenDriveFM:
  - BEV occupancy overlay (pred vs GT)
  - Camera trust score dashboard (6-camera grid)
  - Robustness bar chart
  - Training curve plotter from metrics.csv
"""
from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

TRUST_CMAP = LinearSegmentedColormap.from_list(
    "trust", [(0.85,0.15,0.15), (0.95,0.75,0.10), (0.15,0.75,0.45)])

CAMERA_LABELS = ["FRONT","FRONT-LEFT","FRONT-RIGHT","BACK","BACK-LEFT","BACK-RIGHT"]


# ── BEV Occupancy ─────────────────────────────────────────────────────────────

def render_bev_overlay(pred: np.ndarray, gt: np.ndarray,
                        title: str = "BEV", save_path: Optional[str] = None) -> np.ndarray:
    """
    Overlay GT (red) vs Pred (green) on a dark BEV grid.
    pred, gt: (H,W) uint8 {0,1}
    """
    H, W = pred.shape
    img  = np.zeros((H, W, 3), dtype=np.float32)
    img[gt   == 1] = (0.85, 0.15, 0.15)    # GT red
    img[pred == 1] = (0.15, 0.75, 0.45)    # Pred green
    both = (gt == 1) & (pred == 1)
    img[both]      = (0.95, 0.85, 0.10)    # Overlap yellow

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0a0a10")
    for ax, data, lbl, cmap in [
        (axes[0], gt,   "Ground Truth", "Reds"),
        (axes[1], pred, "Prediction",   "Greens"),
        (axes[2], img,  "Overlay",      None),
    ]:
        if cmap:
            ax.imshow(data, origin="lower", cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(data, origin="lower")
        ax.set_title(lbl, color="white", fontsize=12, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

    patches = [mpatches.Patch(color=(0.85,0.15,0.15), label="GT only"),
               mpatches.Patch(color=(0.15,0.75,0.45), label="Pred only"),
               mpatches.Patch(color=(0.95,0.85,0.10), label="Both (TP)")]
    fig.legend(handles=patches, loc="lower center", ncol=3, framealpha=0,
               labelcolor="white", fontsize=10)
    fig.suptitle(title, color="#e8e8f8", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig.canvas.draw()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    except AttributeError:
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, 1:]  # ARGB -> RGB
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0a10")
    plt.close(fig)
    return buf


# ── Trust Dashboard ───────────────────────────────────────────────────────────

def render_trust_dashboard(camera_images: List[np.ndarray],
                            trust_scores: List[float],
                            perturbations: Optional[List[List[str]]] = None,
                            save_path: Optional[str] = None) -> np.ndarray:
    N     = len(camera_images)
    ncols = 3
    nrows = (N + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows), facecolor="#0a0a10")
    axes = axes.flatten() if nrows > 1 else [axes] * ncols if N == 1 else list(axes.flatten())

    for i, (img, score) in enumerate(zip(camera_images, trust_scores)):
        ax = axes[i]
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        col = TRUST_CMAP(score)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(col); sp.set_linewidth(5)

        label = CAMERA_LABELS[i] if i < len(CAMERA_LABELS) else f"CAM_{i}"
        extra = ""
        if perturbations and i < len(perturbations) and perturbations[i]:
            extra = f"\n⚠ {', '.join(perturbations[i])}"
        ax.set_title(f"{label}\nTrust: {score:.2f}{extra}", color="white", fontsize=10, pad=4)

        # trust bar below image
        bw = img.shape[1]
        bar = np.zeros((max(4, img.shape[0]//20), bw, 3), dtype=np.uint8)
        fill = int(score * bw)
        bar[:, :fill]  = (int(col[0]*255), int(col[1]*255), int(col[2]*255))
        bar[:, fill:]  = (35, 35, 45)
        ax_b = ax.inset_axes([0, -0.06, 1, 0.05])
        ax_b.imshow(bar); ax_b.set_xticks([]); ax_b.set_yticks([])

    for j in range(N, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Camera Trust Score Dashboard", color="#e8e8f8", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.canvas.draw()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    except AttributeError:
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, 1:]  # ARGB -> RGB
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0a10")
    plt.close(fig)
    return buf


# ── Robustness Report ─────────────────────────────────────────────────────────

def plot_robustness_report(results: Dict[str, float],
                            save_path: Optional[str] = None) -> np.ndarray:
    """results: {name: trust_score_mean}"""
    names  = list(results.keys())
    values = [results[n] for n in names]
    colours = [TRUST_CMAP(v) for v in values]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0a0a10")
    ax.set_facecolor("#12121e")
    bars = ax.bar(names, values, color=colours, width=0.5, edgecolor="#2a2a38", linewidth=1.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{v:.3f}", ha="center", va="bottom", color="white",
                fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Mean Trust Score", color="white", fontsize=12)
    ax.set_title("Trust Score Robustness Under Camera Degradations",
                 color="#e8e8f8", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color("#2a2a38")
    if "clean" in results:
        ax.axhline(results["clean"], color="#5588ff", linestyle="--", linewidth=1.5,
                   label=f"Clean: {results['clean']:.3f}")
        ax.legend(facecolor="#1a1a26", labelcolor="white", fontsize=10)
    plt.tight_layout()
    fig.canvas.draw()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    except AttributeError:
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, 1:]  # ARGB -> RGB
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0a10")
    plt.close(fig)
    return buf


# ── Training Curve ────────────────────────────────────────────────────────────

def plot_training_curves(metrics_csv: str, save_path: Optional[str] = None) -> None:
    """Plot loss/ADE/FDE/trust curves from a lightning metrics.csv."""
    import pandas as pd
    df = pd.read_csv(metrics_csv)

    epoch_df = df[df["train/loss_epoch"].notna()].copy() if "train/loss_epoch" in df.columns else df

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0a0a10")
    for ax in axes:
        ax.set_facecolor("#12121e")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_color("#2a2a38")

    def _plot(ax, col_train, col_val, ylabel, title):
        if col_train in df.columns:
            d = df[df[col_train].notna()]
            ax.plot(d["step"], d[col_train], color="#4a9eff", linewidth=1.5, label="train", alpha=0.9)
        if col_val and col_val in df.columns:
            d = df[df[col_val].notna()]
            ax.plot(d["step"], d[col_val], color="#ff6b4a", linewidth=2.0, label="val", alpha=0.9)
        ax.set_xlabel("step", color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.set_title(title, color="#e8e8f8", fontsize=12, pad=6)
        ax.legend(labelcolor="white", facecolor="#1a1a26", fontsize=9)

    _plot(axes[0], "train/loss_step", "val/loss",  "loss",  "Total Loss")
    _plot(axes[1], "train/ADE_step",  "val/ADE",   "m",     "ADE (model vs CV)")
    _plot(axes[2], "train/trust_mean_step", "val/trust_mean", "score", "Trust Score Mean")

    fig.suptitle("OpenDriveFM Training Curves", color="#e8e8f8", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0a10")
    plt.close(fig)
