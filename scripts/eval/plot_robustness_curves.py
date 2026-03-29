"""
Fix 3: plot_robustness_curves.py
Plots robustness curves:
  - IoU vs n_faulted cameras (degradation + dropout)
  - ADE vs n_faulted cameras
  - Trust score vs fault type (bar chart)
Uses existing fault test results + dropout results.
Copy to ~/opendrivefm/scripts/plot_robustness_curves.py
Run: python scripts/plot_robustness_curves.py
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dropout_json", default="artifacts/camera_dropout_results.json")
    ap.add_argument("--out",          default="artifacts/robustness_curves.png")
    args = ap.parse_args()

    # ── Known results from multi-camera fault test (already run) ──────────────
    # From session: n_faulted=0,1,2,3 with Gaussian blur
    blur_results = [
        dict(n=0, IoU=0.1360, ADE=2.7402, trust_faulted=float("nan"), trust_clean=float("nan")),
        dict(n=1, IoU=0.1421, ADE=2.7558, trust_faulted=0.305, trust_clean=0.448),
        dict(n=2, IoU=0.1441, ADE=2.8039, trust_faulted=0.300, trust_clean=0.450),
        dict(n=3, IoU=0.1469, ADE=2.8504, trust_faulted=0.301, trust_clean=0.450),
    ]

    # Trust scores by fault type (from robustness eval)
    fault_types   = ["Clean", "Blur", "Occl.", "Rain", "Noise", "Glare"]
    trust_scores  = [0.795,   0.329,  0.402,   0.340,  0.310,   0.491]
    fault_colors  = ["#2E7D32","#E05C5C","#E05C5C","#E05C5C","#E05C5C","#E07020"]

    # Load dropout results if available
    dropout_available = Path(args.dropout_json).exists()
    dropout_results = []
    if dropout_available:
        dropout_results = json.loads(Path(args.dropout_json).read_text())
        print(f"Loaded dropout results: {len(dropout_results)} entries")
    else:
        print(f"No dropout results found at {args.dropout_json} — skipping dropout curve")

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(1, 3, wspace=0.38)

    # ── Plot 1: IoU vs N Cameras Faulted ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ns_blur = [r["n"] for r in blur_results]
    iou_blur = [r["IoU"] for r in blur_results]

    ax1.plot(ns_blur, iou_blur, "o-", color="#1F4E79", linewidth=2.5,
             markersize=8, label="Blur degradation", zorder=4)

    if dropout_available:
        ns_drop  = [r["n_dropout"] for r in dropout_results]
        iou_drop = [r["IoU"] for r in dropout_results]
        ax1.plot(ns_drop, iou_drop, "s--", color="#E05C5C", linewidth=2,
                 markersize=8, label="Full dropout", zorder=4)

    ax1.axhline(0.1360, color="gray", linestyle=":", linewidth=1.5,
                label="Clean baseline (IoU=0.136)", alpha=0.7)

    for r in blur_results:
        ax1.annotate(f"{r['IoU']:.4f}", (r["n"], r["IoU"]),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, color="#1F4E79")

    ax1.set_title("BEV IoU vs Cameras Faulted", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Cameras Faulted", fontsize=10, labelpad=8)
    ax1.set_ylabel("BEV IoU", fontsize=10, labelpad=8)
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_ylim(0.10, 0.20)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Plot 2: ADE vs N Cameras Faulted ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ade_blur = [r["ADE"] for r in blur_results]

    ax2.plot(ns_blur, ade_blur, "o-", color="#1F4E79", linewidth=2.5,
             markersize=8, label="Blur degradation", zorder=4)

    if dropout_available:
        ade_drop = [r["ADE"] for r in dropout_results]
        ax2.plot(ns_drop, ade_drop, "s--", color="#E05C5C", linewidth=2,
                 markersize=8, label="Full dropout", zorder=4)

    ax2.axhline(3.012, color="#E05C5C", linestyle=":", linewidth=1.5,
                label="CV baseline (ADE=3.01m)", alpha=0.8)
    ax2.axhline(2.740, color="gray", linestyle=":", linewidth=1.5,
                label="Clean baseline (ADE=2.74m)", alpha=0.7)

    for r in blur_results:
        ax2.annotate(f"{r['ADE']:.3f}", (r["n"], r["ADE"]),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, color="#1F4E79")

    ax2.set_title("ADE vs Cameras Faulted", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Cameras Faulted", fontsize=10, labelpad=8)
    ax2.set_ylabel("ADE (m)", fontsize=10, labelpad=8)
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_ylim(2.5, 3.3)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── Plot 3: Trust Score by Fault Type ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    bars = ax3.bar(fault_types, trust_scores, color=fault_colors,
                   width=0.6, zorder=3)
    ax3.axhline(0.15, color="#555555", linestyle=":", linewidth=1.5,
                label="Dropout threshold τ=0.15")

    for bar, val in zip(bars, trust_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    ax3.set_title("Trust Score by Fault Type", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Fault Type", fontsize=10, labelpad=8)
    ax3.set_ylabel("Mean Trust Score (per camera)", fontsize=10, labelpad=8)
    ax3.set_ylim(0, 1.05)
    ax3.set_yticks(np.arange(0, 1.1, 0.2))
    ax3.legend(fontsize=8, loc="upper right")
    ax3.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax3.set_axisbelow(True)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")

if __name__ == "__main__":
    main()
