# OpenDriveFM — Trust-Aware Multi-Camera BEV Occupancy Mapping

> **Project:** Robust Multi-Camera Bird's Eye View (BEV) Occupancy Mapping with Trust-Aware Fusion  
> **Dataset:** nuScenes mini (v1.0-mini, 404 samples, 10 scenes)  
> **Framework:** PyTorch + PyTorch Lightning + Apple Silicon MPS

---

## Project Summary

A computer vision pipeline for autonomous driving that transforms six 2D camera feeds into a unified 3D Bird's Eye View map, with a **Trust-Aware Sensor Fusion** module that assigns per-camera confidence scores — allowing the vehicle to maintain a stable environment map even when a camera faces interference (blur, glare, occlusion, rain, or sensor noise).

---

## Architecture

```
6 RGB Cameras  (B, V=6, T=1, C=3, H=90, W=160)
       │
       ├──► CameraTrustScorer  ──────────────────────────────┐
       │     ├─ CNN branch: learns visual quality features    │
       │     └─ Stats branch: blur variance, luminance,       │  trust ∈ (0,1)
       │                       edge density                   │  per camera
       │                                                      │
       └──► CNN Stem (7×7 + 3×3 conv) → TemporalTransformer  │
                 (d=384, 4 layers, 6 heads)                   │
                       │                                      │
                       └──► TrustWeightedFusion ◄─────────────┘
                                 softmax(trust) × features
                                       │
                     ┌─────────────────┴──────────────────┐
                     │                                    │
              BEVOccupancyHead                       TrajHead
           MLP → (B, 1, 64, 64)              CV prior + residual
           Binary occupancy grid             MLP → (B, 12, 2) xy
                     │                                    │
              64×64 BEV map                  Future trajectory
              (dynamic objects)              (6 seconds @ 2Hz)
```

---

## Three Research Components

### 1. Back-Projection & BEV Decoder (`models/model.py`)
- **CNN Stem** extracts per-camera features at H/4 × W/4 resolution
- **TemporalTransformer** aggregates across time frames (d=384, 4 layers)
- **BEVOccupancyHead**: MLP lifts fused feature → 64×64 occupancy logits
- **TrajHead**: CV-prior residual prediction with time-weighted SmoothL1 loss

### 2. Data Engineering & Infrastructure (`data/`, `scripts/`)
- **LiDAR → BEV labels**: `build_nuscenes_labels.py` projects LiDAR point clouds onto a 50m × 50m BEV grid at 0.78m/cell resolution
- **Multi-camera loader**: `NuScenesMiniMultiView` serves 6-camera tensors with ego-motion, velocity, and future trajectory
- **Loss**: `BCE(pos_weight=10) + 0.7 × Dice` for occupancy; `time-weighted SmoothL1 + L2-reg` for trajectory

### 3. Trust-Aware Module & Robustness Testing (`robustness/`, `scripts/eval_robustness_trust.py`)
- **CameraTrustScorer**: Two-branch scorer (CNN + handcrafted image stats) → trust ∈ (0,1) per camera
- **TrustWeightedFusion**: `softmax(trust)` weighted sum replaces naive mean-pool
- **5 fault-injection types**: Gaussian blur, glare overlay, occlusion patch, rain streaks, salt-pepper noise
- **Ablation**: `--disable_trust` flag to compare with/without trust module

---

## Quickstart

```bash
cd opendrivefm

# One command runs everything:
bash install_and_run.sh

# Or step by step:
pip install -e .                          # Step 0: install
pip install pytorch-lightning einops      # Step 0: deps
python scripts/train_smoke.py --epochs 1  # Step 1: smoke test
```

---

## Project File Structure

```
opendrivefm/
├── install_and_run.sh              ← Run this first!
├── pyproject.toml                  ← Package definition
├── configs/
│   └── default.yaml                ← All hyperparameters
├── src/opendrivefm/
│   ├── models/
│   │   └── model.py                ← OpenDriveFM + CameraTrustScorer (CORE)
│   ├── data/
│   │   ├── nuscenes_mini.py        ← nuScenes dataset loader
│   │   └── synth.py                ← Synthetic data for smoke tests
│   ├── train/
│   │   └── lightning_module.py     ← Training loop + trust loss
│   ├── robustness/
│   │   └── perturbations.py        ← 5 fault-injection types
│   └── utils/
│       └── visualise.py            ← BEV overlay, trust dashboard, charts
├── scripts/
│   ├── prepare_nuscenes_mini.py    ← Build manifest (Step 4)
│   ├── build_nuscenes_labels.py    ← Build LiDAR→BEV labels (Step 5)
│   ├── eval_cv_baseline.py         ← Constant-velocity baseline (Step 6)
│   ├── train_nuscenes_mini_trust.py  ← Trust-aware training (Step 7)
│   ├── train_nuscenes_mini_residual.py ← Original training (ablation)
│   ├── eval_nuscenes_mini_ckpt_residual.py ← Eval IoU + ADE/FDE (Step 8)
│   ├── eval_robustness_trust.py    ← Fault-injection eval (Step 9)
│   ├── bench_latency.py            ← Inference speed (Step 10)
│   └── train_smoke.py              ← Synthetic quick-check
├── tests/
│   └── test_model.py               ← Unit tests (pytest)
├── artifacts/                      ← Generated outputs (auto-created)
│   ├── nuscenes_mini_manifest.jsonl
│   ├── nuscenes_labels/            ← .npz per sample
│   ├── checkpoints_trust/          ← Saved model weights
│   ├── eval_metrics.json           ← IoU + ADE/FDE results
│   ├── robustness_report.json      ← Trust under degradation
│   └── robustness_trust_chart.png  ← Bar chart
├── data/
│   └── nuscenes/v1.0-mini/         ← PUT YOUR DATASET HERE
└── lightning_logs/                 ← TensorBoard logs
```

---

## Loss Function

```
L_total = L_occ + w_traj × L_traj + w_trust × L_trust

L_occ   = BCE(pos_weight=10, capped@15) + 0.7 × Dice
L_traj  = time-weighted SmoothL1(residual, gt − cv_prior) + 0.02 × ‖residual‖²
L_trust = (mean(trust) − 0.75)² − 0.1 × H(trust)
          ↑ pull toward 0.75        ↑ entropy bonus (diverse trust)
```

---

## Training Commands

```bash
# Full trust-aware training (recommended)
python scripts/train_nuscenes_mini_trust.py \
    --manifest artifacts/nuscenes_mini_manifest.jsonl \
    --label_root artifacts/nuscenes_labels \
    --max_epochs 20 --batch_size 2

# Ablation: no trust module
python scripts/train_nuscenes_mini_trust.py ... --disable_trust

# Monitor live: open http://localhost:6006
tensorboard --logdir lightning_logs/
```

---

## Evaluation Commands

```bash
# BEV IoU + trajectory ADE/FDE
python scripts/eval_nuscenes_mini_ckpt_residual.py \
    --ckpt artifacts/checkpoints_trust/last.ckpt \
    --manifest artifacts/nuscenes_mini_manifest.jsonl \
    --label_root artifacts/nuscenes_labels

# Trust robustness under 5 degradation types
python scripts/eval_robustness_trust.py \
    --ckpt artifacts/checkpoints_trust/last.ckpt \
    --manifest artifacts/nuscenes_mini_manifest.jsonl \
    --label_root artifacts/nuscenes_labels \
    --perturb_cam_idx 0    # 0=FRONT camera

# Inference latency on Mac
python scripts/bench_latency.py --views 6 --h 90 --w 160 --iters 100
```

---

## Expected Results (nuScenes mini)

| Metric | Value | Notes |
|--------|-------|-------|
| val/ADE | ~3.0 m | Matches CV baseline (expected on mini) |
| val/FDE | ~6.5 m | Final displacement error at 6s |
| val/occ_iou | ~0.10–0.20 | Low due to class imbalance |
| val/trust_mean | 0.65–0.85 | Mean trust across 6 cameras |
| FRONT blur trust drop | −0.3 to −0.5 | After training on trust loss |
| Mac MPS latency (p50) | ~70ms | ~14 FPS |

---

## Key Findings (Trust-Aware Fusion)

1. **Trust scores respond to image quality** — blur and occlusion cause larger drops than noise
2. **Weighted fusion is more robust** — when one camera fails, model relies on remaining 5
3. **Entropy regularisation** prevents all trust collapsing to a single camera
4. **CV-prior residual training** stabilises trajectory learning on small datasets

---

## Artifacts to Keep

| File/Folder | Keep? | Why |
|-------------|-------|-----|
| `artifacts/nuscenes_labels/` | ✅ Yes | Expensive to rebuild (2–5 min) |
| `artifacts/nuscenes_mini_manifest.jsonl` | ✅ Yes | References to your data paths |
| `artifacts/checkpoints_trust/` | ✅ Yes | Your trained model |
| `artifacts/checkpoints_nuscenes_residual/` | ✅ Yes | Previous best model (comparison) |
| `artifacts/nuscenes_eval_metrics_residual.json` | ✅ Yes | Previous eval results |
| `artifacts/eval_metrics.json` | ✅ Yes | Latest eval output |
| `artifacts/robustness_report.json` | ✅ Yes | Trust robustness results |
| `artifacts/*.tar.gz`, `*.enc` | ⚠️ Optional | Old backups, safe to delete |
| `lightning_logs/` | ✅ Yes | TensorBoard training history |

---

## Dependencies

```
torch >= 2.0
pytorch-lightning >= 2.0
einops >= 0.7
nuscenes-devkit >= 1.1
numpy, Pillow, matplotlib, scipy, pyquaternion, tqdm, pandas
```

Install all: `pip install -e .`
