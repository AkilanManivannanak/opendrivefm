# OpenDriveFM — MLOps One-Pager

> **Built Trust-Aware BEV Perception for Autonomous Driving**
> p95 = 3.22ms · ADE = 2.457m (-18.4% vs baseline) · 317 FPS · 553K params · Apple Silicon

---

## 🎯 Goal & SLOs

| SLO | Target | Achieved | Status |
|-----|--------|----------|--------|
| p50 latency | < 28ms (36Hz) | **3.15ms** | ✅ 8.9× headroom |
| p95 latency | < 35ms | **3.22ms** | ✅ Near-zero jitter |
| Throughput | > 36 FPS | **317 FPS** | ✅ 8.8× above |
| Trajectory ADE | < CV baseline 3.012m | **2.457m** | ✅ -18.4% |
| Trust detection | Detect all 5 fault types | **100%** | ✅ No labels needed |
| BEV IoU | > 0.10 | **0.136** | ✅ |
| Hardware | Consumer device | **MacBook M-series** | ✅ |

---

## 🔄 Data Flow

```
INGEST                STORE              RETRIEVE            INFER             FEEDBACK
──────                ─────              ────────            ─────             ────────
nuScenes              JSONL              DataLoader          CNN Stem          val IoU
v1.0-mini      →      manifest    →      batch=2      →      Trust Score  →    val ADE
404 samples           BEV labels         augment             BEV Lifter        trust score
6 cameras             64×64 NPZ          fault inject        BEV Decoder       per-fault
LiDAR GT              scene splits       (p=0.5/batch)       Traj Head         ranking
```

---

## 🏗️ System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                             │
│  nuScenes → Manifest → Labels → Augment → Model → Checkpoint    │
│  AdamW lr=1e-4 · BCE+Dice · CosineAnnealingLR · 8 train scenes │
└─────────────────────────────────────────────────────────────────┘
                              │
                    best_val_ade.ckpt
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    SERVING PIPELINE                              │
│  6 cameras (90×160) → CNN → Trust → BEV Lift → Decode          │
│  p50=3.15ms · p95=3.22ms · 317 FPS · B=1 · Apple MPS           │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
       OCC MAP HEAD                    TRAJ HEAD
       64×64 binary                    12 waypoints
       IoU=0.136                       ADE=2.457m
```

---

## ⚖️ Trade-offs

| Trade-off | Choice Made | Why |
|-----------|-------------|-----|
| **Latency vs Quality** | 3.15ms / IoU=0.136 | Real-time deployment on edge hardware |
| **Freshness vs Cost** | Single frame inference | No frame buffering needed at 317 FPS |
| **3D voxels vs 2D BEV** | 2D BEV 64×64 | 553K params vs 46M for ProtoOcc |
| **Trust vs Speed** | Lightweight physics gate | Adds <0.1ms, detects all 5 fault types |
| **Binary vs Semantic** | Binary + v13 semantic | Binary stable; semantic needs more data |
| **LiDAR at inference** | Camera-only | Cheaper deployment, no LiDAR sensor |
| **Depth estimation** | 32 depth bins + LiDAR supervision | Train-time LiDAR, inference-time camera only |

---

## 🔧 MLOps Stack

```
Training:       PyTorch Lightning 2.0
Optimizer:      AdamW (lr=1e-4, wd=1e-4)
LR Schedule:    CosineAnnealingLR
Loss:           BCE + Dice (handles 4.3% positive sparsity)
Logging:        Weights & Biases (wandb) + Lightning CSV logs
Checkpointing:  ModelCheckpoint on val ADE (best_val_ade.ckpt)
Profiling:      bench_latency.py — 200 iters, 20 warmup
Eval:           eval_full_metrics_fixed.py
                eval_trust_ablation.py
                eval_worst_camera.py
                eval_camera_dropout.py
Hardware:       Apple M-series MPS (no GPU required)
Versioning:     13 checkpoints (v2→v14) each saved separately
```

---

## 🛡️ Reliability

### Caching
- Pre-allocated MPS tensors → near-zero p95/p50 jitter ratio (1.021)
- Manifest JSONL loaded once at startup

### Fallbacks
- Trust scorer dropout: cameras below τ=0.15 hard-zeroed, weights renormalised
- Demo falls back to webcam if nuScenes manifest not found
- Multiple checkpoint fallback chain (v11 → v9 → v8)

### Observability
- Per-frame LIVE IoU vs LiDAR GT displayed in demo
- Per-camera trust bars update every frame
- FPS + inference time monitored continuously
- Per-fault type trust score logging in eval scripts

### Graceful Degradation
- 0 cameras faulted: full 317 FPS, IoU=0.136
- 1 camera faulted: trust drops to 0.31–0.49, other 5 cameras compensate
- 2 cameras faulted: BEV quality degrades gracefully, no crash
- 3 cameras faulted: system continues running, reduced quality

---

## 📊 Eval Gates & Rollback

```
New checkpoint saved?
        │
        ▼
val ADE < previous best?  ──NO──→  Discard, keep previous ckpt
        │ YES
        ▼
val IoU reasonable (>0.05)?  ──NO──→  Flag for investigation
        │ YES
        ▼
Trust scores differentiate faults?  ──NO──→  Check contrastive loss
        │ YES
        ▼
Promote to best_val_ade.ckpt ✅
```

**Rollback strategy:** Each version (v2→v14) saved separately. Rolling back = load previous `checkpoints_vN/best_val_ade.ckpt`.

---

## 🔴 Postmortem Summary

| Issue | Root Cause | Fix | Impact |
|-------|-----------|-----|--------|
| IoU=0.801 (false win) | Drivable surface labels (79.7% positive) | Switch to object labels (4.3%) | Honest IoU=0.136 |
| Val loss exploding (~26) | lr=1e-3, no schedule | AdamW + CosineAnnealingLR | Loss → 9.5 |
| Data leakage | Per-sample split | Scene-level splits | Realistic metrics |
| Trust scores identical | 90×160 too small for CNN | Physics gate correction | Trust now differentiates |
| v14 ADE=18.78m | LSS needs burn-in epochs | Keep v11 as best | No regression |

---

## 🔮 What I'd Do With More Resources

```
With full nuScenes (700 scenes, 87× more data):
  → Expected IoU: 0.25+ (vs current 0.136)
  → Expected ADE: <2.0m (vs current 2.457m)

With GPU cluster (8× A100):
  → Train v14 LSS properly (needs 200+ epochs)
  → 3D semantic occupancy (17 classes like ProtoOcc)
  → Multi-frame temporal T=8 (vs current T=4)

With production infra:
  → Docker container for inference serving
  → CI/CD: pytest → lint → eval gate → deploy
  → Alert: if live IoU drops >20% vs baseline → page
  → Shadow testing: run v11 + v14 in parallel, compare
  → A/B test: trust-weighted vs uniform fusion in prod
```

---

*OpenDriveFM · LIU Image and Vision Computing · March 2026 · Apple Silicon · 317 FPS*
