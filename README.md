# OpenDriveFM

**Trust-aware multi-camera BEV perception, and the validation harness that measures whether it works.**

Camera-only occupancy and trajectory prediction on nuScenes, plus a fault-detection,
robustness, uncertainty and deployment-parity harness built around it. Every number
below is produced by a script in this repository and written to a JSON artifact in
`outputs/artifacts/`. Each row names the file it came from.

> **Read this first.** Earlier versions of this README reported IoU 0.136, ADE 2.457 m,
> 317 FPS, "+26.6% IoU under sensor faults" and "100% fault detection". None of those
> reproduce from the code and checkpoints in this tree. They are documented as removed
> claims at the bottom of this file, with the measurement that replaced each one. The
> harness that found them is the main contribution of this project.

---

## Verified results

### Camera-fault detection (`scripts/eval/eval_ood_detection.py`)

Can the `CameraTrustScorer` tell a degraded camera from a healthy one? Scored as binary
classification: positives are trust scores for a camera with a fault injected, negatives
are the same camera on the same frame, clean. 82 held-out frames, 5 fault types, 6 cameras,
bootstrap CIs over 1,000 resamples.

| Checkpoint | Pooled AUROC | 95% CI | AP | Artifact |
|---|---|---|---|---|
| v11 baseline | **0.434** | [0.419, 0.449] | 0.491 | `ood_detection_report_v11.json` |
| fix v1 (pooled hinge) | 0.652 | [0.636, 0.668] | 0.745 | `ood_detection_report_v11_trustfix.json` |
| **fix v2 (stratified + worst-case)** | **0.764** | [0.750, 0.777] | 0.814 | `ood_detection_report_v11_trustfix2.json` |

The baseline CI lies **entirely below 0.5**: the scorer was not weak, it was *inverted*.
Trust rose when a camera was degraded.

**Per-fault AUROC, showing why a pooled metric is not enough:**

| Fault | baseline | fix v1 | fix v2 |
|---|---|---|---|
| blur | 0.545 | **0.249** | 0.837 |
| glare | 0.344 | 0.525 | 0.635 |
| occlusion | 0.566 | 0.532 | 0.487 |
| rain | 0.316 | 0.997 | 0.962 |
| noise | 0.377 | 0.975 | 0.941 |

Fix v1 raised pooled AUROC by 55% while **regressing blur from 0.545 to 0.249**. A pooled
hinge let the optimiser sell the hard fault to buy the easy ones. Fix v2 scores every fault
on every batch and weights the loss `0.5·mean + 0.5·max` over faults, so no fault can be
traded away.

**Occlusion remains at chance, and the reason is architectural.** `CameraTrustScorer`
ends its statistics branch with `AdaptiveAvgPool2d(1)`, so every feature is a whole-image
average. A localised opaque patch barely moves a global mean. Fixing it requires spatial
pooling, not more training.

#### Spatial pooling fixes it, and the trade is visible (`--trust_grid 4`)

`CameraTrustScorer(grid=G)` pools into a GxG patch grid and feeds the head both the
**mean and the minimum** across patches. A dead region collapses edge energy in its own
patches while leaving the frame average almost untouched, so the minimum is the statistic
that sees it. `grid=1` reproduces the original architecture exactly, including tensor
shapes, so existing checkpoints keep loading.

Same harness, same 82 held-out frames, same 5 faults x 6 cameras:

| Fault | geometry | grid=1 | grid=4 | delta | worst camera, grid=1 | worst camera, grid=4 |
|---|---|---|---|---|---|---|
| occlusion | localised | 0.487 | **0.689** | **+0.202** | 0.477 | 0.646 |
| blur | localised | 0.838 | **0.920** | +0.083 | 0.726 | 0.863 |
| glare | localised | 0.635 | **0.707** | +0.071 | 0.481 | 0.646 |
| rain | global | 0.962 | 0.943 | -0.018 | 0.879 | 0.920 |
| noise | global | 0.941 | 0.890 | -0.051 | 0.882 | 0.852 |
| **pooled** | | **0.764** | **0.797** | +0.033 | | |
| | | [0.750, 0.777] | [0.784, 0.809] | CIs disjoint | | |

Grouped by fault geometry the result is not a wash, it is a **mechanism**:

- **localised** faults (occlusion, blur, glare): 0.653 -> **0.772** (+0.119)
- **global** faults (rain, noise): 0.951 -> **0.917** (-0.034)

Patch-min pooling buys sensitivity to damage confined to one region and pays for it by
diluting the whole-frame statistics that global corruptions move. That is the trade the
architecture predicts, measured rather than asserted.

The hazard-relevant number is the **worst camera**, not the mean: occlusion's worst case
moves 0.477 -> 0.646, off chance for every one of the six cameras. Pooled AUROC 0.797 is
still **below this repo's own 0.80 release gate**, and the gate was not moved to collect
a passing verdict.

**Choose by deployment, not by pooled AUROC.** A vehicle whose dominant failure mode is
lens occlusion (mud, an insect, a cracked housing) wants `grid=4`. One dominated by
weather wants `grid=1`. Reporting only the pooled number would hide that this is a
choice at all.

Artifacts: `ood_detection_report_grid1_remapped.json`, `ood_detection_report_grid4.json`.

### Measurement integrity: one bug found, one suspicion ruled out

Both surfaced while verifying an unrelated change. The first was a real defect that
produced plausible numbers rather than errors, which is the failure mode that actually
reaches a README. The second was a plausible-sounding bias that measurement refuted, and
it is recorded because a hypothesis you checked and discarded is evidence too.

**1. A rename ran the feature extractor on random weights, and the loader called it 90.5%.**

`CameraTrustScorer.cnn` was refactored into `.trunk` (convolutions) and `.cnn_head`
(classifier) when `grid` was introduced. `load_state_dict(strict=False)` matches on
**name**, so 18 structurally identical tensors stopped loading from every checkpoint
trained before the refactor, and the scorer's entire convolutional branch silently ran
on random initialisation.

Three things conspired to hide it:

- `strict=False` reports missing keys but does not fail.
- The loader reported a **match rate** (90.5%), which reads as a rounding detail rather
  than as "the feature extractor is absent".
- `missing_keys` **understated the damage**: `_NormBase._load_from_state_dict`
  deliberately tolerates an absent `num_batches_tracked` for backward compatibility, so
  18 tensors failed to load and the loader admitted to 16.

Fixes: `src/opendrivefm/validation/ckpt_compat.py` remaps legacy names, refusing any
rename whose shapes disagree; missing trust weights are now a **hard `SystemExit`**, not
a warning, because a warning that says "results below measure nothing" and then prints
the results is a warning nobody acts on; `scripts/eval/inspect_ckpt_trust_keys.py`
prints missing and unexpected keys side by side, where equal counts are the signature of
a rename. `tests/test_ckpt_compat.py` pins all of it (10 tests), asserting on the
**weights** rather than on the loader's report, since the report is what failed.

A related bug in the same load path dropped trust tensors whenever `--trust_grid > 1`,
keyed on the **flag** rather than on the checkpoint. Correct when promoting a grid=1
checkpoint; catastrophic when the checkpoint was already grid=4, where it discarded the
weights that had just been trained and evaluated a random-init scorer at AUROC 0.632.
Now the decision is made by comparing shapes, which cannot encode a stale intent.

**2. A suspected leak, refuted; a real defect found underneath it.**

`CameraTrustScorer._image_stats` ended with `sigmoid(stats - stats.detach().mean(dim=0))`,
and the scorer receives every camera of every frame in one call
(`rearrange(x[:, :, -1], "b v c h w -> (b v) c h w")`). A camera's trust was therefore a
function of its pixels **relative to the other B*V-1 images sharing the forward pass**.

The hypothesis was that this leaks between `eval_ood_detection.py`'s clean pass and its
faulted pass. `scripts/eval/check_batch_contamination.py` measures that directly, by
watching cameras whose pixels are **bit-identical** between the two passes.

**The hypothesis is refuted: contamination ratio 0.000 at every batch size.** A shared
additive centring largely cancels when clean is subtracted from faulted. The clean-vs-
faulted comparison, and every AUROC in this README, was never affected.

**The real defect is one the difference metric could not see.** Score the *same* clean
frames at different batch sizes and compare them directly:

| batch size | max abs delta trust vs batch_size 1 | | |
|---|---|---|---|
| | **uncalibrated** | **calibrated** | |
| 2 | 7.123e-05 | 5.960e-08 | |
| 4 | 1.004e-04 | 1.192e-07 | |
| 8 | **1.206e-04** | **1.192e-07** | float32 noise |

One frame did not get one score. The drift is small, about 0.75% of the 0.016 trust change
a real fault produces, but it is **systematic in batch shape**, and that matters at
deployment: the C++ runtime scores one frame's 6 cameras per call while this eval used 12
images per call. Under batch-relative centring those are **different functions**, so a
trust threshold chosen offline does not mean the same thing on the vehicle and
Python/C++ parity on trust is not well defined.

**The fix** (`scripts/calibrate_trust_stats.py`) is the one BatchNorm already uses: estimate
the centre once over the training split, store it in a buffer, freeze it at eval. Trust
becomes a pure function of one frame (1.2e-07, float32 noise). Ranking is unchanged by
construction and in measurement: **AUROC 0.764 -> 0.763**, identical CI. Uncalibrated
checkpoints fall back to the old path bit-identically, so published numbers stay
reproducible, and `--require_calibrated_trust` turns the fallback into an error for anyone
reporting absolute scores. Pinned by `tests/test_trust_batch_invariance.py` (8 tests).

**A retraction, and the reason it is printed here.** An earlier version of this section
reported that the faulted camera's response fell **36% across batch sizes**. That number
was an artifact of the diagnostic itself: the perturbations draw their patch size and
position from the global RNG, and the seed was set once before the sweep, so each batch
size was scored against a *different set of random faults*. Fault variance was being read
as a batch-size effect. The tell was that the 36% did not move when the underlying cause
was fixed, and a quantity that does not respond to a fix was never measuring it. The sweep
now re-seeds before every pass, both checkpoints then report 0%, and the genuine defect
only became visible once the confound was removed.

Artifacts: `batch_contamination_grid1.json`, `batch_contamination_calibrated.json`,
`ood_detection_report_grid1_calibrated.json`.

### Trust fusion does not convert detection into perception (`scripts/eval/eval_trust_temperature.py`)

`TrustWeightedFusion` computes `softmax(trust, dim=1)`, whose implicit temperature of 1.0
leaves weights near uniform. Sweeping `softmax(trust / T)` over 3 seeds:

| T | weight on faulted cam | blur ΔIoU | sd | relative |
|---|---|---|---|---|
| 1.00 | 0.147 | -0.0015 | 0.0003 | -2.2% |
| 0.25 | 0.104 | -0.0064 | 0.0010 | -9.2% |
| 0.10 | 0.071 | -0.0099 | 0.0010 | -14.2% |
| **0.05** | 0.057 | **-0.0112** | 0.0010 | **-16.1%** |
| 0.02 | 0.042 | -0.0109 | 0.0010 | -15.5% |

At T=0.02 the faulted camera's weight reaches 0.0000 for rain: complete dropout. IoU still
does not improve. **The softmax was never the bottleneck.** Mean ΔIoU is negative at every
temperature.

**A degraded camera is still a useful camera.** Suppressing a blurred CAM_FRONT costs up to
**16.1% ± 1.0pp IoU**, monotone across six temperatures at ~11σ. Graceful degradation is not
sensor dropout. Artifact: `trust_temperature_sweep_3seed.json`.

### Uncertainty: split-conformal trajectory intervals (`scripts/eval/eval_conformal_trajectory.py`)

Distribution-free intervals with the finite-sample corrected quantile `ceil((n+1)(1-α))/n`.
At α=0.1 with n≈41 that is the 93rd percentile, not the 90th; skipping the correction
under-covers.

- Global 90% radius: **5.155 m**
- Per horizon: **0.362 m at t=1 → 10.049 m at t=12** (27.7x growth)
- Coverage holds at nominal under every corruption tested, once binomial sampling error
  (~1.3pp on ~500 points) is accounted for.

Coverage is robust **because the trajectory head barely responds to the cameras**, ADE
moves under 0.15 m under any fault. The predictions are carried by the constant-velocity
prior. Artifact: `conformal_trajectory.json`.

### Coverage-guided scenario fuzzing (`scripts/eval/fuzz_scenarios.py`)

Searches fault type × severity × camera subset × simultaneity. 200 trials each, equal budget.

| Strategy | Cells covered | Worst ΔIoU found |
|---|---|---|
| random | 109/150 (72.7%) | +0.0191 |
| coverage-guided | 150/150 (100%) | +0.0190 |
| **adaptive** | **150/150 (100%)** | **+0.0241** |

Adaptive found a failure **27% worse** than either baseline while still achieving full
coverage. Worst scenario: **occlusion at severity 0.95 on FRONT_LEFT, FRONT_RIGHT and
BACK_LEFT, IoU 0.0436 → 0.0195, a 55% relative collapse**, ADE +0.471 m.

Nine of the ten worst scenarios involve a left-side camera. Both held-out scenes are
Singapore (left-hand traffic), so left cameras carry disproportionate scene content. No
hand-written test would have found this: every pre-existing test perturbs CAM_FRONT only.
Artifacts: `fuzz_report{,_random,_adaptive}.json`.

### Test-suite prioritisation (`scripts/eval/prioritize_tests.py`)

Frames ranked by worst-case degradation across the battery.

| Smoke set | Frames | Share of total degradation captured |
|---|---|---|
| top 10% | 8 | 26.5% |
| top 20% | 16 | 44.0% |
| **top 30%** | **25** | **60.1%** |
| top 50% | 41 | 81.1% |

**Four of five corruptions raise mean IoU.** At IoU ≈ 0.04 with a fixed 0.6 threshold, the
metric is dominated by threshold effects rather than scene understanding, so degrading the
input can nudge predictions toward sparse ground truth. **IoU is not currently a trustworthy
release metric for this model.** Artifact: `test_prioritisation.json`.

### Scene-level OOD, and a 12.7x evaluation-bias result (`scripts/eval/eval_scene_ood.py`)

Mahalanobis distance on backbone features, shrinkage-regularised covariance.

| Evaluation | AUROC | 95% CI |
|---|---|---|
| In-sample (fit and score the same frames) | **0.951** | [0.926, 0.972] |
| **Out-of-sample (fit on 6 scenes, reference = 2 held-out *training* scenes)** | **0.075** | [0.036, 0.121] |

The entire in-sample result was memorisation. Scored honestly the detector inverts: held-out
**training** scenes sit at mean distance 28.2 while the val scenes sit at 14.1.

**Conclusion: scene-level Mahalanobis OOD is not viable at nuScenes-mini scale.** With 10
scenes, scene-to-scene variance exceeds the train/val gap, so a single Gaussian fits a
handful of separate clusters and any unseen scene lands far outside regardless of its split
label. The script emits this warning automatically. Artifacts: `scene_ood_report.json`,
`scene_ood_report_insample.json`.

### VLA: vision-conditioned trajectory decoding (`scripts/train/train_vla_projector.py`)

Frozen BEV backbone → 1.77M-parameter projector (1.73% of 102.6M total) → k=4 prefix
embeddings → frozen GPT-2 decoding residual waypoint tokens. Residual tokenisation against
the constant-velocity prior; structurally constrained decoding (even steps emit x tokens,
odd steps y tokens).

| Predictor | ADE (held-out scenes) |
|---|---|
| MLP residual head | **2.492 m** |
| Constant velocity | 3.012 m |
| VLA (vision-conditioned) | 3.993 m |

**The VLA is worse than assuming constant velocity**, and the vision pathway carries no
scene information:

| Control | ΔADE | Interpretation |
|---|---|---|
| prefix zeroed | +16.065 m | Out of distribution; confounded |
| **prefix shuffled across batch** | **-0.089 m** | Decisive: no scene-specific signal |
| **cameras corrupted** | **+0.000 m** | Decisive: no scene-specific signal |

Zeroing the prefix breaks the model, but a zero prefix is an input the LM never saw in
training, so it proves only that the prefix is load-bearing. Shuffling keeps prefixes
in-distribution while destroying frame correspondence, and ADE does not move. Dataset scale
(322 training frames, 8 scenes) is the binding constraint, not architecture.

**Stratifying by difficulty changes the engineering conclusion.** Terciles by
constant-velocity error, on the same held-out frames:

| Stratum | n | CV | MLP | VLA | MLP vs CV |
|---|---|---|---|---|---|
| easy (CV accurate) | 27 | **0.350** | 1.273 | 1.357 | +264% |
| medium | 27 | 1.573 | **1.493** | 2.346 | -5% |
| hard (CV fails) | 28 | 6.966 | **4.637** | 8.244 | **-33%** |

The pooled ADE says the MLP beats constant velocity by 17%. The strata say something more
useful: the MLP is **33% better exactly where the prior fails**, and **264% worse where the
prior is already right**. A learned residual is worth having only on hard frames, so the
deployable design is not "MLP instead of CV" but **CV gated to a learned residual by
predicted difficulty**. A single pooled number cannot express that, and would have shipped
the worse system.

The VLA loses on all three strata, so the negative result stands.
Artifact: `vla_report.json`.

### Deployment: C++ runtime and Python parity (`cpp/`)

`odfm_parity_check` loads the eager-mode Python outputs and asserts the TorchScript graph
reproduces them in C++.

| Output | max abs diff | Result |
|---|---|---|
| occupancy | 6.676e-06 | PASS |
| trajectory | 3.576e-07 | PASS |
| trust | 0.000e+00 | PASS |
| determinism (repeated forward) | bit-identical | PASS |

The C++ delta on occupancy **equals the eager-vs-traced delta exactly**, so the C++ runtime
is bit-faithful to the traced graph and the entire divergence is Python-side tracing.

**Latency, CPU, Apple Silicon:**

| threads | p50 | p99 | p99.9 | FPS |
|---|---|---|---|---|
| 1 | 26.411 ms | 28.032 | 28.613 | 37.7 |
| **4** | **13.879 ms** | 16.023 | 18.305 | **71.2** |
| 8 | 14.011 ms | 16.969 | 20.441 | 70.3 |

Scaling saturates at 4 threads; 8 is worse on every tail statistic (oversubscription onto
efficiency cores). Two `threads=1` runs disagreed by **53% on p99.9** while p50 differed by
1.5%, so tail latency requires repeated runs and is reported, never gated.

### End-to-end staleness, not model latency (`cpp/src/odfm_runner.cpp`)

Two-thread pipeline: lock-free SPSC ring or seqlock latest-frame buffer, at a fixed sensor
rate. At **100 Hz against ~14 ms inference**:

| | queue (FIFO) | latest (seqlock) |
|---|---|---|
| inference p50 | 13.90 ms | 13.92 ms |
| **end-to-end p50** | **219.00 ms** | **18.98 ms** |
| frames processed | 716 (71.6%) | 704 (70.4%) |

**Same compute, same throughput, same model latency, 11.5x difference in staleness.** At
30 mph, 200 ms of extra staleness is 2.7 m of position error on every tracked obstacle.

Queue staleness equals depth × service time: the ring rounds a requested depth of 8 up to
15 usable slots, 15 × 13.90 ms = 208.5 ms predicted against 205 ms measured.

**How the consumer waits changes the tail more than the model does.** Same pipeline, same
model, three wait strategies at 30 Hz:

| Wait strategy | inference p50 | max | sd | idle waits |
|---|---|---|---|---|
| `yield()` spin | 15.20 ms | 21.91 ms | 1.45 ms | 21,442 |
| `sleep_for(200us)` | 14.32 ms | 49.57 ms | 2.59 ms | n/a |
| **condition variable** | **14.10 ms** | **16.76 ms** | **0.45 ms** | **2,487** |

Spinning steals a core from the inference thread and inflates p50 by 1.1 ms. Sleeping
fixes p50 but quantises wakeups, blowing the maximum to 49.57 ms, a **34 ms** tail on a
33 ms budget, i.e. a dropped frame. The condition variable wins on all four measures at
once, and cuts end-to-end max at 100 Hz from 74.28 ms to **27.25 ms**.

The producer publishes with a release-store and notifies **without holding the mutex**, so
the notify never blocks the sensor thread; the consumer re-checks a sequence counter under
the lock, which closes the lost-wakeup race a bare `wait()` would leave open. Verified
under ThreadSanitizer (`-DODFM_TSAN=ON`).

---

## Reproduce

```bash
CKPT=outputs/artifacts/checkpoints_v11_trustfix2/trust_fixed_v2.ckpt

# detection, before and after
python scripts/eval/eval_ood_detection.py --ckpt outputs/artifacts/checkpoints_v11_temporal/best_val_ade.ckpt --cams all
python scripts/eval/eval_ood_detection.py --ckpt $CKPT --cams all

# the fix itself (trains 0.36% of parameters)
python scripts/train/finetune_trust_head.py --ckpt outputs/artifacts/checkpoints_v11_temporal/best_val_ade.ckpt \
  --out outputs/artifacts/checkpoints_v11_trustfix2/trust_fixed_v2.ckpt

# spatial pooling: the occlusion fix
python scripts/train/finetune_trust_head.py --ckpt $CKPT --trust_grid 4 \
  --fault_sampling stratified --worst_case_w 0.5 \
  --out outputs/artifacts/checkpoints_v11_grid4/trust_grid4.ckpt
python scripts/eval/eval_ood_detection.py --ckpt outputs/artifacts/checkpoints_v11_grid4/trust_grid4.ckpt \
  --trust_grid 4 --cams all --faults all --out outputs/artifacts/ood_detection_report_grid4.json

# measurement-integrity checks (see "Measurement integrity" above)
python -m pytest tests/test_ckpt_compat.py tests/test_trust_batch_invariance.py -q
python scripts/eval/inspect_ckpt_trust_keys.py $CKPT --trust_grid 1
python scripts/eval/check_batch_contamination.py --ckpt $CKPT --fault occlusion \
  --batch_sizes 1,2,4,8

# make trust a pure function of one frame (required for absolute thresholds)
python scripts/calibrate_trust_stats.py --ckpt $CKPT \
  --out outputs/artifacts/checkpoints_v11_trustfix2/trust_fixed_v2_cal.ckpt
python scripts/eval/eval_ood_detection.py \
  --ckpt outputs/artifacts/checkpoints_v11_trustfix2/trust_fixed_v2_cal.ckpt \
  --cams all --faults all --require_calibrated_trust

python scripts/eval/eval_trust_temperature.py   --ckpt $CKPT --seeds 0,1,2
python scripts/eval/eval_conformal_trajectory.py --ckpt $CKPT --alpha 0.1
python scripts/eval/fuzz_scenarios.py           --ckpt $CKPT --trials 200 --strategy adaptive
python scripts/eval/prioritize_tests.py         --ckpt $CKPT
python scripts/eval/eval_scene_ood.py           --ckpt $CKPT
python scripts/train/train_vla_projector.py --train --ckpt $CKPT
python scripts/train/train_vla_projector.py --eval  --ckpt $CKPT --projector outputs/artifacts/vla_projector.pt

# deployment
python scripts/export_torchscript.py --ckpt $CKPT
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
cmake --build cpp/build -j && ctest --test-dir cpp/build --output-on-failure
./cpp/build/odfm_parity_check --model outputs/artifacts/opendrivefm_v11.pt --reference outputs/artifacts/parity_reference.pt
./cpp/build/odfm_runner --fps 100 --seconds 10 --policy latest --threads 4

# release gates
python scripts/ci/check_gates.py --detection outputs/artifacts/ood_detection_report_v11_trustfix2.json
```

CI (`.github/workflows/validation.yml`) runs the C++ tests, a ThreadSanitizer build of the
concurrency tests, the metric unit tests, and the release gates on committed artifacts. The
gate job also asserts the gates still **reject** the known-inverted baseline: a gate that
cannot fail is decoration.

---

## Known limitations

- **IoU is unreliable as a release metric here.** Absolute IoU is ~0.077 and four of five
  corruptions raise it. Threshold effects dominate.
- **Occlusion detection is at chance at `grid=1`** (0.487) because the trust scorer pools
  globally. `--trust_grid 4` raises it to 0.689 at the cost of 0.02-0.05 AUROC on the two
  global faults; neither setting clears the 0.80 pooled gate.
- **Uncalibrated checkpoints are not frame-pure.** Run
  `scripts/calibrate_trust_stats.py` before using any absolute trust score or threshold.
  Without it the same frame scores differently by up to 1.2e-04 depending on batch shape.
  AUROC is unaffected.
- **`num_batches_tracked` is exempt from the missing-weight guard.** PyTorch does not
  report it in `missing_keys`, so a checkpoint missing only those two tensors would load
  silently. Harmless in eval mode, where `running_mean`/`running_var` govern, but it is a
  known hole rather than an unknown one.
- **Scene-level OOD needs far more scenes.** 10 is not enough for a manifold.
- **The trajectory head largely ignores the cameras.** ADE moves under 0.15 m under any
  corruption; the constant-velocity prior carries the prediction.
- **The VLA does not use scene content** and underperforms constant velocity.
- **Three of four saved checkpoints are architecturally incompatible** with the current
  `model.py`. Only `checkpoints_v11_temporal` loads 169/169. Run any script with
  `--dry_run` to check before trusting a number.
- **Tail latency is not reproducible run-to-run** on a laptop under background load.

## Removed claims

| Former claim | What measurement replaced it |
|---|---|
| "Detection rate: 100% across all 5 fault types" | AUROC 0.434 (inverted) before fix, 0.764 after, 0.797 with `--trust_grid 4`; occlusion at chance until spatial pooling |
| "+26.6% IoU under sensor faults" | Trust fusion cannot improve IoU at any temperature, even at full camera dropout, over 3 seeds |
| "IoU = 0.136, ADE = 2.457 m" | Measured IoU 0.0767, ADE 2.763 m on `nuscenes_labels_128` |
| "317 FPS, p50 3.15 ms" | 71.2 FPS, p50 13.879 ms in C++ at 4 CPU threads (the earlier figure was Python/MPS) |
| "GPT-2 fine-tuned on nuScenes expert trajectories" | The manifest has no `ego_future`/`ego_pose`, so all 404 rows produced one all-zero trajectory. Real trajectories live in the label `.npz` files and are what the VLA now uses |

## License

MIT.
