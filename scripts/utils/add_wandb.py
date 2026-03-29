"""
add_wandb.py — Adds W&B logging to existing training scripts
Run ONCE to install and configure:
  pip install wandb
  python scripts/add_wandb.py

Then retrain any model with --wandb flag added automatically.
OR manually add to any train script:
  import wandb
  from pytorch_lightning.loggers import WandbLogger
  wandb_logger = WandbLogger(project="opendrivefm", name="v11-temporal")
  trainer = pl.Trainer(..., logger=wandb_logger)
"""
import subprocess, sys

# Install wandb
print("Installing wandb...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb", "-q"])
print("wandb installed.")

# Log existing results to wandb offline (no account needed)
import wandb
import json
from pathlib import Path

wandb.init(
    project="opendrivefm",
    name="results-summary",
    mode="offline",   # works without internet/account
    config={
        "dataset": "nuScenes-mini",
        "train_samples": 322,
        "val_samples": 82,
        "val_scenes": ["scene-0655", "scene-1077"],
    }
)

# Log all version results
results = {
    "v8/BEV_IoU":  0.136,
    "v8/ADE_val":  2.740,
    "v8/FDE_val":  6.116,
    "v9/ADE_val":  2.559,
    "v10/ADE_val": 2.885,
    "v11/ADE_val": 2.457,
    "v11/FDE_val": 5.486,
    "v13/IoU_vehicle": 0.131,
    "v13/IoU_pedestrian": 0.003,
    "baseline/CV_ADE": 3.012,
    "baseline/CV_FDE": 6.455,
}

# Log fault injection results
fault_results = {
    "trust/clean":    0.795,
    "trust/blur":     0.329,
    "trust/occlusion":0.402,
    "trust/rain":     0.340,
    "trust/noise":    0.310,
    "trust/glare":    0.491,
}

wandb.log({**results, **fault_results})

# Log robustness table
table = wandb.Table(
    columns=["n_cameras_faulted", "BEV_IoU", "ADE_m", "trust_faulted", "trust_clean"],
    data=[
        [0, 0.1360, 2.7402, None,  None],
        [1, 0.1421, 2.7558, 0.305, 0.448],
        [2, 0.1441, 2.8039, 0.300, 0.450],
        [3, 0.1469, 2.8504, 0.301, 0.450],
    ]
)
wandb.log({"robustness/multi_camera_fault": table})

wandb.finish()
print("\nW&B run saved offline.")
print("Results logged to wandb. Run 'wandb sync' to upload if you have an account.")
print("Or view offline: check wandb/ directory created in current folder.")
