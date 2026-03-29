#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-2}"
BATCH="${2:-4}"

python \
  -W "ignore:.*does not have many workers.*:UserWarning" \
  -W "ignore:.*Consider setting `persistent_workers=True`.*:UserWarning" \
  -W "ignore:.*Checkpoint directory .* exists and is not empty.*:UserWarning" \
  -W "ignore:.*LeafSpec.*deprecated.*:DeprecationWarning" \
  scripts/train_smoke.py --clean_ckpt --epochs "$EPOCHS" --batch "$BATCH"

python scripts/eval_ckpt.py --ckpt artifacts/checkpoints/last.ckpt

echo
echo "Artifacts:"
ls -lh artifacts/checkpoints
echo
echo "Metrics:"
cat artifacts/metrics.json
