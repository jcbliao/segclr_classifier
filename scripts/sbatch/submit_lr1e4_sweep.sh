#!/bin/bash
# Submit the requested clean 16-epoch, lr=1e-4 comparison. The flat headline
# run is submitted first; the 42-run LCPN matrix follows (the original 39
# configurations plus raw-embedding Pointwise MLP at n=10/20/40).
set -euo pipefail

REPO=/home/jcbliao/rotation/segclr/gnn_classifier
cd "$REPO"

flat=$(NUM_EMBEDDINGS=40 ARCHITECTURE=mpnn MPNN_LAYERS=2 \
  BATCH_SIZE=1024 NUM_WORKERS=15 LR=1e-4 EPOCHS=16 \
  EXTRA_ARGS="--position --lpe --classifier flat --class-balance sample" \
  sbatch --parsable --job-name=flat_mpnn_n40_lr1e4 \
    --partition=mit_normal_gpu,mit_preemptable --account=mit_general --qos=normal \
    --gres=gpu:1 --cpus-per-task=16 scripts/sbatch/train_gnn.sh)
echo "flat headline: ${flat%%;*}"

echo "LCPN matrix:"
LR=1e-4 WEIGHT_DECAY=1e-5 EPOCHS=16 \
  scripts/sbatch/submit_embedding_sweep.sh
