#!/bin/bash
# Submit via scripts/submit_registered_presynaptic_training.sh to balance
# mean/MLP on mit_preemptable and GraphTransformer on mit_normal_gpu.
#SBATCH --job-name=presyn_native_train
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
case "${SLURM_ARRAY_TASK_ID:?}" in
  0) architecture=mean ;;
  # Native windows have an embedding at every node, like the CAVE windows.
  # graph_transformer_teasar is for mixed embedded/unembedded routing nodes.
  1) architecture=graph_transformer ;;
  2) architecture=pointwise_mlp ;;
  *) exit 2 ;;
esac
factor=${FACTOR:-32}
case "$factor" in
  32) k=9 ;; 16) k=17 ;; 8) k=33 ;; 4) k=65 ;; 2) k=129 ;; 1) k=257 ;; *) exit 2 ;;
esac
exec segclr_db/.venv/bin/python -u scripts/train_gnn.py \
  --dataset presynaptic_dense \
  --presynaptic-database "/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/resampled_teasar_2209/training_local_center/scale${factor}/k${k}" \
  --results-dir "results/presynaptic/new_skeletons_native/local_center/scale${factor}/k${k}" \
  --architecture "$architecture" --num-embeddings "$k" \
  --epochs 30 --resume --amp --class-balance sample \
  --batch-size 4096 --attention-budget 14450688 \
  --presynaptic-cell-cache 1 --num-workers 4 \
  --gt-depth 4 --gt-heads 4 --lr 1e-4 --weight-decay 1e-5
