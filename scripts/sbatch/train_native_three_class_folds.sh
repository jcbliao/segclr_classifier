#!/bin/bash
#SBATCH --job-name=native_three_class_folds
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-14
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task=${SLURM_ARRAY_TASK_ID:?}
if ((task < 0 || task > 14)); then
  echo "Expected array task 0-14" >&2
  exit 1
fi
fold=$((task / 3))
configuration=$((task % 3))
single_args=()
if ((configuration == 0)); then
  experiment=native_single_pre_post
  k=1
  architecture=pointwise_mlp
  single_args=(--single-presynaptic-embedding)
else
  experiment=native_skeletons_pre_post
  k=17
  architecture=pointwise_mlp
  if ((configuration == 2)); then architecture=graph_transformer; fi
fi
base=/orcd/scratch/orcd/013/jcbliao/presynaptic_axons
exec segclr_db/.venv/bin/python -u scripts/train_gnn.py \
  --dataset presynaptic_dense \
  --presynaptic-database "$base/casey_coarse_confidence_with_tc_folds/scale16/k17/conf0.7/fold$fold" \
  --manifest "analysis/presynaptic/$experiment/three_class/manifests/fold$fold.json" \
  --postsynaptic-cache "$base/native_skeletons_pre_post/scale16/k17/conf0.7" \
  --use-postsynaptic "${single_args[@]}" \
  --results-dir "results/presynaptic/$experiment/scale16/k$k/conf0.7/three_class/fold$fold/pre_post" \
  --architecture "$architecture" --num-embeddings "$k" \
  --epochs 30 --resume --amp --class-balance sample \
  --batch-size 4096 --attention-budget 14450688 \
  --mixed-cell-batch-size 16 --presynaptic-cell-cache 16 --num-workers 8 \
  --presynaptic-memmap-root "$base/casey_k17_memmap/scale16/k17" \
  --gt-depth 4 --gt-heads 4 --lr 1e-4 --weight-decay 1e-5
