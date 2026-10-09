#!/bin/bash
#SBATCH --job-name=cave_six_class_folds
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-4%4
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
fold=${SLURM_ARRAY_TASK_ID:?}
base=/orcd/scratch/orcd/013/jcbliao/presynaptic_axons
database="$base/casey_confidence_cave/k10/conf0.7/fold$fold"
exec segclr_db/.venv/bin/python -u scripts/train_gnn.py \
  --dataset presynaptic_cave --presynaptic-database "$database" \
  --manifest "$database/manifest.json" \
  --postsynaptic-cache "$base/cave_skeletons_pre_post/k10/conf0.7/postsynaptic" \
  --use-postsynaptic \
  --results-dir "results/presynaptic/cave_skeletons_pre_post/k10/conf0.7/six_class/fold$fold/pre_post" \
  --architecture pointwise_mlp --num-embeddings 10 \
  --epochs 30 --resume --amp --class-balance sample \
  --batch-size 4096 --attention-budget 14450688 \
  --mixed-cell-batch-size 16 --presynaptic-cell-cache 16 --num-workers 8 \
  --gt-depth 4 --gt-heads 4 --lr 1e-4 --weight-decay 1e-5
