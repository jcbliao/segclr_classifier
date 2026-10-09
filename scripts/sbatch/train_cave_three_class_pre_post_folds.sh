#!/bin/bash
#SBATCH --job-name=cave_three_class_folds
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-9
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task=${SLURM_ARRAY_TASK_ID:?}
if ((task < 0 || task > 9)); then exit 1; fi
fold=$((task / 2))
architecture=pointwise_mlp
if ((task % 2 == 1)); then architecture=graph_transformer; fi
base=/orcd/scratch/orcd/013/jcbliao/presynaptic_axons
exec segclr_db/.venv/bin/python -u scripts/train_gnn.py \
  --dataset presynaptic_cave \
  --presynaptic-database "$base/casey_confidence_cave/k10/conf0.7/fold$fold" \
  --manifest "analysis/presynaptic/cave_skeletons_pre_post/three_class/manifests/fold$fold.json" \
  --postsynaptic-cache "$base/cave_skeletons_pre_post/k10/conf0.7/postsynaptic" \
  --use-postsynaptic \
  --results-dir "results/presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/fold$fold/pre_post" \
  --architecture "$architecture" --num-embeddings 10 \
  --epochs 30 --resume --amp --class-balance sample \
  --batch-size 4096 --attention-budget 14450688 \
  --mixed-cell-batch-size 16 --presynaptic-cell-cache 16 --num-workers 8 \
  --gt-depth 4 --gt-heads 4 --lr 1e-4 --weight-decay 1e-5
