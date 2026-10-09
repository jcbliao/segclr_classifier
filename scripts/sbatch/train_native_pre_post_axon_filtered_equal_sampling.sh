#!/bin/bash
#SBATCH --job-name=native_axon_equal
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
architectures=(mean pointwise_mlp graph_transformer)
task=${SLURM_ARRAY_TASK_ID:?}
architecture=${architectures[$((task % 3))]}
post_args=()
condition=pre_only
if ((task >= 6)); then
  post_args=(--append-presynaptic-mean)
  condition=pre_mean_control
elif ((task >= 3)); then
  post_args=(--use-postsynaptic)
  condition=pre_post
fi
base=/orcd/scratch/orcd/013/jcbliao/presynaptic_axons
exec segclr_db/.venv/bin/python -u scripts/train_gnn.py \
  --dataset presynaptic_dense \
  --presynaptic-database "$base/casey_coarse_confidence_with_tc_folds/scale16/k17/conf0.7/fold0" \
  --postsynaptic-cache "$base/native_skeletons_pre_post/scale16/k17/conf0.7" \
  "${post_args[@]}" \
  --presynaptic-compartment-filter "$base/compartment_filtered/scale16/k17/conf0.7/fold0/geodesic_exclusion20um" \
  --results-dir "results/presynaptic/native_pre_post_axon_filtered_equal_sampling/scale16/k17/conf0.7/fold0/$condition" \
  --architecture "$architecture" --num-embeddings 17 \
  --epochs 30 --resume --amp --class-balance equal \
  --batch-size 4096 --attention-budget 14450688 \
  --mixed-cell-batch-size 16 --presynaptic-cell-cache 16 --num-workers 8 \
  --presynaptic-memmap-root "$base/casey_k17_memmap/scale16/k17" \
  --gt-depth 4 --gt-heads 4 --lr 1e-4 --weight-decay 1e-5
