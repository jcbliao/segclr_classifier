#!/bin/bash
#SBATCH --job-name=cave_compile_cache
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/orcd/scratch/orcd/013/jcbliao/torchinductor_cache/segclr_static_b128
py=/home/jcbliao/.conda/envs/segclr/bin/python
plan=$PWD/analysis/presynaptic/cave_aug_benchmark_plan.json
base=/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/cave_aug_compile_cache_${SLURM_JOB_ID}
rm -rf "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
/usr/bin/time -f 'cold_compile_seconds=%e peak_kb=%M' "$py" -u scripts/generate_all_cave_augmentations.py --plan "$plan" --output "$base/cold" --batch-size 32 --variants-per-forward 4 --compile
/usr/bin/time -f 'warm_cache_seconds=%e peak_kb=%M' "$py" -u scripts/generate_all_cave_augmentations.py --plan "$plan" --output "$base/warm" --batch-size 32 --variants-per-forward 4 --compile
