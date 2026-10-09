#!/bin/bash
#SBATCH --job-name=cave_compile_pair
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/orcd/scratch/orcd/013/jcbliao/torchinductor_cache/segclr_static_b128
exec /home/jcbliao/.conda/envs/segclr/bin/python -u scripts/generate_all_cave_augmentations.py \
 --plan analysis/presynaptic/cave_aug_pilot_plan.json \
 --output /orcd/scratch/orcd/013/jcbliao/presynaptic_axons/cave_aug_compile_paired_${SLURM_JOB_ID} \
 --batch-size 32 --variants-per-forward 4 --compare-compile
