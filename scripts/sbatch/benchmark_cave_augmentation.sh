#!/bin/bash
#SBATCH --job-name=cave_aug_bench
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
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
py=/home/jcbliao/.conda/envs/segclr/bin/python
plan=$PWD/analysis/presynaptic/cave_aug_benchmark_plan.json
base=/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/cave_aug_benchmark_${SLURM_JOB_ID}
/usr/bin/time -f 'baseline_seconds=%e peak_kb=%M' "$py" -u scripts/generate_all_cave_augmentations.py --plan "$plan" --output "$base/baseline" --batch-size 32 --variants-per-forward 1
/usr/bin/time -f 'group4_seconds=%e peak_kb=%M' "$py" -u scripts/generate_all_cave_augmentations.py --plan "$plan" --output "$base/group4" --batch-size 32 --variants-per-forward 4
/usr/bin/time -f 'compile_group4_seconds=%e peak_kb=%M' "$py" -u scripts/generate_all_cave_augmentations.py --plan "$plan" --output "$base/compile_group4" --batch-size 32 --variants-per-forward 4 --compile
