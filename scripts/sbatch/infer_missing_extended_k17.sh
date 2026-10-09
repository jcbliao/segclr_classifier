#!/bin/bash
#SBATCH --job-name=extended_teasar_infer
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH=/orcd/home/002/jcbliao/rotation/segclr/segclr_db_named_v5/_pkgroot
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/jcbliao/.conda/envs/segclr/bin/python -u - <<'PY'
import json
import os
from pathlib import Path
import subprocess

base = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_k17_full_tc_source/scale16/k17')
roots = [int(x) for x in (base / 'missing_inference_roots.txt').read_text().split()]
root_id = roots[int(os.environ['SLURM_ARRAY_TASK_ID'])]
routes = {r['root_id']: r for r in json.loads(Path(
    '/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_registered_teasar_111nm_20260911/routes.json'
).read_text())}
route = routes[root_id]
status = base / 'status'
status.mkdir(parents=True, exist_ok=True)
one = status / f'infer_{root_id}.txt'
one.write_text(f'{root_id}\n')
env = dict(os.environ, INFERENCE_BASE=route['base'], SKELETON_NAME=route['skeleton_name'],
           INFERENCE_ROOTS=str(one), INFERENCE_STATUS_PREFIX=f'extended_{root_id}')
subprocess.run(['/home/jcbliao/.conda/envs/segclr/bin/python', '-u',
                'scripts/infer_named_teasar.py'], env=env, check=True)
PY
