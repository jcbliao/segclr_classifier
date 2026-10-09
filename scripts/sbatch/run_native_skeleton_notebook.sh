#!/bin/bash
#SBATCH --job-name=native_skeleton_stats
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/native_skeleton_stats_%j.out
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export PATH="$PWD/segclr_db/.venv/bin:$PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
path = Path('analysis/presynaptic/new_skeletons_native/skeleton_stats.ipynb')
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=3600, kernel_name='python3',
                        resources={'metadata': {'path': str(Path.cwd())}})
try:
    client.execute()
finally:
    temporary = path.with_suffix('.tmp.ipynb')
    nbformat.write(nb, temporary)
    temporary.replace(path)
print(f'Executed {path}', flush=True)
PY
