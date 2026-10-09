"""Materialize the exact presynaptic cell arrays as memory-mapped .npy files.

The training sampler and window index are unchanged. Only repeated decompression
of cell, geometry, and embedding NPZ files is replaced by shared page-cache IO.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset_presynaptic import load_cell_arrays

DATABASE = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_k17_full_tc_source/scale16/k17')
OUTPUT = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_k17_memmap/scale16/k17')


def build_one(root_id: int, database: Path, output: Path) -> None:
    destination = output / str(root_id)
    marker = destination / 'complete.json'
    if marker.is_file():
        return
    if destination.exists():
        raise ValueError(f'Incomplete cache directory: {destination}')
    temporary = output / f'.{root_id}.{os.getpid()}.partial'
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        arrays = load_cell_arrays(database / 'cells' / f'{root_id}.npz', 'new')
        for name, array in arrays.items():
            np.save(temporary / f'{name}.npy', array, allow_pickle=False)
        (temporary / 'complete.json').write_text(json.dumps({
            'root_id': root_id, 'variant': 'new', 'arrays': sorted(arrays),
        }) + '\n')
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=DATABASE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error('shard must be in [0, shards)')
    roots = sorted(int(root_id) for root_id in json.loads((args.database / 'manifest.json').read_text())['cells'])
    args.output.mkdir(parents=True, exist_ok=True)
    for index, root_id in enumerate(roots):
        if index % args.shards == args.shard:
            build_one(root_id, args.database, args.output)
            if index % 50 == 0:
                print(f'shard {args.shard}: cached {root_id}', flush=True)
    print(f'shard {args.shard}: complete', flush=True)


if __name__ == '__main__':
    main()
