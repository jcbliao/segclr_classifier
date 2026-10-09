"""Validate the complete mesh/skeleton source and write its Globus viewer link."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import subprocess
import os
from urllib.parse import quote

import numpy as np

sys.path.insert(0, '/orcd/home/002/jcbliao/rotation/skeletonization')
from skeletonization.precomputed import write_skeleton, viewer_state
from skeletonization.verify import verify_mesh, verify_skeleton


def fingerprint(out, root):
    return {name: [p.stat().st_size, p.stat().st_mtime_ns] for name, p in (
        ('mesh', out / 'mesh' / root), ('index', out / 'mesh' / f'{root}.index'),
        ('skeleton', out / 'skeletons' / root))}


def validate_cell(out, root, skeletons):
    rid = int(root)
    if not (out / 'skeletons' / root).exists():
        with np.load(skeletons / f'{root}.npz') as data, tempfile.TemporaryDirectory(dir=out) as tmp:
            write_skeleton(tmp, rid, data['vertices'], data['edges'], radius=data['radius'])
            (Path(tmp) / root).replace(out / 'skeletons' / root)
    marker = out / 'validated' / f'{root}.json'
    before = fingerprint(out, root)
    if marker.exists() and json.loads(marker.read_text()) == before:
        return
    verify_mesh(out / 'mesh', rid)
    verify_skeleton(out / 'skeletons', rid)
    if fingerprint(out, root) != before:
        raise RuntimeError(f'Files changed during validation: {root}')
    temporary = marker.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(before))
    temporary.replace(marker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skeletons', type=Path, default=Path('/orcd/scratch/orcd/013/jcbliao/skeletons/segclr/skeletons'))
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--num-tasks', type=int, default=1)
    parser.add_argument('--watch-jobs', help='Comma-separated SLURM rebuild job IDs')
    args = parser.parse_args()
    out = args.output
    roots = (out / 'teasar_roots.txt').read_text().split()
    (out / 'validated').mkdir(exist_ok=True)
    if args.validation_only:
        remaining = set(roots[args.task_id::args.num_tasks])
        while remaining:
            for root in sorted(remaining.copy()):
                if (out / 'status' / f'{root}.json').exists():
                    validate_cell(out, root, args.skeletons)
                    remaining.remove(root)
                    print(f'Validated {root}; {len(remaining)} pending in shard', flush=True)
            if not remaining:
                return
            if not args.watch_jobs:
                return
            result = subprocess.run(['squeue', '-h', '-j', args.watch_jobs, '-o', '%i'], capture_output=True, text=True, check=True)
            if not result.stdout.strip():
                raise RuntimeError(f'Rebuild jobs ended with {len(remaining)} cells missing in this shard')
            time.sleep(30)
        return
    missing = [r for r in roots if not (out / 'status' / f'{r}.json').exists()]
    if missing:
        raise RuntimeError(f'{len(missing)} meshes incomplete; first roots: {missing[:10]}')
    for i, root in enumerate(roots):
        validate_cell(out, root, args.skeletons)
        if i % 100 == 0:
            print(f'Verified {i+1}/{len(roots)}', flush=True)
    base = 'https://g-6235c1.d1c26e.5898.data.globus.org'
    relative = 'microns/cleaned_teasar'
    selected = 864691134989909114
    with np.load(args.skeletons / f'{selected}.npz') as data:
        position = data['vertices'].mean(axis=0)
        projection_scale = max(30000., float(np.linalg.norm(np.ptp(data['vertices'], axis=0))) * 1.25)
    kwargs = dict(position=position, name='Cleaned TEASAR meshes and skeletons')
    state = viewer_state(base, relative, selected, **kwargs)
    state.update(projectionScale=projection_scale, showSlices=False)
    (out / 'viewer_state.json').write_text(json.dumps(state, indent=2))
    link = 'https://spelunker.cave-explorer.org/#!' + quote(json.dumps(state, separators=(',', ':')), safe='')
    (out / 'viewer_link.txt').write_text(link+'\n')
    (out / 'complete.json').write_text(json.dumps(dict(n_roots=len(roots), root_ids=roots,
        mesh_url=f'{base}/{relative}/mesh', skeleton_url=f'{base}/{relative}/skeletons',
        validation='skeletonization.verify for every root'), indent=2))
    print(f'Complete, verified Globus layer: {out}', flush=True)


if __name__ == '__main__':
    main()
