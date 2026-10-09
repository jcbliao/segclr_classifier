"""Rebuild retained TEASAR healed meshes into one precomputed mesh source."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', type=Path, required=True, help='One root ID per line')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task-id', type=int, help='Process one root by its zero-based index')
    parser.add_argument('--num-tasks', type=int, default=1)
    parser.add_argument('--skeletons', type=Path, default=Path('/orcd/scratch/orcd/013/jcbliao/skeletons/segclr/skeletons'))
    parser.add_argument('--skeletonization-repo', type=Path, default=Path('/orcd/home/002/jcbliao/rotation/skeletonization'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.skeletonization_repo))
    from skeletonization import cave_io
    from skeletonization.healing import component_report, remove_cavities
    from skeletonization.multires import write_multires_mesh
    from skeletonization.precomputed import write_skeleton
    mesh_dir = args.output / 'mesh'
    healed_dir = args.output / 'healed_npz'
    mesh_dir.mkdir(parents=True, exist_ok=True)
    healed_dir.mkdir(parents=True, exist_ok=True)
    status_dir = args.output / 'status'
    status_dir.mkdir(exist_ok=True)
    roots = args.roots.read_text().split()
    if args.task_id is not None:
        roots = roots[args.task_id::args.num_tasks]
    for root in roots:
        rid = int(root)
        original = args.skeletons / f'{root}.npz'
        skeleton_dir = args.output / 'skeletons'
        if not (skeleton_dir / root).exists():
            with np.load(original) as data, tempfile.TemporaryDirectory(dir=args.output, prefix=f'.skeleton_{root}_') as tmp:
                staged = Path(tmp)
                write_skeleton(staged, rid, data['vertices'], data['edges'], radius=data['radius'])
                if json.loads((staged / 'info').read_text()) != json.loads((skeleton_dir / 'info').read_text()):
                    raise ValueError('Skeleton metadata differs from shared layer')
                (staged / root).replace(skeleton_dir / root)
        done = status_dir / f'{root}.json'
        if done.exists() and (mesh_dir / root).exists() and (mesh_dir / f'{root}.index').exists():
            continue
        cached = healed_dir / f'{root}.npz'
        if cached.exists():
            with np.load(cached) as data:
                hv, hf = data['vertices'], data['faces']
        else:
            print(f'Fetching and healing {rid}', flush=True)
            with tempfile.TemporaryDirectory() as tmp:
                v, f = cave_io.fetch_mesh(rid, cache_dir=tmp)
            hv, hf, _, _, report = remove_cavities(v, f, report=component_report(v, f))
            kept = np.unique(f[report['classification']['keep'][report['face_labels']]]).astype(np.int32)
            if original.exists():
                with np.load(original) as data:
                    if 'heal_kept_vertices' in data:
                        np.testing.assert_array_equal(kept, data['heal_kept_vertices'],
                            err_msg='Fetched mesh differs from original TEASAR input')
                    if 'healed_n_vertices' in data:
                        assert len(hv) == int(data['healed_n_vertices'][0])
            temporary = cached.with_suffix(f'.{os.getpid()}.tmp.npz')
            np.savez_compressed(temporary, vertices=hv, faces=hf, heal_kept_vertices=kept)
            temporary.replace(cached)
        # Export privately; shared info is initialized once before submission.
        with tempfile.TemporaryDirectory(dir=args.output, prefix=f'.export_{root}_') as tmp:
            staged = Path(tmp)
            write_multires_mesh(staged, rid, hv, hf, num_lod=0, workers=1)
            info = json.loads((staged / 'info').read_text())
            if info != json.loads((mesh_dir / 'info').read_text()):
                raise ValueError('Export metadata differs from shared mesh layer')
            from cloudvolume.datasource.precomputed.mesh.multilod import MultiLevelPrecomputedMeshManifest
            MultiLevelPrecomputedMeshManifest.from_binary((staged / f'{root}.index').read_bytes(), segment_id=rid)
            for name in [root, f'{root}.index']:
                (staged / name).replace(mesh_dir / name)
        temporary = done.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(dict(root_id=root, vertices=len(hv), faces=len(hf),
            mesh_bytes=(mesh_dir / root).stat().st_size)))
        temporary.replace(done)
        print(f'Exported {rid}', flush=True)


if __name__ == '__main__':
    main()
