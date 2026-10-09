"""Export TEASAR NPZ geometry in the CAVE/MeshParty HDF5 layout.

Validate every export with CAVEclient's actual HDF5 download parser.
Missing compartments and level-2 IDs are omitted, never inferred.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from caveclient.skeletonservice import SkeletonClient


def export(source, dest):
    with np.load(source, allow_pickle=False) as s:
        v, e, r = (s[k] for k in ('vertices', 'edges', 'radius'))
        rid = int(s['root_id'].item())
        if v.ndim != 2 or v.shape[1] != 3 or e.ndim != 2 or e.shape[1] != 2:
            raise ValueError(f'Invalid geometry: {source}')
        if len(r) != len(v) or (e.size and (e.min() < 0 or e.max() >= len(v))):
            raise ValueError(f'Invalid indices/radii: {source}')
        meta = dict(root_id=rid, skeleton_name='teasar_target', units='nm',
                    source=str(source), target_spacing_nm=float(s['target_spacing_nm']),
                    subdivision=str(s['subdivision'].item()),
                    missing_properties=['compartment', 'lvl2_ids'])
        tmp = dest.with_suffix('.partial.h5')
        if not dest.exists():
            with h5py.File(tmp, 'w') as f:
                f.attrs['file_version'] = 2
                f.create_dataset('vertices', data=v, compression='gzip')
                f.create_dataset('edges', data=e, compression='gzip')
                f.create_dataset('meta', data=np.bytes_(json.dumps(meta)))
                # MeshParty stores vertex properties as JSON scalar datasets.
                f.create_group('vertex_properties').create_dataset(
                    'radius', data=json.dumps(r.tolist(), separators=(',', ':')))
        check = tmp if tmp.exists() else dest
        actual = SkeletonClient._parse_h5gz_to_dict(check.read_bytes())
        for key, expected in [('vertices', v), ('edges', e), ('radius', r)]:
            np.testing.assert_array_equal(actual[key], expected)
        assert actual['meta']['root_id'] == rid
        if check == tmp:
            tmp.chmod(0o640)
            tmp.replace(dest)
        return dict(root_id=rid, file=dest.name, nodes=len(v), edges=len(e),
                    bytes=dest.stat().st_size)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = sorted(args.source.glob('*.npz'))
    if args.limit:
        sources = sources[:args.limit]
    if not sources:
        raise ValueError('No source skeletons found')
    inventory = []
    for i, source in enumerate(sources, 1):
        inventory.append(export(source, args.out / f'{source.stem}.h5'))
        if i == 1 or i % 100 == 0 or i == len(sources):
            print(f'Exported and verified {i}/{len(sources)}', flush=True)
    report = dict(source=str(args.source), cells=len(inventory),
                  verification='CAVEclient HDF5 parser; exact array readback for every file',
                  nodes=sum(x['nodes'] for x in inventory),
                  edges=sum(x['edges'] for x in inventory), skeletons=inventory)
    (args.out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
