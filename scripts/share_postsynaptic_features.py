"""Replace duplicated bundle vectors with shared per-cell features and row maps.

Run on a CPU allocation with a checkout containing the shared-feature loader.
Each alignment file is replaced atomically; rerunning safely handles mixed formats.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.postsynaptic_cache import load_postsynaptic_features


def atomic_npz(path, **arrays):
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.npz')
    try:
        np.savez_compressed(temporary, **arrays)
        os.chmod(temporary, 0o640)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def share(bundle):
    bundle = Path(bundle).resolve()
    caches = [bundle/'cave_skeletons_pre_post/k10/conf0/postsynaptic',
              bundle/'native_skeletons_pre_post/scale16/k17/conf0']
    roots = [set(p.name for p in (cache/'cells').glob('*.npz')) for cache in caches]
    if roots[0] != roots[1]:
        raise ValueError('Postsynaptic cache cell sets differ')
    shared = bundle/'postsynaptic_features/cells'
    shared.mkdir(parents=True, exist_ok=True)
    os.chmod(shared.parent, 0o2750)
    os.chmod(shared, 0o2750)
    before = sum(p.stat().st_size for cache in caches for p in (cache/'cells').glob('*.npz'))
    for index, name in enumerate(sorted(roots[0])):
        paths = [cache/'cells'/name for cache in caches]
        original = [load_postsynaptic_features(path) for path in paths]
        # Byte keys preserve floats exactly, including signed zeros.
        combined = np.ascontiguousarray(np.concatenate(original))
        keys = combined.view(np.dtype((np.void, combined.dtype.itemsize*combined.shape[1]))).ravel()
        _, selected, inverse = np.unique(keys, return_index=True, return_inverse=True)
        target = shared/name
        features = combined[selected]
        if target.exists():
            # Already published maps may be in use: never change their row meanings.
            with np.load(target, allow_pickle=False) as z:
                existing = z['post_x']
            if existing.tobytes() != features.tobytes():
                raise ValueError(f'Existing shared features differ: {target}')
        else:
            atomic_npz(target, post_x=features)
        start = 0
        for path, expected in zip(paths, original):
            with np.load(path, allow_pickle=False) as z:
                metadata = {key:z[key].copy() for key in z.files
                            if key not in ('post_x', 'feature_path', 'feature_rows')}
            rows = inverse[start:start+len(expected)].astype(np.int32)
            start += len(expected)
            reconstructed = features[rows]
            if reconstructed.dtype != expected.dtype or reconstructed.tobytes() != expected.tobytes():
                raise ValueError(f'Feature reconstruction differs: {path}')
            atomic_npz(path, **metadata, feature_rows=rows,
                       feature_path=np.asarray(os.path.relpath(target, path.parent)))
            actual = load_postsynaptic_features(path)
            if actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
                raise ValueError(f'Published features differ: {path}')
        if index % 200 == 0:
            print(f'Verified {index+1}/{len(roots[0])} cells', flush=True)
    after = sum(p.stat().st_size for cache in caches for p in (cache/'cells').glob('*.npz'))
    feature_bytes = sum(p.stat().st_size for p in shared.glob('*.npz'))
    receipt = dict(complete=True, cells=len(roots[0]), original_alignment_bytes=before,
                   alignment_bytes=after, shared_feature_bytes=feature_bytes,
                   checks=['All cell feature arrays byte-identical',
                           'Synapse IDs and eligibility metadata preserved'],
                   loader='data.postsynaptic_cache.load_postsynaptic_features')
    path = bundle/'postsynaptic_features/complete.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt, indent=2)+'\n')
    os.chmod(temporary, 0o640)
    temporary.replace(path)
    for cache in caches:
        marker = cache/'complete.json'
        if marker.exists():
            metadata = json.loads(marker.read_text())
            metadata['shared_feature_store'] = os.path.relpath(shared.parent, cache)
            metadata['feature_storage'] = 'shared-postsynaptic-features-v1'
            temporary = marker.with_suffix('.tmp')
            temporary.write_text(json.dumps(metadata, indent=2)+'\n')
            os.chmod(temporary, 0o640)
            temporary.replace(marker)
    readme = bundle/'README.md'
    note = ('\n## Shared postsynaptic features\n\n'
            'Postsynaptic vectors are stored once per cell in `postsynaptic_features/`.\n'
            'CAVE and native caches contain separate row maps, synapse IDs and eligibility.\n'
            'Use a repository checkout containing `data/postsynaptic_cache.py`; older\n'
            'loaders that require inline `post_x` arrays cannot read these row maps.\n'
            'See `postsynaptic_features/complete.json` for conversion verification.\n')
    if readme.exists() and '## Shared postsynaptic features' not in readme.read_text():
        with readme.open('a') as stream:
            stream.write(note)
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    share(parser.parse_args().bundle)
