"""Join verified postsynaptic embeddings to native bin16 synapse rows."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

BASE = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons')
DATABASE = BASE / 'casey_coarse_confidence_with_tc_folds/scale16/k17/conf0.7/fold0'
EMBEDDINGS = Path('/orcd/scratch/orcd/013/jcbliao/segclr/postsynaptic_site_embeddings_v1718')
OUTPUT = BASE / 'native_skeletons_pre_post/scale16/k17/conf0.7'


def prepare(database=DATABASE, embeddings=EMBEDDINGS, output=OUTPUT):
    manifest = json.loads((database / 'manifest.json').read_text())
    summary = json.loads((embeddings / 'summary.json').read_text())
    embedding_manifest = json.loads((embeddings / 'manifest.json').read_text())
    if summary['n_sites'] != embedding_manifest['n_sites']:
        raise ValueError('Postsynaptic verification is incomplete')
    metadata = json.loads((database / 'metadata.json').read_text())
    assert metadata['factor'] == 16 and metadata['k_observed'] == 17
    assert embedding_manifest['materialization_version'] == 1718
    output.mkdir(parents=True, exist_ok=True)
    (output / 'cells').mkdir(exist_ok=True)
    marker = output / 'complete.json'
    expected = dict(format='postsynaptic-native-site-cache-v1', database=str(database.resolve()),
        embeddings=str(embeddings.resolve()), checkpoint_sha256=embedding_manifest['checkpoint_sha256'],
        exclusion_policy='unresolved_root_or_empty_mask')
    if marker.exists():
        previous = json.loads(marker.read_text())
        if any(previous.get(key) != value for key, value in expected.items()):
            raise ValueError('Existing cache has different provenance')
        print('Prepared cache already complete', flush=True)
        return
    c = duckdb.connect()
    c.execute('SET threads=4')
    roots = pd.DataFrame({'presynaptic_root_id': np.asarray([int(r) for r in manifest['cells']], np.int64)})
    frame = c.execute('''select e.synapse_id, e.presynaptic_root_id, e.postsynaptic_root_id,
        e.status, e.embedding from read_parquet(?) e join roots r using(presynaptic_root_id)''',
        [str(embeddings / 'embeddings/*.parquet')]).df()
    if frame.synapse_id.duplicated().any():
        raise ValueError('Duplicate inference synapse IDs')
    unexpected = frame[(frame.postsynaptic_root_id != 0) & ~frame.status.isin(['ok', 'empty_mask'])]
    if len(unexpected):
        raise ValueError(f'{len(unexpected)} resolved sites lack embeddings; repair inference before training')
    grouped = frame.groupby('presynaptic_root_id').indices
    counts = dict(n_cells=0, n_site_rows=0, n_unresolved_excluded=0, n_empty_mask_excluded=0)
    for rid in sorted(int(r) for r in manifest['cells']):
        rows = frame.iloc[grouped.get(rid, [])].set_index('synapse_id')
        with np.load(database / 'cells' / f'{rid}.npz') as z:
            ids = z['new_synapse_id'].copy()
        positions = rows.index.get_indexer(ids)
        if (positions < 0).any():
            raise ValueError(f'Cell {rid}: {int((positions < 0).sum())} native synapse IDs absent from inference')
        aligned = rows.iloc[positions]
        unresolved = aligned.postsynaptic_root_id.to_numpy(np.int64) == 0
        empty_mask = aligned.status.to_numpy() == 'empty_mask'
        eligible = ~unresolved & ~empty_mask
        vectors = np.zeros((len(ids), 64), np.float32)
        if eligible.any():
            vectors[eligible] = np.stack(aligned.embedding.to_numpy()[eligible]).astype(np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError(f'Cell {rid}: nonfinite postsynaptic vectors')
        dest = output / 'cells' / f'{rid}.npz'
        tmp = dest.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, synapse_ids=ids, eligible=eligible, post_x=vectors)
        os.replace(tmp, dest)
        counts['n_cells'] += 1
        counts['n_site_rows'] += len(ids)
        counts['n_unresolved_excluded'] += int(unresolved.sum())
        counts['n_empty_mask_excluded'] += int((empty_mask & ~unresolved).sum())
        if counts['n_cells'] % 100 == 0:
            print(counts, flush=True)
    result = dict(**expected, **counts, uniqueness='complete sorted presynaptic node set within root; postsynaptic points do not alter uniqueness',
        duplicate_policy='first eligible valid site in native row order', embedding_dim=64)
    tmp = marker.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    os.replace(tmp, marker)
    print(result, flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database', type=Path, default=DATABASE)
    p.add_argument('--embeddings', type=Path, default=EMBEDDINGS)
    p.add_argument('--output', type=Path, default=OUTPUT)
    a = p.parse_args()
    prepare(a.database, a.embeddings, a.output)
