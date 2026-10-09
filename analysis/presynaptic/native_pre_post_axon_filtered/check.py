"""Verify paired filtered indices against independent mask/deduplication on real cells."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from data.dataset_presynaptic import PresynapticWindowDataset
BASE = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons')
DATABASE = BASE / 'casey_coarse_confidence_with_tc_folds/scale16/k17/conf0.7/fold0'
FILTER = BASE / 'compartment_filtered/scale16/k17/conf0.7/fold0/geodesic_exclusion20um'
POST = BASE / 'native_skeletons_pre_post/scale16/k17/conf0.7'
manifest = json.loads((DATABASE / 'manifest.json').read_text())
cells = {}
for split in ('train', 'test'):
    candidates = [(r, v) for r, v in manifest['cells'].items() if v['split'] == split]
    cells.update(candidates[:6])
manifest['cells'] = cells
for split in ('train', 'test'):
    kwargs = dict(database=DATABASE, postsynaptic_cache=POST, compartment_filter=FILTER)
    pre = PresynapticWindowDataset(manifest, split, 'new', **kwargs)
    post = PresynapticWindowDataset(manifest, split, 'new', use_postsynaptic=True, **kwargs)
    np.testing.assert_array_equal(pre.index_root_ids, post.index_root_ids)
    np.testing.assert_array_equal(pre.index_window_rows, post.index_window_rows)
    for rid in np.unique(pre.index_root_ids):
        with np.load(DATABASE / 'cells' / f'{rid}.npz') as source, np.load(FILTER / 'cells' / f'{rid}.npz') as masks, np.load(POST / 'cells' / f'{rid}.npz') as paired:
            np.testing.assert_array_equal(source['new_synapse_id'], masks['synapse_ids'])
            np.testing.assert_array_equal(source['new_synapse_id'], paired['synapse_ids'])
            valid = source['new_valid_k_window'] & masks['keep'] & paired['eligible']
            seen, expected = set(), []
            for row in np.flatnonzero(valid):
                offsets = source['new_window_offsets']
                key = np.sort(source['new_window_members'][offsets[row]:offsets[row + 1]]).tobytes()
                if key not in seen:
                    seen.add(key); expected.append(row)
            selected = np.flatnonzero(pre.index_root_ids == rid)
            np.testing.assert_array_equal(pre.index_window_rows[selected], expected)
            if len(selected):
                item = post[int(selected[0])]
                np.testing.assert_array_equal(item.postsynaptic_embedding.numpy()[0], paired['post_x'][expected[0]])
    print(f'PASS {split}: {len(pre):,} paired filtered windows', flush=True)
