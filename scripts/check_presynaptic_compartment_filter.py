"""Verify mask alignment, independent vote decisions, and loader means on real cells."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset_presynaptic import PresynapticWindowDataset
from filter_presynaptic_compartments import DATABASE, MEMMAP, OUTPUT

manifest = json.loads((DATABASE / 'manifest.json').read_text())
summary = json.loads((OUTPUT / 'summary.json').read_text())
subset = dict(manifest)
subset['cells'] = dict(list(manifest['cells'].items())[:12])
dendrite = summary['classes'].index('dendrite')
checked = 0
for split in ['train','test']:
    ds = PresynapticWindowDataset(subset, split, 'new', database=DATABASE,
        memmap_root=MEMMAP, compartment_filter=OUTPUT)
    expected = 0
    for rid, info in subset['cells'].items():
        if info['split'] != split:
            continue
        with np.load(OUTPUT / 'cells' / f'{rid}.npz') as z:
            record = json.loads(str(z['audit']))
            expected += record['unique_retained']
            votes = z['votes']; keep = z['keep']; rows = z['window_rows']
            np.testing.assert_array_equal(keep, (votes.sum(1) > 0) & (2*votes[:,dendrite] <= votes.sum(1)))
            selected = np.flatnonzero(ds.index_root_ids == int(rid))
            np.testing.assert_array_equal(ds.index_window_rows[selected], rows)
            for local_index in [0,len(selected)//2,len(selected)-1] if len(selected) else []:
                item = ds[int(selected[local_index])]
                got = item.x[item.has_segclr].numpy().mean(0)
                np.testing.assert_allclose(got, z['mean_embeddings'][local_index], rtol=1e-5, atol=1e-5)
                checked += 1
    assert len(ds) == expected, (len(ds), expected)
print(f'PASS: both split indices, mask decisions, and {checked} loader means')
