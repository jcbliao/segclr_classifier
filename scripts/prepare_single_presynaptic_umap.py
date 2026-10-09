"""Use the nearest observed raw vector for each previously sampled pre window."""
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / 'analysis/presynaptic/compartment_filtered/casey_l2'
OUTPUT = REPO / 'analysis/presynaptic/native_single_pre_post/umap'
MEMMAP = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_k17_memmap/scale16/k17')


def main():
    settings = json.loads((SOURCE / 'settings.json').read_text())
    with np.load(SOURCE / 'balanced_mean_embeddings.npz') as data:
        selected = np.ones(len(data['cell_type']), dtype=bool)
        samples = {key: data[key][selected].copy() for key in data.files if key != 'x'}
        source_indices = np.flatnonzero(selected)
    roots = samples['root_id']; rows = samples['window_row']
    x = np.empty((len(roots), 64), np.float32)
    nodes = np.empty(len(roots), np.int64)
    for root in np.unique(roots):
        selected = np.flatnonzero(roots == root)
        folder = MEMMAP / str(root)
        centers = np.load(folder / 'centers.npy', mmap_mode='r')
        ids = np.load(folder / 'observed_ids.npy', mmap_mode='r')
        vectors = np.load(folder / 'observed_x.npy', mmap_mode='r')
        lookup = {int(node): i for i, node in enumerate(ids)}
        nodes[selected] = centers[rows[selected]]
        positions = [lookup[int(node)] for node in nodes[selected]]
        x[selected] = vectors[positions]
        np.testing.assert_array_equal(x[selected], np.asarray(vectors[positions], dtype=np.float32))
    assert np.isfinite(x).all()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / 'single_presynaptic_embeddings.npz', x=x, node_id=nodes,
        source_indices=source_indices, **samples)
    unique = len(set(zip(roots.tolist(), nodes.tolist())))
    settings.update(plot_title='Single nearest presynaptic embedding',
        unit='one raw 64D nearest-observed presynaptic vector per sampled window',
        n_points=len(x), sampling='same 2,000 retained windows per class across all six Casey L2 classes; preserve duplicate central nodes',
        unique_root_node_pairs=unique, original_sample_source=str(SOURCE / 'balanced_mean_embeddings.npz'),
        training_selection_rule='same nearest observed node selection as --single-presynaptic-embedding; UMAP retains earlier compartment-filtered window cohort')
    (OUTPUT / 'settings.json').write_text(json.dumps(settings, indent=2))
    (OUTPUT / 'dimension_comparison_settings.json').write_bytes((SOURCE / 'dimension_comparison_settings.json').read_bytes())
    print(f'Saved {len(x):,} single vectors from {unique:,} unique root/node pairs to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
