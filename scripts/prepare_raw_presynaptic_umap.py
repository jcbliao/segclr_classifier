"""Sample unaveraged SegCLR vectors from the windows used in the two-class UMAP."""
import json
from pathlib import Path
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'analysis/presynaptic/compartment_filtered/casey_l2'
OUTPUT = SOURCE / 'raw_martinotti_neurogliaform'
MEMMAP = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_k17_memmap/scale16/k17')


def main():
    settings = json.loads((SOURCE / 'settings.json').read_text())
    with np.load(SOURCE / 'balanced_mean_embeddings.npz') as z:
        selected = np.isin(z['cell_type'], ['MartFam', 'NglFam'])
        roots = z['root_id'][selected]
        rows = z['window_row'][selected]
        labels = z['cell_type'][selected]
    pools = {name: [] for name in ['MartFam', 'NglFam']}
    for root in np.unique(roots):
        idx = roots == root
        folder = MEMMAP / str(root)
        members = np.load(folder / 'members.npy', mmap_mode='r')
        offsets = np.load(folder / 'offsets.npy', mmap_mode='r')
        ids = np.load(folder / 'observed_ids.npy', mmap_mode='r')
        x = np.load(folder / 'observed_x.npy', mmap_mode='r')
        nodes = np.unique(np.concatenate([members[offsets[row]:offsets[row+1]] for row in rows[idx]]))
        positions = np.flatnonzero(np.isin(ids, nodes))
        assert len(np.unique(ids[positions])) == len(positions)
        name = labels[idx][0]
        pools[name].append((np.array(x[positions]), np.array(ids[positions]), np.repeat(root, len(positions))))
    rng = np.random.default_rng(settings['seed'])
    samples = []; sample_nodes = []; sample_roots = []; sample_labels = []; counts = {}
    n = min(2000, min(sum(len(v[0]) for v in pool) for pool in pools.values()))
    for name, pool in pools.items():
        x, nodes, roots = [np.concatenate([v[i] for v in pool]) for i in range(3)]
        counts[name] = len(x)
        chosen = rng.choice(len(x), n, replace=False)
        samples.append(x[chosen]); sample_nodes.append(nodes[chosen]); sample_roots.append(roots[chosen]); sample_labels.extend([name]*n)
    order = rng.permutation(2*n)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / 'balanced_raw_embeddings.npz', x=np.concatenate(samples)[order],
        cell_type=np.asarray(sample_labels)[order], root_id=np.concatenate(sample_roots)[order],
        node_id=np.concatenate(sample_nodes)[order])
    settings.update(plot_title='Raw presynaptic SegCLR embeddings', unit='one unaveraged 64D vector per observed node',
        sampling='uniform without replacement within class from unique observed nodes in the original sampled retained windows',
        available_per_class=counts, n_points=2*n, per_class=n, original_sample_source=str(SOURCE / 'balanced_mean_embeddings.npz'))
    (OUTPUT / 'settings.json').write_text(json.dumps(settings, indent=2))
    (OUTPUT / 'dimension_comparison_settings.json').write_bytes((SOURCE / 'dimension_comparison_settings.json').read_bytes())
    print(json.dumps(dict(available=counts, sampled_per_class=n, output=str(OUTPUT)), indent=2), flush=True)


if __name__ == '__main__':
    main()
