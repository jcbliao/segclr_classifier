"""Exactly class-balanced, unsupervised UMAP of retained window means."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnn.hierarchy import parse_hierarchy

DEFAULT = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/compartment_filtered/scale16/k17/conf0.7/fold0/geodesic_exclusion20um')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=DEFAULT)
    p.add_argument('--output', type=Path, default=Path('analysis/presynaptic/compartment_filtered/casey_l2'))
    p.add_argument('--per-class', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--hierarchy-level', type=int, default=2)
    a = p.parse_args()
    if a.per_class < 1:
        p.error('per-class must be positive')
    summary = json.loads((a.cache / 'summary.json').read_text())
    audit = pd.read_csv(a.cache / 'cell_audit.csv', dtype={'root_id':str})
    manifest = json.loads((Path(summary['database']) / 'manifest.json').read_text())
    hierarchy = parse_hierarchy(manifest['hierarchy_tree'])
    if not 0 <= a.hierarchy_level < hierarchy.depth:
        p.error('hierarchy level out of range')
    label_map = {name:hierarchy.label_paths[name][a.hierarchy_level] for name in audit.cell_type.unique()}
    audit['source_cell_type'] = audit.cell_type
    audit['cell_type'] = audit.cell_type.map(label_map)
    totals = audit.groupby('cell_type').unique_retained.sum()
    if (totals == 0).any():
        raise ValueError(f'Classes entirely removed: {totals[totals == 0].index.tolist()}')
    n = min(a.per_class, int(totals.min()))
    names = sorted(totals.index)
    rng = np.random.default_rng(a.seed)
    pools = {}
    # Independent random priorities give an exact uniform reservoir over windows in each class.
    for row in tqdm(audit.itertuples(), total=len(audit), desc='Sample retained means', file=sys.stderr):
        with np.load(a.cache / 'cells' / f'{row.root_id}.npz') as z:
            x = z['mean_embeddings']; rows = z['window_rows']
        if not len(x):
            continue
        count = len(x)
        entry = dict(priority=rng.random(count), x=x, window_row=rows,
            root_id=np.full(count, row.root_id), split=np.full(count, row.split),
            source_cell_type=np.full(count, row.source_cell_type))
        if row.cell_type in pools:
            entry = {key:np.concatenate([pools[row.cell_type][key], val]) for key,val in entry.items()}
        chosen = np.argsort(entry['priority'])[:n]
        pools[row.cell_type] = {key:val[chosen] for key,val in entry.items()}
    assert all(len(pools[name]['x']) == n for name in names)
    samples = {key:np.concatenate([pools[name][key] for name in names]) for key in ['x','window_row','root_id','split','source_cell_type']}
    samples['cell_type'] = np.repeat(names, n)
    order = rng.permutation(len(samples['x']))
    samples = {key:val[order] for key,val in samples.items()}
    assert np.isfinite(samples['x']).all()
    a.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output / 'balanced_mean_embeddings.npz', **samples)
    settings = dict(cache=str(a.cache.resolve()), classifier=summary['model'], classifier_sha256=summary['model_sha256'],
        voting_rule=summary['rule'], unit='one raw 64-dimensional mean per unique retained presynaptic window',
        sampling='uniform without replacement within cell type; exactly equal counts across classes',
        per_class=n, available_per_class={name:int(totals[name]) for name in names}, n_points=len(samples['x']),
        seed=a.seed, n_neighbors=15, min_dist=0.1, metric='euclidean', n_epochs=500,
        standardization=False, supervised=False, umap_version=umap.__version__)
    settings.update(hierarchy_level=a.hierarchy_level, source_to_class=label_map,
                    label_source='source database Casey hierarchy_tree, zero-based L2')
    print(json.dumps(settings, indent=2), flush=True)
    reducer = umap.UMAP(n_neighbors=min(15,len(samples['x'])-1), min_dist=0.1, metric='euclidean',
        n_epochs=500, random_state=a.seed, n_jobs=1, verbose=True, tqdm_kwds={'file':sys.stderr})
    coordinates = reducer.fit_transform(samples['x'])
    assert np.isfinite(coordinates).all()
    np.savez_compressed(a.output / 'umap_coordinates.npz', coordinates=coordinates, **samples)
    reducer.tqdm_kwds = {}
    joblib.dump(reducer, a.output / 'reducer.joblib')
    frame = pd.DataFrame({key:val for key,val in samples.items() if key != 'x'})
    frame['UMAP1'] = coordinates[:,0]; frame['UMAP2'] = coordinates[:,1]
    frame.to_csv(a.output / 'umap_coordinates.csv', index=False)
    colors = dict(zip(names, plt.get_cmap('tab20').colors))
    fig,ax = plt.subplots(figsize=(11,8))
    ax.scatter(coordinates[:,0], coordinates[:,1], c=[colors[name] for name in samples['cell_type']],
        s=3, alpha=0.55, linewidths=0, rasterized=True)
    ax.legend(handles=[Line2D([],[],marker='o',linestyle='',color=colors[name],label=f'{name} (n={n:,})')
        for name in names], loc='center left', bbox_to_anchor=(1,0.5), frameon=False)
    ax.set(title='Presynaptic window means after dendrite-majority filtering', xlabel='UMAP 1', ylabel='UMAP 2')
    fig.tight_layout()
    fig.savefig(a.output / 'mean_embedding_umap.png', dpi=250)
    fig.savefig(a.output / 'mean_embedding_umap.pdf')
    plt.close(fig)
    (a.output / 'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    (a.output / 'filter_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    audit.to_csv(a.output / 'cell_audit.csv',index=False)
    class_audit = audit.groupby('cell_type')[['valid_sites','removed_sites','unique_before','unique_removed','unique_retained']].sum()
    class_audit['removed_fraction'] = class_audit.unique_removed / class_audit.unique_before
    class_audit.to_csv(a.output / 'cell_type_audit.csv')
    print('UMAP COMPLETE', flush=True)

if __name__ == '__main__':
    main()
