"""Compare 2D/3D UMAP on identical means balanced by Casey L2 class."""
import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
import umap

HERE = Path(__file__).resolve().parents[1]
DEFAULT = HERE / 'analysis/presynaptic/compartment_filtered/casey_l2'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT)
    p.add_argument('--palette-json', type=Path, help='optional exact cell-type to color map')
    p.add_argument('--fit-only', action='store_true', help='fit 3D now; defer palette and figure export')
    a = p.parse_args()
    with np.load(a.output / 'balanced_mean_embeddings.npz') as z:
        samples = {key:z[key].copy() for key in z.files}
    x = samples['x']; labels = samples['cell_type']; names, counts = np.unique(labels, return_counts=True)
    assert len(set(counts)) == 1 and np.isfinite(x).all()
    settings = json.loads((a.output / 'settings.json').read_text())
    np.testing.assert_array_equal(labels, np.asarray([settings['source_to_class'][name]
        for name in samples['source_cell_type']]))
    coordinates = {}
    for dim in [2,3]:
        path = a.output / ('umap_coordinates.npz' if dim == 2 else 'umap_coordinates_3d.npz')
        if path.exists():
            with np.load(path) as z:
                for key in samples:
                    np.testing.assert_array_equal(samples[key], z[key])
                coordinates[dim] = z['coordinates'].copy()
        else:
            reducer = umap.UMAP(n_components=dim, n_neighbors=settings['n_neighbors'],
                min_dist=settings['min_dist'], metric=settings['metric'], n_epochs=settings['n_epochs'],
                random_state=settings['seed'], n_jobs=1, verbose=True, tqdm_kwds={'file':sys.stderr})
            coordinates[dim] = reducer.fit_transform(x)
            reducer.tqdm_kwds = {}
            joblib.dump(reducer, a.output / f'reducer_{dim}d.joblib')
            np.savez_compressed(path, coordinates=coordinates[dim], **samples)
        assert coordinates[dim].shape == (len(x),dim) and np.isfinite(coordinates[dim]).all()
    if a.fit_only:
        print(f'2D and 3D coordinates ready; same {len(x):,} inputs verified', flush=True)
        return
    colors = (json.loads(a.palette_json.read_text()) if a.palette_json else
        {name:to_hex(plt.get_cmap('tab10')(i)) for i,name in enumerate(names)})
    if set(names) - colors.keys():
        raise ValueError('Palette omits cell types')
    rng = np.random.default_rng(settings['seed'])
    evaluation = np.concatenate([rng.choice(np.flatnonzero(labels == name),min(100,int(count)),replace=False)
        for name,count in zip(names,counts)])
    metrics = []
    payload = {}
    for dim, coords in coordinates.items():
        frame = pd.DataFrame({key:val for key,val in samples.items() if key != 'x'})
        for axis in range(dim):
            frame[f'UMAP{axis+1}'] = coords[:,axis]
        frame.to_csv(a.output / f'umap_coordinates_{dim}d.csv',index=False)
        fig = plt.figure(figsize=(11,8))
        ax = fig.add_subplot(111, projection='3d' if dim == 3 else None)
        kwargs = dict(c=[colors[name] for name in labels],s=3,alpha=.55,linewidths=0,rasterized=True)
        if dim == 3:
            ax.scatter(*coords.T,**kwargs,depthshade=False)
            ax.set_zlabel('UMAP 3')
        else:
            ax.scatter(*coords.T,**kwargs)
        ax.set(title=f'{dim}D UMAP of presynaptic window means · Casey L2',xlabel='UMAP 1',ylabel='UMAP 2')
        ax.legend(handles=[Line2D([],[],marker='o',linestyle='',color=colors[name],label=f'{name} (n={count:,})')
            for name,count in zip(names,counts)],loc='center left',bbox_to_anchor=(1.02,.5),frameon=False)
        fig.tight_layout()
        fig.savefig(a.output / f'mean_embedding_umap_{dim}d.png',dpi=250,bbox_inches='tight')
        fig.savefig(a.output / f'mean_embedding_umap_{dim}d.pdf',bbox_inches='tight')
        if dim == 2:
            fig.savefig(a.output / 'mean_embedding_umap.png',dpi=250,bbox_inches='tight')
            fig.savefig(a.output / 'mean_embedding_umap.pdf',bbox_inches='tight')
        plt.close(fig)
        neighbor_indices = NearestNeighbors(n_neighbors=16).fit(coords).kneighbors(return_distance=False)[:,:15]
        purity = float((labels[neighbor_indices] == labels[:,None]).mean())
        score = float(trustworthiness(x[evaluation],coords[evaluation],n_neighbors=15,metric=settings['metric']))
        metrics.append(dict(dimensions=dim,n_points=len(x),trustworthiness_15=score,
            trustworthiness_evaluation_points=len(evaluation),cell_type_neighbor_agreement_15=purity))
        traces = []
        for name,count in zip(names,counts):
            idx = np.flatnonzero(labels == name)
            trace = dict(type='scatter3d' if dim == 3 else 'scattergl',mode='markers',
                name=f'{name} (n={count:,})',x=coords[idx,0].tolist(),y=coords[idx,1].tolist(),
                marker=dict(color=colors[name],size=2 if dim == 3 else 3,opacity=.55),
                customdata=np.column_stack([samples['root_id'][idx],samples['window_row'][idx],samples['split'][idx],samples['source_cell_type'][idx]]).tolist(),
                hovertemplate=name+'<br>root %{customdata[0]}<br>window %{customdata[1]}<br>%{customdata[2]}<br>source type %{customdata[3]}<extra></extra>')
            if dim == 3:
                trace['z'] = coords[idx,2].tolist()
            traces.append(trace)
        payload[str(dim)] = traces
    html = '''<!doctype html><html><head><meta charset="utf-8"><title>Presynaptic mean UMAP: 2D / 3D</title>
<script src="https://cdn.plot.ly/plotly-3.1.0.min.js"></script>
<style>body{font:15px system-ui;margin:20px}#plot{height:85vh;min-height:600px}button{margin:0 8px 0 0;padding:8px 18px}</style></head>
<body><h2>Dendrite-filtered presynaptic window means · Casey L2 classes</h2><p>Equal window counts per Casey L2 class; identical samples in 2D and 3D. Drag to rotate 3D; click legend entries to hide classes; double-click to isolate.</p>
<button onclick="draw(2)">2D UMAP</button><button onclick="draw(3)">3D UMAP</button><div id="plot"></div>
<script>const data=PAYLOAD;function draw(dim){Plotly.react('plot',data[dim],{title:dim+'D UMAP',margin:{l:60,r:240,t:60,b:60},
xaxis:{title:'UMAP 1'},yaxis:{title:'UMAP 2'},scene:{xaxis:{title:'UMAP 1'},yaxis:{title:'UMAP 2'},zaxis:{title:'UMAP 3'},aspectmode:'data'},legend:{itemsizing:'constant'}},{responsive:true,displaylogo:false})}draw(2)</script></body></html>'''
    (a.output / 'mean_embedding_umap_2d_3d.html').write_text(html.replace('PAYLOAD',json.dumps(payload,separators=(',',':')).replace('</','<\\/')))
    pd.DataFrame(metrics).to_csv(a.output / 'dimension_comparison.csv',index=False)
    settings.update(dimensions=[2,3],colors=colors,palette_source=str(a.palette_json.resolve()) if a.palette_json else
        'tab10; labels from the source manifest Casey hierarchy at L2',
        comparison=metrics,comparison_note='Trustworthiness on the same 100 windows per class; neighbor agreement is descriptive, not a held-out classification score')
    (a.output / 'dimension_comparison_settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    print(json.dumps(metrics,indent=2),flush=True)
    print('2D/3D UMAP EXPORT COMPLETE',flush=True)

if __name__ == '__main__':
    main()
