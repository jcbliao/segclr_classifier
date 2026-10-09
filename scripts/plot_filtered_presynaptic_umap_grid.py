"""Compare UMAP parameters on the existing, identical Casey L2 samples."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import umap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[1] / 'analysis/presynaptic/compartment_filtered/casey_l2')
    parser.add_argument('--dimensions', type=int, choices=[2, 3], default=2)
    parser.add_argument('--classes', nargs='+', help='Refit on only these ground-truth classes')
    parser.add_argument('--samples-name', default='balanced_mean_embeddings.npz')
    parser.add_argument('--refit-baseline', action='store_true')
    args = parser.parse_args()
    source = args.input
    dim = args.dimensions
    output = source / ('parameter_grid' if dim == 2 else 'parameter_grid_3d')
    if args.classes:
        output = output / '_'.join(args.classes)
    output.mkdir(parents=True, exist_ok=True)
    settings = json.loads((source / 'settings.json').read_text())
    with np.load(source / args.samples_name) as data:
        x, labels = data['x'], data['cell_type']
        if args.classes:
            missing = set(args.classes) - set(labels)
            if missing:
                raise ValueError(f'Unknown classes: {missing}')
            selected = np.isin(labels, args.classes)
            source_indices = np.flatnonzero(selected)
            x, labels = x[selected], labels[selected]
            np.savez_compressed(output / 'input_samples.npz', x=x, cell_type=labels, source_indices=source_indices)
    names = np.unique(labels)
    palette_path = source / 'dimension_comparison_settings.json'
    palette_settings = json.loads(palette_path.read_text()) if palette_path.exists() else {}
    colors = palette_settings.get('colors', palette_settings.get('palette', {}))
    if not colors:
        colors = {name: matplotlib.colors.to_hex(plt.get_cmap('tab10')(i)) for i, name in enumerate(names)}
    point_colors = [colors[name] for name in labels]
    neighbors, distances = [2, 4, 15], [0.001, 0.01, 0.1]
    fig, axes = plt.subplots(3, 3, figsize=(16, 14), subplot_kw={'projection': '3d'} if dim == 3 else {})
    payload = []
    for row, nn in enumerate(neighbors):
        for col, md in enumerate(distances):
            path = output / f'coordinates_n{nn}_d{md:g}.npz'
            if path.exists():
                with np.load(path) as saved:
                    coords = saved['coordinates']
            elif not args.classes and not args.refit_baseline and nn == settings['n_neighbors'] and md == settings['min_dist']:
                with np.load(source / ('umap_coordinates.npz' if dim == 2 else 'umap_coordinates_3d.npz')) as saved:
                    coords = saved['coordinates']
                np.savez_compressed(path, coordinates=coords)
            else:
                print(f'Fitting n_neighbors={nn}, min_dist={md}', flush=True)
                coords = umap.UMAP(n_components=dim, n_neighbors=nn, min_dist=md, metric=settings['metric'],
                    n_epochs=settings['n_epochs'], random_state=settings['seed'], n_jobs=1).fit_transform(x)
                np.savez_compressed(path, coordinates=coords)
            assert coords.shape == (len(x), dim) and np.isfinite(coords).all()
            ax = axes[row, col]
            ax.scatter(*coords.T, c=point_colors, s=2, alpha=.55, linewidths=0, rasterized=True,
                       **({'depthshade': False} if dim == 3 else {}))
            ax.set(title=f'n_neighbors = {nn}, min_dist = {md:g}', xlabel='UMAP 1', ylabel='UMAP 2')
            if dim == 3:
                ax.set_zlabel('UMAP 3')
                traces = []
                for name in names:
                    idx = labels == name
                    traces.append(dict(type='scatter3d', mode='markers', name=str(name),
                        x=coords[idx, 0].tolist(), y=coords[idx, 1].tolist(), z=coords[idx, 2].tolist(),
                        marker=dict(color=colors[name], size=2, opacity=.55)))
                payload.append(dict(title=f'n_neighbors = {nn}, min_dist = {md:g}', traces=traces))
            print(f'Ready: {path.name}', flush=True)
    title = settings.get('plot_title', 'Presynaptic window means')
    fig.suptitle(f'{title} · Casey L2 · identical {len(x):,} samples', fontsize=16)
    fig.legend(handles=[Line2D([], [], marker='o', linestyle='', color=colors[name], label=name)
        for name in names], loc='lower center', ncol=len(names), frameon=False)
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(output / 'umap_parameter_grid.png', dpi=220)
    fig.savefig(output / 'umap_parameter_grid.pdf')
    plt.close(fig)
    if dim == 3:
        html = '''<!doctype html><html><head><meta charset="utf-8"><title>3D UMAP parameter grid</title>
<script src="https://cdn.plot.ly/plotly-3.1.0.min.js"></script>
<style>body{font:15px system-ui;margin:20px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr))}.panel{height:480px}</style></head>
<body><h2>3D UMAP · Casey L2 · identical 12,000 windows</h2><p>Rows: neighbors 2, 4, 15. Columns: min_dist 0.001, 0.01, 0.1. Drag each panel to rotate; click legend entries to hide classes.</p><div class="grid" id="grid"></div>
<script>const panels=PAYLOAD;panels.forEach((p,i)=>{const el=document.createElement('div');el.className='panel';el.id='panel'+i;document.getElementById('grid').appendChild(el);Plotly.newPlot(el,p.traces,{title:{text:p.title,font:{size:14}},margin:{l:0,r:0,t:45,b:0},scene:{xaxis:{title:'UMAP 1'},yaxis:{title:'UMAP 2'},zaxis:{title:'UMAP 3'},aspectmode:'data'},legend:{orientation:'h',font:{size:10}}},{responsive:true,displaylogo:false})});</script></body></html>'''
        html = html.replace('identical 12,000 windows', f'identical {len(x):,} windows · ' + ', '.join(names))
        (output / 'umap_parameter_grid.html').write_text(html.replace('PAYLOAD', json.dumps(payload, separators=(',', ':')).replace('</', '<\\/')))
    (output / 'settings.json').write_text(json.dumps(dict(source=str(source.resolve()),
        dimensions=dim, n_neighbors=neighbors, min_dist=distances, metric=settings['metric'],
        n_epochs=settings['n_epochs'], seed=settings['seed'], n_points=len(x),
        classes=names.tolist(), input_samples=args.samples_name, palette=colors, umap_version=umap.__version__, baseline_reused=not (bool(args.classes) or args.refit_baseline)), indent=2))
    print(f'Grid saved to {output}', flush=True)


if __name__ == '__main__':
    main()
