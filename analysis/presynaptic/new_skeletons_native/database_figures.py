"""Cached diagnostics for points assigned to >5-site resampled components."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import skeleton_stats as stats


def load_diagnostics():
    db = stats.PRESYNAPTIC_DATABASE
    paths = sorted((db/'cells').glob('*.npz'))
    metadata = json.loads((db/'metadata.json').read_text())
    assert len(paths) == metadata['n_cells'] == 2209
    key = stats._digest({'version': 1, 'files': stats._stamp(paths),
                        'metadata': stats._stamp([db/'metadata.json', db/'manifest.json'])})
    cache = Path(__file__).resolve().parent/'diagnostic_cache'/key
    if (cache/'complete.json').exists():
        print('Loading cached native distance diagnostics', flush=True)
        with np.load(cache/'distances.npz') as z:
            distances = z['distance_nm']
        return distances, pd.read_csv(cache/'per_cell.csv', dtype={'root_id':str}), metadata
    labels = json.loads((db/'manifest.json').read_text())['cells']
    inventory = {str(r['root_id']):r for r in metadata['inventory']}
    pieces, rows = [], []
    for i,path in enumerate(paths,1):
        with np.load(path) as cell:
            selected = cell['new_nearest_component_retained']
            distances = cell['new_nearest_observed_distance_nm'][selected]
            assert np.isfinite(distances).all()
        pieces.append(distances)
        audit = inventory[path.stem]
        assert len(distances) == audit['rows_nearest_retained_component']
        assert (distances <= 2000).sum() == audit['retained_rows_within_2um']
        rows.append(dict(root_id=path.stem,
            cell_type=labels.get(path.stem,{}).get('cell_type') or 'Unlabeled',
            n_presynaptic_points=len(distances),
            mean_distance_um=distances.mean()/1000 if len(distances) else np.nan,
            n_presynaptic_points_over_2um=int((distances>2000).sum()),
            fraction_presynaptic_points_over_2um=float((distances>2000).mean()) if len(distances) else np.nan,
            total_components=audit['components'],retained_components=audit['retained_components'],
            total_edges=audit['edges'],retained_edges=audit['retained_edges'],
            retained_unique_accepted_sites=audit['retained_unique_accepted_sites']))
        if i==1 or i%250==0 or i==len(paths):
            print(f'Native diagnostics: {i}/{len(paths)} cells',flush=True)
    distances=np.concatenate(pieces)
    frame=pd.DataFrame(rows)
    assert len(distances)==metadata['totals']['rows_nearest_retained_component']
    cache.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache/'distances.npz', distance_nm=distances)
    frame.to_csv(cache/'per_cell.csv',index=False)
    stats._json(cache/'complete.json', {'n_cells':len(paths),'n_distances':len(distances)})
    return distances,frame,metadata
