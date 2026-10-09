"""Build training windows for priority cells from the audited native geometry cache."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.build_dense_presynaptic import RUN, CHECKPOINT
from data.build_presynaptic_axon_database import _atomic_json, _atomic_savez
from data.skeleton_subsampling import nested_skeletons, NATIVE_SCALE_WINDOWS as SCALE_WINDOWS
from data.native_centered_windows import (
    CenteredWindows, CENTER_POLICY, DEFAULT_SYNAPSE_MATCH_CUTOFF_NM,
)

NATIVE = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/resampled_teasar_2209/native')
ROUTES = Path('/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_registered_teasar_111nm_20260911/routes.json')
OUT = NATIVE.parent / 'training_local_center'


def build(rid, route, out, factors):
    embedding = Path(route['base'])/'named_embeddings'/route['skeleton_name']/RUN/CHECKPOINT/f'{rid}.npz'
    with np.load(embedding) as z:
        assert int(z['root_id']) == rid and str(z['skeleton_name']) == route['skeleton_name']
        assert str(z['run_id']) == RUN and str(z['checkpoint_id']) == CHECKPOINT
        ids, x = z['node_ids'], z['embeddings']
        np.testing.assert_array_equal(ids, np.arange(len(ids)))
        assert x.shape == (len(ids), 64) and np.isfinite(x).all()
    with np.load(NATIVE/'cells'/f'{rid}.npz') as z:
        original = z['original_node_ids']
        assert len(original) and original.max() < len(ids)
        xyz, synapse_ids = z['new_synapse_xyz_nm'], z['new_synapse_id']
        pos,edges,lengths=z['new_pos_nm'],z['new_edges'],z['new_edge_length_nm']
        scales=nested_skeletons(pos,edges,lengths,original)
    for factor in factors:
        k = SCALE_WINDOWS[factor]
        dest = out/f'scale{factor}'/f'k{k}'/'cells'/f'{rid}.npz'
        if dest.exists():
            continue
        geometry = out/'topology'/'full'/f'{rid}.npz'
        geo = scales[factor]
        if not geometry.exists():
            _atomic_savez(geometry, root_id=rid, skeleton_name=route['skeleton_name'], **scales[1])
        selector=CenteredWindows(pos,edges,lengths,original,geo,xyz)
        center_ids=original[selector.centers]
        arrays=selector.arrays(synapse_ids,k)
        valid=np.flatnonzero(arrays['new_valid_k_window'])
        starts=arrays['new_window_offsets'][valid]
        assert np.array_equal(original[arrays['new_window_members'][starts]], center_ids[valid])
        _atomic_savez(dest, root_id=rid, k_observed=k, factor=factor,
                      center_policy=CENTER_POLICY, new_center_original_node_ids=center_ids,
                      skeleton_name=route['skeleton_name'], geometry_path=str(geometry.resolve()),
                      embedding_path=str(embedding.resolve()), **arrays)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--factors', type=int, nargs='+', choices=list(SCALE_WINDOWS), default=[32])
    p.add_argument('--task-id', type=int, default=0)
    p.add_argument('--num-tasks', type=int, default=1)
    p.add_argument('--limit', type=int)
    p.add_argument('--finalize', action='store_true')
    a = p.parse_args()
    routes = {r['root_id']:r for r in json.loads(ROUTES.read_text()) if r['priority']}
    manifest = json.loads((NATIVE/'manifest.json').read_text())
    assert set(map(int, manifest['cells'])) == set(routes)
    assert all(v.get('cell_type') and v.get('split') for v in manifest['cells'].values())
    manifest.update(training_source='registered priority native skeletons', routes=str(ROUTES))
    if a.finalize:
        for factor in a.factors:
            d = a.out/f'scale{factor}'/f'k{SCALE_WINDOWS[factor]}'
            built = {int(f.stem) for f in (d/'cells').glob('*.npz')}
            missing = sorted(set(routes)-built)
            if missing:
                raise RuntimeError(f'scale{factor}: {len(missing)} missing cells')
            assert built == set(routes)
            valid_windows=0
            for path in (d/'cells').glob('*.npz'):
                with np.load(path) as z:
                    valid=z['new_valid_k_window']
                    assert int(z['k_observed']) == SCALE_WINDOWS[factor]
                    assert str(z['center_policy']) == CENTER_POLICY
                    assert np.isfinite(z['new_cable_length_nm'][valid]).all()
                    valid_windows+=int(valid.sum())
            _atomic_json(d/'manifest.json', manifest)
            _atomic_json(d/'metadata.json', dict(format='dense-presynaptic-v1',factor=factor,
                k_observed=SCALE_WINDOWS[factor],n_cells=len(built),missing_root_ids=[],
                routes=str(ROUTES), native_source=str(NATIVE), soma_radius_nm=5000,
                match_cutoff_nm=DEFAULT_SYNAPSE_MATCH_CUTOFF_NM,
                window_policy='fixed node count; no cable-length cutoff',
                center_policy=CENTER_POLICY,
                valid_site_windows=valid_windows))
        return
    roots = sorted(routes)[a.task_id::a.num_tasks]
    if a.limit is not None:
        roots = roots[:a.limit]
    for rid in roots:
        if all((a.out/f'scale{f}'/f'k{SCALE_WINDOWS[f]}'/'cells'/f'{rid}.npz').exists() for f in a.factors):
            continue
        build(rid, routes[rid], a.out, a.factors)
        print(f'Built {rid}', flush=True)


if __name__ == '__main__':
    main()
