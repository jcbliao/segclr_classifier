"""Complete all extended-axon cells omitted by the priority k17 build."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import lance
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.build_dense_presynaptic import window_arrays
from data.build_presynaptic_axon_database import _atomic_json, _atomic_savez, _induced_cut
from data.skeleton_subsampling import nested_skeletons
from data.synapses import build_client, fetch_synapses

ROUTES = Path('/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_registered_teasar_111nm_20260911/routes.json')
STORE = Path('/orcd/compute/sdorkenw/001/segclr-db/microns')
SOURCE = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/resampled_teasar_2209/training_local_center/scale16/k17')
OUTPUT = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_k17_full_tc_source/scale16/k17')
COMPARISON = ROOT / 'data/v1718_extended_axon_neurons_casey_comparison.csv'
RUN = 'resnet_860b_reshuffled__20260603_150412'
CHECKPOINT = 'checkpoint_e0_s95000'


def missing_roots() -> tuple[list[int], dict[int, dict], dict[int, dict]]:
    source = json.loads((SOURCE / 'manifest.json').read_text())
    rows = list(csv.DictReader(COMPARISON.open(newline='')))
    eligible = {int(r['root_id']): r for r in rows
                if r['axon_status'] in {'axon_partially_extended', 'axon_fully_extended'}}
    roots = sorted(root_id for root_id in eligible if str(root_id) not in source['cells'])
    routes = {r['root_id']: r for r in json.loads(ROUTES.read_text())}
    if len(roots) != 14 or any(r not in routes for r in roots):
        raise ValueError(f'Expected 14 registered missing extended-axon roots; got {len(roots)}')
    return roots, routes, eligible


def token() -> str:
    value = os.environ.get('CAVE_TOKEN')
    if value:
        return value
    path = Path.home() / '.cloudvolume/secrets/global.daf-apis.com-cave-secret.json'
    return json.loads(path.read_text())['token']


def build(root_id: int, route: dict, output: Path) -> None:
    destination = output / 'cells' / f'{root_id}.npz'
    if destination.exists():
        print(f'Already built {root_id}', flush=True)
        return
    embedding = (Path(route['base']) / 'named_embeddings' / route['skeleton_name']
                 / RUN / CHECKPOINT / f'{root_id}.npz')
    if not embedding.is_file():
        raise FileNotFoundError(f'Named TEASAR embeddings missing: {embedding}')
    dense = Path(route['base']) / 'resampled_111nm' / f'{root_id}.npz'
    with np.load(dense) as raw:
        pos, edges = raw['vertices'], raw['edges']
    with np.load(embedding) as emb:
        ids = emb['node_ids']
        if len(ids) != len(pos) or emb['embeddings'].shape != (len(pos), 64):
            raise ValueError(f'Incomplete embedding for {root_id}')
        np.testing.assert_array_equal(ids, np.arange(len(pos)))

    dim = lance.dataset(str(STORE / 'dims/cells.lance')).to_table(
        columns=['soma_x_nm', 'soma_y_nm', 'soma_z_nm'], filter=f'root_id = {root_id}'
    ).to_pandas()
    if len(dim) != 1:
        raise ValueError(f'Expected one soma row for {root_id}')
    soma = dim.iloc[0].to_numpy(float)
    if not np.isfinite(soma).all():
        raise ValueError(f'Missing soma coordinates for {root_id}')
    cave = lance.dataset(str(STORE / 'skeletons/skeleton_nodes.lance')).to_table(
        columns=['node_id', 'x_nm', 'y_nm', 'z_nm'], filter=f'root_id = {root_id}',
        use_scalar_index=False,
    ).to_pandas().sort_values('node_id')
    cave_pos = cave[['x_nm', 'y_nm', 'z_nm']].to_numpy(float)
    if not len(cave_pos):
        raise ValueError(f'Missing CAVE skeleton for {root_id}')
    center_xyz = cave_pos[int(cKDTree(cave_pos).query(soma)[1])]
    keep, _, cut_pos, cut_edges, lengths, _ = _induced_cut(pos, edges, center_xyz)
    geometry = nested_skeletons(cut_pos, cut_edges, lengths, np.flatnonzero(keep))[16]
    geometry_path = output / 'topology/scale16' / f'{root_id}.npz'
    _atomic_savez(geometry_path, root_id=root_id, skeleton_name=route['skeleton_name'],
                  root_xyz_nm=center_xyz, **geometry)

    sites = fetch_synapses(build_client(token()), [root_id], 'outgoing')
    sites_path = output / 'sites' / f'{root_id}.parquet'
    sites_path.parent.mkdir(parents=True, exist_ok=True)
    sites.to_parquet(sites_path, index=False)
    windows = window_arrays(sites, geometry['pos_nm'], geometry['edges'],
                            geometry['edge_length_nm'], 17)
    n_valid = int(windows['new_valid_k_window'].sum())
    if n_valid == 0:
        raise ValueError(f'No valid k17 presynaptic windows for {root_id}')
    _atomic_savez(destination, root_id=root_id, k_observed=17, factor=16,
                  skeleton_name=route['skeleton_name'],
                  geometry_path=str(geometry_path.resolve()),
                  embedding_path=str(embedding.resolve()), **windows)
    print(f'Built {root_id}: {n_valid} valid windows from {len(sites)} synapses', flush=True)


def finalize(output: Path) -> None:
    roots, _, eligible = missing_roots()
    source = json.loads((SOURCE / 'manifest.json').read_text())
    metadata = json.loads((SOURCE / 'metadata.json').read_text())
    cells = dict(source['cells'])
    dest_cells = output / 'cells'
    dest_cells.mkdir(parents=True, exist_ok=True)
    for root_id, info in source['cells'].items():
        link = dest_cells / f'{root_id}.npz'
        target = (SOURCE / 'cells' / f'{root_id}.npz').resolve()
        if not link.exists():
            link.symlink_to(target)
        elif link.resolve() != target:
            raise ValueError(f'Wrong cell link: {link}')
    for root_id in roots:
        path = dest_cells / f'{root_id}.npz'
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as z:
            n_valid = int(z['new_valid_k_window'].sum())
            if n_valid == 0:
                raise ValueError(f'No valid windows for {root_id}')
        cells[str(root_id)] = dict(cell_type=eligible[root_id]['cell_classification'], split='train')
        metadata['valid_site_windows'] += n_valid
    source['cells'] = cells
    source['training_source'] = 'priority k17 plus all 14 omitted extended-axon cells'
    metadata['n_cells'] = len(cells)
    metadata['missing_root_ids'] = []
    metadata['added_extended_axon_roots'] = roots
    _atomic_json(output / 'manifest.json', source)
    _atomic_json(output / 'metadata.json', metadata)
    print(f'Finalized {len(cells)}-cell k17 database; all extended-axon roots built', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['build', 'finalize'])
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    roots, routes, _ = missing_roots()
    if args.mode == 'finalize':
        finalize(args.output)
    else:
        build(roots[args.task_id], routes[roots[args.task_id]], args.output)


if __name__ == '__main__':
    main()
