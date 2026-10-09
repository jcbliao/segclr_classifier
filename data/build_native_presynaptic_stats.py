"""Build geometry-only resampled TEASAR statistics for the exact k10 cohort.

Reuses saved synaptic coordinates and soma-cut centers; no CAVE query or
embedding inference is required. Original resampled files are linked for exact
soma-boundary reconstruction in the upstream analyses.
"""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

REPO = Path(__file__).resolve().parents[1]
SOURCE_DB = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/k10')
ROUTES = Path('/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_registered_teasar_111nm_20260911/routes.json')
OUTPUT = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/resampled_teasar_2209/native')
RADIUS = 5000.
CUTOFF = 2000.


def stamp(path):
    st = path.stat()
    return [str(path.resolve()), st.st_size, st.st_mtime_ns]


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def build_cell(task):
    rid, source, out = task
    original = SOURCE_DB/'cells'/f'{rid}.npz'
    source, out = Path(source), Path(out)
    dest = out/'cells'/f'{rid}.npz'
    audit = out/'audit'/f'{rid}.json'
    signature = dict(version=1, source=stamp(source), sites=stamp(original),
                     soma_radius_nm=RADIUS, match_cutoff_nm=CUTOFF, min_component_sites=5)
    if dest.exists() and audit.exists():
        saved = json.loads(audit.read_text())
        if saved['signature'] == signature:
            return saved
    with np.load(original) as old:
        center = old['root_xyz_nm'].astype(np.float64)
        site_ids = old['new_synapse_id']
        xyz = old['new_synapse_xyz_nm']
    with np.load(source) as raw:
        assert int(raw['root_id'].item()) == int(rid)
        pos, edges = raw['vertices'], raw['edges']
        assert float(raw['spacing_nm']) == 111.
    keep = np.linalg.norm(pos.astype(np.float64)-center, axis=1) > RADIUS
    remap = np.full(len(pos), -1, np.int64)
    remap[keep] = np.arange(keep.sum())
    edges = remap[edges[keep[edges].all(axis=1)]]
    pos = pos[keep].astype(np.float32)
    lengths = np.linalg.norm(pos[edges[:,0]].astype(np.float64)-pos[edges[:,1]], axis=1).astype(np.float32)
    assert len(pos) and np.isfinite(pos).all() and np.isfinite(xyz).all()
    assert not len(lengths) or lengths.max() <= 111.5
    distance, nearest = cKDTree(pos).query(xyz)
    accepted = distance <= CUTOFF
    graph = coo_matrix((np.ones(len(edges), np.uint8), (edges[:,0],edges[:,1])),
                       shape=(len(pos),len(pos))).tocsr()
    ncomponents, components = connected_components(graph, directed=False)
    accepted_rows = np.flatnonzero(accepted)
    _, first = np.unique(site_ids[accepted_rows], return_index=True)
    unique_rows = accepted_rows[first]
    counts = np.bincount(components[nearest[unique_rows]], minlength=ncomponents)
    retained = counts > 5
    node_keep = retained[components]
    edge_keep = node_keep[edges].all(axis=1)
    site_keep = node_keep[nearest]
    tmp = dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, root_id=np.array([int(rid)],np.uint64),
        root_xyz_nm=center.astype(np.float32), new_pos_nm=pos, new_edges=edges.astype(np.int32),
        new_edge_length_nm=lengths, new_synapse_id=site_ids, new_synapse_xyz_nm=xyz,
        new_nearest_observed_node=nearest.astype(np.int32),
        new_nearest_observed_distance_nm=distance, new_within_2um=accepted,
        new_nearest_component_retained=site_keep, original_node_ids=np.flatnonzero(keep),
        component_presynaptic_sites=counts, skeleton_source=str(source))
    tmp.replace(dest)
    result = dict(root_id=rid, signature=signature, nodes=len(pos), edges=len(edges),
        retained_nodes=int(node_keep.sum()), retained_edges=int(edge_keep.sum()),
        components=ncomponents, retained_components=int(retained.sum()),
        unique_accepted_sites=len(unique_rows), retained_unique_accepted_sites=int(counts[retained].sum()),
        presynaptic_rows=len(site_ids), rows_nearest_retained_component=int(site_keep.sum()),
        retained_rows_within_2um=int((site_keep & accepted).sum()),
        edge_max_nm=float(lengths.max()) if len(lengths) else None)
    atomic_json(audit, result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUTPUT)
    ap.add_argument('--workers', type=int, default=8)
    args=ap.parse_args()
    for d in ('cells','originals','audit'):
        (args.out/d).mkdir(parents=True,exist_ok=True)
    routes={str(r['root_id']):r for r in json.loads(ROUTES.read_text())}
    roots=sorted(p.stem for p in (SOURCE_DB/'cells').glob('*.npz'))
    assert len(roots)==2209
    tasks=[]
    for rid in roots:
        source=Path(routes[rid]['base'])/'resampled_111nm'/f'{rid}.npz'
        assert source.is_file(), source
        link=args.out/'originals'/source.name
        if link.is_symlink():
            assert link.resolve()==source.resolve()
        elif not link.exists():
            link.symlink_to(source)
        else:
            raise ValueError(f'Expected source symlink: {link}')
        tasks.append((rid,str(source),str(args.out)))
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,row in enumerate(pool.map(build_cell,tasks,chunksize=1),1):
            results.append(row)
            if i==1 or i%100==0 or i==len(roots):
                print(f'Native geometry: {i}/{len(roots)} cells',flush=True)
    assert {p.stem for p in (args.out/'cells').glob('*.npz')}==set(roots)
    previous=json.loads((REPO/'data/manifest.json').read_text())
    manifest=dict(previous, cells={rid:previous['cells'].get(rid,{}) for rid in roots},
        source_database=str(SOURCE_DB), routes=str(ROUTES),
        skeleton_sources={rid:str(Path(routes[rid]['base'])/'resampled_111nm'/f'{rid}.npz') for rid in roots})
    atomic_json(args.out/'manifest.json',manifest)
    totals={key:sum(r[key] for r in results) for key in (
        'nodes','edges','retained_nodes','retained_edges','components','retained_components',
        'unique_accepted_sites','retained_unique_accepted_sites','presynaptic_rows',
        'rows_nearest_retained_component','retained_rows_within_2um')}
    atomic_json(args.out/'metadata.json',dict(format='native-presynaptic-statistics-v1',
        n_cells=len(roots), soma_radius_nm=RADIUS,match_cutoff_nm=CUTOFF,
        min_component_sites=5,spacing_nm=111,embeddings_required=False,
        mapping='nearest resampled node after soma cut', totals=totals,
        source_database=str(SOURCE_DB),routes=str(ROUTES),inventory=results))
    print(json.dumps(totals,indent=2),flush=True)


if __name__=='__main__':
    main()
