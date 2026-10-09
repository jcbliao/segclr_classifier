"""Prepare branch-preserving topology now; build presynaptic windows after inference.

Stages: --prepare, --topology (array), --sites (array), --windows (array),
--finalize. All outputs live separately from the previous presynaptic database.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.build_presynaptic_axon_database import (
    _atomic_json, _atomic_savez, _induced_cut, _sym_edges, _window,
    SOMA_RADIUS_NM, SYNAPSE_MATCH_CUTOFF_NM,
)
from data.geodesic_window import build_csr_from_edges, _window_laplacian_pos_enc, DEFAULT_POS_DIM
from data.skeleton_subsampling import nested_skeletons, SCALE_WINDOWS

SOURCE = Path('/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_testing_teasar_111nm_20260910_resumed')
DEFAULT_OUT = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/dense_teasar_multiscale_v2')
NAME = 'teasar_testing_111nm_20260910'
RUN = 'resnet_860b_reshuffled__20260603_150412'
CHECKPOINT = 'checkpoint_e0_s95000'
PROOFREAD = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/k10/proofread_axon_root_ids.json')


def config_dir(out, factor):
    return out / f'scale{factor}' / f'k{SCALE_WINDOWS[factor]}'


def prepare(out, source):
    from segclr_db import store as st
    from data.build_dataset import stratified_split
    cohort = pd.read_csv(source/'cells.csv')
    labels = dict(zip(cohort.pt_root_id.astype(int), cohort.cell_type))
    splits = stratified_split(labels)
    previous = json.loads((ROOT/'data/manifest.json').read_text())
    # Preserve existing cell splits; use the same seed-0 stratification for
    # new cells. One shared manifest serves every resolution.
    cells = {str(r): dict(cell_type=label, split=previous['cells'].get(str(r), {}).get('split', splits[r]))
             for r, label in labels.items()}
    proof = json.loads(PROOFREAD.read_text())
    assert proof['materialization_version'] == 1718
    proof_ids = set(proof['root_ids'])
    store = st.open_store('/orcd/compute/sdorkenw/001/segclr-db','microns')
    dimensions = st.scan(store,'cells',root_ids=list(labels),
                         columns=['root_id','soma_x_nm','soma_y_nm','soma_z_nm']).to_pandas()
    soma = {str(int(row.root_id)): [row.soma_x_nm,row.soma_y_nm,row.soma_z_nm]
            for row in dimensions.itertuples()
            if np.isfinite([row.soma_x_nm,row.soma_y_nm,row.soma_z_nm]).all()}
    soma_sources={r:'segclr_db.cells' for r in soma}
    for row in cohort.itertuples():
        rid=str(int(row.pt_root_id))
        xyz=[row.pt_x_nm,row.pt_y_nm,row.pt_z_nm]
        if rid not in soma and np.isfinite(xyz).all():
            soma[rid]=xyz
            soma_sources[rid]='saved CAVE cell_type_multifeature_combo soma, nm'
    payload = dict(cells=cells,split_seed=0,split_fracs=[.8,.2],label_mat_version=1718,
                   skeleton_name=NAME,source=str(source),scales=SCALE_WINDOWS,
                   soma_radius_nm=SOMA_RADIUS_NM,
                   match_cutoff_nm=SYNAPSE_MATCH_CUTOFF_NM,
                   soma_cut_center='CAVE node nearest cached soma',
                   proofread_root_ids=sorted(proof_ids & set(labels)),soma_positions=soma,soma_sources=soma_sources,
                   proofread_source=str(PROOFREAD),
                   split_policy='Preserve previous splits; seed-0 stratification for new cells',
                   hierarchy_tree=previous.get('hierarchy_tree'))
    if payload['hierarchy_tree'] is None: payload.pop('hierarchy_tree')
    dest=out/'manifest.json'
    if dest.exists() and json.loads(dest.read_text()) != json.loads(json.dumps(payload)):
        raise ValueError('Existing preparation differs; use a new output directory')
    _atomic_json(dest,payload)
    print(f'Prepared {len(cells)} cells; {len(payload["proofread_root_ids"])} proofread axons; {len(soma)} soma coordinates',flush=True)


def topology(rid, out, manifest):
    from segclr_db import store as st
    source=Path(manifest['source'])/'resampled_111nm'/f'{rid}.npz'
    with np.load(source) as raw:
        pos,edges=raw['vertices'],raw['edges']
    digest=hashlib.sha256(pos.tobytes()+edges.tobytes()).hexdigest()
    soma=manifest['soma_positions'].get(str(rid))
    if soma is None:
        # Retain topology for every skeleton, but exclude these cells from
        # presynaptic training because the required soma cut cannot be made.
        keep=np.ones(len(pos),bool)
        root_xyz=np.full(3,np.nan)
        cut_pos,cut_edges=pos,edges
        weights=np.linalg.norm(pos[edges[:,0]].astype(float)-pos[edges[:,1]],axis=1)
    else:
        store=st.open_store('/orcd/compute/sdorkenw/001/segclr-db','microns')
        # Only coordinates are needed. The shared scalar index has restricted
        # files; reading the accessible data directly preserves the same rows.
        frame=st.open_table(store,'skeleton_nodes').to_table(
            columns=['node_id','x_nm','y_nm','z_nm'],filter=f'root_id = {rid}',
            use_scalar_index=False).to_pandas().sort_values('node_id')
        cave_pos=frame[['x_nm','y_nm','z_nm']].to_numpy(np.float64)
        if not len(cave_pos): raise ValueError(f'Missing default CAVE skeleton: {rid}')
        center=int(cKDTree(cave_pos).query(soma)[1])
        root_xyz=cave_pos[center]
        keep,_,cut_pos,cut_edges,weights,_=_induced_cut(pos,edges,root_xyz)
    scales=nested_skeletons(cut_pos,cut_edges,weights,np.flatnonzero(keep))
    stats=[]
    for factor, arrays in scales.items():
        dest=out/'topology'/f'scale{factor}'/f'{rid}.npz'
        _atomic_savez(dest,root_id=np.array([rid]),root_xyz_nm=root_xyz,soma_cut_applied=soma is not None,
                      source_digest=digest,source_node_count=len(pos),post_cut_node_count=int(keep.sum()),
                      factor=factor,skeleton_name=NAME,**arrays)
        stats.append(dict(factor=factor,nodes=len(arrays['pos_nm']),edges=len(arrays['edges']),
                          target_nodes=(int(keep.sum())+factor-1)//factor,
                          target_exceeded=len(arrays['pos_nm'])>((int(keep.sum())+factor-1)//factor)))
    _atomic_json(out/'topology'/f'{rid}.json',dict(root_id=rid,soma_cut_applied=soma is not None,scales=stats))


def sites(rid,out):
    """Use the same CAVE query/normalization as the original presynaptic builder."""
    from data.synapses import build_client,fetch_synapses
    token_path=Path.home()/'.cloudvolume/secrets/global.daf-apis.com-cave-secret.json'
    token=os.environ.get('CAVE_TOKEN') or json.loads(token_path.read_text())['token']
    frame=fetch_synapses(build_client(token),[rid],'outgoing')
    dest=out/'sites'/f'{rid}.parquet'
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_suffix(f'.tmp.{os.getpid()}.parquet')
    frame.to_parquet(temp,index=False)
    os.replace(temp,dest)


def window_arrays(points,pos,edges,lengths,k,*,compute_lpe=True):
    """K includes the nearest embedding itself; all retained nodes are observed."""
    import torch
    xyz=points[['cell_x_nm','cell_y_nm','cell_z_nm']].to_numpy(float)
    n=len(points)
    distances=np.full(n,np.inf)
    centers=np.full(n,-1,np.int32)
    if len(pos) and n:
        distances,centers=cKDTree(pos).query(xyz)
    offsets,neighbors,weights=build_csr_from_edges(_sym_edges(edges),np.r_[lengths,lengths][:,None],len(pos))
    observed=np.ones(len(pos),bool)
    valid=np.zeros(n,bool)
    radii=np.full(n,np.nan,np.float32)
    cable_lengths=np.full(n,np.nan,np.float64)
    parts,pes,bounds=[],[],[0]
    cached={}
    for i,center in enumerate(centers):
        if distances[i] <= SYNAPSE_MATCH_CUTOFF_NM:
            center=int(center)
            if center not in cached:
                found=_window(center,offsets,neighbors,weights,observed,k)
                if found is None: cached[center]=None
                else:
                    nodes,radius=found
                    local={int(v):j for j,v in enumerate(nodes)}
                    pairs=[]
                    cable=0.0
                    for v in nodes:
                        for p in range(offsets[v],offsets[v+1]):
                            w=int(neighbors[p])
                            if w in local:
                                pairs.append((local[int(v)],local[w]))
                                if int(v)<w:
                                    cable+=float(weights[p])
                    pe=None
                    if compute_lpe:
                        edge_index=torch.tensor(pairs,dtype=torch.long).reshape(-1,2).T
                        pe=_window_laplacian_pos_enc(edge_index,len(nodes),DEFAULT_POS_DIM).numpy()
                    cached[center]=(nodes,radius,pe,cable)
            result=cached[center]
            if result is not None:
                nodes,radius,pe,cable=result
                cable_lengths[i]=cable
                valid[i]=True; radii[i]=radius
                parts.append(nodes)
                if pe is not None: pes.append(pe)
        bounds.append(bounds[-1]+(k if valid[i] else 0))
    return dict(new_synapse_id=points.synapse_id.to_numpy(np.int64),
                new_nearest_observed_node=np.asarray(centers,np.int32),
                new_nearest_observed_distance_nm=np.asarray(distances,np.float32),
                new_valid_k_window=valid,new_radius_nm=radii,
                new_cable_length_nm=cable_lengths,
                new_window_offsets=np.array(bounds,np.int64),
                new_window_members=np.concatenate(parts) if parts else np.empty(0,np.int32),
                new_window_lpe=np.concatenate(pes) if pes else np.empty((0,DEFAULT_POS_DIM),np.float32))


def windows(rid,out,manifest):
    source=Path(manifest['source'])
    embedding=source/'named_embeddings'/NAME/RUN/CHECKPOINT/f'{rid}.npz'
    # Validate full inference before producing any training artifact. Features
    # are shared across all six resolutions; do not replicate them six times.
    with np.load(embedding) as z:
        if str(z['skeleton_name'])!=NAME or str(z['run_id'])!=RUN or str(z['checkpoint_id'])!=CHECKPOINT:
            raise ValueError('Wrong embedding provenance')
        ids=z['node_ids']; x=z['embeddings']
        if int(z['root_id']) != rid or x.shape!=(len(ids),64) or not np.isfinite(x).all():
            raise ValueError('Invalid embedding arrays')
        np.testing.assert_array_equal(ids,np.arange(len(ids)))
    points=pd.read_parquet(out/'sites'/f'{rid}.parquet')
    for factor,k in SCALE_WINDOWS.items():
        geometry=out/'topology'/f'scale{factor}'/f'{rid}.npz'
        with np.load(geometry) as z:
            if not bool(z['soma_cut_applied']): raise ValueError('Missing soma cut')
            if int(z['source_node_count'])!=len(ids): raise ValueError('Incomplete inference')
            arrays=window_arrays(points,z['pos_nm'],z['edges'],z['edge_length_nm'],k)
        _atomic_savez(config_dir(out,factor)/'cells'/f'{rid}.npz',
                      root_id=rid,k_observed=k,factor=factor,skeleton_name=NAME,
                      geometry_path=str(geometry.resolve()),embedding_path=str(embedding.resolve()),**arrays)


def native(rid,out,manifest):
    """Geometry-only scale-1 adapter for the existing skeleton-statistics code."""
    points=pd.read_parquet(out/'sites'/f'{rid}.parquet')
    with np.load(out/'topology/scale1'/f'{rid}.npz') as geo:
        if not bool(geo['soma_cut_applied']): raise ValueError('Missing soma cut')
        pos=geo['pos_nm']; edges=geo['edges']
        xyz=points[['cell_x_nm','cell_y_nm','cell_z_nm']].to_numpy(float)
        distance=np.full(len(points),np.inf)
        nearest=np.full(len(points),-1,np.int32)
        if len(pos) and len(points): distance,nearest=cKDTree(pos).query(xyz)
        _atomic_savez(out/'native/cells'/f'{rid}.npz',root_id=rid,
                      root_xyz_nm=geo['root_xyz_nm'],new_pos_nm=pos,new_edges=edges,
                      new_edge_length_nm=geo['edge_length_nm'],
                      new_synapse_id=points.synapse_id.to_numpy(np.int64),
                      new_nearest_observed_node=np.asarray(nearest,np.int32),
                      new_within_match_cutoff=distance<=SYNAPSE_MATCH_CUTOFF_NM,
                      new_nearest_observed_distance_nm=distance,
                      original_node_ids=geo['original_node_ids'],skeleton_name=NAME)


def native_finalize(out,manifest):
    eligible=set(manifest['proofread_root_ids']) & {int(r) for r in manifest['soma_positions']}
    built={int(p.stem) for p in (out/'native/cells').glob('*.npz')}
    missing=sorted(eligible-built)
    _atomic_json(out/'native/manifest.json',manifest)
    _atomic_json(out/'native/metadata.json',dict(soma_radius_nm=SOMA_RADIUS_NM,
                 match_cutoff_nm=SYNAPSE_MATCH_CUTOFF_NM,skeleton_name=NAME,scale=1,
                 n_cells=len(built),missing_root_ids=missing,embeddings_required=False))
    if missing: raise RuntimeError(f'{len(missing)} native cells missing')


def finalize(out,manifest):
    eligible=set(manifest['proofread_root_ids']) & {int(r) for r in manifest['soma_positions']}
    for factor,k in SCALE_WINDOWS.items():
        directory=config_dir(out,factor)
        built={int(p.stem) for p in (directory/'cells').glob('*.npz')}
        missing=sorted(eligible-built)
        _atomic_json(directory/'manifest.json',manifest)
        _atomic_json(directory/'metadata.json',dict(format='dense-presynaptic-v1',factor=factor,
                     k_observed=k,skeleton_name=NAME,missing_root_ids=missing,n_cells=len(built)))
        if missing: raise RuntimeError(f'scale{factor}: {len(missing)} eligible cells missing')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    modes=ap.add_mutually_exclusive_group(required=True)
    for mode in ['prepare','topology','sites','windows','finalize','native','native-finalize']: modes.add_argument('--'+mode,action='store_true')
    ap.add_argument('--out',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--source',type=Path,default=SOURCE)
    ap.add_argument('--task-id',type=int,default=0)
    ap.add_argument('--num-tasks',type=int,default=1)
    ap.add_argument('--limit',type=int)
    args=ap.parse_args()
    if args.prepare: return prepare(args.out,args.source)
    manifest=json.loads((args.out/'manifest.json').read_text())
    if args.finalize: return finalize(args.out,manifest)
    if args.native_finalize: return native_finalize(args.out,manifest)
    mode=next(m for m in ['topology','sites','windows','native'] if getattr(args,m))
    roots=sorted(int(r) for r in manifest['cells'])
    if mode!='topology':
        roots=sorted(set(roots)&set(manifest['proofread_root_ids'])&{int(r) for r in manifest['soma_positions']})
    roots=roots[args.task_id::args.num_tasks]
    if args.limit is not None: roots=roots[:args.limit]
    errors=[]
    for rid in roots:
        marker=(args.out/'topology'/f'{rid}.json' if mode=='topology' else
                args.out/'sites'/f'{rid}.parquet' if mode=='sites' else
                args.out/'native/cells'/f'{rid}.npz' if mode=='native' else
                config_dir(args.out,32)/'cells'/f'{rid}.npz')
        if marker.exists(): continue
        try:
            if mode=='sites': sites(rid,args.out)
            else: globals()[mode](rid,args.out,manifest)
            print(f'{mode}: {rid} done',flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            errors.append(dict(root_id=rid,error=str(exc)))
    _atomic_json(args.out/'status'/f'{mode}_{args.task_id}.json',dict(errors=errors,cells=len(roots)))
    if errors: raise RuntimeError(f'{len(errors)} cells failed; rerun to resume')


if __name__=='__main__': main()
