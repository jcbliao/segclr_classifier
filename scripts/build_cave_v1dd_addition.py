"""Build matched K=10 presynaptic CAVE datasets with/without V1DD training cells."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
BASE = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/cave_v1dd_addition')
INFERENCE = Path('/orcd/scratch/orcd/013/jcbliao/segclr/v1dd_candidates_v1196')
MICRONS = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/k10')
FOLDS = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_coarse_confidence_with_tc_folds/scale16/k17/conf0')
N_FOLDS = 5
RUN_FOLDS = (0,)
K = 10


def save_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(payload,indent=2)+'\n')
    tmp.replace(path)


def cohort():
    frame = pd.read_parquet(INFERENCE/'cohort.parquet')
    assert len(frame)==json.loads((INFERENCE/'manifest.json').read_text())['n_cells'] and frame.status_axon.eq(True).all()
    return frame


def settings():
    metadata = json.loads((MICRONS/'metadata.json').read_text())
    assert metadata['soma_radius_nm']==5000 and metadata['k_observed']==K
    cutoff = metadata.get('synapse_match_cutoff_nm',metadata.get('match_cutoff_nm'))
    if cutoff is None:
        raise ValueError('MICrONS cache does not record a synapse cutoff')
    return float(cutoff)


def topology_sites(rank,world):
    from segclr_db import store as st
    from segclr_db.skeletons import SkeletonCache
    from caveclient import CAVEclient
    from data.build_presynaptic_axon_database import _induced_cut,_atomic_savez
    from data.synapses import fetch_synapses
    BASE.mkdir(parents=True,exist_ok=True)
    for folder in ['topology','sites','cells','status']:
        (BASE/folder).mkdir(exist_ok=True)
    cache = SkeletonCache(st.open_store('/orcd/compute/sdorkenw/001/segclr-db','v1dd'))
    client = CAVEclient('v1dd_public',server_address='https://global.em.brain.allentech.org',version=1196)
    rows = []
    for row in list(cohort().itertuples())[rank::world]:
        rid = int(row.pt_root_id)
        geo = BASE/'topology'/f'{rid}.npz'
        if not geo.exists():
            skeleton = cache.get_skeleton(rid,fetch_if_missing=False)
            soma = np.array([row.pt_position_x,row.pt_position_y,row.pt_position_z])
            assert np.isfinite(soma).all()
            root_node = int(cKDTree(skeleton.coords).query(soma)[1])
            root_xyz = skeleton.coords[root_node]
            keep,remap,pos,edges,weights,distance = _induced_cut(skeleton.coords,skeleton.edges,root_xyz)
            _atomic_savez(geo,root_id=np.array(rid),root_xyz_nm=root_xyz,root_cave_node_original=np.array(root_node),
                original_node_ids=np.flatnonzero(keep),pos_nm=pos,edges=edges,edge_length_nm=weights)
        path = BASE/'sites'/f'{rid}.parquet'
        if path.exists():
            sites = pd.read_parquet(path)
        else:
            sites = fetch_synapses(client,[rid],'outgoing',synapse_table='synapses_v1dd',sleep_s=.2)
            tmp = path.with_suffix('.tmp.parquet')
            sites.to_parquet(tmp,index=False)
            tmp.replace(path)
        assert sites.empty or sites.cell_root_id.eq(rid).all()
        assert not sites.synapse_id.duplicated().any()
        rows.append(dict(root_id=rid,n_sites=len(sites)))
        print('PREPARED',rid,len(sites),flush=True)
    save_json(BASE/'status'/f'topology_sites_{rank}.json',{'cells':rows})


def windows(rank,world):
    from data.build_presynaptic_axon_database import _window_arrays,_atomic_savez
    verified = json.loads((INFERENCE/'verified_summary.json').read_text())
    assert verified['status']=='verified' and verified['n_cells']==len(cohort())
    cutoff = settings()
    rows = []
    for row in list(cohort().itertuples())[rank::world]:
        rid = int(row.pt_root_id)
        target = BASE/'cells'/f'{rid}.npz'
        if target.exists():
            with np.load(target) as z:
                assert int(z['synapse_match_cutoff_nm'])==cutoff and int(z['k_observed'][0])==K
        else:
            points = pd.read_parquet(BASE/'sites'/f'{rid}.parquet')
            with np.load(BASE/'topology'/f'{rid}.npz') as geo, np.load(INFERENCE/'embeddings'/f'{rid}.npz') as emb:
                ids = geo['original_node_ids']
                np.testing.assert_array_equal(emb['node_ids'],np.arange(len(emb['node_ids'])))
                pos,edges,weights = geo['pos_nm'],geo['edges'],geo['edge_length_nm']
                observed = np.ones(len(pos),bool)
                windows = _window_arrays(points,pos,edges,weights,observed,K,match_cutoff_nm=cutoff)
                arrays = dict(root_id=np.array([rid],np.uint64),k_observed=np.array([K],np.int32),
                    root_xyz_nm=geo['root_xyz_nm'],root_cave_node_original=geo['root_cave_node_original'],
                    synapse_match_cutoff_nm=np.array(cutoff),cave_pos_nm=pos,cave_edges=edges,
                    cave_edge_length_nm=weights,cave_observed_node_ids=np.arange(len(pos),dtype=np.int32),
                    cave_observed_x=emb['embeddings'][ids],cave_original_node_ids=ids)
                arrays.update({f'cave_{key}':value for key,value in windows.items()})
                _atomic_savez(target,**arrays)
        with np.load(target) as z:
            n_valid = int(z['cave_valid_k_window'].sum())
            rows.append(dict(root_id=rid,n_valid_windows=n_valid))
            print('WINDOWS',rid,n_valid,flush=True)
    save_json(BASE/'status'/f'windows_{rank}.json',{'cells':rows})


def unique_windows(path):
    with np.load(path) as z:
        valid = np.flatnonzero(z['cave_valid_k_window'])
        offsets,members = z['cave_window_offsets'],z['cave_window_members']
        keys = set()
        for row in valid:
            nodes = members[int(offsets[row]):int(offsets[row+1])]
            assert len(nodes)==K
            keys.add(np.sort(nodes).tobytes())
        return len(keys)


def v1dd_folds(frame):
    assignments = {}
    total_counts = [0]*N_FOLDS
    for family,group in frame.groupby('training_label',sort=True):
        class_counts = [0]*N_FOLDS
        for fine,sub in group.groupby('cell_type',sort=True):
            rng = random.Random(int.from_bytes(hashlib.sha256(fine.encode()).digest()[:8],'big'))
            roots = sorted(int(x) for x in sub.pt_root_id)
            rng.shuffle(roots)
            tie_order = list(range(N_FOLDS))
            rng.shuffle(tie_order)
            order = {fold:i for i,fold in enumerate(tie_order)}
            for rid in roots:
                fold = min(range(N_FOLDS),key=lambda f:(class_counts[f],total_counts[f],order[f]))
                assignments[rid]=fold
                class_counts[fold]+=1
                total_counts[fold]+=1
    return assignments


def finalize():
    frame = cohort()
    cutoff = settings()
    originals = [json.loads((FOLDS/f'fold{fold}'/'manifest.json').read_text()) for fold in RUN_FOLDS]
    microns_roots = {int(r) for r in originals[0]['cells']}
    v1dd_roots = set(map(int,frame.pt_root_id))
    assert not microns_roots & v1dd_roots
    inventory = []
    eligible = set()
    for rid in sorted(microns_roots|v1dd_roots):
        path = BASE/'cells'/f'{rid}.npz'
        domain = 'microns' if rid in microns_roots else 'v1dd'
        if domain=='microns':
            source = MICRONS/'cells'/path.name
            if not source.exists():
                raise FileNotFoundError(source)
            if not path.exists():
                path.symlink_to(source)
            assert path.resolve()==source.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        n_windows = unique_windows(path)
        inventory.append(dict(root_id=rid,source_dataset=domain,n_unique_windows=n_windows,
                              exclusion_reason='' if n_windows else 'no_valid_unique_presynaptic_window'))
        if n_windows:
            eligible.add(rid)
    pd.DataFrame(inventory).to_csv(BASE/'window_inventory.csv',index=False)
    vframe = frame[frame.pt_root_id.isin(eligible)]
    assignments = v1dd_folds(vframe)
    pd.DataFrame([dict(root_id=int(r.pt_root_id),fine_type=r.cell_type,training_label=r.training_label,held_out_fold=assignments[int(r.pt_root_id)])
                  for r in vframe.itertuples()]).to_csv(BASE/'v1dd_fold_assignments.csv',index=False)
    for fold,original in enumerate(originals):
        microns = {rid:{**info,'source_dataset':'microns'} for rid,info in original['cells'].items() if int(rid) in eligible}
        v1dd = {str(int(r.pt_root_id)):dict(cell_type=r.training_label,source_dataset='v1dd',
            nucleus_id=int(r.target_id),v1dd_fine_type=r.cell_type,
            split='test' if assignments[int(r.pt_root_id)]==fold else 'train') for r in vframe.itertuples()}
        for condition in ['no_v1dd','with_v1dd']:
            # Shared held-out cells in both conditions; only training membership differs.
            cells = {**microns,**{rid:info for rid,info in v1dd.items() if condition=='with_v1dd' or info['split']=='test'}}
            manifest = {**original,'cells':cells,'training_condition':condition,
                'casey_coarse_confidence_min':None,
                'source_database':str(BASE),'v1dd_materialization':1196,
                'comparison':'paired MICrONS+V1DD test sets; V1DD train cells present only in with_v1dd',
                'heuristics':dict(soma_radius_nm=5000,synapse_match_cutoff_nm=cutoff,k_observed=K,exact_membership_deduplication=True)}
            save_json(BASE/'manifests'/condition/f'fold{fold}.json',manifest)
        a = json.loads((BASE/'manifests/no_v1dd'/f'fold{fold}.json').read_text())['cells']
        b = json.loads((BASE/'manifests/with_v1dd'/f'fold{fold}.json').read_text())['cells']
        assert {r:i for r,i in a.items() if i['split']=='test'}=={r:i for r,i in b.items() if i['split']=='test'}
        assert {r:i for r,i in a.items() if i['source_dataset']=='microns'}=={r:i for r,i in b.items() if i['source_dataset']=='microns'}
        assert all(b[r]['source_dataset']=='v1dd' and b[r]['split']=='train' for r in set(b)-set(a))
    # Exercise the actual loader on both domains before releasing GPU training.
    from data.dataset_presynaptic import PresynapticWindowDataset
    example = json.loads((BASE/'manifests/with_v1dd/fold0.json').read_text())
    for domain in ['microns','v1dd']:
        candidates = {rid:info for rid,info in example['cells'].items() if info['source_dataset']==domain and info['split']=='train'}
        rid = next(iter(candidates))
        probe = {**example,'cells':{rid:candidates[rid]}}
        dataset = PresynapticWindowDataset(probe,'train','cave',database=BASE,cell_cache_size=1)
        graph = dataset[0]
        assert graph.x.shape==(K,64) and graph.pos_enc.shape[0]==K
        assert np.isfinite(graph.x.numpy()).all()
    metadata = dict(format='presynaptic-axon-npz-v1',complete=True,k_observed=K,n_folds=N_FOLDS,run_folds=list(RUN_FOLDS),
        microns_confidence_min=None,microns_source_database=str(MICRONS),microns_source_folds=str(FOLDS),
        v1dd_source_inference=str(INFERENCE),soma_radius_nm=5000,synapse_match_cutoff_nm=cutoff,
        v1dd_candidate_counts=frame.training_label.value_counts().to_dict(),
        v1dd_eligible_counts=vframe.training_label.value_counts().to_dict(),
        n_microns_eligible=len(microns_roots & eligible),
        excluded_root_ids=sorted((microns_roots|v1dd_roots)-eligible))
    save_json(BASE/'metadata.json',metadata)
    print(json.dumps(metadata,indent=2),flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['topology_sites','windows','finalize'])
    ap.add_argument('--rank',type=int,default=0)
    ap.add_argument('--world',type=int,default=1)
    args = ap.parse_args()
    assert 0<=args.rank<args.world
    if args.stage=='finalize':
        finalize()
    elif args.stage=='windows':
        windows(args.rank,args.world)
    else:
        topology_sites(args.rank,args.world)


if __name__=='__main__':
    main()
