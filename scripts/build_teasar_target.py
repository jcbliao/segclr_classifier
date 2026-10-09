"""Register rounded equal-edge subdivisions as teasar_target; geometry only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

from resample_teasar_dense import densify
from registered_teasar_pipeline import BASE, atomic_json, ingest_batch

NAME='teasar_target'
TARGET_NM=111.0
OUT=BASE.parent/'teasar_target'
DB='/orcd/compute/sdorkenw/001/segclr-db'


def source_path(row):
    return Path(row['source']) if row['source'] else BASE/'skeletons'/f"{row['root_id']}.npz"


def signature(path):
    s=path.stat()
    return dict(path=str(path.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns)


def prepare(row,out):
    from segclr_db.results import Skeleton
    rid=int(row['root_id']); source=source_path(row)
    stamp=signature(source)
    dest=out/'skeletons'/f'{rid}.npz'
    with np.load(source) as raw:
        assert int(raw['root_id'].item())==rid
        v,e,r=densify(raw['vertices'],raw['edges'],raw['radius'],TARGET_NM,subdivision='round')
        np.testing.assert_array_equal(v[:len(raw['vertices'])],raw['vertices'].astype(np.float32))
        raw_lengths=np.linalg.norm(raw['vertices'][raw['edges'][:,1]].astype(float)-raw['vertices'][raw['edges'][:,0]],axis=1)
        pieces=np.maximum(1,np.rint(raw_lengths/TARGET_NM).astype(np.int64))
        lengths=np.linalg.norm(v[e[:,1]].astype(float)-v[e[:,0]],axis=1)
        assert len(e)==int(pieces.sum())
        assert len(v)-len(e)==len(raw['vertices'])-len(raw['edges'])
        # World-coordinate float32 rounding can perturb subedges by <0.5 nm.
        np.testing.assert_allclose(lengths,np.repeat(raw_lengths/pieces,pieces),atol=.5,rtol=0)
        assert signature(source)==stamp, 'Source changed during resampling'
        report=dict(root_id=rid,skeleton_name=NAME,target_nm=TARGET_NM,
            subdivision='max(1, round(L/d)); equal subdivision',rounding_ties='to even',
            source=stamp,original_nodes=len(raw['vertices']),nodes=len(v),edges=len(e),
            edge_sum_nm=float(lengths.sum()),edge_sum_squared_nm2=float(np.square(lengths).sum()),
            edge_min_nm=float(lengths.min()) if len(e) else None,
            edge_max_nm=float(lengths.max()) if len(e) else None,
            edge_percentiles_nm=np.percentile(lengths,[0,5,25,50,75,95,100]).tolist() if len(e) else [],
            edges_below_target=int(np.sum(lengths<TARGET_NM)),
            edges_above_target=int(np.sum(lengths>TARGET_NM)),
            histogram_bins_nm=list(range(0,202)),
            histogram_counts=np.histogram(lengths,bins=np.arange(202))[0].tolist(),
            radius_method='linear interpolation of original clearance radii')
    if dest.exists():
        with np.load(dest) as old:
            assert float(old['target_spacing_nm'])==TARGET_NM
            for key,value in [('vertices',v),('edges',e),('radius',r)]:
                np.testing.assert_array_equal(old[key],value)
    else:
        tmp=dest.with_suffix('.partial.npz')
        np.savez_compressed(tmp,root_id=np.array([rid],np.uint64),vertices=v,edges=e,radius=r,
            original_node_count=report['original_nodes'],target_spacing_nm=TARGET_NM,
            subdivision='round',skeleton_name=NAME)
        tmp.replace(dest)
    return [Skeleton(root_id=rid,coords=v,edges=e,radii=r,skeleton_version=0,skeleton_name=NAME)],report


def main():
    from segclr_db import store as st
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['preflight','prepare','verify'])
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--task-id',type=int,default=0)
    p.add_argument('--num-tasks',type=int,default=1)
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    rows=json.loads((BASE/'routes.json').read_text())
    assert len(rows)==len({r['root_id'] for r in rows})==2442
    for sub in ('status','skeletons'):
        (a.out/sub).mkdir(parents=True,exist_ok=True)
    st.COMPACT_SMALL_FRAGMENTS=sys.maxsize
    store=st.open_store(DB,'microns')
    if a.mode=='preflight':
        missing=[str(source_path(r)) for r in rows if not source_path(r).is_file()]
        assert not missing, missing
        existing=st.scan(store,'named_skeletons',columns=['root_id','n_nodes'],
                         filter=f"skeleton_name = '{NAME}'").to_pylist()
        plan=dict(skeleton_name=NAME,target_nm=TARGET_NM,cells=len(rows),
            priority_cells=sum(r['priority'] for r in rows),database=DB,dataset='microns',
            source_routes=str(BASE/'routes.json'),source='original unresampled TEASAR',
            inference_requested=False,existing_named_rows=len(existing),
            rounding_ties='to even',sources={str(r['root_id']):signature(source_path(r)) for r in rows})
        atomic_json(a.out/'plan.json',plan)
        print(json.dumps({k:v for k,v in plan.items() if k!='sources'},indent=2),flush=True)
    elif a.mode=='prepare':
        plan=json.loads((a.out/'plan.json').read_text())
        selected=rows[a.task_id::a.num_tasks]
        if a.limit is not None: selected=selected[:a.limit]
        batch=[]; nodes=0
        for row in selected:
            rid=row['root_id']; stamp=signature(source_path(row))
            assert stamp==plan['sources'][str(rid)], 'Source differs from frozen plan'
            marker=a.out/'status'/f'prepared_{rid}.json'
            if marker.exists():
                saved=json.loads(marker.read_text())
                assert saved['source']==stamp and saved['target_nm']==TARGET_NM
                continue
            item=prepare(row,a.out); batch.append(item); nodes+=item[1]['nodes']
            if len(batch)>=8 or nodes>=1_000_000:
                ingest_batch(batch,store,status_dir=a.out/'status')
                batch=[]; nodes=0
        if batch: ingest_batch(batch,store,status_dir=a.out/'status')
        print(f'PREPARED rank {a.task_id}: {len(selected)} cells',flush=True)
    else:
        inventory=[]
        plan=json.loads((a.out/'plan.json').read_text())
        for row in rows:
            assert signature(source_path(row))==plan['sources'][str(row['root_id'])]
            inventory.append(json.loads((a.out/'status'/f"prepared_{row['root_id']}.json").read_text()))
        stored=st.scan(store,'named_skeletons',columns=['root_id','n_nodes','n_edges'],
                       filter=f"skeleton_name = '{NAME}'").to_pylist()
        assert len(stored)==2442 and {r['root_id'] for r in stored}=={r['root_id'] for r in rows}
        lookup={r['root_id']:r for r in inventory}
        for r in stored:
            assert r['n_nodes']==lookup[r['root_id']]['nodes'] and r['n_edges']==lookup[r['root_id']]['edges']
        edges=sum(r['edges'] for r in inventory)
        mean=sum(r['edge_sum_nm'] for r in inventory)/edges
        variance=sum(r['edge_sum_squared_nm2'] for r in inventory)/edges-mean**2
        summary=dict(skeleton_name=NAME,cells=len(stored),nodes=sum(r['nodes'] for r in inventory),
            edges=edges,target_nm=TARGET_NM,mean_edge_nm=mean,std_edge_nm=float(np.sqrt(max(0,variance))),
            min_edge_nm=min(r['edge_min_nm'] for r in inventory if r['edges']),
            max_edge_nm=max(r['edge_max_nm'] for r in inventory if r['edges']),
            edges_below_target=sum(r['edges_below_target'] for r in inventory),
            edges_above_target=sum(r['edges_above_target'] for r in inventory),
            database=DB,dataset='microns',geometry_readback_verified=True,inference_submitted=False,
            histogram_bins_nm=list(range(0,202)),
            histogram_counts=np.sum([r['histogram_counts'] for r in inventory],axis=0).tolist())
        atomic_json(a.out/'completion.json',summary)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
        bins=np.asarray(summary['histogram_bins_nm'])
        ax.bar(bins[:-1],summary['histogram_counts'],width=1,align='edge',color='#287c9e')
        ax.axvline(TARGET_NM,color='black',linestyle='--',label=f'Target {TARGET_NM:g} nm')
        ax.axvline(mean,color='#d16c2a',label=f'Mean {mean:.2f} nm')
        ax.set(xlabel='Edge length (nm)',ylabel='Edges',xlim=(0,180),
               title=f'teasar_target · {len(stored):,} cells · rounded equal subdivisions')
        ax.legend(); ax.grid(axis='y',alpha=.2)
        for ext in ('png','pdf'):
            fig.savefig(a.out/f'edge_length_histogram.{ext}',dpi=180)
        plt.close(fig)
        print(json.dumps({k:v for k,v in summary.items() if not k.startswith('histogram')},indent=2),flush=True)


if __name__=='__main__': main()
