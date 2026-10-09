"""Build an audited k<=10 presynaptic partner-fragment database and five-fold logits.

Each record is an incoming synapse onto a classified neuron. Skeletons belong
to its presynaptic partner. Start at the nearest skeleton node, grow along
geodesic edges, and mean-pool up to ten nodes in that connected component.
No soma, distance, axon-proofreading, or minimum component-size exclusions.
Missing skeletons/embeddings are reported, never generated or imputed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import heapq
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.subcompartment_predict_all import (
    DB_ROOT, EMBED_RUN, EMBED_CKPT, NEURONS, HIERARCHY, OUT as MODEL_PLAN_OUT,
    atomic_json, load_models, sha,
    unique_embedding_rows,
)

OUT=Path('/orcd/scratch/orcd/013/jcbliao/incoming_presynaptic_fragments_k10_v1')


def nearest_fragment(center, adjacency, k=10):
    """Shortest-path order, deterministic node-id tie break, undersized allowed."""
    heap=[(0.,int(center))];seen=set();members=[]
    while heap and len(members)<k:
        distance,node=heapq.heappop(heap)
        if node in seen:continue
        seen.add(node);members.append(node)
        for other,weight in adjacency[node]:
            if other not in seen:heapq.heappush(heap,(distance+weight,other))
    return np.asarray(members,np.int32)


def fragments_for_root(synapses,nodes,edges,embeddings):
    """Return identity-preserving rows, explicit coverage failures, and means."""
    node_ids=np.asarray(nodes['node_id'],np.int32)
    positions=np.column_stack([np.asarray(nodes[f'{axis}_nm']) for axis in 'xyz']).astype(np.float64)
    order=np.argsort(node_ids);node_ids=node_ids[order];positions=positions[order]
    if len(np.unique(node_ids))!=len(node_ids):raise ValueError('duplicate skeleton node IDs')
    adjacency=[[] for _ in node_ids]
    src=np.asarray(edges['src'],np.int32);dst=np.asarray(edges['dst'],np.int32)
    if len(src):
        ends=np.column_stack([np.searchsorted(node_ids,src),np.searchsorted(node_ids,dst)])
        if ends.max()>=len(node_ids) or not np.array_equal(node_ids[ends],np.column_stack([src,dst])):
            raise ValueError('edge endpoints absent from skeleton')
        for a,b in ends:
            weight=float(np.linalg.norm(positions[a]-positions[b]))
            adjacency[a].append((int(b),weight));adjacency[b].append((int(a),weight))
    eids=np.asarray(embeddings['node_id'],np.int32)
    vectors=np.asarray(embeddings['embedding'].combine_chunks().values).reshape(-1,64)
    eids,vectors=unique_embedding_rows(eids,vectors)
    if not np.isfinite(vectors).all():raise ValueError('nonfinite node embeddings')
    xyz=np.column_stack([np.asarray(synapses[f'partner_{axis}_nm']) for axis in 'xyz']).astype(np.float64)
    n=len(synapses);members=[[] for _ in range(n)];counts=np.zeros(n,np.int16)
    missing=np.zeros(n,np.int16);anchor=np.full(n,-1,np.int32);distance=np.full(n,np.nan,np.float32)
    means=np.full((n,64),np.nan,np.float32);status=np.repeat('missing_skeleton',n).astype(object)
    raw_fragments=[np.empty((0,64),np.float32) for _ in range(n)]
    if len(node_ids):
        distance,centers=cKDTree(positions).query(xyz)
        cache={}
        for row,center in enumerate(centers):
            center=int(center);anchor[row]=node_ids[center]
            if center not in cache:
                local=nearest_fragment(center,adjacency);ids=node_ids[local]
                loc=np.searchsorted(eids,ids);valid=loc<len(eids)
                valid[valid]&=eids[loc[valid]]==ids[valid]
                num_missing=int((~valid).sum())
                mean=vectors[loc].mean(0,dtype=np.float64).astype(np.float32) if not num_missing else None
                raw=vectors[loc] if not num_missing else np.empty((0,64),np.float32)
                cache[center]=(ids.tolist(),num_missing,mean,raw)
            ids,num_missing,mean,raw=cache[center]
            members[row]=ids;counts[row]=len(ids);missing[row]=num_missing
            status[row]='missing_embeddings' if num_missing else 'ready'
            if mean is not None:means[row]=mean;raw_fragments[row]=raw
    result={name:synapses[name] for name in ['synapse_id','cell_root_id','partner_root_id']}
    result.update(anchor_node_id=pa.array(anchor),anchor_distance_nm=pa.array(distance),
        fragment_node_ids=pa.array(members,type=pa.list_(pa.int32())),node_count=pa.array(counts),
        n_embeddings_available=pa.array(counts-missing),
        n_embeddings_used=pa.array(np.where(status=='ready',counts,0).astype(np.int16)),
        missing_embedding_count=pa.array(missing),status=pa.array(status.tolist()),
        mean_embedding=pa.FixedSizeListArray.from_arrays(pa.array(means.reshape(-1)),64))
    offsets=np.r_[0,np.cumsum([len(v) for v in raw_fragments])].astype(np.int32)
    flat=np.concatenate(raw_fragments,axis=0) if n else np.empty((0,64),np.float32)
    result['fragment_embeddings']=pa.ListArray.from_arrays(pa.array(offsets),
        pa.FixedSizeListArray.from_arrays(pa.array(flat.reshape(-1)),64))
    return pa.table(result)


def prepare(out,tasks,prior_predictions=None,source=None):
    import duckdb
    import lance
    from segclr_db import store as st
    out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():
        existing=json.loads((out/'plan.json').read_text())
        actual_source=Path(existing['source'])
        if source and source.resolve()!=actual_source.resolve():raise ValueError('existing source differs')
        if existing['tasks']!=tasks or actual_source.stat().st_size!=existing['source_size'] or actual_source.stat().st_mtime_ns!=existing['source_mtime_ns']:
            raise ValueError('existing preparation differs; use a new output directory')
        if existing.get('prior_predictions')!=(str(prior_predictions.resolve()) if prior_predictions else None):
            raise ValueError('existing root cohort differs; use a new output directory')
        print('Reusing frozen plan '+sha(out/'plan.json'),flush=True)
        return
    previous=json.loads((MODEL_PLAN_OUT/'plan.json').read_text())
    roots=sorted(c['root_id'] for c in previous['cells'] if c['cell_type'] in NEURONS)
    source=source or ROOT/'data/synapse_cache/postsynaptic_sites.parquet'
    c=duckdb.connect();c.execute('SET threads=4')
    c.register('neurons',pa.table({'root_id':pa.array(roots,pa.int64())}))
    c.read_parquet(str(source)).create_view('source')
    query='SELECT s.* FROM source s JOIN neurons n ON s.cell_root_id=n.root_id'
    prior_metadata=None
    if prior_predictions:
        c.read_parquet(str(prior_predictions)).create_view('prior')
        # Only use presynaptic roots previously predicted at synapses onto neurons.
        # Root IDs define the cohort; retain every cached synapse from those roots.
        c.execute('CREATE TABLE prior_partners AS SELECT DISTINCT p.subject_root_id AS root_id FROM prior p JOIN neurons n ON p.root_id=n.root_id')
        query+=' JOIN prior_partners p ON p.root_id=s.partner_root_id'
        c.execute('CREATE TABLE prior_targets AS SELECT DISTINCT root_id FROM prior')
        query+=' JOIN prior_targets t ON t.root_id=s.cell_root_id'
        source_json=prior_predictions.parent/'collina_source.json'
        prior_metadata=json.loads(source_json.read_text()) if source_json.exists() else None
    selected=c.sql(query).arrow().read_all()
    assert len(selected)==len(np.unique(np.asarray(selected['synapse_id'])))
    partners=np.unique(np.asarray(selected['partner_root_id']))
    boundaries=np.linspace(0,len(partners),tasks+1,dtype=int)
    shards=[]
    for rank in range(tasks):
        rids=partners[boundaries[rank]:boundaries[rank+1]]
        if not len(rids):continue
        c.register('selected',selected)
        subset=c.sql(f'SELECT * FROM selected WHERE partner_root_id BETWEEN {int(rids[0])} AND {int(rids[-1])} ORDER BY partner_root_id,synapse_id').arrow().read_all()
        path=out/'cohort'/f'{rank:03d}.parquet';path.parent.mkdir(exist_ok=True)
        pq.write_table(subset,path,compression='zstd')
        shards.append(dict(rank=rank,lo=int(rids[0]),hi=int(rids[-1]),roots=len(rids),synapses=len(subset),source=str(path)))
    store=st.open_store(DB_ROOT,'microns');tables={}
    for name,dim in [('skeleton_nodes',None),('skeleton_edges',None),('node_embeddings',64)]:
        ds=st.open_table(store,name,dim)
        tables[name]=dict(uri=ds.uri,version=ds.version)
    plan=dict(models=previous['models'],logit_classes=list(HIERARCHY.level_classes[0]),
        embedding_run_id=EMBED_RUN,embedding_checkpoint_id=EMBED_CKPT,
        source=str(source),source_size=source.stat().st_size,source_mtime_ns=source.stat().st_mtime_ns,
        mat_version=1718,neuron_roots=roots,synapses=len(selected),partner_roots=len(partners),
        tasks=tasks,shards=shards,tables=tables,k=10,aggregation='arithmetic mean of up to 10 nearest geodesic nodes',
        skeleton='default CAVE partner skeleton',soma_exclusion_nm=None,match_cutoff_nm=None,
        proofreading_filter=False,missing_embeddings='report; do not impute or generate',
        classifier_training_soma_exclusion_nm=10000,
        prior_predictions=str(prior_predictions.resolve()) if prior_predictions else None,
        prior_predictions_sha256=sha(prior_predictions) if prior_predictions else None,
        prior_prediction_source=prior_metadata)
    atomic_json(out/'plan.json',plan)
    print(json.dumps({k:plan[k] for k in ['synapses','partner_roots','tasks','tables']}),flush=True)


def grouped(table):
    if not len(table):return {}
    ids=np.asarray(table['root_id']);order=np.argsort(ids,kind='stable')
    table=table.take(pa.array(order));ids=ids[order]
    starts=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1,len(ids)]
    return {int(ids[a]):table.slice(int(a),int(b-a)) for a,b in zip(starts[:-1],starts[1:])}


def build(out,rank,workers,limit_batches=0):
    import duckdb
    import lance
    from scripts.subcompartment_predict_fast import fold_predictor
    import torch
    plan=json.loads((out/'plan.json').read_text());digest=sha(out/'plan.json')
    shard=next(s for s in plan['shards'] if s['rank']==rank)
    cohort=pq.read_table(shard['source'])
    c=duckdb.connect();c.execute(f'SET threads={workers}');c.execute("SET memory_limit='20GB'")
    c.execute('SET temp_directory=?',[str(out/f'tmp_{rank}')])
    c.register('cohort',cohort)
    roots=c.sql('SELECT DISTINCT partner_root_id AS root_id FROM cohort').arrow().read_all()
    c.register('wanted',roots)
    torch.set_num_threads(workers)
    models=load_models(plan,'cpu');predict=fold_predictor(models,[m['classes'] for m in plan['models']])
    datasets={name:lance.dataset(info['uri'],version=info['version']) for name,info in plan['tables'].items()}
    root_ids=np.sort(np.asarray(roots['root_id']));begin=time.perf_counter()
    out_parts=out/'parts'/f'{rank:03d}';out_parts.mkdir(parents=True,exist_ok=True)
    stats=dict(rank=rank,synapses=0,ready=0,missing_skeleton=0,missing_embeddings=0,
        fewer_than_10=0,missing_embedding_nodes=0,parts=0,seconds=0.,plan_sha256=digest)
    # Range scans prune Lance fragments without requiring private scalar indices.
    # Each root belongs to one batch; selected roots are joined before Python decoding.
    for batch,start in enumerate(range(0,len(root_ids),2000)):
        if limit_batches and batch>=limit_batches:break
        dest=out_parts/f'{batch:05d}.parquet'
        if dest.exists() and 'fragment_embeddings' in pq.read_schema(dest).names:
            table=pq.read_table(dest)
        else:
            ids=root_ids[start:start+2000];lo,hi=int(ids[0]),int(ids[-1])
            batch_cohort=cohort.filter(pc.and_(pc.greater_equal(cohort['partner_root_id'],lo),pc.less_equal(cohort['partner_root_id'],hi)))
            c.register('wanted_batch',pa.table({'root_id':ids}))
            loaded={}
            for name,columns in [('skeleton_nodes',['root_id','node_id','x_nm','y_nm','z_nm']),
                                 ('skeleton_edges',['root_id','src','dst']),
                                 ('node_embeddings',['root_id','node_id','embedding'])]:
                clause=f'root_id >= {lo} AND root_id <= {hi}'
                if name=='node_embeddings':clause+=f" AND run_id = '{EMBED_RUN}' AND checkpoint_id = '{EMBED_CKPT}'"
                scanner=datasets[name].scanner(columns=columns,filter=clause,use_scalar_index=False,
                    batch_size=65536,batch_readahead=1,fragment_readahead=2,io_buffer_size=64*1024**2)
                c.register('rows',scanner.to_reader())
                loaded[name]=grouped(c.sql('SELECT r.* FROM rows r JOIN wanted_batch w USING(root_id)').arrow().read_all())
                c.unregister('rows')
            ids_in=np.asarray(batch_cohort['partner_root_id']);bounds=np.r_[0,np.flatnonzero(ids_in[1:]!=ids_in[:-1])+1,len(ids_in)]
            def one(bounds_pair):
                a,b=bounds_pair;rid=int(ids_in[a]);syn=batch_cohort.slice(int(a),int(b-a))
                defaults={
                    'skeleton_nodes':pa.table({'node_id':pa.array([],pa.int32()),**{f'{axis}_nm':pa.array([],pa.float32()) for axis in 'xyz'}}),
                    'skeleton_edges':pa.table({'src':pa.array([],pa.int32()),'dst':pa.array([],pa.int32())}),
                    'node_embeddings':pa.table({'node_id':pa.array([],pa.int32()),'embedding':pa.array([],pa.list_(pa.float32(),64))})}
                return fragments_for_root(syn,*[loaded[name].get(rid,defaults[name]) for name in ['skeleton_nodes','skeleton_edges','node_embeddings']])
            with ThreadPoolExecutor(max_workers=workers) as pool:
                table=pa.concat_tables(list(pool.map(one,zip(bounds[:-1],bounds[1:]))))
            ready=np.asarray(table['status'])=='ready';features=np.asarray(table['mean_embedding'].combine_chunks().values).reshape(-1,64)
            logits=np.full((len(table),5,4),np.nan,np.float32)
            valid=np.flatnonzero(ready)
            with torch.inference_mode():
                for pos in range(0,len(valid),32768):
                    index=valid[pos:pos+32768]
                    logits[index]=predict(torch.from_numpy(features[index].copy())).numpy().transpose(1,0,2)
            if not np.isfinite(logits[ready]).all():raise ValueError('nonfinite predictions')
            for fold in range(5):
                table=table.append_column(f'fold{fold}_logits',pa.FixedSizeListArray.from_arrays(pa.array(logits[:,fold].reshape(-1)),4))
            table=append_presynaptic_classifications(table,plan['logit_classes'])
            metadata={b'plan_sha256':digest.encode(),b'logit_classes':json.dumps(plan['logit_classes']).encode()}
            table=table.replace_schema_metadata(metadata)
            tmp=dest.with_suffix(f'.tmp.{os.getpid()}');pq.write_table(table,tmp,compression='zstd');tmp.replace(dest)
        if table.schema.metadata.get(b'plan_sha256')!=digest.encode():raise ValueError('stale output plan')
        status=np.asarray(table['status']);stats['synapses']+=len(table)
        for label in ['ready','missing_skeleton','missing_embeddings']:stats[label]+=int((status==label).sum())
        stats['fewer_than_10']+=int(((np.asarray(table['node_count'])<10)&(status=='ready')).sum())
        stats['missing_embedding_nodes']+=int(np.asarray(table['missing_embedding_count']).sum())
        stats['parts']+=1;stats['seconds']=time.perf_counter()-begin
        atomic_json(out/'status'/f'{rank:03d}.json',stats);print(json.dumps(stats),flush=True)
    suffix='pilot' if limit_batches else 'complete'
    atomic_json(out/'status'/f'{rank:03d}_{suffix}.json',stats)


def finalize(out):
    import duckdb
    plan=json.loads((out/'plan.json').read_text());reports=[]
    for shard in plan['shards']:
        report=json.loads((out/'status'/f"{shard['rank']:03d}_complete.json").read_text())
        if report['synapses']!=shard['synapses']:raise ValueError('incomplete shard')
        reports.append(report)
    summary={key:sum(r[key] for r in reports) for key in ['synapses','ready','missing_skeleton','missing_embeddings','fewer_than_10','parts']}
    if summary['synapses']!=plan['synapses']:raise ValueError('incomplete cohort')
    summary.update(plan_sha256=sha(out/'plan.json'),folds=5,logit_classes=plan['logit_classes'],
        dataset_glob=str(out/'parts/*/[0-9][0-9][0-9][0-9][0-9].parquet'),coverage_complete=not(summary['missing_skeleton'] or summary['missing_embeddings']))
    c=duckdb.connect(str(out/'fragments.duckdb'))
    c.read_parquet(summary['dataset_glob']).create_view('fragments',replace=True)
    c.execute("CREATE OR REPLACE VIEW classified_fragments AS SELECT * FROM fragments WHERE status='ready'")
    for fold in range(5):
        c.execute(f'CREATE OR REPLACE VIEW fold{fold}_axon_fragments AS SELECT * FROM classified_fragments WHERE presynaptic_fold{fold}_is_axon')
    summary['axon_filter']='hard argmax == axon, separately for each fold; applied to the mean of up to ten embeddings'
    summary['axon_counts_by_fold']={str(fold):c.sql(f'SELECT count(*) FROM fold{fold}_axon_fragments').fetchone()[0] for fold in range(5)}
    distribution=c.sql('SELECT status,n_embeddings_used,count(*) AS synapses FROM fragments GROUP BY status,n_embeddings_used ORDER BY status,n_embeddings_used').fetchdf()
    distribution.to_csv(out/'embedding_count_distribution.csv',index=False)
    summary['embedding_count_distribution']=distribution.to_dict('records')
    c.close()
    summary['database']=str(out/'fragments.duckdb')
    atomic_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)


def append_presynaptic_classifications(table,classes):
    """Hard per-fold axon membership, with null labels for missing features."""
    classes=np.asarray(classes);ready=np.asarray(table['status'])=='ready'
    votes=np.zeros(len(table),np.int8)
    for fold in range(5):
        logits=np.asarray(table[f'fold{fold}_logits'].combine_chunks().values).reshape(-1,len(classes))
        labels=np.empty(len(table),object);labels[:]=None
        labels[ready]=classes[logits[ready].argmax(1)]
        is_axon=ready&(labels=='axon');votes+=is_axon
        table=table.append_column(f'presynaptic_fold{fold}_class',pa.array(labels.tolist(),pa.string()))
        table=table.append_column(f'presynaptic_fold{fold}_is_axon',pa.array(is_axon))
    table=table.append_column('presynaptic_axon_fold_count',pa.array(votes))
    return table.append_column('presynaptic_axon_all_folds',pa.array(votes==5))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','build','finalize'])
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--tasks',type=int,default=32)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit-batches',type=int,default=0)
    p.add_argument('--prior-predictions',type=Path)
    p.add_argument('--source',type=Path)
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.output,a.tasks,a.prior_predictions,a.source)
    elif a.stage=='build':build(a.output,a.rank,a.workers,a.limit_batches)
    else:finalize(a.output)


if __name__=='__main__':main()
