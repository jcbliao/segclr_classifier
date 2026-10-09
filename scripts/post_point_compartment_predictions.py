"""Parallel, identity-preserving postsynaptic point features and five-fold calls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.subcompartment_predict_all import load_models,HIERARCHY,sha,atomic_json,save_npz,EMBED_CKPT
from scripts.subcompartment_predict_fast import fold_predictor

DEFAULT_OUT=Path('/orcd/scratch/orcd/013/jcbliao/incoming_presynaptic_fragments_k10_collina_v2')
POINTS=Path('/orcd/scratch/orcd/013/jcbliao/postsynaptic_point_embeddings')


def load_root(root,parts):
    ids=[];vectors=[];positions=[]
    for part in sorted(parts,key=lambda p:p.get('start',0)):
        with np.load(part['path']) as z:
            if int(z['root_id'])!=root:raise ValueError('postsynaptic root mismatch')
            ids.append(z['synapse_ids'].copy());vectors.append(z['embeddings'].copy());positions.append(z['positions_nm'].copy())
    ids=np.concatenate(ids);vectors=np.concatenate(vectors);positions=np.concatenate(positions)
    if len(np.unique(ids))!=len(ids):raise ValueError('repeated synapse IDs in post cache')
    if vectors.shape!=(len(ids),64) or not np.isfinite(vectors).all():raise ValueError('invalid post embeddings')
    return root,ids,vectors,positions


def embed_supplement(out):
    import pandas as pd
    from scripts.generate_cave_embedding_augmentations import FastAugInference,CHECKPOINT
    if Path(CHECKPOINT).stem!=EMBED_CKPT:raise ValueError('supplement checkpoint mismatch')
    frame=pd.read_parquet(out/'chc_incoming_supplement.parquet')
    engine=FastAugInference('clean',0,0,batch_size=32,num_threads=8)
    for root,rows in frame.groupby('cell_root_id'):
        path=out/'post_point_supplement'/f'{int(root)}.npz'
        rows=rows.sort_values('synapse_id');xyz=rows[[f'cell_{a}_nm' for a in 'xyz']].to_numpy(np.float32)
        if path.exists():
            with np.load(path) as saved:
                if int(saved['root_id'])!=int(root) or str(saved['checkpoint_id'])!=EMBED_CKPT:raise ValueError('stale supplemental embedding identity')
                np.testing.assert_array_equal(saved['synapse_ids'],rows.synapse_id.to_numpy(np.int64))
                np.testing.assert_array_equal(saved['positions_nm'],xyz)
                if saved['embeddings'].shape!=(len(rows),64) or not np.isfinite(saved['embeddings']).all():raise ValueError('invalid supplemental embeddings')
            continue
        nodes=np.arange(len(rows));got,vectors=engine.embed(int(root),xyz,nodes)
        np.testing.assert_array_equal(got,nodes)
        if not np.isfinite(vectors).all():raise ValueError('nonfinite supplemental embeddings')
        save_npz(path,root_id=np.asarray(int(root),np.int64),synapse_ids=rows.synapse_id.to_numpy(np.int64),
            positions_nm=xyz,embeddings=vectors.astype(np.float32),checkpoint_id=EMBED_CKPT)
        print(json.dumps(dict(root_id=int(root),embeddings=len(rows))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--tasks',type=int,default=32)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--supplement-only',action='store_true')
    p.add_argument('--embed-supplement',action='store_true');a=p.parse_args()
    if a.embed_supplement:embed_supplement(a.output);return
    plan=json.loads((a.output/'plan.json').read_text());digest=sha(a.output/'plan.json')
    manifest=json.loads((POINTS/'validated_manifest.json').read_text())
    if Path(manifest['checkpoint']).stem!=EMBED_CKPT:raise ValueError('postsynaptic checkpoint mismatch')
    grouped=defaultdict(list)
    for part in manifest['parts']:grouped[part['root_id']].append(part)
    if a.supplement_only:
        grouped={int(path.stem):[dict(path=str(path),start=0)] for path in (a.output/'post_point_supplement').glob('*.npz')}
    expected=sorted(plan['neuron_roots'])
    if a.supplement_only:
        expected=sorted(set(pq.read_table(a.output/'chc_incoming_supplement.parquet',columns=['cell_root_id'])['cell_root_id'].to_pylist()))
    roots=[root for root in expected[a.rank::a.tasks] if root in grouped]
    missing=[root for root in expected[a.rank::a.tasks] if root not in grouped]
    torch.set_num_threads(a.workers);models=load_models(plan,'cpu');predict=fold_predictor(models,[m['classes'] for m in plan['models']])
    stats=dict(rank=a.rank,cells=0,points=0,missing_roots=missing,plan_sha256=digest)
    begin=time.perf_counter();destination=a.output/'postsynaptic_points';destination.mkdir(exist_ok=True)
    # Bounded prefetch: two decoded cells, four BLAS threads for batched inference.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[]
        pending=iter(roots)
        for _ in range(2):
            root=next(pending,None)
            if root is not None:futures.append(pool.submit(load_root,root,grouped[root]))
        while futures:
            root,ids,x,positions=futures.pop(0).result();next_root=next(pending,None)
            if next_root is not None:futures.append(pool.submit(load_root,next_root,grouped[next_root]))
            dest=destination/f'{root}.parquet'
            if not dest.exists():
                logits=np.empty((5,len(ids),4),np.float32)
                with torch.inference_mode():
                    for start in range(0,len(ids),8192):
                        logits[:,start:start+8192]=predict(torch.from_numpy(x[start:start+8192])).numpy()
                table=pa.table(dict(cell_root_id=pa.array(np.repeat(root,len(ids)),pa.int64()),synapse_id=pa.array(ids,pa.int64()),
                    post_embedding=pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1)),64),
                    post_position_nm=pa.FixedSizeListArray.from_arrays(pa.array(positions.reshape(-1)),3),
                    postsynaptic_n_embeddings_used=pa.array(np.ones(len(ids),np.int16))))
                for fold in range(5):
                    table=table.append_column(f'postsynaptic_fold{fold}_logits',pa.FixedSizeListArray.from_arrays(pa.array(logits[fold].reshape(-1)),4))
                    table=table.append_column(f'postsynaptic_fold{fold}_class',pa.array(np.asarray(HIERARCHY.level_classes[0])[logits[fold].argmax(1)]))
                table=table.replace_schema_metadata({b'plan_sha256':digest.encode(),b'logit_classes':json.dumps(list(HIERARCHY.level_classes[0])).encode()})
                temp=dest.with_suffix(f'.tmp.{os.getpid()}.parquet');pq.write_table(table,temp,compression='zstd');temp.replace(dest)
            elif pq.read_schema(dest).metadata.get(b'plan_sha256')!=digest.encode():raise ValueError('stale postsynaptic output')
            stats['cells']+=1;stats['points']+=len(ids);stats['seconds']=time.perf_counter()-begin
            atomic_json(a.output/'post_status'/f"{'supplement_' if a.supplement_only else ''}{a.rank:03d}.json",stats)
            print(json.dumps(stats),flush=True)


if __name__=='__main__':main()
