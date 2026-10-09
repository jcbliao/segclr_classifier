"""Operational parity with trained classifiers and distinct synapse records."""
import numpy as np
import pyarrow as pa
import pytest
import torch
import json
import os
import pyarrow.parquet as pq
from scripts.predict_incoming_fragment_cell_types import (
    checkpoint,load_checkpoint,vectorized_predictor,padded_fragments,nullable_vectors,infer,sha,prepare_posts,writer_payload,finalize,
)

DEVICE=os.environ.get('FRAGMENT_TEST_DEVICE','cpu')


@pytest.mark.parametrize('family,classes,k',[('cave_n10','three',10),('single_pre_post','six',1)])
def test_vectorized_matches_training_path(family,classes,k):
    torch.set_num_threads(1);torch.manual_seed(73)
    models=[load_checkpoint(checkpoint(family,classes,f))[0] for f in range(5)]
    x=torch.randn(3,k,64,device=DEVICE);counts=torch.tensor([1,min(5,k),k],device=DEVICE)
    mask=(torch.arange(k,device=DEVICE)[None,:]<counts[:,None]).float();post=torch.randn(3,64,device=DEVICE)
    predict=vectorized_predictor(models,DEVICE)
    with torch.inference_mode():
        raw=predict(x,mask,post)
        decoded=models[0].cls_head.distribution_from_head_logits(tuple(h.reshape(15,-1) for h in raw))
        flat=x[mask.bool()];batch=torch.repeat_interleave(torch.arange(3,device=DEVICE),counts)
        for f,model in enumerate(models):
            hidden=model(flat,None,batch,postsynaptic_embedding=post)
            expected=model.cls_head.predict_distribution(hidden)
            torch.testing.assert_close(expected['predictions'],model.cls_head.predict_top_down(hidden))
            for head,a in zip(expected['head_logits'],raw):torch.testing.assert_close(head,a[f],rtol=2e-5,atol=2e-5)
            for level,probs in enumerate(expected['level_probabilities']):
                torch.testing.assert_close(probs.sum(1),torch.ones(3,device=DEVICE))
                torch.testing.assert_close(probs,decoded['level_probabilities'][level].reshape(5,3,-1)[f],rtol=2e-5,atol=2e-5)
            if classes=='six':
                # Conditional children sum to the parent's joint probability.
                for node,head in zip(model.cls_head.nodes,expected['head_logits']):
                    level=node['child_level'];parent=node['parent_global_idx']
                    parent_prob=expected['level_probabilities'][level-1][:,parent] if level else torch.ones(3,device=DEVICE)
                    got=expected['level_probabilities'][level][:,node['child_global_indices']]
                    torch.testing.assert_close(got,parent_prob[:,None]*head.softmax(-1))
                torch.testing.assert_close(expected['level_probabilities'][-1].sum(1),torch.ones(3,device=DEVICE))


def test_same_fragment_distinct_synapses_stay_separate():
    raw=np.arange(5*64,dtype=np.float32).reshape(5,64)
    values=pa.FixedSizeListArray.from_arrays(pa.array(np.tile(raw,(2,1)).reshape(-1)),64)
    table=pa.table(dict(synapse_id=[11,12],status=['ready','ready'],n_embeddings_used=[5,5],
        fragment_embeddings=pa.ListArray.from_arrays(pa.array([0,5,10]),values)))
    padded,mask=padded_fragments(table)
    assert padded.shape==(2,10,64)
    np.testing.assert_array_equal(padded[0],padded[1]);assert list(mask.sum(1))==[5,5]
    array=nullable_vectors(np.ones((2,3),np.float32),np.array([True,False]))
    assert array.to_pylist()==[[1.,1.,1.],None]


def test_end_to_end_twenty_outputs_keep_synapse_identity(tmp_path):
    (tmp_path/'plan.json').write_text('{"test":true,"synapses":3}')
    digest=sha(tmp_path/'plan.json');(tmp_path/'cell_types').mkdir()
    models=[]
    for family in ('cave_n10','single_pre_post'):
        for classes in ('three','six'):
            for fold in range(5):
                # Stable existing weights exercise the full pipeline while
                # new CAVE-six training is still in progress.
                source='cave_n10' if classes=='three' else 'single_pre_post'
                path=checkpoint(source,classes,fold);model,_=load_checkpoint(path)
                models.append(dict(family=family,classes=classes,fold=fold,path=str(path),sha256=sha(path),
                                   level_classes=model.cls_head.hierarchy.level_classes))
    (tmp_path/'cell_types/plan.json').write_text(json.dumps(dict(models=models,fragment_plan_sha256=digest)))
    (tmp_path/'parts/000').mkdir(parents=True);(tmp_path/'cohort').mkdir();(tmp_path/'postsynaptic_points').mkdir();(tmp_path/'status').mkdir()
    vector=np.ones((3,5,64),np.float32)
    raw=pa.ListArray.from_arrays(pa.array([0,5,10,15]),pa.FixedSizeListArray.from_arrays(pa.array(vector.reshape(-1)),64))
    table=pa.table(dict(synapse_id=[11,12,13],cell_root_id=[100,101,100],partner_root_id=[200]*3,
        anchor_node_id=[0]*3,status=['ready']*3,n_embeddings_used=[5]*3,fragment_embeddings=raw,
        mean_embedding=pa.FixedSizeListArray.from_arrays(pa.array(np.ones(3*64,np.float32)),64),
        **{f'presynaptic_fold{f}_is_axon':[True,f%2==0,False] for f in range(5)}))
    pq.write_table(table.replace_schema_metadata({b'plan_sha256':digest.encode()}),tmp_path/'parts/000/00000.parquet')
    pq.write_table(table.select(['synapse_id','cell_root_id']),tmp_path/'cohort/000.parquet')
    for root,ids in [(100,[13,11]),(101,[12])]:
        post=pa.table(dict(synapse_id=ids,cell_root_id=[root]*len(ids),
            post_embedding=pa.FixedSizeListArray.from_arrays(pa.array(np.repeat(np.asarray(ids,np.float32),64)),64),
            postsynaptic_fold0_class=['axon']*len(ids)))
        pq.write_table(post,tmp_path/f'postsynaptic_points/{root}.parquet')
    (tmp_path/'status/000_complete.json').write_text(json.dumps(dict(synapses=3,parts=1)))
    prepare_posts(tmp_path,2)
    assert (tmp_path/'cell_types/post_shards/rank=0').is_dir()
    infer(tmp_path,0,2,DEVICE,2,0)
    result=pq.read_table(tmp_path/'cell_types/parts/000/00000.parquet')
    assert result['synapse_id'].to_pylist()==[11,12,13]
    assert result['cell_root_id'].to_pylist()==[100,101,100]
    for family in ('cave_n10','single_pre_post'):
        for classes in ('three','six'):
            for f in range(5):
                level=0 if classes=='three' else 2
                name=f'{family}_{classes}_fold{f}'
                keep=[True,f%2==0,False]
                assert result[name+'_included'].to_pylist()==keep
                used=5 if family=='cave_n10' else 1
                assert result[name+'_n_embeddings_used'].to_pylist()==[used,used if keep[1] else 0,0]
                probs=result[f'{name}_level{level}_probabilities'].to_pylist()
                for included,p in zip(keep,probs):
                    assert (p is not None)==included
                    if included:assert abs(sum(p)-1)<1e-5
    info=next(m for m in models if m['family']=='cave_n10' and m['classes']=='six' and m['fold']==0)
    payload=writer_payload(result,info)
    assert payload['synapse_ids'].tolist()==[11,12]
    assert payload['subject_root_ids'].tolist()==[200,200]
    assert payload['labels'].shape==(2,3)
    assert payload['logits'][-1].shape==(2,6)
    assert payload['labels'][:,0].tolist()==['neuron','neuron']
    assert np.isfinite(payload['uncertainty']).all()
    report=json.loads((tmp_path/'cell_types/status/000_complete.json').read_text())
    for rank in range(1,32):
        (tmp_path/'cell_types/status'/f'{rank:03d}_complete.json').write_text(json.dumps({**report,'rank':rank,'rows':0,'parts':0}))
    (tmp_path/'summary.json').write_text(json.dumps(dict(synapses=3,axon_counts_by_fold={str(f):2 if f%2==0 else 1 for f in range(5)})))
    finalize(tmp_path,2)
    assert json.loads((tmp_path/'cell_types/summary.json').read_text())['score_columns_validated']
    col='cave_n10_three_fold0_level0_probabilities';bad=result[col].to_pylist();bad[0][0]+=0.5
    corrupted=result.set_column(result.column_names.index(col),col,pa.array(bad,pa.list_(pa.float32())))
    pq.write_table(corrupted,tmp_path/'cell_types/parts/000/00000.parquet')
    with pytest.raises(ValueError,match='invalid probability'):
        finalize(tmp_path,2)
