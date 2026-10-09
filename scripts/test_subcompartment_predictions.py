"""Model-output and five-fold database contracts for compartment inference."""
import json
import numpy as np
import pytest
from scripts.subcompartment_predict_all import db_payload, HIERARCHY, register, sha, EMBED_RUN, EMBED_CKPT
from segclr_db import store as st
from segclr_db.writer import SegCLRWriter
from segclr_db.database import SegCLRDatabase
from segclr_db.results import Skeleton
from segclr_db.skeletons import SkeletonCache


def test_raw_logits_reordered_without_softmax_or_aggregation():
    raw=np.array([[10,-2,3,1],[-3,7,2,1]],np.float32)
    labels,entropy,logits=db_payload(raw,['soma','dendrite','axon','astrocytic_process'])
    np.testing.assert_array_equal(logits[0],raw[:,[3,2,1,0]])
    np.testing.assert_array_equal(labels[:,0],['soma','dendrite'])
    assert entropy.shape==(2,1) and (entropy>=0).all()
    np.testing.assert_array_equal(raw,[[10,-2,3,1],[-3,7,2,1]])


@pytest.mark.parametrize('raw,classes',[(np.array([[np.nan,0,0,0]]),['soma','dendrite','axon','astrocytic_process']),
    (np.zeros((1,4)),['soma','soma','axon','astrocytic_process']),
    (np.zeros((4,)),['soma','dendrite','axon','astrocytic_process'])])
def test_invalid_model_output_rejected(raw,classes):
    with pytest.raises(ValueError):db_payload(raw,classes)


def test_five_folds_raw_teasar_cave_and_presynaptic_roundtrip(tmp_path):
    store=st.init_store(tmp_path/'db','microns',datastack='minnie65_phase3_v1',mat_version=1718)
    writer=SegCLRWriter(store=store);db=SegCLRDatabase(store=store)
    writer.register_experiment('resnet_860b_reshuffled',{},embedding_dim=64)
    writer.register_run('resnet_860b_reshuffled',tmp_path/'embedding','20260603_150412')
    writer.register_checkpoints(EMBED_RUN,[(EMBED_CKPT,0,95000,str(tmp_path/'embedding.pt'))])
    fold_plan=tmp_path/'folds.json';fold_plan.write_text(json.dumps(dict(assignments={'10':0,'11':1})))
    models=[]
    for fold in range(5):
        path=tmp_path/f'fold{fold}'/'model.pt';path.parent.mkdir();path.write_bytes(f'fold {fold}'.encode())
        models.append(dict(fold=fold,path=str(path),sha256=sha(path),best_epoch=fold+1))
    plan=dict(models=models,fold_plan=str(fold_plan),fold_plan_sha256=sha(fold_plan),uncertainty='entropy')
    runs=register(writer,plan)
    assert len(set(runs))==5
    cache=SkeletonCache(store)
    geometry=Skeleton(root_id=10,coords=np.array([[0,0,0],[100,0,0]],np.float32),edges=np.array([[0,1]],np.int32))
    cache.store_skeletons([geometry]);cache.store_named_skeleton('teasar_registered_20260911',geometry)
    experiment=db.experiment('resnet_860b_reshuffled')
    writer.add_named_point_embeddings(experiment,10,'presynaptic_sites',geometry.coords,np.ones((2,64)),[90,91],checkpoint_id=EMBED_CKPT)
    raw=np.array([[7,-3,2,1],[1,2,8,0]],np.float32)
    labels,entropy,logits=db_payload(raw,['soma','dendrite','axon','astrocytic_process'])
    for run in runs:
        for name in [None,'teasar_registered_20260911','presynaptic_sites']:
            writer.add_cell_predictions(run,10,np.arange(2),labels,entropy,logits,
                skeleton_name=name,synapse_ids=np.array([90,91]) if name=='presynaptic_sites' else None)
            got=db.get_predictions(run,10,skeleton_name=name,with_logits=True)
            np.testing.assert_array_equal(got.logits[0],logits[0]);np.testing.assert_array_equal(got.labels,labels)
            assert writer.add_cell_predictions(run,10,np.arange(2),labels,entropy,logits,skeleton_name=name)==0
    assert db.list_prediction_runs().agg_spec_id.isna().all()
    assert db.list_prediction_runs().window_nm.isna().all()
    assert db.verify()==[]


def test_repeated_export_rows_do_not_pool_or_create_duplicate_predictions():
    from scripts.subcompartment_predict_all import unique_embedding_rows
    first=np.array([1.,2.],np.float32)
    nodes,x=unique_embedding_rows(np.array([1,0,1]),np.array([[4.,5.],first,[4.+1e-7,5.]],np.float32))
    np.testing.assert_array_equal(nodes,[0,1]);np.testing.assert_array_equal(x,[first,[4.,5.]])
    with pytest.raises(AssertionError):
        unique_embedding_rows(np.array([0,0]),np.array([[1.,2.],[4.,5.]]))


def test_vectorized_real_folds_match_serial_logits():
    import torch
    from scripts.subcompartment_predict_all import load_models, OUT, db_payload
    from scripts.subcompartment_predict_fast import fold_predictor
    torch.set_num_threads(1)
    plan=json.loads((OUT/'plan.json').read_text())
    models=load_models(plan,'cpu')
    predict=fold_predictor(models,[m['classes'] for m in plan['models']])
    x=torch.randn(79,64,generator=torch.Generator().manual_seed(17))
    with torch.inference_mode():
        actual=predict(x).numpy()
        expected=np.stack([db_payload(m(x).numpy(),info['classes'])[2][0] for m,info in zip(models,plan['models'])])
    np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=3e-5)
    np.testing.assert_array_equal(actual.argmax(2),expected.argmax(2))


def test_batched_geometry_checks_original_ids_and_embedding_partition():
    import pyarrow as pa
    from scripts.subcompartment_predict_fast import check_geometry,partition_embeddings
    items=[dict(root_id=10,skeleton_name='raw',embedding_skeleton_name='dense',n_nodes=2),
           dict(root_id=11,skeleton_name='raw',embedding_skeleton_name='dense',n_nodes=1)]
    coords_type=pa.list_(pa.list_(pa.float32(),3))
    table=pa.table(dict(root_id=[10,10,11,11],skeleton_name=['raw','dense','raw','dense'],
        coords=pa.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6],[7,8,9]],[[0,0,0]],[[0,0,0],[2,2,2]]],type=coords_type)))
    check_geometry(table,items)
    wrong=pa.table(dict(root_id=[10,10],skeleton_name=['raw','dense'],
        coords=pa.array([[[1,2,3],[4,5,6]],[[4,5,6],[1,2,3]]],type=coords_type)))
    with pytest.raises(AssertionError):check_geometry(wrong,items[:1])
    embeddings=pa.table(dict(root_id=[11,10,10],node_id=[0,1,0],
        embedding=pa.FixedSizeListArray.from_arrays(pa.array(np.repeat([11.,101.,100.],64),type=pa.float32()),64)))
    got=partition_embeddings(embeddings,items)
    np.testing.assert_array_equal(got[10][0],[0,1]);np.testing.assert_array_equal(got[10][1][:,0],[100,101])
    np.testing.assert_array_equal(got[11][1][:,0],[11])
    with pytest.raises(AssertionError):partition_embeddings(embeddings.slice(0,2),items)


def test_async_writer_keeps_cell_geometry_and_fold_identity(tmp_path):
    from scripts.subcompartment_predict_fast import write_batch
    from scripts.subcompartment_predict_all import destination
    items=[dict(root_id=10,geometry='teasar',skeleton_name='raw'),
           dict(root_id=11,geometry='presynaptic',skeleton_name='presynaptic_sites')]
    loaded=[(items[0],np.arange(2),np.zeros((2,64)),{}),
            (items[1],np.arange(1),np.ones((1,64)),dict(synapse_ids=np.array([91]),positions_nm=np.ones((1,3))))]
    logits=np.arange(5*3*4,dtype=np.float32).reshape(5,3,4)
    assert write_batch(tmp_path,loaded,logits,[2,1],'frozen')==3
    for index,i in enumerate(items):
        with np.load(destination(tmp_path,i)) as z:
            np.testing.assert_array_equal(z['logits'],logits[:,:2] if index==0 else logits[:,2:])
            assert str(z['plan_sha256'])=='frozen' and str(z['skeleton_name'])==i['skeleton_name']
            if index==1:np.testing.assert_array_equal(z['synapse_ids'],[91])


def test_batched_reader_reports_missing_nodes_and_reuses_cached_backfills(tmp_path):
    from scripts.subcompartment_predict_fast import reader,complete_embeddings
    from scripts.subcompartment_predict_all import save_npz
    import pyarrow as pa
    store=st.init_store(tmp_path/'db','microns',datastack='minnie65_phase3_v1',mat_version=1718)
    geometry=Skeleton(root_id=10,coords=np.array([[1,2,3],[4,5,6],[7,8,9]],np.float32),edges=np.empty((0,2),np.int32))
    cache=SkeletonCache(store)
    cache.store_named_skeleton('raw',geometry);cache.store_named_skeleton('dense',geometry)
    st.append(store,'named_node_embeddings',dict(root_id=pa.array([10,10],pa.int64()),
        skeleton_name=['dense','dense'],node_id=pa.array([0,2],pa.int32()),run_id=[EMBED_RUN]*2,
        checkpoint_id=[EMBED_CKPT]*2,embedding=pa.FixedSizeListArray.from_arrays(pa.array(np.ones(128),pa.float32()),64)),dim=64)
    tables={}
    for name in ['named_skeletons','named_node_embeddings']:
        ds=st.open_table(store,name,64 if name=='named_node_embeddings' else None)
        tables[name]=dict(uri=ds.uri,version=ds.version)
    plan=dict(tables=tables)
    item=dict(root_id=10,geometry='teasar',skeleton_name='raw',embedding_skeleton_name='dense',n_nodes=3)
    loaded,_=reader(plan,tmp_path)([item])
    np.testing.assert_array_equal(loaded[0][3]['_missing_node_ids'],[1])
    np.testing.assert_array_equal(loaded[0][3]['_original_coords'],geometry.coords)
    save_npz(tmp_path/'embedding_backfill/teasar/10.npz',root_id=10,skeleton_name='raw',
        node_ids=np.array([1],np.int32),embeddings=np.full((1,64),9,np.float32),run_id=EMBED_RUN,checkpoint_id=EMBED_CKPT)
    reused,_=reader(plan,tmp_path)([item])
    assert '_missing_node_ids' not in reused[0][3]
    complete=complete_embeddings(reused,tmp_path,'plan',lambda:pytest.fail('cached vectors must not be regenerated'))
    np.testing.assert_array_equal(complete[0][1],[0,1,2]);np.testing.assert_array_equal(complete[0][2][:,0],[1,9,1])


def test_missing_original_nodes_are_embedded_without_changing_existing_vectors(tmp_path):
    from scripts.subcompartment_predict_fast import complete_embeddings
    class Engine:
        def embed(self,root,coords,ids):
            assert root==10
            np.testing.assert_array_equal(coords,[[1,2,3],[4,5,6],[7,8,9]])
            np.testing.assert_array_equal(ids,[1])
            return ids,np.full((1,64),9,np.float32)
    item=dict(root_id=10,geometry='teasar',skeleton_name='raw',n_nodes=3)
    loaded=[(item,np.array([0,2],np.int32),np.ones((2,64),np.float32),
        dict(_missing_node_ids=np.array([1],np.int32),_original_coords=np.array([[1,2,3],[4,5,6],[7,8,9]],np.float32)))]
    result=complete_embeddings(loaded,tmp_path,'plan',lambda:Engine())
    np.testing.assert_array_equal(result[0][1],[0,1,2]);np.testing.assert_array_equal(result[0][2][:,0],[1,9,1])
    assert result[0][3]=={}
    with np.load(tmp_path/'embedding_backfill/teasar/10.npz') as z:
        np.testing.assert_array_equal(z['node_ids'],[1]);assert str(z['checkpoint_id'])==EMBED_CKPT
