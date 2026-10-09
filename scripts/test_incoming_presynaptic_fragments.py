import numpy as np
import pyarrow as pa
from scripts.incoming_presynaptic_fragments import nearest_fragment, fragments_for_root


def fixture(embedded=(0,1,2,3)):
    # Node 3 is physically close but disconnected. Node 2 is reachable via 1.
    nodes=pa.table(dict(node_id=[0,1,2,3],x_nm=[0.,5.,10.,.1],y_nm=[0.]*4,z_nm=[0.]*4))
    edges=pa.table(dict(src=[0,1],dst=[1,2]))
    e=np.repeat(np.asarray(embedded,np.float32)[:,None],64,axis=1)
    embeddings=pa.table(dict(node_id=pa.array(embedded,pa.int32()),
        embedding=pa.FixedSizeListArray.from_arrays(pa.array(e.reshape(-1)),64)))
    syn=pa.table(dict(synapse_id=[10,11],cell_root_id=[100,101],partner_root_id=[200,200],
        partner_x_nm=[0.,.1],partner_y_nm=[0.,0.],partner_z_nm=[0.,0.]))
    return syn,nodes,edges,embeddings


def test_geodesic_and_undersized_component():
    result=fragments_for_root(*fixture())
    assert result['fragment_node_ids'].to_pylist()==[[0,1,2],[3]]
    assert result['node_count'].to_pylist()==[3,1]
    assert result['n_embeddings_used'].to_pylist()==[3,1]
    assert result['n_embeddings_available'].to_pylist()==[3,1]
    assert result['status'].to_pylist()==['ready','ready']
    np.testing.assert_array_equal(result['mean_embedding'].to_pylist(),np.repeat([[1.],[3.]],64,axis=1))
    assert result['synapse_id'].to_pylist()==[10,11]
    assert result['cell_root_id'].to_pylist()==[100,101]


def test_missing_embeddings_do_not_change_membership_or_pool_partial_data():
    result=fragments_for_root(*fixture((0,2,3)))
    assert result['fragment_node_ids'].to_pylist()==[[0,1,2],[3]]
    assert result['missing_embedding_count'].to_pylist()==[1,0]
    assert result['n_embeddings_used'].to_pylist()==[0,1]
    assert result['n_embeddings_available'].to_pylist()==[2,1]
    assert result['status'].to_pylist()==['missing_embeddings','ready']
    assert np.isnan(result['mean_embedding'][0].as_py()).all()


def test_nearest_ten_cap_and_shortest_path_order():
    adjacency=[[] for _ in range(12)]
    for i in range(11):adjacency[i].append((i+1,1.));adjacency[i+1].append((i,1.))
    assert nearest_fragment(0,adjacency).tolist()==list(range(10))
    assert nearest_fragment(5,adjacency,4).tolist()==[5,4,6,3]


def test_missing_skeleton_is_explicit():
    syn,nodes,edges,embeddings=fixture()
    result=fragments_for_root(syn,nodes.slice(0,0),edges.slice(0,0),embeddings.slice(0,0))
    assert result['status'].to_pylist()==['missing_skeleton']*2
    assert result['node_count'].to_pylist()==[0,0]
    assert result['fragment_node_ids'].to_pylist()==[[],[]]


def test_repeated_identical_cache_rows_count_only_once():
    import pytest
    syn,nodes,edges,embeddings=fixture()
    duplicate=pa.concat_tables([embeddings,embeddings.slice(0,1)])
    result=fragments_for_root(syn,nodes,edges,duplicate)
    assert result['n_embeddings_used'].to_pylist()==[3,1]
    np.testing.assert_array_equal(result['fragment_embeddings'][0].as_py(),np.repeat([[0.],[1.],[2.]],64,axis=1))
    conflicting=pa.table({'node_id':pa.array([0],pa.int32()),
        'embedding':pa.array([[99.]*64],pa.list_(pa.float32(),64))})
    with pytest.raises(AssertionError):
        fragments_for_root(syn,nodes,edges,pa.concat_tables([embeddings,conflicting]))


def test_final_database_retains_count_distribution(tmp_path):
    import json
    import duckdb
    import pyarrow.parquet as pq
    from scripts.incoming_presynaptic_fragments import finalize,append_presynaptic_classifications
    plan=dict(shards=[dict(rank=0,synapses=2)],synapses=2,logit_classes=['astrocytic_process','axon','dendrite','soma'])
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    (tmp_path/'status').mkdir();(tmp_path/'parts/000').mkdir(parents=True)
    report=dict(synapses=2,ready=2,missing_skeleton=0,missing_embeddings=0,fewer_than_10=2,parts=1)
    (tmp_path/'status/000_complete.json').write_text(json.dumps(report))
    table=fragments_for_root(*fixture())
    for fold in range(5):
        logits=np.array([[0,2,0,0],[0,2 if fold==0 else 0,1,0]],np.float32)
        table=table.append_column(f'fold{fold}_logits',pa.FixedSizeListArray.from_arrays(pa.array(logits.reshape(-1)),4))
    table=append_presynaptic_classifications(table,plan['logit_classes'])
    pq.write_table(table,tmp_path/'parts/000/00000.parquet')
    finalize(tmp_path)
    with duckdb.connect(str(tmp_path/'fragments.duckdb'),read_only=True) as c:
        assert c.sql('select n_embeddings_used from classified_fragments order by synapse_id').fetchall()==[(3,),(1,)]
        assert c.sql('select count(*) from fold0_axon_fragments').fetchone()==(2,)
        assert c.sql('select count(*) from fold1_axon_fragments').fetchone()==(1,)
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['embedding_count_distribution']==[
        dict(status='ready',n_embeddings_used=1,synapses=1),dict(status='ready',n_embeddings_used=3,synapses=1)]
