"""Topology, geodesic, cutoff and training-loader regression checks."""
import sys
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from data.skeleton_subsampling import nested_skeletons,contract_degree_two,SCALE_WINDOWS
from data.build_dense_presynaptic import window_arrays,native
from data.build_presynaptic_axon_database import _induced_cut,_atomic_savez
from data.dataset_presynaptic import PresynapticWindowDataset


def branched():
    # Three bent arms, with different lengths, around a common branch point.
    v=[[0.,0.,0.]]; edges=[]
    for arm,size in enumerate([137,91,53]):
        previous=0
        for i in range(1,size+1):
            point=[0.,0.,0.]; point[arm]=i*100.; point[(arm+1)%3]=20*np.sin(i/4)
            v.append(point); edges.append((previous,len(v)-1)); previous=len(v)-1
    v=np.array(v,np.float32); e=np.array(edges,np.int32)
    w=np.linalg.norm(v[e[:,0]].astype(float)-v[e[:,1]],axis=1)
    return v,e,w


def test_nested_preserves_branches_and_cable():
    v,e,w=branched()
    original=np.arange(len(v))
    degree=np.bincount(e.ravel(),minlength=len(v))
    anchors=set(np.flatnonzero(degree!=2))
    levels=nested_skeletons(v,e,w,original)
    previous=set(original)
    original_graph=coo_matrix((np.r_[w,w],(np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(len(v),len(v))).tocsr()
    for factor,level in levels.items():
        ids=level['original_node_ids']; edges=level['edges']; weights=level['edge_length_nm']
        assert anchors<=set(ids)<=previous
        previous=set(ids)
        assert len(ids)==max(len(anchors),(len(v)+factor-1)//factor)
        assert len(edges)==len(ids)-1
        np.testing.assert_allclose(weights.sum(),w.sum(),rtol=1e-12)
        np.testing.assert_array_equal(level['pos_nm'],v[ids])
        graph=coo_matrix((np.r_[weights,weights],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(ids),len(ids))).tocsr()
        for root in anchors:
            old=dijkstra(original_graph,indices=root)
            new=dijkstra(graph,indices=int(np.where(ids==root)[0][0]))
            np.testing.assert_allclose(new,old[ids],rtol=1e-12)


def test_mandatory_nodes_and_disconnected():
    v=np.arange(15).reshape(5,3)
    e=np.array([[0,1],[0,2],[0,3]])
    levels=nested_skeletons(v,e,np.ones(3),np.arange(5))
    assert len(levels[32]['pos_nm'])==5  # branch, three tips, isolated vertex
    assert len(nested_skeletons(np.empty((0,3)),np.empty((0,2),int),[],[])[32]['pos_nm'])==0


def test_reject_cycle():
    with pytest.raises(ValueError,match='forest'):
        contract_degree_two(np.zeros((3,3)),[[0,1],[1,2],[2,0]],[1,1,1],np.arange(3),2)


def test_cut_before_contraction():
    pos=np.column_stack([np.arange(-10,11)*1000,np.zeros(21),np.zeros(21)])
    edges=np.column_stack([np.arange(20),np.arange(1,21)])
    keep,_,v,e,w,_=_induced_cut(pos,edges,np.zeros(3))
    assert keep.sum()==10  # boundary at 5 um is removed too
    levels=nested_skeletons(v,e,w,np.flatnonzero(keep))
    for level in levels.values():
        assert np.all(np.linalg.norm(level['pos_nm'],axis=1)>5000)
        assert all(np.sign(level['pos_nm'][u,0])==np.sign(level['pos_nm'][v,0]) for u,v in level['edges'])


def test_windows_and_loader():
    n=230
    pos=np.column_stack([np.arange(n)*100,np.zeros(n),np.zeros(n)]).astype(np.float32)
    edges=np.column_stack([np.arange(n-1),np.arange(1,n)])
    points=pd.DataFrame(dict(synapse_id=[1,2,3],cell_x_nm=[10000,10000,10000],cell_y_nm=[0,0,3000],cell_z_nm=[0,0,0]))
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp); (out/'cells').mkdir()
        original=np.arange(n)*2
        emb=np.arange(n*2*64,dtype=np.float32).reshape(n*2,64)
        _atomic_savez(out/'geometry.npz',pos_nm=pos,edges=edges,original_node_ids=original)
        _atomic_savez(out/'embeddings.npz',node_ids=np.arange(n*2),embeddings=emb)
        for factor,k in SCALE_WINDOWS.items():
            arrays=window_arrays(points,pos,edges,np.full(n-1,100.),k)
            assert arrays['new_valid_k_window'].tolist()==[True,True,True]
            assert np.diff(arrays['new_window_offsets']).tolist()==[k,k,k]
            assert arrays['new_window_members'][0]==100  # closest point is included
            _atomic_savez(out/'cells/1.npz',geometry_path=str(out/'geometry.npz'),embedding_path=str(out/'embeddings.npz'),**arrays)
            ds=PresynapticWindowDataset({'cells':{'1':{'cell_type':'L2IT','split':'train'}}},'train','new',out)
            assert len(ds)==1  # duplicate synaptic centers deduplicated
            item=ds[0]
            assert item.num_nodes==k and int(item.has_segclr.sum())==k
            assert item.pos_enc.shape==(k,8)
            nodes=arrays['new_window_members'][:k]
            np.testing.assert_array_equal(item.x.numpy(),emb[original[nodes]])
            assert item.edge_index.shape[1]==2*(k-1)


def test_small_component_rejected():
    points=pd.DataFrame(dict(synapse_id=[1],cell_x_nm=[0],cell_y_nm=[0],cell_z_nm=[0]))
    z=window_arrays(points,np.zeros((1,3)),np.empty((0,2),int),np.empty(0),7)
    assert not z['new_valid_k_window'][0]
    assert z['new_window_offsets'].tolist()==[0,0]


def test_synapse_beyond_5um_rejected():
    pos=np.column_stack([np.arange(20)*100,np.zeros(20),np.zeros(20)]).astype(np.float32)
    edges=np.column_stack([np.arange(19),np.arange(1,20)])
    points=pd.DataFrame(dict(synapse_id=[1],cell_x_nm=[1000],cell_y_nm=[6000],cell_z_nm=[0]))
    z=window_arrays(points,pos,edges,np.full(19,100.),7)
    assert not z['new_valid_k_window'][0]


def test_native_statistics_without_embeddings():
    from analysis.presynaptic.new_skeletons.skeleton_stats import derived_edge_upstream_counts
    # A chain crossing the soma cutoff; upstream counts begin at x=6000.
    pos=np.column_stack([np.arange(12)*1000,np.zeros(12),np.zeros(12)]).astype(np.float32)
    edges=np.column_stack([np.arange(11),np.arange(1,12)])
    keep,_,v,e,w,_=_induced_cut(pos,edges,np.zeros(3))
    points=pd.DataFrame(dict(synapse_id=[1,2,3],cell_x_nm=[6000,8000,8000],cell_y_nm=[0,0,3000],cell_z_nm=[0,0,0]))
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp); (out/'sites').mkdir()
        points.to_parquet(out/'sites/1.parquet')
        _atomic_savez(out/'original.npz',vertices=pos,edges=edges)
        _atomic_savez(out/'topology/scale1/1.npz',pos_nm=v,edges=e,edge_length_nm=w,
                      root_xyz_nm=np.zeros(3),soma_cut_applied=True,original_node_ids=np.flatnonzero(keep))
        native(1,out,{})
        lengths,branches,synapses,stats=derived_edge_upstream_counts(out/'native/cells/1.npz',out/'original.npz',5000)
        np.testing.assert_array_equal(lengths,w)
        np.testing.assert_array_equal(branches,np.zeros(len(e)))
        np.testing.assert_array_equal(synapses,[1,1,3,3,3])
        assert stats['unmatched_presynaptic_site_rows']==0
