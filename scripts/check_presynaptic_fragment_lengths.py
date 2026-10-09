"""Regression checks: cable is the edge sum, and long windows remain valid."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.build_dense_presynaptic import window_arrays
from data.skeleton_subsampling import NATIVE_SCALE_WINDOWS
from data.native_centered_windows import CenteredWindows
from data.skeleton_subsampling import nested_skeletons

points=pd.DataFrame(dict(cell_x_nm=[0.],cell_y_nm=[0.],cell_z_nm=[0.],synapse_id=[1]))
# Three 10-micron arms: cable=30 microns, geodesic radius=10 microns.
pos=np.array([[0,0,0],[10000,0,0],[0,10000,0],[0,0,10000]],float)
edges=np.array([[0,1],[0,2],[0,3]])
for compute_lpe in (False,True):
    arrays=window_arrays(points,pos,edges,np.full(3,10000.),4,compute_lpe=compute_lpe)
    assert arrays['new_valid_k_window'].tolist()==[True]
    np.testing.assert_allclose(arrays['new_cable_length_nm'],[30000.])
    np.testing.assert_allclose(arrays['new_radius_nm'],[10000.])
    assert len(arrays['new_window_members'])==4
    if compute_lpe:
        assert arrays['new_window_lpe'].shape[0]==4
# Preserve summed curved-path cable weights rather than endpoint distances.
arrays=window_arrays(points,pos[:2],edges[:1],np.array([27000.]),2,compute_lpe=False)
assert arrays['new_valid_k_window'][0]
assert arrays['new_cable_length_nm'][0]==27000.
arrays=window_arrays(points,pos,edges,np.full(3,10000.),5,compute_lpe=False)
assert not arrays['new_valid_k_window'][0]
assert np.isnan(arrays['new_cable_length_nm'][0])
assert NATIVE_SCALE_WINDOWS=={1:257,2:129,4:65,8:33,16:17,32:9}
assert all((k-1)*factor == 256 for factor,k in NATIVE_SCALE_WINDOWS.items())
# A near-end full-resolution center would normally be removed at coarse scales.
pos=np.column_stack([np.arange(601)*100.,np.zeros((601,2))])
edges=np.column_stack([np.arange(600),np.arange(1,601)])
original=np.arange(601)+1000
xyz=np.array([[500.,0,0],[12100.,0,0],[500.,5000.,0]])
scales=nested_skeletons(pos,edges,np.full(600,100.),original)
points=pd.DataFrame(xyz,columns=['cell_x_nm','cell_y_nm','cell_z_nm'])
points['synapse_id']=np.arange(3)
previous=set(original)
for factor,geo in scales.items():
    ids=geo['original_node_ids']
    assert set(ids).issubset(previous)
    previous=set(ids)
    assert len(ids)==(601+factor-1)//factor
    np.testing.assert_allclose(geo['edge_length_nm'].sum(),60000.)
    selector=CenteredWindows(pos,edges,np.full(600,100.),original,geo,xyz)
    np.testing.assert_array_equal(original[selector.centers],[1005,1121,1005])
    arrays=selector.arrays(points.synapse_id.to_numpy(),NATIVE_SCALE_WINDOWS[factor],compute_lpe=False)
    assert arrays['new_valid_k_window'].tolist()==[True,True,False]
    np.testing.assert_array_equal(np.diff(arrays['new_window_offsets'])[:2],NATIVE_SCALE_WINDOWS[factor])
    first=arrays['new_window_members'][arrays['new_window_offsets'][:2]]
    np.testing.assert_array_equal(original[first],[1005,1121])
    for row in range(2):
        lo,hi=arrays['new_window_offsets'][row:row+2]
        members=arrays['new_window_members'][lo:hi]
        np.testing.assert_allclose(arrays['new_cable_length_nm'][row],np.ptp(pos[members,0]))
# Center between two immutable anchors is inserted locally; no branch is moved.
pos=np.array([[0,0,0],[1000,0,0],[10000,0,0],[0,10000,0],[0,0,10000]],float)
edges=np.array([[0,1],[1,2],[0,3],[0,4]])
lengths=np.array([1000.,9000.,10000.,10000.])
coarse=dict(original_node_ids=np.array([0,2,3,4]),edges=np.array([[0,1],[0,2],[0,3]]),
            edge_length_nm=np.array([10000.,10000.,10000.]))
selector=CenteredWindows(pos,edges,lengths,np.arange(5),coarse,np.array([[1000.,0,0]]))
arrays=selector.arrays([1],5)
assert arrays['new_window_members'][0]==1
assert set(arrays['new_window_members'])==set(range(5))
assert arrays['new_cable_length_nm'][0]==30000.
assert arrays['new_window_edges'].shape==(8,2)
assert arrays['new_window_lpe'].shape[0]==5
print('PASS: edge-sum cable, long-window retention, LPE compatibility, undersized components, node counts',flush=True)
print('PASS: exact full-resolution centers at every scale, local degree-2 replacement, anchor-edge insertion, cable preservation, unchanged matching cutoff',flush=True)
