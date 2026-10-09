"""Test rounded subdivision and immutable named-store ingestion in a temporary DB."""
import tempfile
from pathlib import Path
import numpy as np
from resample_teasar_dense import densify
from registered_teasar_pipeline import ingest_batch
from segclr_db import store as st
from segclr_db.results import Skeleton

d=111.
lengths=d*np.array([0.,.5,1.4,1.5,2.5,3.6])
v=np.column_stack([np.r_[0.,lengths],np.zeros((7,2))])
e=np.column_stack([np.zeros(6,int),np.arange(1,7)])
r=np.arange(7,dtype=float)+1
rv,re,rr=densify(v,e,r,d,subdivision='round')
counts=np.array([1,1,1,2,2,4])
assert len(re)==counts.sum()
assert len(rv)-len(re)==len(v)-len(e)
np.testing.assert_array_equal(rv[:len(v)],v.astype(np.float32))
np.testing.assert_array_equal(rr[:len(v)],r)
actual=np.linalg.norm(rv[re[:,1]].astype(float)-rv[re[:,0]],axis=1)
np.testing.assert_allclose(actual,np.repeat(lengths/counts,counts),atol=1e-4)
assert np.any(actual>d) and np.any(actual<d)
start=0
for (a,b),n in zip(e,counts):
    chain=re[start:start+n]
    assert chain[0,0]==a and chain[-1,1]==b
    np.testing.assert_array_equal(chain[:-1,1],chain[1:,0])
    np.testing.assert_allclose(rr[chain[:,0]],np.linspace(r[a],r[b],n+1)[:-1])
    start+=n
cv,ce,_=densify(v,e,r,d)
assert len(ce)==np.maximum(1,np.ceil(lengths/d)).sum()
assert np.linalg.norm(cv[ce[:,1]].astype(float)-cv[ce[:,0]],axis=1).max()<=d+1e-4
empty=densify(np.empty((0,3)),np.empty((0,2),int),np.empty(0),subdivision='round')
assert all(len(x)==0 for x in empty)
for spacing in (0.,-1.,float('nan')):
    try: densify(v,e,r,spacing,subdivision='round')
    except ValueError: pass
    else: raise AssertionError('Invalid target accepted')
with tempfile.TemporaryDirectory() as folder:
    base=Path(folder); (base/'status').mkdir()
    store=st.init_store(base/'db','microns',datastack='minnie65_phase3_v1',mat_version=1718)
    value=Skeleton(root_id=123,coords=rv,edges=re,radii=rr,skeleton_version=0,skeleton_name='teasar_target')
    batch=[([value],dict(root_id=123))]
    ingest_batch(batch,store,status_dir=base/'status')
    ingest_batch(batch,store,status_dir=base/'status')
    assert st.count_rows(store,'named_skeletons')==1
    changed=rv.copy(); changed[0,0]+=1
    wrong=Skeleton(root_id=123,coords=changed,edges=re,radii=rr,skeleton_version=0,skeleton_name='teasar_target')
    try: ingest_batch([([wrong],dict(root_id=123))],store,status_dir=base/'status')
    except AssertionError: pass
    else: raise AssertionError('Conflicting existing geometry accepted')
    for table in ('skeleton_manifest','skeleton_nodes','skeleton_edges'):
        assert st.count_rows(store,table)==0
print('PASS: rounded equal subdivisions, tie handling, short/zero edges, radii, original vertices, ceil regression, DB roundtrip, idempotence, conflict rejection',flush=True)
