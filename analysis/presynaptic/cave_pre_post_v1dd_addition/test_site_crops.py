"""Verify raw-chunk reuse, exact crop axes, padding, and historical root masks."""
import sys,threading
from pathlib import Path
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from v1dd_postsynaptic_crops import V1DDPostCrops
c=V1DDPostCrops.__new__(V1DDPostCrops);c.raw=OrderedDict();c.raw_pending={};c.raw_bytes=0;c.raw_guard=threading.Lock();c.root_guard=threading.Lock();c.root_map={};c.timestamp=object();calls={'image':0,'seg':0,'roots':0}
def volume(seg):
 def download(box,**kwargs):
  calls['seg' if seg else 'image']+=1
  a=np.full(tuple(np.asarray(box.maxpt)-box.minpt),7 if seg else 9,np.uint64 if seg else np.uint8)
  if seg:a[120:]=8
  return a
 return SimpleNamespace(dtype=np.dtype(np.uint64 if seg else np.uint8),download=download,chunk_size=np.full(3,256),voxel_offset=np.zeros(3,dtype=int),bounds=SimpleNamespace(minpt=np.zeros(3,dtype=int),maxpt=np.full(3,256)))
v=(volume(False),volume(True));c._volumes=lambda:v
def roots(ids,timestamp):
 assert timestamp is c.timestamp;calls['roots']+=1;return np.array([101 if i==7 else 102 for i in ids])
c.client=SimpleNamespace(chunkedgraph=SimpleNamespace(get_roots=roots))
em,mask=c.fetch_crop(np.full(3,128),101)
assert em.shape==(1,129,129,129) and mask.shape==em.shape
assert np.all(em.numpy()==9);assert mask.numpy()[0,:,:,:56].all();assert not mask.numpy()[0,:,:,56:].any()
_,other=c.fetch_crop(np.full(3,128),102);assert np.array_equal(~mask.numpy(),other.numpy())
assert calls=={'image':1,'seg':1,'roots':1},calls
em,mask=c.fetch_crop(np.full(3,32),101);assert not em.numpy()[0,:32].any();assert em.numpy()[0,32:,32:,32:].all()
assert calls=={'image':1,'seg':1,'roots':1}
print('PASS exact 129-cubed crop axes, historical root mapping, masks, padding and shared raw chunk reuse')
