"""Focused newest-SegCLR, six-morphology-class training; no production scan."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json,os,pickle,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('LATEST_SEGCLR_WORK','/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/latest100k_morpho6_centers_v1')
os.environ['LATEST_SEGCLR_SCOPE']='training'
from scripts import latest_segclr_features as f


def plan():
 import numpy as np
 import pyarrow as pa
 import pyarrow.parquet as pq
 from segclr_db import store as st
 from segclr_db.skeletons import skeleton_content_hash
 if (f.WORK/'plan.json').exists():
  print('Existing focused plan',flush=True);return
 f.WORK.mkdir(parents=True,exist_ok=True,mode=0o2775)
 # Exercise the archive API on a tiny independent table before expensive work.
 import tempfile,duckdb
 with tempfile.TemporaryDirectory(dir=f.WORK) as tmp,duckdb.connect() as test:
  test.sql('SELECT 1 AS root_id, [1.0::FLOAT,2.0::FLOAT] AS embedding').write_parquet(str(Path(tmp)/'archive'),compression='zstd',row_group_size=8192,per_thread_output=True)
 if f.DB.exists():raise ValueError('incomplete focused point database requires inspection')
 store=st.open_store('/orcd/compute/sdorkenw/001/segclr-db','microns')
 roots=json.loads((f.OLD_BASE/'casey_confidence_cave/k10/conf0/fold0/manifest.json').read_text())['cells']
 ids=','.join(roots)
 manifest=st.open_table(store,'node_manifest').scanner(columns=['root_id','content_hash'],filter=f"node_set_id='cave_skeleton_v4' AND root_id IN ({ids})",use_scalar_index=False).to_table().to_pylist()
 hashes={str(row['root_id']):row['content_hash'] for row in manifest}
 canonical_cells=0
 with f.open_db() as c:
  c.execute('CREATE TABLE requests(root_id BIGINT,x FLOAT,y FLOAT,z FLOAT)')
  c.execute('CREATE TABLE canonical(root_id BIGINT,node_id INTEGER,x_nm FLOAT,y_nm FLOAT,z_nm FLOAT)')
  for family,directory in [('cave',f.OLD_BASE/'casey_confidence_cave/k10/conf0/fold0'),('native',f.OLD_BASE/'casey_coarse_confidence_with_tc_folds/scale16/k17/conf0/fold0')]:
   for i,rid in enumerate(sorted(roots)):
    with np.load(directory/'cells'/f'{rid}.npz') as z:
     if 'geometry_path' in z:
      selected=np.unique(z['new_nearest_observed_node'][z['new_valid_k_window']])
      with np.load(str(z['geometry_path'])) as geo:xyz=geo['pos_nm'][selected].copy()
     else:xyz=z['cave_pos_nm'][z['cave_observed_node_ids']]
    f.append_points(c,np.repeat(int(rid),len(xyz)),xyz)
    if family=='cave':
     cache=ROOT/'data/skeleton_cache'/f'{rid}.pkl'
     if cache.exists():
      try:
       with cache.open('rb') as src:skeleton=pickle.load(src)
       trusted=skeleton_content_hash(skeleton)==hashes.get(rid)
      except (ImportError,AttributeError,ValueError):trusted=False
      if trusted:
       xyz=np.asarray(skeleton.coords,np.float32)
       c.register('canonical_cell',pa.table(dict(root_id=np.repeat(int(rid),len(xyz)),node_id=np.arange(len(xyz),dtype=np.int32),x_nm=xyz[:,0],y_nm=xyz[:,1],z_nm=xyz[:,2])))
       c.execute('INSERT INTO canonical SELECT * FROM canonical_cell');c.unregister('canonical_cell');canonical_cells+=1
    if i%200==0:print('training inputs',family,i,flush=True)
  c.read_parquet(str(f.OLD_POST/'sites/*.parquet')).create_view('sites')
  c.read_parquet(str(f.OLD_POST/'embeddings/*.parquet')).create_view('post')
  c.execute("INSERT INTO requests SELECT s.postsynaptic_root_id,CAST(s.x_nm AS FLOAT),CAST(s.y_nm AS FLOAT),CAST(s.z_nm AS FLOAT) FROM sites s JOIN post e USING(synapse_id,presynaptic_root_id,postsynaptic_root_id) WHERE e.status='ok' AND s.postsynaptic_root_id!=0")
  c.execute('CREATE TABLE points AS SELECT row_number() OVER ()::BIGINT AS point_id,* FROM (SELECT DISTINCT root_id,x,y,z FROM requests)')
  c.execute('DROP TABLE requests')
  print('matching only verified local training geometry; no shared-node scan',flush=True)
  c.execute('CREATE TABLE point_nodes AS SELECT p.point_id,n.root_id,n.node_id FROM canonical n JOIN points p ON n.root_id=p.root_id AND n.x_nm=p.x AND n.y_nm=p.y AND n.z_nm=p.z QUALIFY row_number() OVER(PARTITION BY p.point_id ORDER BY n.node_id)=1')
  ds=st.open_table(store,'node_embeddings',64);version=ds.version
  c.register('shared',ds.scanner(columns=['root_id','node_id','embedding'],filter=f"run_id='{f.RUN}' AND checkpoint_id='{f.CHECKPOINT}' AND node_set_id='cave_skeleton_v4' AND root_id IN ({ids})",use_scalar_index=False).to_reader())
  c.execute('CREATE TABLE cached_raw AS SELECT p.point_id,e.embedding FROM shared e JOIN point_nodes p USING(root_id,node_id) WHERE e.embedding IS NOT NULL AND len(e.embedding)=64 AND isfinite(list_sum(e.embedding))')
  c.execute('CREATE TABLE cached AS SELECT point_id,first(embedding) AS embedding FROM cached_raw GROUP BY point_id HAVING count(DISTINCT embedding)=1')
  c.execute('DROP TABLE cached_raw')
  c.execute('CREATE TABLE missing AS SELECT p.* FROM points p ANTI JOIN cached USING(point_id)')
  total=c.sql('SELECT count(*) FROM points').fetchone()[0];cached=c.sql('SELECT count(*) FROM cached').fetchone()[0]
  (f.WORK/'requests').mkdir(exist_ok=True);chunks=[]
  for i,batch in enumerate(c.sql('SELECT * FROM missing ORDER BY z,y,x,root_id').fetch_record_batch(8192)):
   path=f.WORK/'requests'/f'{i:06d}.parquet';pq.write_table(pa.Table.from_batches([batch]),path,compression='zstd');chunks.append(dict(path=str(path),rows=len(batch),sha256=f.sha(path)))
  c.execute('CHECKPOINT')
 p=dict(experiment_id=f.EXPERIMENT,run_id=f.RUN,checkpoint_id=f.CHECKPOINT,checkpoint_path=str(f.WEIGHTS),checkpoint_sha256=f.sha(f.WEIGHTS),points=total,cached_points=cached,missing_points=total-cached,chunks=chunks,embedding_table_version=version,training_cells=len(roots),verified_canonical_cells=canonical_cells,scope='training_only',classes=['BasketFam','BipFam','MartFam','NglFam','pyramidal','thalamocortical'],label_confidence_min=0,embedding_dim=64,em_volume='em_clahe2d_flash',seg_volume='seg_1718_zarr3_flash',precision='tf32',segclr_source_commit='afb3d8c')
 f.save(f.WORK/'plan.json',p);f.save(f.STATE/'morpho6_feature_plan.json',p)
 print('FOCUSED TRAINING PLAN READY',total,'points;',cached,'cached;',total-cached,'missing',flush=True)


def register():
 from segclr_db import store as st
 from segclr_db.writer import SegCLRWriter
 from scripts.predict_incoming_fragment_cell_types import load_checkpoint
 from scripts import conf0_presynaptic_deployment as training
 plan=json.loads((training.STATE/'plan.json').read_text());feature_plan=json.loads((f.WORK/'plan.json').read_text())
 if len(plan['models'])!=10:raise ValueError('expected two six-class configurations, five folds')
 store=st.open_store('/orcd/compute/sdorkenw/001/segclr-db','microns');writer=SegCLRWriter(store=store)
 receipt=f.WORK/'classifier_registry.json'
 old=json.loads(receipt.read_text()) if receipt.exists() else dict(timestamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
 f.save(receipt,dict(old,complete=False))
 records=[]
 for item in plan['models']:
  path=Path(item['checkpoint']);model,manifest=load_checkpoint(path)
  digest=f.sha(path);result=path.parent.with_suffix('.json');report=json.loads(result.read_text())
  if report['args']['manifest']!=str(manifest) or json.loads(manifest.read_text())['casey_coarse_confidence_min']!=0:raise ValueError('wrong training cohort')
  hierarchy='casey_neuron_six_class_v1'
  registered=st.scan(store,'label_hierarchies',filter=f"hierarchy_id='{hierarchy}'").to_pylist()
  classes=[list(level) for level in model.cls_head.hierarchy.level_classes]
  if len(registered)!=1 or json.loads(registered[0]['level_classes_json'])!=classes:raise ValueError('class ordering drift')
  directory=f.WORK/'models'/digest;directory.mkdir(parents=True,exist_ok=True,mode=0o2775);directory.chmod(0o2775)
  for source,name in [(path,'checkpoint.pt'),(manifest,'manifest.json'),(result,'training_result.json')]:
   dest=directory/name
   if not dest.exists():
    temporary=dest.with_suffix(dest.suffix+'.tmp');shutil.copyfile(source,temporary);temporary.chmod(0o664);temporary.replace(dest)
   if f.sha(dest)!=f.sha(source):raise ValueError('shared artifact hash mismatch')
  experiment=f"casey_six_latest100k_{item['family']}_fold{item['fold']}_{digest[:12]}"
  spec=dict(model_config=asdict(model.config),head_nodes=model.cls_head.nodes,level_classes=classes,label_confidence_min=0,fold=item['fold'],family=item['family'],training_args=report['args'],manifest_sha256=f.sha(manifest),checkpoint_sha256=digest,embedding_run_id=f.RUN,embedding_checkpoint_id=f.CHECKPOINT,embedding_checkpoint_sha256=feature_plan['checkpoint_sha256'],feature_plan=str(f.WORK/'plan.json'),feature_plan_sha256=f.sha(f.WORK/'plan.json'),postsynaptic_input='same newest SegCLR checkpoint; 64D concatenation',architecture='same pointwise MLP L2 and ResNet classifier trunk as original deployment',run_timestamp_semantics='UTC artifact registration time; historical training launch time is not asserted')
  writer.register_agg_spec(method='pointwise_mlp_mean',k_nearest=10 if item['family']=='cave_n10' else 1)
  writer.register_experiment(experiment,spec,kind='classifier',hierarchy_id=hierarchy)
  run=writer.register_run(experiment,path.parent,old['timestamp'],seed=report['args']['seed'])
  checkpoint='checkpoint_'+digest[:16]
  writer.register_checkpoints(run.run_id,[(checkpoint,int(report['best_epoch']),-1,str(directory/'checkpoint.pt'))])
  records.append(dict(family=item['family'],classes='six',fold=item['fold'],experiment_id=experiment,run_id=run.run_id,checkpoint_id=checkpoint,path=str(directory/'checkpoint.pt'),sha256=digest))
 f.save(receipt,dict(old,complete=True,models=records,embedding_run_id=f.RUN,label_confidence_min=0,predictions_written=False))
 print('TEN NEWEST-SEGCLR MORPHOLOGY MODELS REGISTERED',flush=True)


def check_sparse():
 import tempfile,numpy as np
 from data.dataset_presynaptic import load_cell_arrays,PresynapticWindowDataset
 base=f.OLD_BASE/'casey_coarse_confidence_with_tc_folds/scale16/k17/conf0/fold0'
 manifest=json.loads((base/'manifest.json').read_text())
 rid=next(r for r,info in manifest['cells'].items() if info['split']=='train')
 source=base/'cells'/f'{rid}.npz'
 with np.load(source) as z:arrays={k:z[k].copy() for k in z.files}
 if 'geometry_path' not in arrays:raise ValueError('expected named-node fixture')
 dense=load_cell_arrays(source,'new')
 centers=np.unique(arrays['new_nearest_observed_node'][arrays['new_valid_k_window']]).astype(np.int32)
 with tempfile.TemporaryDirectory(dir=f.WORK) as tmp:
  tmp=Path(tmp);(tmp/'cells').mkdir()
  emb=tmp/'embedding.npz'
  np.savez(emb,node_ids=dense['original_node_ids'][centers],local_node_ids=centers,embeddings=dense['observed_x'][centers])
  arrays['embedding_path']=np.asarray(str(emb));arrays['center_embeddings_only']=np.asarray(True)
  np.savez(tmp/'cells'/f'{rid}.npz',**arrays)
  sparse=load_cell_arrays(tmp/'cells'/f'{rid}.npz','new',True)
  np.testing.assert_array_equal(sparse['pos'],dense['pos']);np.testing.assert_array_equal(sparse['edges'],dense['edges'])
  np.testing.assert_array_equal(sparse['observed_ids'],centers);np.testing.assert_array_equal(sparse['observed_x'],dense['observed_x'][centers])
  try:load_cell_arrays(tmp/'cells'/f'{rid}.npz','new')
  except ValueError:pass
  else:raise ValueError('sparse features must reject whole-window use')
  probe=dict(manifest,cells={rid:manifest['cells'][rid]})
  post=f.OLD_BASE/'native_skeletons_pre_post/scale16/k17/conf0'
  opts=dict(postsynaptic_cache=post,use_postsynaptic=True,single_presynaptic_embedding=True)
  old=PresynapticWindowDataset(probe,'train','new',database=base,**opts)
  new=PresynapticWindowDataset(probe,'train','new',database=tmp,**opts)
  for key in ('index_root_ids','index_window_rows','window_sizes','index_labels'):
   np.testing.assert_array_equal(getattr(old,key),getattr(new,key))
  import torch
  for i in range(min(8,len(old))):
   a,b=old[i],new[i]
   for key in ('x','edge_index','postsynaptic_embedding','y_levels'):torch.testing.assert_close(getattr(a,key),getattr(b,key),rtol=0,atol=0)
 f.save(f.WORK/'center_selection_check.json',dict(passed=True,root_id=rid,geometry_and_windows_unchanged=True,training_index_unchanged=True,features_identical_before_embedding_swap=True))
 print('CENTER-ONLY REPRESENTATION PRESERVES EXACT GEOMETRY, WINDOWS, INDEX AND INPUTS',flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('plan','smoke','embed','assemble','register','check-sparse'));p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=32);a=p.parse_args()
 if a.stage=='plan':plan()
 elif a.stage=='smoke':f.smoke()
 elif a.stage=='embed':f.embed(a.rank,a.world)
 elif a.stage=='assemble':f.assemble()
 elif a.stage=='check-sparse':check_sparse()
 else:register()
