"""Inventory the pinned latest SegCLR checkpoint and downstream feature schemas."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('segclr_db',root/'logs/segclr_import_upstream/src/__init__.py',submodule_search_locations=[str(root/'logs/segclr_import_upstream/src')])
module=importlib.util.module_from_spec(spec);sys.modules['segclr_db']=module;spec.loader.exec_module(module)
from segclr_db import store as st
run='resnet50_1280b_slab1280_50um_mac_partial_aug_2_style_bf16_100k__20261022_140504'
store=st.open_store('/orcd/compute/sdorkenw/001/segclr-db','microns')
report={'run_id':run,'checkpoint_id':'checkpoint_e0_s100000'}
for name in ('runs','checkpoints'):
 report[name]=st.scan(store,name,filter=f"run_id = '{run}'").to_pylist()
for name in ('node_embeddings','agg_embeddings'):
 ds=st.open_table(store,name,64)
 report[name]={'schema':str(ds.schema),'version':ds.version,'rows_for_pinned_checkpoint':ds.count_rows(f"run_id = '{run}' AND checkpoint_id = 'checkpoint_e0_s100000'")}
base=Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons')
report['training_npz']={}
for family,path in [('cave',base/'casey_confidence_cave/k10/conf0/fold0'),('native',base/'casey_coarse_confidence_with_tc_folds/scale16/k17/conf0/fold0')]:
 manifest=json.loads((path/'manifest.json').read_text());rid=next(iter(manifest['cells']))
 with np.load(path/'cells'/f'{rid}.npz') as z:
  report['training_npz'][family]={'root_id':rid,'keys':{k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files}}
source=Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/incoming_presynaptic_fragments_k10_collina_v2')
report['parquet_schemas']={}
for pattern in ('parts/*/*.parquet','postsynaptic_points/*.parquet','cell_types/post_shards/*/*.parquet'):
 path=next(source.glob(pattern));report['parquet_schemas'][pattern]={'path':str(path),'schema':str(pq.read_schema(path))}
out=root/'analysis/db_deployment/presynaptic/latest_segclr';out.mkdir(parents=True,exist_ok=True)
(out/'inventory.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('training_npz','parquet_schemas')},default=str),flush=True)
