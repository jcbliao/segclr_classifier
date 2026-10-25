"""Read-only exact-coordinate coverage audit of the pinned SegCLR checkpoint."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import latest_segclr_features as f
from segclr_db import store as st
import duckdb
scratch=Path('/orcd/scratch/orcd/013/jcbliao/latest_embedding_coverage_audit');scratch.mkdir(exist_ok=True)
c=duckdb.connect(str(scratch/'audit.duckdb'));c.execute('SET threads=4');c.execute("SET memory_limit='24GB'");c.execute('SET temp_directory=?',[str(scratch/'spill')])
source='/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/latest100k_morpho6_centers_v1/points.duckdb'
c.execute("ATTACH '"+source+"' AS old (READ_ONLY)")
store=st.open_store('/orcd/compute/sdorkenw/001/segclr-db','microns')
ds=st.open_table(store,'node_embeddings',64)
print('current embeddings version',ds.version,flush=True)
c.register('shared',ds.scanner(columns=['run_id','checkpoint_id','node_set_id','root_id','node_id'],filter=f"run_id='{f.RUN}' AND checkpoint_id='{f.CHECKPOINT}' AND embedding IS NOT NULL",use_scalar_index=False).to_reader())
if not c.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='current_keys'").fetchone()[0]:c.execute('CREATE TABLE current_keys AS SELECT * FROM shared')
report={'run_id':f.RUN,'checkpoint_id':f.CHECKPOINT,'embedding_table_version':ds.version,'required_points':c.execute('SELECT count(*) FROM old.points').fetchone()[0],'previously_cached':c.execute('SELECT count(*) FROM old.cached').fetchone()[0]}
report['node_sets']=c.execute('SELECT node_set_id,count(*) FROM current_keys GROUP BY 1').fetchall()
print('node sets for checkpoint',report['node_sets'],flush=True)
c.execute("CREATE OR REPLACE TABLE newly_available AS SELECT DISTINCT m.point_id FROM old.missing m JOIN old.point_nodes p ON m.point_id=p.point_id JOIN current_keys k ON k.root_id=p.root_id AND k.node_id=p.node_id WHERE k.node_set_id='cave_skeleton_v4'")
report['newly_available_canonical']=c.execute('SELECT count(*) FROM newly_available').fetchone()[0]
report['missing_without_local_canonical_mapping']=c.execute('SELECT count(*) FROM old.missing ANTI JOIN old.point_nodes USING(point_id)').fetchone()[0]
print('canonical lookup',report,flush=True)
sets=[r[0] for r in report['node_sets']]
roots=[r[0] for r in c.execute('SELECT DISTINCT root_id FROM old.missing').fetchall()]
rootfilter=','.join(str(int(r)) for r in roots)
setfilter=','.join("'"+s.replace("'","''")+"'" for s in sets)
nodes=st.open_table(store,'nodes');report['nodes_version']=nodes.version
print('scanning shared coordinates for',len(roots),'missing-point roots; node sets',sets,flush=True)
c.register('shared_nodes',nodes.scanner(columns=['root_id','node_id','node_set_id','x_nm','y_nm','z_nm'],filter=f'node_set_id IN ({setfilter})',use_scalar_index=False,batch_size=8192,batch_readahead=1,fragment_readahead=1).to_reader())
c.execute('CREATE OR REPLACE TABLE missing_node_matches AS SELECT DISTINCT m.point_id,n.root_id,n.node_id,n.node_set_id FROM shared_nodes n JOIN old.missing m ON n.root_id=m.root_id AND n.x_nm=m.x AND n.y_nm=m.y AND n.z_nm=m.z')
print('coordinate matches',c.execute('SELECT count(DISTINCT point_id) FROM missing_node_matches').fetchone()[0],flush=True)
c.execute('CREATE OR REPLACE TABLE available_missing AS SELECT DISTINCT m.point_id,m.root_id,m.node_id,m.node_set_id FROM missing_node_matches m JOIN current_keys k ON m.root_id=k.root_id AND m.node_id=k.node_id AND m.node_set_id=k.node_set_id')
report['previously_missing_now_in_db']=c.execute('SELECT count(DISTINCT point_id) FROM available_missing').fetchone()[0]
report['required_not_found_in_node_embeddings']=report['required_points']-report['previously_cached']-report['previously_missing_now_in_db']
for name in ['node_sets','agg_specs']:
 try:report[name+'_registry']=st.open_table(store,name).to_table().to_pylist()
 except Exception as e:report[name+'_registry_error']=str(e)
agg=st.open_table(store,'agg_embeddings',64)
c.register('aggs',agg.scanner(columns=['agg_spec_id','node_set_id'],filter=f"run_id='{f.RUN}' AND checkpoint_id='{f.CHECKPOINT}'",use_scalar_index=False).to_reader())
report['aggregated_embedding_groups']=c.execute('SELECT agg_spec_id,node_set_id,count(*) FROM aggs GROUP BY 1,2').fetchall()
(scratch/'coverage.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
c.execute('CHECKPOINT')
print('COVERAGE AUDIT COMPLETE',json.dumps(report,default=str),flush=True)
