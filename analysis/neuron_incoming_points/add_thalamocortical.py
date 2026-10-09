"""Query missing thalamocortical incoming synapses and atomically extend cache."""
import sys,json,os
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
import duckdb
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from data.synapses import build_client,fetch_synapses
out=Path(__file__).resolve().parent
cache=out.parents[1]/'data/synapse_cache'
missing=pd.read_csv(out/'cells_without_cached_points.csv')
ids=missing.loc[missing.cell_types.eq('[thalamocortical]'),'root_id'].astype('int64').tolist()
if not ids:
 print('No missing thalamocortical cells');sys.exit(0)
secret=json.loads((Path.home()/'.cloudvolume/secrets/global.daf-apis.com-cave-secret.json').read_text())
client=build_client(secret['token'])
print(f'Querying {len(ids)} thalamocortical cells',flush=True)
frame=fetch_synapses(client,ids,'incoming')
assert set(frame.cell_root_id)<=set(ids)
assert frame.synapse_id.is_unique
extra=out/'thalamocortical_additional_incoming.parquet'
frame.to_parquet(extra,index=False)
print(f'Retrieved {len(frame)} points',flush=True)
c=duckdb.connect();c.execute('SET threads=4')
p=cache/'postsynaptic_sites.parquet'
c.read_parquet(str(p)).create_view('existing')
c.read_parquet(str(extra)).create_view('extra')
assert c.sql('select count(*) from existing join extra using(synapse_id)').fetchone()[0]==0
c.execute('create view merged as select * from existing union all select * from extra')
assert c.sql('select count(*)=count(distinct synapse_id) from merged').fetchone()[0]
temp=cache/'postsynaptic_sites.extending.parquet'
c.execute('copy (select * from merged order by cell_root_id,synapse_id) to ? (format parquet,compression zstd,row_group_size 100000)',[str(temp)])
stats=c.sql('select count(*),count(distinct cell_root_id),count(distinct partner_root_id),count(*) filter(where partner_root_id=0) from merged').fetchone()
assert c.read_parquet(str(temp)).count('*').fetchone()[0]==stats[0]
summary=json.loads((cache/'summary.json').read_text())
inc=summary['modes']['incoming']
inc.update(rows=stats[0],cells_with_synapses=stats[1],distinct_partners=stats[2],rows_with_unresolved_partner=stats[3])
inc['n_cells_queried']=inc.get('n_cells_queried',summary['n_cells'])+len(ids)
inc['cells_with_none']=inc['n_cells_queried']-stats[1]
inc.setdefault('extensions',[]).append(dict(queried_ts=datetime.now(timezone.utc).isoformat(),root_ids=ids,rows_added=len(frame),source=str(extra),zero_synapse_root_ids=sorted(set(ids)-set(frame.cell_root_id))))
inc['updated_ts']=datetime.now(timezone.utc).isoformat()
os.replace(temp,p)
summary_temp=cache/'summary.extending.json'
summary_temp.write_text(json.dumps(summary,indent=2)+'\n');os.replace(summary_temp,cache/'summary.json')
print('Cache updated:',stats,flush=True)
print(frame.groupby('cell_root_id').size().to_string(),flush=True)
