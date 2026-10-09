"""Collect outgoing synapses on all neuronal roots at materialization 1718."""
import sys,json
from pathlib import Path
import lance,duckdb,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from data.synapses import build_client,fetch_synapses
out=Path(__file__).resolve().parent;repo=out.parents[1]
l=lance.dataset('/orcd/compute/sdorkenw/001/segclr-db/microns/dims/cell_labels.lance').to_table().to_pandas()
l=l[(l.label_set=='cell_type')&~l.label.isin(['astrocyte','oligo','microglia','OPC'])]
assert set(l.mat_version)=={1718}
c=duckdb.connect();c.execute('SET threads=4');c.register('labels',l)
c.execute('create table neurons as select root_id,list_sort(list(distinct label)) cell_types from labels group by 1')
sources=[str(repo/'data/synapse_cache/presynaptic_sites.parquet')]
extra=repo/'analysis/neuron_incoming_points/thalamocortical_additional_outgoing.parquet'
if extra.exists():sources.append(str(extra))
c.read_parquet(sources).create_view('cached')
missing=[x[0] for x in c.sql('select root_id from neurons except select distinct cell_root_id from cached').fetchall()]
print('Querying missing neuronal roots:',missing,flush=True)
queried=out/'additional_outgoing.parquet'
if missing and not queried.exists():
 secret=json.loads((Path.home()/'.cloudvolume/secrets/global.daf-apis.com-cave-secret.json').read_text())
 client=build_client(secret['token']);frame=fetch_synapses(client,missing,'outgoing');frame.to_parquet(queried,index=False)
if queried.exists():sources.append(str(queried))
c.read_parquet(sources).create_view('all_sites')
c.execute('create view selected as select s.*,n.cell_types from all_sites s join neurons n on s.cell_root_id=n.root_id')
assert c.sql('select count(*)=count(distinct synapse_id) from selected').fetchone()[0]
assert c.sql("select count(*) from selected where mode<>'outgoing'").fetchone()[0]==0
c.execute('copy (select * from selected order by cell_root_id,synapse_id) to ? (format parquet,compression zstd)',[str(out/'points.parquet')])
n,cells=c.sql('select count(*),count(distinct cell_root_id) from selected').fetchone()
summary=dict(n_points=n,n_cells_with_points=cells,n_neurons=len(l.root_id.unique()),materialization_version=1718,mode='outgoing',coordinates='cell_x_nm,cell_y_nm,cell_z_nm',sources=sources,additional_queried_roots=missing)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
