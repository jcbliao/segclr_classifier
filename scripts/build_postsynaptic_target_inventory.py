"""Resume a v1718 outgoing-target inventory for the established Casey 0.7 cohort."""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import duckdb
import pandas as pd
from caveclient import CAVEclient
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.synapses import fetch_synapses, _retry

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'analysis/postsynaptic_targets_conf0.7'
COHORT = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_coarse_confidence_with_tc/scale16/k17/conf0.7/cohort.csv')
OUT.mkdir(exist_ok=True, parents=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--stage', choices=['prepare', 'skeletons', 'export', 'all'], default='all')
args = parser.parse_args()
c = duckdb.connect(str(OUT / 'targets.duckdb'))
c.execute('SET threads=2')
if args.stage in ('prepare', 'all'):
    c.execute('create or replace table cohort as select * from read_csv_auto(?, types={\'root_id\':\'BIGINT\',\'casey_cell_id\':\'VARCHAR\'})', [str(COHORT)])
    c.execute('create or replace table synapses as select s.* from read_parquet(?) s join cohort c on s.cell_root_id=c.root_id', [str(REPO/'data/synapse_cache/presynaptic_sites.parquet')])
    client = CAVEclient('minnie65_public', version=1718)
    missing = [r[0] for r in c.sql('select root_id from cohort except select distinct cell_root_id from synapses').fetchall()]
    extra = OUT / 'additional_synapses.parquet'
    if not extra.exists():
        frame = fetch_synapses(client, missing, 'outgoing', sleep_s=.2)
        frame.to_parquet(extra, index=False)
        (OUT/'additional_queries.json').write_text(json.dumps({'root_ids':missing, 'rows':len(frame), 'materialization':1718},indent=2))
    c.execute('insert into synapses select * from read_parquet(?)',[str(extra)])
    assert c.sql('select count(*)=count(distinct synapse_id) from synapses').fetchone()[0]
    c.execute('create or replace table connections as select cell_root_id as presynaptic_root_id, partner_root_id as postsynaptic_root_id, count(*) as n_synapses from synapses where partner_root_id<>0 group by 1,2')
    c.execute('create or replace table targets as select postsynaptic_root_id as root_id, sum(n_synapses)::BIGINT as n_synapses, count(*) as n_presynaptic_cells from connections group by 1')
    print('targets',c.sql('select count(*) from targets').fetchone(),flush=True)
    # Reuse previously measured segmented volumes; cache is for the same datastack.
    c.execute('create or replace table cached_volumes as select v.* from read_parquet(?) v join targets t using(root_id)',[str(REPO/'data/partner_volume_cache/parts/*.parquet')])
    assert c.sql('select count(*)=count(distinct root_id) from cached_volumes').fetchone()[0]
    print('cached volumes',c.sql('select count(*) from cached_volumes').fetchone(),flush=True)
c.execute('create table if not exists skeleton_status(root_id BIGINT, skeleton_version INTEGER, available BOOLEAN, checked_at VARCHAR, primary key(root_id,skeleton_version))')
if args.stage in ('skeletons', 'all'):
    client = CAVEclient('minnie65_public', version=1718)
    versions = sorted([v for v in client.skeleton.get_versions() if isinstance(v,int) and v>0], reverse=True)
    (OUT/'skeleton_versions.json').write_text(json.dumps(versions))
    print('skeleton versions',versions,flush=True)
    for version in versions:
        roots=[r[0] for r in c.execute('select root_id from targets except select root_id from skeleton_status where skeleton_version=?',[version]).fetchall()]
        for start in range(0,len(roots),1000):
            batch=roots[start:start+1000]
            result=_retry(lambda: client.skeleton.skeletons_exist(root_ids=batch,skeleton_version=version), what=f'skeleton existence v{version}')
            if isinstance(result,bool): result={batch[0]:result}
            assert set(result)==set(batch), 'Incomplete skeleton existence response'
            stamp=datetime.now(timezone.utc).isoformat()
            rows=pd.DataFrame([(r,version,result[r],stamp) for r in batch],columns=['root_id','skeleton_version','available','checked_at'])
            c.execute('insert or replace into skeleton_status select * from rows')
            if start%10000==0: print('skeleton',version,start,len(roots),flush=True)
versions = json.loads((OUT/'skeleton_versions.json').read_text()) if (OUT/'skeleton_versions.json').exists() else [4,3,2,1]
parts = sorted((OUT/'volume_cache/parts').glob('*.parquet'))
if parts:
    c.execute('create or replace table new_volumes as select * from read_parquet(?)', [[str(p) for p in parts]])
    c.execute('delete from cached_volumes where root_id in (select root_id from new_volumes)')
    c.execute('insert into cached_volumes select * from new_volumes')
c.execute(f'''create or replace table inventory as select t.*, v.n_l2_chunks, v.n_l2_sizes_missing,
case when v.error is null then v.volume_nm3 end as volume_nm3,
case when v.error is null then v.volume_nm3/1e9 end as volume_um3,
case when v.root_id is null then 'not_queried' when v.error is not null then 'error' when v.n_l2_sizes_missing>0 then 'lower_bound' else 'complete' end as volume_status,
v.error as volume_error, s.cave_skeleton_available, s.available_skeleton_versions
from targets t left join cached_volumes v using(root_id)
left join (select root_id, case when bool_or(available) then true when count(*)={len(versions)} then false else null end as cave_skeleton_available, string_agg(cast(skeleton_version as varchar), ',' order by skeleton_version) filter(where available) as available_skeleton_versions from skeleton_status group by root_id) s using(root_id)''')
assert c.sql('select count(*)=count(distinct root_id) from inventory').fetchone()[0]
assert c.sql('select sum(n_synapses) from targets').fetchone()[0] == c.sql('select count(*) from synapses where partner_root_id<>0').fetchone()[0]
c.execute(f"copy (select root_id from targets order by root_id) to '{OUT/'root_ids.txt'}' (header false)")
for table in ['inventory','connections','cohort','skeleton_status']:
    c.execute(f"copy (select * from {table}) to '{OUT/table}.parquet' (format parquet)")
c.execute(f"copy (select * from inventory order by n_synapses desc, root_id) to '{OUT/'targets.csv'}' (header, delimiter ',')")
summary={'cohort_path':str(COHORT),'datastack':'minnie65_public','materialization':1718,'built_at':datetime.now(timezone.utc).isoformat(),'n_cohort':c.sql('select count(*) from cohort').fetchone()[0],'n_synapses':c.sql('select count(*) from synapses').fetchone()[0],'unresolved_synapses':c.sql('select count(*) from synapses where partner_root_id=0').fetchone()[0],'n_targets':c.sql('select count(*) from targets').fetchone()[0],'volume_status':dict(c.sql('select volume_status,count(*) from inventory group by 1').fetchall()),'skeleton_available':c.sql('select count(*) from inventory where cave_skeleton_available').fetchone()[0],'skeleton_versions_requested':versions,'skeleton_status_counts':c.sql('select skeleton_version, count(*), count(*) filter(where available) from skeleton_status group by 1 order by 1').fetchall(),'skeleton_unknown':c.sql('select count(*) from inventory where cave_skeleton_available is null').fetchone()[0]}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if not (OUT/'missing_volume_roots.parquet').exists():
    c.execute(f"copy (select root_id as partner_root_id from inventory where volume_status in ('not_queried','error') order by root_id) to '{OUT/'missing_volume_roots.parquet'}' (format parquet)")
print(summary,flush=True)
