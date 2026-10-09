"""Read-only completion audit of the full fragment/cell-type dataset."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

import duckdb
import lance
import pyarrow.parquet as pq

from scripts.predict_incoming_fragment_cell_types import DEFAULT,sha
from segclr_db import store as st
from segclr_db.hierarchies import casey_hierarchies


def audit(out,workers):
    out=out.resolve();dest=out/'cell_types'
    source_plan=json.loads((out/'plan.json').read_text())
    model_plan=json.loads((dest/'plan.json').read_text())
    summary=json.loads((dest/'summary.json').read_text())
    digest=sha(dest/'plan.json');source_digest=sha(out/'plan.json')
    assert summary['plan_sha256']==digest and summary['score_columns_validated']
    assert model_plan['fragment_plan_sha256']==source_digest
    expected={(family,classes,fold) for family in ['cave_n10','single_pre_post'] for classes in ['three','six'] for fold in range(5)}
    assert {(m['family'],m['classes'],m['fold']) for m in model_plan['models']}==expected
    assert len(model_plan['models'])==20
    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashes=list(pool.map(lambda m:sha(Path(m['path'])),model_plan['models']))
    assert hashes==[m['sha256'] for m in model_plan['models']]
    files=sorted((dest/'parts').glob('*/[0-9][0-9][0-9][0-9][0-9].parquet'))
    def footer(path):
        f=pq.ParquetFile(path);meta=f.schema_arrow.metadata
        assert meta[b'cell_type_plan_sha256']==digest.encode()
        assert meta[b'fragment_plan_sha256']==source_digest.encode()
        return f.metadata.num_rows
    with ThreadPoolExecutor(max_workers=workers) as pool:rows=sum(pool.map(footer,files))
    assert rows==source_plan['synapses']==summary['rows']==summary['distinct_synapses']
    c=duckdb.connect()
    c.execute(f'SET threads={workers}');c.execute("SET memory_limit='36GB'")
    c.execute('SET temp_directory=?',[str(dest/'completion_audit_tmp')])
    c.read_parquet(str(dest/'parts/*/[0-9][0-9][0-9][0-9][0-9].parquet')).create_view('predictions')
    source=c.read_parquet(str(out/'parts/*/[0-9][0-9][0-9][0-9][0-9].parquet'))
    source.create_view('source_fragments')
    columns=[name for name in source.columns if name not in ['mean_embedding','fragment_embeddings']]
    def differences(columns,other):
        projection=','.join('"'+name.replace('"','""')+'"' for name in columns)
        return c.sql(f'SELECT count(*) FROM (SELECT {projection} FROM predictions EXCEPT ALL SELECT {projection} FROM {other})').fetchone()[0]
    source_difference=differences(columns,'source_fragments')
    assert source_difference==0,'source identities, fragment data or pre classifications changed'
    source_rows,source_unique=c.sql('SELECT count(*),count(DISTINCT synapse_id) FROM source_fragments').fetchone()
    assert source_rows==source_unique==rows
    post=c.read_parquet(str(dest/'post_shards/rank=*/*.parquet'),hive_partitioning=False)
    post.create_view('source_posts')
    post_columns=[name for name in post.columns if name!='post_embedding']
    post_difference=differences(post_columns,'source_posts')
    assert post_difference==0,'postsynaptic data or classifications changed'
    assert c.sql('SELECT count(*) FROM source_posts').fetchone()[0]==rows
    targets=[int(r[0]) for r in c.sql('SELECT DISTINCT cell_root_id FROM predictions ORDER BY cell_root_id').fetchall()]
    assert targets==sorted(source_plan['neuron_roots'])
    partners=c.sql('SELECT count(DISTINCT partner_root_id) FROM predictions').fetchone()[0]
    assert partners==source_plan['partner_roots']
    missing=c.sql("SELECT count(*) FROM predictions WHERE status='missing_embeddings'").fetchone()[0]
    assert missing==json.loads((out/'summary.json').read_text())['missing_embeddings']
    c.close()
    for db in [out/'fragments.duckdb',dest/'predictions.duckdb']:
        c=duckdb.connect(str(db),read_only=True)
        views=c.sql('SELECT view_name,sql FROM duckdb_views() WHERE NOT internal').fetchall()
        for _,sql in views:
            assert '/orcd/scratch/orcd/013/jcbliao' not in sql,'view uses private scratch path'
        if db.name=='predictions.duckdb':
            names={name for name,_ in views}
            assert {f'{family}_{classes}_fold{fold}' for family,classes,fold in expected}<=names
            assert c.sql('SELECT count(*) FROM predictions').fetchone()[0]==rows
        c.close()
    staged=st.Store(dest/'staged_registry','microns')
    specs=lance.dataset(staged.table_uri('agg_specs')).to_table().to_pylist()
    expected_specs={'geodesic_mean_k10':('geodesic_mean',10),
        'pointwise_mlp_mean_k10':('pointwise_mlp_mean',10),'pointwise_mlp_mean_k1':('pointwise_mlp_mean',1)}
    assert {r['agg_spec_id'] for r in specs}==set(expected_specs)
    assert all(r['window_nm'] is None and (r['method'],r['k_nearest'])==expected_specs[r['agg_spec_id']] for r in specs)
    shared=st.Store(Path('/orcd/compute/sdorkenw/001/segclr-db'),'microns')
    hierarchy_table=lance.dataset(shared.table_uri('label_hierarchies'))
    hierarchies={h.hierarchy_id:h for h in casey_hierarchies()}
    clause='hierarchy_id IN ('+','.join(st.sql_literal(k) for k in hierarchies)+')'
    actual=hierarchy_table.scanner(filter=clause,use_scalar_index=False).to_table().to_pylist()
    assert len(actual)==3 and all(r['content_hash']==hierarchies[r['hierarchy_id']].content_hash for r in actual)
    parity=json.loads((dest/'real_data_parity_audit.json').read_text())
    assert parity['passed'] and len(parity['models'])==20
    assert parity['raw_fragment_mean_verified'] and parity['presynaptic_subcompartment_parity'] and parity['postsynaptic_single_point_subcompartment_parity']
    old=Path('/orcd/scratch/orcd/013/jcbliao/segclr')
    assert old.is_symlink() and old.resolve()==Path('/orcd/compute/sdorkenw/001/jcbliao/segclr')
    assert not old.with_name(old.name+'.relocation_backup').exists()
    report=dict(passed=True,synapses=rows,targets=len(targets),partners=partners,model_folds=20,
        prediction_records=sum(summary['counts'].values()),parts=len(files),
        all_source_fragment_fields_preserved=source_difference==0,
        all_postsynaptic_fields_preserved=post_difference==0,
        score_columns_validated=True,model_hashes_verified=True,part_metadata_verified=True,
        missing_selected_node_synapses_excluded=missing,shared_view_paths_verified=True,
        casey_hierarchies_verified=3,shared_hierarchy_registry_version=hierarchy_table.version,
        k_nearest_specs_staged=True,storage_move_verified=True,shared_database_predictions_written=False)
    (dest/'completion_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--workers',type=int,default=16);a=p.parse_args();audit(a.output,a.workers)
