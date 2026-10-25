"""Import frozen incoming-fragment cell typing, without repeating inference.

preflight checks every referenced node and biological synapse and runs a real
Parquet round trip in an isolated v7 store. import requires that successful
preflight. Each rank/model is reconciled against both fact tables on resume;
partial writes are cleared only inside this deployment's own run/root scope.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
import time

# Use the pinned upstream checkout without changing the user's editable install.
checkout = Path(os.environ['SEGCLR_IMPORT_CHECKOUT'])
spec = importlib.util.spec_from_file_location('segclr_db', checkout/'src/__init__.py', submodule_search_locations=[str(checkout/'src')])
module = importlib.util.module_from_spec(spec)
sys.modules['segclr_db'] = module
spec.loader.exec_module(module)
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from segclr_db import store as st
from segclr_db.identity import PredictionRun
from segclr_db.writer import SegCLRWriter

SOURCE = Path(os.environ.get('SEGCLR_IMPORT_SOURCE','/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/incoming_presynaptic_fragments_k10_collina_v2'))
DB_ROOT = Path('/orcd/compute/sdorkenw/001/segclr-db')
STATE = Path(os.environ.get('SEGCLR_IMPORT_STATE','logs/presynaptic_import'))
NODE_SET = 'cave_skeleton_v4'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def file_fingerprint(files):
    return hashlib.sha256(json.dumps([[str(f), f.stat().st_size, f.stat().st_mtime_ns] for f in sorted(files)]).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temp.replace(path)


def prefix(m):
    return f"{m['family']}_{m['classes']}_fold{m['fold']}"


def ids(m, digest):
    # This is the import registration time, not the historical training launch.
    identity = json.loads((STATE/'run_identity.json').read_text())
    if identity['plan_sha256'] != digest:
        raise ValueError('registration identity belongs to another deployment')
    experiment = f"casey_{m['classes']}_{m['family']}_fold{m['fold']}_{m['sha256'][:12]}"
    run = experiment+'__'+identity['timestamp']
    prediction = run+'.checkpoint_best.incoming_fragments_'+digest[:16]
    return experiment, run, prediction


def artifacts():
    plan_path = SOURCE/'cell_types/plan.json'
    plan = json.loads(plan_path.read_text())
    digest = sha(plan_path)
    summary = json.loads((SOURCE/'cell_types/summary.json').read_text())
    audit = json.loads((SOURCE/'cell_types/completion_audit.json').read_text())
    if not audit['passed'] or summary['plan_sha256'] != digest or sha(SOURCE/'plan.json') != plan['fragment_plan_sha256']:
        raise ValueError('frozen plans/audit disagree')
    models = plan['models']
    expected = {(family, classes, fold) for family in ('cave_n10','single_pre_post') for classes in ('three','six') for fold in range(5)}
    if {(m['family'],m['classes'],m['fold']) for m in models} != expected:
        raise ValueError('expected exactly four configurations with five folds')
    files = sorted((SOURCE/'cell_types/parts').glob('*/*.parquet'))
    if len(files) != audit['parts']:
        raise ValueError('source part count changed')
    fingerprint = []
    rows = 0
    for f in files:
        pf = pq.ParquetFile(f)
        md = pf.schema_arrow.metadata or {}
        if md.get(b'cell_type_plan_sha256', b'').decode() != digest or md.get(b'fragment_plan_sha256', b'').decode() != plan['fragment_plan_sha256']:
            raise ValueError(f'part provenance changed: {f}')
        rows += pf.metadata.num_rows
        stat = f.stat()
        fingerprint.append([str(f.relative_to(SOURCE)), stat.st_size, stat.st_mtime_ns])
    if rows != audit['synapses']:
        raise ValueError('source row count changed')
    for m in models:
        if sha(m['path']) != m['sha256'] or sha(m['manifest']) != m['manifest_sha256']:
            raise ValueError(f'model artifact changed: {prefix(m)}')
    return plan, digest, summary, files, hashlib.sha256(json.dumps(fingerprint).encode()).hexdigest()


def model_columns(m):
    p = prefix(m)
    cols = ['cell_root_id','partner_root_id','anchor_node_id','synapse_id',p+'_included']
    for level in range(len(m['level_classes'])):
        cols += [f'{p}_level{level}_{suffix}' for suffix in ('class_index','logits','probabilities')]
    return cols


def payload(table, m):
    p = prefix(m)
    table = table.filter(table[p+'_included'])
    n = len(table)
    labels, uncertainty, logits = [], [], []
    for level, classes in enumerate(m['level_classes']):
        name = f'{p}_level{level}'
        index = np.asarray(table[name+'_class_index'], np.int64)
        scores = np.asarray(table[name+'_logits'].combine_chunks().values, np.float32).reshape(n,len(classes))
        probs = np.asarray(table[name+'_probabilities'].combine_chunks().values, np.float32).reshape(n,len(classes))
        if np.any(index < 0) or np.any(index >= len(classes)) or not np.isfinite(scores).all() or not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1):
            raise ValueError('invalid saved scores or class index')
        labels.append(np.asarray(classes,object)[index])
        uncertainty.append(1-probs[np.arange(n),index])
        logits.append(scores)
    raw_nodes = np.asarray(table['anchor_node_id'], np.int64)
    if np.any(raw_nodes < 0) or np.any(raw_nodes > np.iinfo(np.int32).max):
        raise ValueError('node ID outside v7 int32 range')
    return dict(root_ids=np.asarray(table['cell_root_id'],np.int64),
                subject_root_ids=np.asarray(table['partner_root_id'],np.int64),
                node_ids=raw_nodes.astype(np.int32), synapse_ids=np.asarray(table['synapse_id'],np.int64),
                labels=np.column_stack(labels), uncertainty=np.column_stack(uncertainty), logits=logits,
                node_set_id=NODE_SET)


def expected_tables(data):
    n = len(data['root_ids'])
    labels = pa.table(dict(root_id=pa.array(data['root_ids']), subject_root_id=pa.array(data['subject_root_ids']),
        node_id=pa.array(data['node_ids']), synapse_id=pa.array(data['synapse_ids']),
        pred_labels=pa.array(data['labels'].tolist(),pa.list_(pa.string())),
        pred_uncertainty=pa.array(data['uncertainty'].tolist(),pa.list_(pa.float32()))))
    depth = len(data['logits'])
    logits = pa.table(dict(root_id=pa.array(np.tile(data['root_ids'],depth)),
        node_id=pa.array(np.tile(data['node_ids'],depth)), synapse_id=pa.array(np.tile(data['synapse_ids'],depth)),
        level=pa.array(np.repeat(np.arange(depth,dtype=np.int8),n)),
        logits=pa.array([row for scores in data['logits'] for row in scores.tolist()],pa.list_(pa.float32()))))
    return labels, logits


def equal_rows(actual, expected, keys):
    if len(actual) != len(expected):
        return False
    actual = actual.cast(expected.schema)
    return actual.take(pc.sort_indices(actual,sort_keys=[(k,'ascending') for k in keys])).equals(
        expected.take(pc.sort_indices(expected,sort_keys=[(k,'ascending') for k in keys])))


def verify_group(store, run_id, roots, data):
    root_clause = ','.join(str(int(r)) for r in roots)
    predicate = f"prediction_run_id = {st.sql_literal(run_id)} AND node_set_id = {st.sql_literal(NODE_SET)} AND root_id IN ({root_clause})"
    expected = expected_tables(data)
    counts, matches = [], []
    for name, target in zip(('predictions','prediction_logits'), expected):
        actual = st.scan(store,name,columns=target.column_names,filter=predicate)
        counts.append(len(actual))
        matches.append(equal_rows(actual,target,['synapse_id'] if name=='predictions' else ['level','synapse_id']))
    return all(matches), counts, predicate


def reconcile_group(store, writer, run_id, roots, data):
    ok, counts, predicate = verify_group(store,run_id,roots,data)
    if not ok:
        if any(counts):
            print('reconciling partial group',run_id,counts,flush=True)
            st.delete(store,'predictions',predicate)
            st.delete(store,'prediction_logits',predicate)
        writer.add_predictions(run_id,**data)
        ok,counts,_ = verify_group(store,run_id,roots,data)
        if not ok:
            raise ValueError(f'post-write mismatch: {run_id}: {counts}')
    return counts


def publish_model_artifacts(models):
    """Copy immutable weights and provenance out of the private home directory."""
    paths = {}
    root = SOURCE/'cell_types/models'
    root.mkdir(exist_ok=True,mode=0o2775)
    root.chmod(0o2775)
    for m in models:
        directory = root/m['sha256']
        directory.mkdir(exist_ok=True,mode=0o2775)
        directory.chmod(0o2775)
        sources = [(Path(m['path']),directory/('checkpoint_'+m['sha256']+'.pt'),m['sha256']),
                   (Path(m['manifest']),directory/'manifest.json',m['manifest_sha256'])]
        training = Path(m['path']).parent.with_suffix('.json')
        sources.append((training,directory/'training_result.json',sha(training)))
        for source,dest,expected in sources:
            if not dest.exists():
                temp = dest.with_suffix(dest.suffix+'.tmp')
                shutil.copyfile(source,temp)
                if sha(temp) != expected:
                    raise ValueError('copied artifact hash mismatch')
                temp.chmod(0o664)
                temp.replace(dest)
            if sha(dest) != expected:
                raise ValueError(f'published artifact changed: {dest}')
        save(directory/'frozen_model.json',m)
        (directory/'frozen_model.json').chmod(0o664)
        paths[prefix(m)] = str(sources[0][1])
    return paths


def register(store, plan, digest, models, shared_paths=None):
    writer = SegCLRWriter(store=store)
    for m in models:
        experiment, run_id, prediction_id = ids(m,digest)
        training_path = Path(m['path']).parent.with_suffix('.json')
        training = json.loads(training_path.read_text())
        full_spec = dict(frozen_model=m, training_args=training['args'], training_result_path=str(training_path),
            run_timestamp_semantics='Import registration time in UTC; original training launch time is not asserted.',
            training_result_sha256=sha(training_path), manifest_path=m['manifest'], manifest_sha256=m['manifest_sha256'],
            deployment_plan_sha256=digest, fragment_plan_sha256=plan['fragment_plan_sha256'],
            deployment_source=str(SOURCE), postsynaptic_input='exact matched postsynaptic point embedding; 64D concatenation',
            selection_filter=plan['filter'], score_semantics=plan['scores'],
            uncertainty='1 minus saved joint probability at the saved top-down class, per level',
            training_data_links='Not asserted: original label-version/node-label provenance has not been verified; frozen manifest and training arguments retained.')
        if 'axon_filter_embedding_run_id' in plan:
            full_spec['axon_filter_embedding_run_id']=plan['axon_filter_embedding_run_id']
        writer.register_agg_spec(method='pointwise_mlp_mean',k_nearest=m['aggregation']['k_nearest'],agg_spec_id=m['agg_spec_id'])
        writer.register_experiment(experiment,full_spec,kind='classifier',hierarchy_id=m['hierarchy_id'],
            notes='Frozen presynaptic classifier artifact; learned pooling belongs to these weights. Exact class/head order in spec.')
        timestamp = json.loads((STATE/'run_identity.json').read_text())['timestamp']
        run = writer.register_run(experiment,Path(m['path']).parent,timestamp,seed=training['args'].get('seed'))
        if run.run_id != run_id:
            raise ValueError('run identity mismatch')
        checkpoint_path = shared_paths[prefix(m)] if shared_paths else m['path']
        writer.register_checkpoints(run_id,[('checkpoint_best',int(training['best_epoch']),-1,checkpoint_path)])
        writer.register_prediction_run(PredictionRun(classifier_run_id=run_id,classifier_checkpoint_id='checkpoint_best',
            embedding_run_id=plan['embedding_run_id'],embedding_checkpoint_id=plan['embedding_checkpoint_id'],
            agg_spec_id=m['agg_spec_id'],prediction_run_id=prediction_id,
            notes=json.dumps(dict(source=str(SOURCE/'cell_types'), plan_sha256=digest, model_sha256=m['sha256'],
                input='presynaptic learned pooling plus matched postsynaptic point',filter=plan['filter'],
                scores=plan['scores'],uncertainty='1 minus chosen-class joint probability',
                probabilities='Full joint probabilities retained in source Parquet; decode conditional logits using frozen head topology.'))))
    return writer


def pilot(shared, plan, digest, files):
    with tempfile.TemporaryDirectory(prefix='segclr_import_pilot_',dir=STATE) as tmp:
        store = st.init_store(tmp,'microns',datastack='minnie65_phase3_v1',mat_version=1718)
        # Copy actual provenance rows rather than inventing test identities.
        for name in ('label_hierarchies','experiments','runs','checkpoints'):
            st.append(store,name,st.scan(shared,name))
        for m in plan['models']:
            table = pq.read_table(files[0],columns=model_columns(m)).slice(0,256)
            data = payload(table,m)
            writer = register(store,plan,digest,[m])
            run_id = ids(m,digest)[2]
            roots = np.unique(np.asarray(table['cell_root_id'],np.int64))
            # Exercise a crash between label and logit commits, then a repeat.
            incomplete = dict(data,logits=None,record_work=False)
            writer.add_predictions(run_id,**incomplete)
            reconcile_group(store,writer,run_id,roots,data)
            reconcile_group(store,writer,run_id,roots,data)
            ok,counts,_ = verify_group(store,run_id,roots,data)
            if not ok:
                raise ValueError(f'round-trip mismatch for {prefix(m)}: {counts}')
    return {'model_folds':len(plan['models']),'exact_labels_uncertainty_logits_and_synapse_identities':True,
            'interrupted_write_recovery_and_duplicate_free_retry':True}


def preflight():
    STATE.mkdir(parents=True,exist_ok=True)
    plan,digest,summary,files,fingerprint = artifacts()
    if not (STATE/'run_identity.json').exists():
        save(STATE/'run_identity.json',dict(plan_sha256=digest,timestamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')))
    store = st.open_store(DB_ROOT,'microns')
    hierarchies = {r['hierarchy_id']:r for r in st.scan(store,'label_hierarchies').to_pylist()}
    for m in plan['models']:
        if json.loads(hierarchies[m['hierarchy_id']]['level_classes_json']) != m['level_classes']:
            raise ValueError('hierarchy class order changed')
    if st.read_meta(store).mat_version != 1718:
        raise ValueError('materialization changed')
    report = dict(passed=False,plan_sha256=digest,source_fingerprint=fingerprint,
        importer_sha256=sha(__file__), source=str(SOURCE), database_root=str(DB_ROOT), models=[dict(model=prefix(m),experiment_id=ids(m,digest)[0],classifier_run_id=ids(m,digest)[1],prediction_run_id=ids(m,digest)[2],expected_predictions=summary['counts'][prefix(m)]) for m in plan['models']])
    print('verifying frozen artifacts and real v7 round trips',flush=True)
    report['pilot'] = pilot(store,plan,digest,files)
    save(STATE/'preflight.json',report)
    # Only narrow identity columns are read; wide embeddings and score columns
    # never enter this join. DuckDB may spill to the allocated scratch disk.
    scratch = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_import')
    scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as tmp, duckdb.connect(str(Path(tmp)/'preflight.duckdb')) as c:
        c.execute("SET threads=4")
        c.execute("SET memory_limit='20GB'")
        c.execute('SET temp_directory=?',[str(Path(tmp)/'spill')])
        c.read_parquet([str(f) for f in files],hive_partitioning=False).create_view('source')
        included = ' OR '.join('"'+prefix(m)+'_included"' for m in plan['models'])
        c.execute(f'CREATE TABLE refs AS SELECT cell_root_id,partner_root_id,anchor_node_id,synapse_id FROM source WHERE {included}')
        report['included_synapses'],unique = c.sql('SELECT count(*),count(DISTINCT synapse_id) FROM refs').fetchone()
        if unique != report['included_synapses']:
            raise ValueError('duplicate synapse identities')
        report['table_versions'] = {}
        ds = st.open_table(store,'synapses')
        report['table_versions']['synapses'] = ds.version
        c.register('catalog',ds.scanner(columns=['synapse_id','pre_root_id','post_root_id']).to_reader())
        print('checking biological synapse foreign keys',flush=True)
        bad = c.sql('SELECT r.* FROM refs r ANTI JOIN catalog s ON r.synapse_id=s.synapse_id AND r.partner_root_id=s.pre_root_id AND r.cell_root_id=s.post_root_id').to_arrow_table()
        report['missing_or_mismatched_synapses'] = len(bad)
        save(STATE/'preflight.json',report)
        if len(bad):
            report['missing_synapse_examples'] = bad.slice(0,20).to_pylist()
            save(STATE/'preflight.json',report)
            raise ValueError(f'{len(bad)} saved synapses absent or mismatched in the shared catalog')
        c.execute('CREATE TABLE wanted_nodes AS SELECT DISTINCT partner_root_id AS root_id,anchor_node_id AS node_id FROM refs')
        ds = st.open_table(store,'nodes')
        report['table_versions']['nodes'] = ds.version
        # Some shared index files are owner-only. A sequential data scan avoids
        # depending on those indexes and still checks every referenced node.
        c.register('all_nodes',ds.scanner(columns=['root_id','node_id'],filter=f"node_set_id = '{NODE_SET}'",use_scalar_index=False).to_reader())
        save(STATE/'preflight.json',report)
        print('checking exact skeleton-node foreign keys against',ds.count_rows(),'nodes',flush=True)
        c.execute('CREATE TABLE found_nodes AS SELECT n.root_id,n.node_id FROM all_nodes n SEMI JOIN wanted_nodes w USING(root_id,node_id)')
        missing = c.sql('SELECT * FROM wanted_nodes ANTI JOIN found_nodes USING(root_id,node_id)').to_arrow_table()
        report['missing_nodes'] = len(missing)
        report['missing_node_examples'] = missing.slice(0,20).to_pylist()
        report['unique_referenced_nodes'] = c.sql('SELECT count(*) FROM wanted_nodes').fetchone()[0]
        save(STATE/'preflight.json',report)
        if len(missing):
            save(STATE/'preflight.json',report)
            raise ValueError(f'{len(missing)} saved anchors absent in shared nodes')
        # Source ranks split target cells. Repartition once by target identity
        # so each retry boundary contains whole cells and bounded input volume.
        staged = scratch/('by_target_'+digest[:16])
        marker = staged/'ready.json'
        if marker.exists():
            ready = json.loads(marker.read_text())
            if ready['source_fingerprint'] != fingerprint or ready['files_fingerprint'] != file_fingerprint(list(staged.glob('*/*.parquet'))):
                raise ValueError('staged import inputs changed')
        else:
            if staged.exists():
                raise ValueError('incomplete staged input directory requires inspection')
            temp_stage = Path(tmp)/'by_target'
            columns = sorted(set(col for m in plan['models'] for col in model_columns(m)))
            query = 'SELECT '+','.join('"'+col+'"' for col in columns)+', CAST(hash(cell_root_id) % 32 AS INTEGER) AS import_group FROM source'
            print('repartitioning saved scores into 32 groups of whole target cells',flush=True)
            c.sql(query).write_parquet(str(temp_stage),partition_by=['import_group'],compression='zstd',row_group_size=32768)
            temp_stage.rename(staged)
            save(marker,dict(source_fingerprint=fingerprint,files_fingerprint=file_fingerprint(list(staged.glob('*/*.parquet')))))
        staged_files = sorted(staged.glob('*/*.parquet'))
        report['staged_files_fingerprint'] = file_fingerprint(staged_files)
        report['group_files'] = {directory.name:[str(f) for f in sorted(directory.glob('*.parquet'))] for directory in sorted(staged.iterdir()) if directory.is_dir()}
        c.read_parquet([str(f) for f in staged_files],filename=True,hive_partitioning=False).create_view('staged')
        if c.sql('SELECT count(*) FROM staged').fetchone()[0] != summary['rows']:
            raise ValueError('staged source row count changed')
        c.execute("CREATE TABLE membership AS SELECT DISTINCT cell_root_id, regexp_extract(filename,'/(import_group=[^/]+)/',1) AS rank FROM staged")
        if c.sql('SELECT count(*) FROM (SELECT cell_root_id FROM membership GROUP BY cell_root_id HAVING count(*) != 1)').fetchone()[0]:
            raise ValueError('a target belongs to multiple ranks')
        report['rank_roots'] = {rank:[r[0] for r in c.execute('SELECT cell_root_id FROM membership WHERE rank=? ORDER BY cell_root_id',[rank]).fetchall()] for (rank,) in c.sql('SELECT DISTINCT rank FROM membership').fetchall()}
        c.close()
    report['passed'] = True
    save(STATE/'preflight.json',report)
    print('PREFLIGHT PASSED',json.dumps({k:v for k,v in report.items() if k not in ('rank_roots','models')}),flush=True)


def run_import():
    # A single importer owns this deployment; parallel duplicate launches fail.
    import fcntl
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'import.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan,digest,summary,files,fingerprint = artifacts()
        check = json.loads((STATE/'preflight.json').read_text())
        if not check['passed'] or check['plan_sha256'] != digest or check['source_fingerprint'] != fingerprint or check['importer_sha256'] != sha(__file__):
            raise ValueError('successful preflight for these exact artifacts/importer is required')
        staged_files = [Path(f) for group in check['group_files'].values() for f in group]
        if file_fingerprint(staged_files) != check['staged_files_fingerprint']:
            raise ValueError('staged input files changed after preflight')
        store = st.open_store(DB_ROOT,'microns')
        shared_paths = publish_model_artifacts(plan['models'])
        writer = register(store,plan,digest,plan['models'],shared_paths)
        progress_path = STATE/'progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else dict(plan_sha256=digest,complete=False,groups={})
        if progress['plan_sha256'] != digest:
            raise ValueError('progress belongs to another deployment')
        total = {prefix(m):0 for m in plan['models']}
        for rank,roots in sorted(check['rank_roots'].items()):
            rank_files = check['group_files'][rank]
            for m in plan['models']:
                p = prefix(m)
                table = pq.ParquetDataset(rank_files,partitioning=None).read(columns=model_columns(m))
                data = payload(table,m)
                prediction_id = ids(m,digest)[2]
                # Recovery touches only this deployment's run and rank targets.
                counts = reconcile_group(store,writer,prediction_id,roots,data)
                total[p] += len(data['root_ids'])
                progress['groups'][rank+'/'+p] = dict(predictions=counts[0],logits=counts[1],verified=True)
                save(progress_path,progress)
                print('verified',rank,p,counts,flush=True)
        if total != summary['counts']:
            raise ValueError(f'import totals disagree: {total}')
        # Check whole-run counts too: catches unexpected rows outside all groups.
        for m in plan['models']:
            predicate = f"prediction_run_id = {st.sql_literal(ids(m,digest)[2])}"
            expected = total[prefix(m)]
            if st.count_rows(store,'predictions',filter=predicate) != expected or st.count_rows(store,'prediction_logits',filter=predicate) != expected*len(m['level_classes']):
                raise ValueError('whole-run row count mismatch')
        progress.update(complete=True,counts=total,total_predictions=sum(total.values()))
        save(progress_path,progress)
        save(SOURCE/'cell_types/segclr_db_import.json',dict(progress,models=check['models'],database_root=str(DB_ROOT),node_set_id=NODE_SET))
        print('IMPORT COMPLETE',sum(total.values()),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('preflight','import'))
    args = parser.parse_args()
    if args.command == 'preflight':
        preflight()
    else:
        run_import()
