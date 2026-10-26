"""Publish frozen five-fold radius predictions, with exact/resumable verification.

Run through scripts/sbatch/export_presynaptic_radius.sh. Does not run inference.
Only this content-addressed deployment's fact rows may be reconciled on retry.
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import importlib.util
import json
import os
import multiprocessing
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import time

# Bound scanner prefetch independently of the number of fold workers.
os.environ.setdefault('LANCE_DEFAULT_IO_BUFFER_SIZE', str(64 * 2**20))
ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = Path(os.environ.get('SEGCLR_EXPORT_CHECKOUT', ROOT / 'logs/segclr_radius_export_upstream'))
if not (CHECKOUT/'src/__init__.py').is_file():
    raise FileNotFoundError('Set SEGCLR_EXPORT_CHECKOUT to the SegCLR DB source checkout; '
                            'see docs/presynaptic_geodesic20um.md for the pinned setup.')
spec = importlib.util.spec_from_file_location('segclr_db', CHECKOUT/'src/__init__.py',
    submodule_search_locations=[str(CHECKOUT/'src')])
module = importlib.util.module_from_spec(spec)
sys.modules['segclr_db'] = module
spec.loader.exec_module(module)

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from segclr_db import store as st
from segclr_db.database import SegCLRDatabase
from segclr_db.identity import LabelHierarchy, PredictionRun
from segclr_db.skeletons import skeleton_content_hash
from segclr_db.writer import SegCLRWriter

RUN = Path(os.environ.get('PRESYNAPTIC_STATE', ROOT / 'analysis/presynaptic/latest100k_geodesic20um'))
STATE = RUN / 'segclr_db_export'
DB_ROOT = Path('/orcd/compute/sdorkenw/001/segclr-db')
POST = Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/latest100k_morpho6_centers_v1/training_post')
NODE_SET = 'cave_skeleton_v4'
SCAN_OPTIONS = dict(batch_size=65536, batch_readahead=1, fragment_readahead=1,
                    io_buffer_size=64 * 2**20)
WORKER_MEMORY_MIB = 8192
COORDINATOR_MEMORY_MIB = 8192


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temp.chmod(0o664)
    temp.replace(path)


def copy_verified(source, dest):
    source, dest = Path(source), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True, mode=0o2775)
    digest = sha(source)
    if not dest.exists():
        temp = dest.with_suffix(dest.suffix+'.tmp')
        shutil.copyfile(source, temp)
        if sha(temp) != digest:
            raise ValueError(f'copy mismatch: {source}')
        temp.chmod(0o664)
        temp.replace(dest)
    if sha(dest) != digest:
        raise ValueError(f'immutable shared artifact changed: {dest}')
    return digest


def matrix(table, name, width, dtype=np.float32):
    return np.asarray(table[name].combine_chunks().values, dtype=dtype).reshape(len(table), width)


def payload(table, model):
    n = len(table)
    levels, heads = model['level_classes'], model['local_heads']
    logits = [np.zeros((n, len(v)), np.float32) for v in levels]
    for i, head in enumerate(heads):
        scores = matrix(table, f'head_{i}_logits', head['n_children'])
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite logits')
        exp = np.exp(scores-scores.max(axis=1, keepdims=True))
        conditional = exp/exp.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(conditional,
            matrix(table, f'head_{i}_probabilities', head['n_children']), rtol=2e-5, atol=2e-6)
        logits[head['child_level']][:, head['child_global_indices']] = scores
    indices = matrix(table, 'top_down_predictions', len(levels), np.int64)
    labels, uncertainty = [], []
    for level, classes in enumerate(levels):
        selected = indices[:, level]
        assert np.all((selected >= 0) & (selected < len(classes)))
        probs = matrix(table, f'level_{level}_joint_probabilities', len(classes))
        assert np.isfinite(probs).all() and np.all((probs >= 0) & (probs <= 1))
        np.testing.assert_allclose(probs.sum(axis=1), 1, rtol=2e-5, atol=2e-6)
        labels.append(np.asarray(classes, object)[selected])
        uncertainty.append(1-probs[np.arange(n), selected])
    # The stored labels must form a legal top-down path, including unary branches.
    label_matrix = np.column_stack(labels)
    allowed = {tuple(v) for v in model['label_paths'].values()}
    assert all(tuple(row) in allowed for row in np.unique(label_matrix.astype(str), axis=0))
    return dict(root_ids=np.asarray(table['postsynaptic_root_id'], np.int64),
        subject_root_ids=np.asarray(table['root_id'], np.int64),
        node_ids=np.asarray(table['canonical_anchor_node_id'], np.int32),
        synapse_ids=np.asarray(table['synapse_id'], np.int64), labels=label_matrix,
        uncertainty=np.column_stack(uncertainty), logits=logits, node_set_id=NODE_SET,
        record_work=False)


def expected_tables(data):
    n = len(data['root_ids'])
    labels = pa.table(dict(root_id=data['root_ids'], subject_root_id=data['subject_root_ids'],
        node_id=data['node_ids'], synapse_id=data['synapse_ids'],
        pred_labels=pa.array(data['labels'].tolist(), pa.list_(pa.string())),
        pred_uncertainty=pa.array(data['uncertainty'].tolist(), pa.list_(pa.float32()))))
    depth = len(data['logits'])
    logits = pa.table(dict(root_id=np.tile(data['root_ids'], depth),
        node_id=np.tile(data['node_ids'], depth), synapse_id=np.tile(data['synapse_ids'], depth),
        level=np.repeat(np.arange(depth, dtype=np.int8), n),
        logits=pa.array([row for scores in data['logits'] for row in scores.tolist()], pa.list_(pa.float32()))))
    return labels, logits


def verify(store, prediction_id, data):
    predicate = f'prediction_run_id = {st.sql_literal(prediction_id)}'
    counts, matches = [], []
    for name, target in zip(('predictions', 'prediction_logits'), expected_tables(data)):
        actual = (st.open_table(store, name).scanner(columns=target.column_names,
            filter=predicate, **SCAN_OPTIONS).to_table() if st.table_exists(store, name)
            else target.slice(0, 0))
        counts.append(len(actual))
        keys = [('synapse_id', 'ascending')]
        if name == 'prediction_logits':
            keys.insert(0, ('level', 'ascending'))
        matches.append(len(actual) == len(target) and
            actual.cast(target.schema).take(pc.sort_indices(actual, sort_keys=keys)).equals(
                target.take(pc.sort_indices(target, sort_keys=keys))))
    return all(matches), counts, predicate


def reconcile(store, writer, model, data):
    prediction_id = model['prediction_run_id']
    ok, counts, predicate = verify(store, prediction_id, data)
    if not ok:
        if any(counts):
            print('Repairing incomplete rows for this deployment only:', prediction_id, counts, flush=True)
            for name in ('predictions', 'prediction_logits'):
                # Other folds may commit while this fold repairs its own rows.
                for attempt in range(8):
                    try:
                        st.delete(store, name, predicate)
                        break
                    except Exception as exc:
                        if not st._is_commit_conflict(exc) or attempt == 7:
                            raise
                        time.sleep(0.1 * 2**attempt)
        writer.add_predictions(prediction_id, **data)
        ok, counts, _ = verify(store, prediction_id, data)
        if not ok:
            raise ValueError(f'exact database round-trip failed: {prediction_id}: {counts}')
    return counts


def register(store, context, models):
    writer = SegCLRWriter(store=store)
    writer.register_hierarchy(context['hierarchy'])
    for model in models:
        # Each learned pooling spec is tied to its fold's checkpoint.
        writer.register_agg_spec(method='pointwise_mlp_mean',
            window_nm=int(context['plan']['window_nm']) if context['plan'].get('k_observed') is None else None,
            k_nearest=context['plan'].get('k_observed'),
            params=dict(selection=context['plan']['selection'], presynaptic_seeded_only=True,
                soma_exclusion_nm=5000, soma_policy='Preserve source cut; explicitly soma-free records retain full skeleton. Per-cell flags in canonical_skeleton_audit.json.',
                source_anchor_eligibility=context['plan']['anchor_match_policy'],
                checkpoint_sha256=model['checkpoint_sha256'], pooling='mean of learned two-layer pointwise MLP outputs',
                postsynaptic_input='64D matched postsynaptic embedding concatenated after pooling',
                source_plan_sha256=context['digest']), agg_spec_id=model['agg_spec_id'])
        writer.register_experiment(model['experiment_id'], dict(model=model,
            plan=context['plan'], shared_bundle=str(context['bundle']),
            score_semantics=context['score_semantics'],
            uncertainty='1 minus joint probability at selected top-down class; not SNGP variance',
            evaluation_population='held-out membership-deduplicated presynaptic windows, one fold per cell',
            run_timestamp_semantics='UTC export registration time, not original training start'),
            kind='classifier', hierarchy_id=context['hierarchy'].hierarchy_id,
            notes='Presynaptic-seeded learned pooling, all branching local heads, full neuron morphology hierarchy')
        registered = writer.register_run(model['experiment_id'], Path(model['shared_checkpoint']).parent,
            context['timestamp'], seed=model['training_args'].get('seed'))
        assert registered.run_id == model['classifier_run_id']
        writer.register_checkpoints(registered.run_id,
            [('checkpoint_best', model['best_epoch'], -1, model['shared_checkpoint'])])
        writer.register_prediction_run(PredictionRun(classifier_run_id=registered.run_id,
            classifier_checkpoint_id='checkpoint_best', embedding_run_id=context['plan']['feature_run_id'],
            embedding_checkpoint_id='checkpoint_e0_s100000', agg_spec_id=model['agg_spec_id'],
            window_nm=int(context['plan']['window_nm']) if context['plan'].get('k_observed') is None else None,
            prediction_run_id=model['prediction_run_id'],
            notes=json.dumps(dict(source=str(context['bundle']), plan_sha256=context['digest'],
                score_semantics=context['score_semantics'], local_heads=model['local_heads'],
                probabilities=f"{context['bundle']}/all_lcpn_predictions/fold{model['fold']}.parquet",
                identity='root_id=postsynaptic target; subject_root_id=classified presynaptic cell; node_id=canonical presynaptic anchor',
                uncertainty='1 minus selected-class joint probability'))))
    return writer


def prepare(store):
    plan, complete = read(RUN/'plan.json'), read(RUN/'completion.json')
    digest = sha(RUN/'plan.json')
    assert complete['complete'] and complete['plan_sha256'] == digest
    assert complete['classified_windows'] == sum(m['test_windows'] for m in plan['models'])
    assert st.read_meta(store).mat_version == 1718
    if st.schema_drift(store):
        raise ValueError(f'schema drift: {st.schema_drift(store)}')
    identity_file = STATE/'run_identity.json'
    if not identity_file.exists():
        save(identity_file, dict(plan_sha256=digest,
            timestamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')))
    identity = read(identity_file)
    assert identity['plan_sha256'] == digest
    tag = f'geo{int(plan["window_nm"]):d}nm' if plan.get('k_observed') is None else f'k{plan["k_observed"]}'
    # Preserve the already-published 20um deployment's identity on retry.
    legacy = RUN.name == 'latest100k_geodesic20um'
    bundle_tag = 'geodesic20um' if legacy else tag
    bundle = Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows') / (f'latest100k_{bundle_tag}_full_hierarchy_'+digest[:16])
    bundle.mkdir(exist_ok=True, mode=0o2775)
    bundle.chmod(0o2775)
    manifest = read(RUN/'manifests/fold0.json')
    hierarchy = LabelHierarchy.from_tree('morpho_neuron_full_v1718_'+digest[:12],
        {'neuron': manifest['hierarchy_tree']['neuron']})
    assert [list(x) for x in hierarchy.level_classes] == plan['hierarchy']['level_classes']
    context = dict(plan=plan, digest=digest, timestamp=identity['timestamp'], bundle=bundle,
        hierarchy=hierarchy, score_semantics='Raw conditional logits from EVERY local head, scattered into registered level class order. Apply softmax separately over each head child_global_indices, never over the whole level. Unary positions have zero logits. Top-down labels use parent routing. Full conditional and joint probabilities are preserved in shared source Parquet.')
    for filename in ('plan.json', 'completion.json'):
        copy_verified(RUN/filename, bundle/filename)
    copy_verified(__file__, bundle/('export_presynaptic_radius_db_'+sha(__file__)[:12]+'.py'))
    models = []
    for fold in range(5):
        m = read(RUN/f'all_lcpn_predictions/fold{fold}.json')
        assert m['plan_sha256'] == digest and m['fold'] == fold
        assert sha(m['checkpoint']) == m['checkpoint_sha256']
        assert sha(RUN/f'manifests/fold{fold}.json') == m['manifest_sha256']
        assert sha(RUN/f'all_lcpn_predictions/fold{fold}.parquet') == m['parquet_sha256']
        assert m['local_heads'] == plan['hierarchy']['local_heads']
        assert m['level_classes'] == plan['hierarchy']['level_classes']
        training = read(Path(m['checkpoint']).parent.with_suffix('.json'))
        m = {k: m[k] for k in ('fold','windows','checkpoint_sha256','parquet_sha256','manifest_sha256','local_heads','level_classes')}
        m.update(best_epoch=int(training['best_epoch']), training_args=training['args'],
            label_paths=hierarchy.label_paths, hierarchy_id=hierarchy.hierarchy_id)
        model_tag = 'geo20um' if legacy else tag
        experiment = f'presynaptic_{model_tag}_full_lcpn_fold{fold}_{digest[:12]}'
        m.update(experiment_id=experiment, classifier_run_id=experiment+'__'+identity['timestamp'],
            agg_spec_id=f'pre_{model_tag}_pointwise_mlp_fold{fold}_{m["checkpoint_sha256"][:12]}')
        m['prediction_run_id'] = m['classifier_run_id']+'.checkpoint_best.heldout_all_heads'
        model_dir = bundle/'models'/f'fold{fold}'
        checkpoint = read(RUN/f'all_lcpn_predictions/fold{fold}.json')['checkpoint']
        m['shared_checkpoint'] = str(model_dir/'checkpoint_best.pt')
        for source, dest in ((checkpoint, model_dir/'checkpoint_best.pt'),
                (Path(checkpoint).parent.with_suffix('.json'), model_dir/'training_result.json'),
                (RUN/f'configs/{fold:02d}.json', model_dir/'config.json'),
                (RUN/f'manifests/fold{fold}.json', model_dir/'manifest.json'),
                (RUN/f'all_lcpn_predictions/fold{fold}.parquet', bundle/f'all_lcpn_predictions/fold{fold}.parquet'),
                (RUN/f'all_lcpn_predictions/fold{fold}.json', bundle/f'all_lcpn_predictions/fold{fold}.json')):
            copy_verified(source, dest)
        save(model_dir/'registration.json', m)
        models.append(m)
    context['models'] = models
    return context


def map_anchors(store, context):
    hashes = {int(r['root_id']):r['content_hash'] for r in st.scan(store, 'node_manifest',
        columns=['root_id','content_hash'], filter=f"node_set_id='{NODE_SET}'").to_pylist()}
    radius = Path(read(RUN/'configs/00.json')['presynaptic_database'])
    source = Path(read(ROOT/'analysis/db_deployment/presynaptic/latest_segclr/morpho6/configs/00.json')['presynaptic_database'])
    records = []
    with duckdb.connect() as c:
        c.execute('SET threads=4')
        c.execute("SET memory_limit='16GB'")
        c.read_parquet(str(POST/'sites/*.parquet')).create_view('sites')
        for model in context['models']:
            fold = model['fold']
            table = pq.read_table(RUN/f'all_lcpn_predictions/fold{fold}.parquet')
            pre = np.asarray(table['root_id'], np.int64)
            window = np.asarray(table['window_row'], np.int64)
            synapses = np.asarray(table['synapse_id'], np.int64)
            anchor = np.full(len(table), -1, np.int32)
            split = read(RUN/f'manifests/fold{fold}.json')['cells']
            for i, rid in enumerate(np.unique(pre)):
                assert split[str(rid)]['split'] == 'test'
                at = np.flatnonzero(pre == rid)
                with (ROOT/'data/skeleton_cache'/f'{rid}.pkl').open('rb') as f:
                    skeleton = pickle.load(f)
                assert skeleton_content_hash(skeleton) == hashes[int(rid)], f'canonical skeleton changed: {rid}'
                with np.load(source/'cells'/f'{rid}.npz') as old:
                    center = old['root_xyz_nm'].astype(np.float64)
                    soma_cut = bool(old['soma_cut_applied']) if 'soma_cut_applied' in old else True
                if soma_cut:
                    assert np.isfinite(center).all(), f'missing soma-cut center: {rid}'
                    keep = np.linalg.norm(skeleton.coords.astype(np.float64)-center, axis=1) > 5000
                else:
                    keep = np.ones(len(skeleton.coords), dtype=bool)
                canonical = np.flatnonzero(keep)
                with np.load(radius/'cells'/f'{rid}.npz') as z:
                    np.testing.assert_array_equal(skeleton.coords[keep].astype(np.float32), z['cave_pos_nm'])
                    remap = np.full(len(keep), -1, np.int64)
                    remap[keep] = np.arange(len(canonical))
                    edges = skeleton.edges
                    np.testing.assert_array_equal(remap[edges[keep[edges[:,0]] & keep[edges[:,1]]]], z['cave_edges'])
                    rows = window[at]
                    np.testing.assert_array_equal(z['cave_synapse_id'][rows], synapses[at])
                    assert z['cave_valid_k_window'][rows].all()
                    local = z['cave_nearest_observed_node'][rows]
                    assert np.isin(local, z['cave_observed_node_ids']).all()
                    anchor[at] = canonical[local]
                records.append(dict(root_id=int(rid), content_hash=hashes[int(rid)], fold=fold,
                    source_soma_cut_applied=soma_cut))
                if i % 100 == 0:
                    print('Canonical anchor audit', fold, i, flush=True)
            assert (anchor >= 0).all()
            c.register('wanted', pa.table(dict(synapse_id=synapses, presynaptic_root_id=pre, row=np.arange(len(table)))))
            matched = c.sql('SELECT w.row,s.postsynaptic_root_id FROM wanted w JOIN sites s USING(synapse_id,presynaptic_root_id) ORDER BY w.row').to_arrow_table()
            np.testing.assert_array_equal(np.asarray(matched['row']), np.arange(len(table)))
            post = np.asarray(matched['postsynaptic_root_id'], np.int64)
            assert (post > 0).all()
            mapping = pa.table(dict(synapse_id=synapses, presynaptic_root_id=pre,
                postsynaptic_root_id=post, canonical_anchor_node_id=anchor))
            path = context['bundle']/f'anchor_mappings/fold{fold}.parquet'
            path.parent.mkdir(exist_ok=True, mode=0o2775)
            pq.write_table(mapping, path, compression='zstd')
            path.chmod(0o664)
            # Validate all logits/probabilities before any shared registry/fact writes.
            payload(table.append_column('postsynaptic_root_id', pa.array(post))
                .append_column('canonical_anchor_node_id', pa.array(anchor)), model)
            print('Validated fold', fold, len(table), flush=True)
        c.read_parquet(str(context['bundle']/'anchor_mappings/*.parquet')).create_view('wanted_all')
        catalog = st.open_table(store, 'synapses')
        c.register('catalog', catalog.scanner(columns=['synapse_id','pre_root_id','post_root_id'],
            use_scalar_index=False,batch_size=65536,batch_readahead=1,fragment_readahead=1).to_reader())
        c.execute('CREATE TABLE matched_catalog AS SELECT c.* FROM catalog c SEMI JOIN wanted_all w USING(synapse_id)')
        missing = c.sql('SELECT count(*) FROM wanted_all ANTI JOIN matched_catalog USING(synapse_id)').fetchone()[0]
        conflict = c.sql('SELECT count(*) FROM wanted_all w JOIN matched_catalog c USING(synapse_id) WHERE w.presynaptic_root_id!=c.pre_root_id OR w.postsynaptic_root_id!=c.post_root_id').fetchone()[0]
        assert missing == conflict == 0
        assert c.sql('SELECT count(DISTINCT synapse_id) FROM wanted_all').fetchone()[0] == sum(m['windows'] for m in context['models'])
    assert len(records) == context['plan']['cells'] and len({r['root_id'] for r in records}) == len(records)
    save(context['bundle']/'canonical_skeleton_audit.json', records)


def load_fold(context, model):
    fold = model['fold']
    table = pq.read_table(context['bundle']/f'all_lcpn_predictions/fold{fold}.parquet')
    mapping = pq.read_table(context['bundle']/f'anchor_mappings/fold{fold}.parquet')
    assert table['synapse_id'].equals(mapping['synapse_id'])
    assert table['root_id'].equals(mapping['presynaptic_root_id'])
    return table.append_column('postsynaptic_root_id', mapping['postsynaptic_root_id']).append_column(
        'canonical_anchor_node_id', mapping['canonical_anchor_node_id'])


def pilot(store, context):
    with tempfile.TemporaryDirectory(prefix='radius_export_pilot_', dir=STATE) as temp:
        isolated = st.init_store(temp, 'microns', datastack='minnie65_phase3_v1', mat_version=1718)
        for name in ('label_hierarchies','experiments','runs','checkpoints'):
            st.append(isolated, name, st.scan(store, name))
        writer = register(isolated, context, context['models'])
        for model in context['models']:
            data = payload(load_fold(context, model).slice(0,128), model)
            partial = dict(data, logits=None)
            writer.add_predictions(model['prediction_run_id'], **partial)
            reconcile(isolated, writer, model, data)
            reconcile(isolated, writer, model, data)
            fetched = SegCLRDatabase(store=isolated).get_predictions(model['prediction_run_id'], with_logits=True)
            lookup = {int(s): i for i,s in enumerate(data['synapse_ids'])}
            order = np.asarray([lookup[int(s)] for s in fetched.synapse_ids])
            np.testing.assert_array_equal(fetched.labels, data['labels'][order])
            for actual, expected in zip(fetched.logits, data['logits'], strict=True):
                np.testing.assert_array_equal(actual, expected[order])
    print('Five-model pilot passed, including interrupted writes, duplicate-free retry, and public API logits', flush=True)


def publish_fold(db_root, bundle, model):
    """One independent worker; registry writes and receipts belong to the parent."""
    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)
    store = st.open_store(db_root, 'microns')
    writer = SegCLRWriter(store=store)
    print('Publishing fold', model['fold'], 'worker', os.getpid(), flush=True)
    data = payload(load_fold({'bundle': Path(bundle)}, model), model)
    counts = reconcile(store, writer, model, data)
    assert counts == [model['windows'], model['windows']*len(model['level_classes'])]
    target = int(data['root_ids'][0])
    fetched = SegCLRDatabase(store=store).get_predictions(model['prediction_run_id'],
        root_ids=[target], with_logits=True)
    lookup = {int(s):i for i,s in enumerate(data['synapse_ids'])}
    order = np.asarray([lookup[int(s)] for s in fetched.synapse_ids])
    np.testing.assert_array_equal(fetched.labels, data['labels'][order])
    roots = ','.join(str(int(r)) for r in np.unique(fetched.subject_root_ids))
    nodes = st.open_table(store, 'nodes').scanner(
        columns=['root_id','node_id','x_nm','y_nm','z_nm'],
        filter=f"node_set_id='{NODE_SET}' AND root_id IN ({roots})",
        use_scalar_index=False, **SCAN_OPTIONS).to_table().to_pandas().set_index(['root_id','node_id'])
    coordinates = nodes.loc[list(zip(fetched.subject_root_ids, fetched.node_ids)),
        ['x_nm','y_nm','z_nm']].to_numpy(np.float32)
    assert np.isfinite(coordinates).all()
    for rid in np.unique(fetched.subject_root_ids):
        with (ROOT/'data/skeleton_cache'/f'{rid}.pkl').open('rb') as f:
            skeleton = pickle.load(f)
        at = fetched.subject_root_ids == rid
        np.testing.assert_array_equal(coordinates[at], skeleton.coords[fetched.node_ids[at]])
    for actual, expected in zip(fetched.logits, data['logits'], strict=True):
        np.testing.assert_array_equal(actual, expected[order])
    print('Verified fold', model['fold'], counts, flush=True)
    return dict(fold=model['fold'], prediction_run_id=model['prediction_run_id'],
        predictions=counts[0], level_logit_rows=counts[1], exact_roundtrip_verified=True,
        worker_pid=os.getpid())


def publish_parallel(db_root, bundle, models, workers):
    """Spawn fresh processes: never fork Lance/Arrow's running thread pools."""
    if workers < 1 or len({m['prediction_run_id'] for m in models}) != len(models):
        raise ValueError('Positive worker count and distinct prediction runs are required')
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(publish_fold, str(db_root), str(bundle), model) for model in models]
        for future in as_completed(futures):
            yield future.result()


def worker_capacity(folds, requested=None, environ=None):
    """Use all capacity: four CPUs/8 GiB per fold plus 8 GiB for coordination.

    Bounded scans measured 2.7 GiB on a full 355k-window fold and 20 GiB for
    a complete five-worker production export. The reservations retain headroom.
    """
    env = os.environ if environ is None else environ
    cpus = int(env.get('SLURM_CPUS_PER_TASK', len(os.sched_getaffinity(0))))
    memory_mb = int(env.get('SLURM_MEM_PER_NODE', 0))
    if not memory_mb and env.get('SLURM_MEM_PER_CPU'):
        memory_mb = int(env['SLURM_MEM_PER_CPU']) * cpus
    if not memory_mb:
        # Outside Slurm, use currently available memory rather than total RAM.
        import psutil
        memory_mb = psutil.virtual_memory().available // 2**20
    capacity = min(folds, cpus // 4,
                   (memory_mb - COORDINATOR_MEMORY_MIB) // WORKER_MEMORY_MIB)
    if capacity < 1:
        raise ValueError('Export requires at least four CPUs and 16 GiB of available memory')
    if requested is not None and not 1 <= requested <= capacity:
        raise ValueError(f'--workers must be between 1 and allocated capacity {capacity}')
    return capacity if requested is None else requested


def main():
    global RUN, STATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=RUN)
    parser.add_argument('--workers', type=int, help='Optional cap; default uses all available fold capacity')
    args = parser.parse_args()
    RUN = args.state.resolve()
    STATE = RUN/'segclr_db_export'
    workers = worker_capacity(len(read(RUN/'plan.json')['models']), args.workers)
    print('Parallel export workers:', workers, flush=True)
    os.umask(0o002)
    STATE.mkdir(exist_ok=True)
    with (STATE/'export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = st.open_store(DB_ROOT, 'microns')
        context = prepare(store)
        map_anchors(store, context)
        pilot(store, context)
        receipt = dict(complete=False, database_root=str(DB_ROOT), dataset='microns',
            shared_bundle=str(context['bundle']), plan_sha256=context['digest'],
            hierarchy_id=context['hierarchy'].hierarchy_id, node_set_id=NODE_SET,
            score_semantics=context['score_semantics'], canonical_cells_verified=context['plan']['cells'],
            pilot_passed=True, parallel_workers=workers, models=context['models'], folds=[])
        save(STATE/'preflight.json', receipt)
        # Two experiments may finish together. Registry replace operations are
        # brief and serialized; all independent fact writes remain concurrent.
        with (ROOT/'logs/presynaptic_registry.lock').open('a') as registry_lock:
            fcntl.flock(registry_lock, fcntl.LOCK_EX)
            register(store, context, context['models'])
        for result in publish_parallel(DB_ROOT, context['bundle'], context['models'], workers):
            receipt['folds'].append(result)
            receipt['folds'].sort(key=lambda item: item['fold'])
            save(STATE/'progress.json', receipt)
        assert {r['fold'] for r in receipt['folds']} == {m['fold'] for m in context['models']}
        receipt.update(complete=True, predictions=sum(x['predictions'] for x in receipt['folds']),
            level_logit_rows=sum(x['level_logit_rows'] for x in receipt['folds']),
            completed_utc=datetime.now(timezone.utc).isoformat(), importer_sha256=sha(__file__),
            coordinate_read_note='Public return_coords=True encounters an existing restricted nodes scalar index. Coordinate verification used the same nodes table with use_scalar_index=False; public labels/logits retrieval passed.')
        save(STATE/'receipt.json', receipt)
        save(context['bundle']/'segclr_db_export_receipt.json', receipt)
        print('EXPORT COMPLETE', receipt['predictions'], receipt['level_logit_rows'], flush=True)


if __name__ == '__main__':
    main()
