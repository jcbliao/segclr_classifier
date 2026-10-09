"""Prepare and infer the frozen registered-cell cohort, presynaptic cells first."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_registered_teasar_111nm_20260911')
OLD = BASE.parent / 'segclr_testing_teasar_111nm_20260910_resumed'
NAME = 'teasar_registered_111nm_20260911'
RAW_NAME = 'teasar_registered_20260911'
RUN = 'resnet_860b_reshuffled__20260603_150412'
CHECKPOINT = 'checkpoint_e0_s95000'


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def plan():
    import numpy as np
    rows = json.loads((BASE / 'plan.json').read_text())
    for row in rows:
        rid = row['root_id']
        reuse = False
        if row['previous_dense'] and row['source']:
            with np.load(row['source']) as source, np.load(OLD / 'skeletons' / f'{rid}.npz') as previous:
                reuse = all(np.array_equal(source[k], previous[k], equal_nan=True)
                            for k in ('vertices', 'edges', 'radius'))
        row['base'] = str(OLD if reuse else BASE)
        row['skeleton_name'] = 'teasar_testing_111nm_20260910' if reuse else NAME
    assert len(rows) == len({r['root_id'] for r in rows}) == 2442
    atomic_json(BASE / 'routes.json', rows)
    summary = {}
    for phase in ('priority', 'remaining'):
        cohort = [r for r in rows if r['priority'] == (phase == 'priority')]
        # Two of every three cells go to preemptable; the third to normal GPU.
        for partition in ('preemptable', 'normal'):
            selected = [r for i, r in enumerate(cohort) if (i % 3 != 2) == (partition == 'preemptable')]
            (BASE / f'{phase}_{partition}_roots.txt').write_text(''.join(f"{r['root_id']}\n" for r in selected))
        summary[phase] = dict(cells=len(cohort), reuse_dense=sum(r['base'] == str(OLD) for r in cohort),
                              generation_needed=sum(r['source'] is None for r in cohort))
    atomic_json(BASE / 'routing_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def prepare(row):
    import numpy as np
    from segclr_db.results import Skeleton
    from resample_teasar_dense import densify
    rid = row['root_id']
    dest_base = Path(row['base'])
    dense_path = dest_base / 'resampled_111nm' / f'{rid}.npz'
    source = Path(row['source']) if row['source'] else BASE / 'skeletons' / f'{rid}.npz'
    if not source.exists():
        generation_roots = BASE / 'status' / f'generate_{rid}.txt'
        generation_roots.write_text(f'{rid}\n')
        skel_repo = Path('/home/jcbliao/rotation/skeletonization')
        subprocess.run(['/home/jcbliao/.conda/envs/segclr/bin/python', '-u',
                        str(skel_repo / 'scripts/skeletonize_segclr.py'),
                        '--roots', str(generation_roots), '--task-id', '0', '--num-tasks', '1',
                        '--out', str(BASE / 'generated' / str(rid))], cwd=skel_repo,
                       env=dict(os.environ, PYTHONPATH=str(skel_repo)), check=True)
        generated = BASE / 'generated' / str(rid) / 'skeletons' / f'{rid}.npz'
        if not generated.exists():
            raise RuntimeError(f'No skeleton generated for {rid}')
        source.symlink_to(generated)
    with np.load(source) as raw:
        assert int(raw['root_id'][0]) == rid
        raw_v, raw_e, raw_r = (raw[k] for k in ('vertices', 'edges', 'radius'))
        if not dense_path.exists():
            v, e, radii = densify(raw_v, raw_e, raw_r, spacing=111.0)
            temp = dense_path.with_suffix('.partial.npz')
            np.savez_compressed(temp, root_id=np.array([rid], np.uint64), vertices=v,
                                edges=e, radius=radii, spacing_nm=111,
                                original_node_count=len(raw_v))
            temp.replace(dense_path)
        with np.load(dense_path) as dense:
            assert int(dense['root_id'][0]) == rid
            v, e, radii = (dense[k] for k in ('vertices', 'edges', 'radius'))
            np.testing.assert_array_equal(v[:len(raw_v)], raw_v.astype(np.float32))
            assert np.isfinite(v).all() and len(v) == len(radii)
            assert not e.size or (e.min() >= 0 and e.max() < len(v))
            lengths = np.linalg.norm(v[e[:, 0]].astype(float) - v[e[:, 1]], axis=1)
            assert not len(lengths) or lengths.max() <= 111.5
            assert len(v) - len(e) == len(raw_v) - len(raw_e)
            values = []
            if row['base'] == str(BASE):
                values.append(Skeleton(root_id=rid, coords=raw_v, edges=raw_e,
                              radii=raw_r, skeleton_version=0, skeleton_name=RAW_NAME))
            values.append(Skeleton(root_id=rid, coords=v, edges=e, radii=radii,
                          skeleton_version=0, skeleton_name=row['skeleton_name']))
            report = dict(root_id=rid, skeleton_name=row['skeleton_name'],
                          source=str(source), dense=str(dense_path),
                          original_nodes=len(raw_v), nodes=len(v), spacing_nm=111,
                          max_edge_nm=float(lengths.max()) if len(lengths) else 0)
            return values, report


def ingest_batch(batch, store, status_dir=None):
    """Serialize only the batched commit; build and verify arrays outside the lock."""
    import numpy as np
    import pyarrow as pa
    from datetime import datetime
    from segclr_db import store as st
    from segclr_db import schema
    started = time.monotonic()
    values = [v for skeletons, _ in batch for v in skeletons]
    keys = [(v.root_id, v.skeleton_name) for v in values]
    assert len(keys) == len(set(keys))
    records = [dict(root_id=v.root_id, skeleton_name=v.skeleton_name,
                    coords=v.coords.tolist(), edges=v.edges.tolist(),
                    radii=v.radii.tolist(), compartments=v.compartments.tolist(),
                    skeleton_version=v.skeleton_version, n_nodes=len(v),
                    n_edges=len(v.edges), created_ts=datetime.now()) for v in values]
    table = pa.Table.from_pylist(records, schema=schema.get('named_skeletons').schema_for(None))
    del records
    clause = ' OR '.join(f'(root_id = {rid} AND skeleton_name = {st.sql_literal(name)})'
                         for rid, name in keys)
    with (OLD / '.ingestion.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = st.scan(store, 'named_skeletons', columns=['root_id', 'skeleton_name'],
                           filter=clause).to_pylist()
        present = {(r['root_id'], r['skeleton_name']) for r in existing}
        assert len(present) == len(existing), 'Duplicate skeleton identities'
        missing = [i for i, key in enumerate(keys) if key not in present]
        if missing:
            st.append(store, 'named_skeletons', table.take(pa.array(missing, type=pa.int64())))
    # Preserve immutable retry checks and full readback, including raw geometry.
    got = st.scan(store, 'named_skeletons', filter=clause).to_pylist()
    lookup = {(r['root_id'], r['skeleton_name']): r for r in got}
    assert len(got) == len(lookup) == len(values)
    for v in values:
        r = lookup[(v.root_id, v.skeleton_name)]
        assert r['skeleton_version'] == v.skeleton_version
        for field in ('coords', 'edges', 'radii', 'compartments'):
            actual = np.asarray(r[field]).reshape(getattr(v, field).shape)
            np.testing.assert_array_equal(actual, getattr(v, field))
    for _, report in batch:
        atomic_json((Path(status_dir) if status_dir is not None else BASE / 'status') /
                    ('prepared_%s.json' % report['root_id']), report)
    print(f'BATCH VERIFIED: {len(batch)} cells, {len(missing)} new skeletons, '
          f'{time.monotonic() - started:.1f}s ingest+verify', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['plan', 'prepare', 'infer', 'verify'])
    ap.add_argument('--phase', choices=['priority', 'remaining'])
    ap.add_argument('--partition', choices=['preemptable', 'normal', 'balanced_normal'])
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--world', type=int, default=1)
    args = ap.parse_args()
    for sub in ('status', 'skeletons', 'resampled_111nm', 'logs'):
        (BASE / sub).mkdir(exist_ok=True)
    if args.mode == 'plan':
        return plan()
    rows = json.loads((BASE / 'routes.json').read_text())
    rows = [r for r in rows if r['priority'] == (args.phase == 'priority')]
    if args.mode == 'prepare':
        from segclr_db import store as st
        # Skeleton rows contain large nested arrays. The generic million-row
        # compaction target repeatedly rewrites the entire table during ingest.
        # Defer that optional maintenance for these worker processes only.
        st.COMPACT_SMALL_FRAGMENTS = sys.maxsize
        store = st.open_store('/orcd/compute/sdorkenw/001/segclr-db', 'microns')
        failures, batch = [], []
        batch_nodes = 0
        started = time.monotonic()
        for row in rows[args.rank::args.world]:
            if (BASE / 'status' / ('prepared_%s.json' % row['root_id'])).exists():
                continue
            try:
                item = prepare(row)
                batch.append(item)
                batch_nodes += sum(len(v) for v in item[0])
                if len(batch) >= 8 or batch_nodes >= 1_000_000:
                    ingest_batch(batch, store)
                    batch, batch_nodes = [], 0
            except Exception as exc:
                import traceback
                traceback.print_exc()
                failures.append(dict(root_id=row['root_id'], error=str(exc)))
                # An ingest failure must stop this rank; never discard its batch.
                if batch:
                    raise
        if batch:
            ingest_batch(batch, store)
        atomic_json(BASE / 'status' / f'prepare_{args.phase}_{args.rank}.json', dict(failures=failures))
        print(f'RANK FINISHED in {time.monotonic() - started:.1f}s', flush=True)
        print(st.profile_summary(), flush=True)
        if failures:
            raise RuntimeError(f'{len(failures)} preparation failures')
    elif args.mode == 'infer':
        ids = (BASE / f'{args.phase}_{args.partition}_roots.txt').read_text().split()
        lookup = {r['root_id']: r for r in rows}
        failures = []
        for text in ids[args.rank::args.world]:
            rid = int(text)
            row = lookup[rid]
            assert (BASE / 'status' / f'prepared_{rid}.json').exists()
            one = BASE / 'status' / f'infer_{rid}.txt'
            one.write_text(f'{rid}\n')
            env = dict(os.environ, INFERENCE_BASE=row['base'], SKELETON_NAME=row['skeleton_name'],
                       INFERENCE_ROOTS=str(one), INFERENCE_STATUS_PREFIX=f'registered_{rid}')
            result = subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/infer_named_teasar.py')], env=env)
            if result.returncode:
                failures.append(rid)
        if failures:
            raise RuntimeError(f'Inference failed for {failures}')
    else:
        failures = []
        for row in rows:
            rid = row['root_id']
            directory = Path(row['base']) / 'named_embeddings' / row['skeleton_name'] / RUN / CHECKPOINT
            status = directory / f'registered_{rid}_0.json'
            if not status.exists() or json.loads(status.read_text()) != dict(cells=1, failures=[]):
                failures.append(rid)
        atomic_json(BASE / f'{args.phase}_completion.json', dict(cells=len(rows), incomplete=failures))
        print(f'{args.phase}: {len(rows) - len(failures)}/{len(rows)} cells verified', flush=True)
        if failures:
            raise RuntimeError(f'{len(failures)} cells incomplete')


if __name__ == '__main__':
    main()
