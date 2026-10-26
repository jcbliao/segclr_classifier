"""Real concurrent five-fold ingestion, repair and retry in an isolated store."""
from copy import deepcopy
from pathlib import Path
import pickle
import tempfile
import time
import os
import sys
import argparse
from contextlib import ExitStack
from unittest.mock import patch
import resource

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    # Integration-only dependencies are intentionally absent during CPU test
    # discovery on machines without the lab's SegCLR DB checkout/data.
    from scripts import export_presynaptic_radius_db as e
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-readonly',action='store_true')
    parser.add_argument('--capacity-only',action='store_true')
    args = parser.parse_args()
    if args.benchmark_readonly:
        receipt = e.read(e.STATE/'receipt.json')
        started = time.monotonic()
        with ExitStack() as guards:
            for name in ('append','delete','replace','upsert'):
                guards.enter_context(patch.object(e.st,name,side_effect=AssertionError('Read-only benchmark attempted a mutation')))
            result = e.publish_fold(e.DB_ROOT,receipt['shared_bundle'],receipt['models'][1])
        result.update(peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            seconds=time.monotonic()-started, read_only=True)
        e.save(ROOT/'logs/radius_export_memory_benchmark.json',result)
        print('FULL FOLD READ-ONLY MEMORY BENCHMARK',result,flush=True)
        return
    assert e.worker_capacity(5, environ={'SLURM_CPUS_PER_TASK':'20','SLURM_MEM_PER_NODE':'49152'}) == 5
    assert e.worker_capacity(5, environ={'SLURM_CPUS_PER_TASK':'20','SLURM_MEM_PER_NODE':'32768'}) == 3
    assert e.worker_capacity(5, environ={'SLURM_CPUS_PER_TASK':'4','SLURM_MEM_PER_NODE':'49152'}) == 1
    assert e.worker_capacity(5, environ={'SLURM_CPUS_PER_TASK':'40','SLURM_MEM_PER_CPU':'4096'}) == 5
    assert e.worker_capacity(5, 2, environ={'SLURM_CPUS_PER_TASK':'20','SLURM_MEM_PER_NODE':'49152'}) == 2
    try:
        e.worker_capacity(5, 5, environ={'SLURM_CPUS_PER_TASK':'8','SLURM_MEM_PER_NODE':'131072'})
    except ValueError:
        pass
    else:
        raise AssertionError('oversubscribed worker request was accepted')
    try:
        e.worker_capacity(5, environ={'SLURM_CPUS_PER_TASK':'20','SLURM_MEM_PER_NODE':'8192'})
    except ValueError:
        pass
    else:
        raise AssertionError('insufficient memory was accepted')
    if args.capacity_only:
        e.save(ROOT/'logs/radius_export_capacity_test.json',dict(passed=True,
            default_workers=5,default_cpus=20,default_memory_gib=48,
            checks=['memory cap','CPU cap','fold cap','explicit cap','memory per CPU','insufficient memory']))
        print('RESOURCE-AWARE CAPACITY TEST PASSED',flush=True)
        return
    original = e.read(e.STATE/'receipt.json')
    shared = e.st.open_store(e.DB_ROOT, 'microns')
    original_versions = {name:e.st.open_table(shared,name).version for name in ('predictions','prediction_logits')}
    with tempfile.TemporaryDirectory(prefix='parallel_radius_test_', dir=ROOT/'logs') as temp:
        temp = Path(temp)
        store = e.st.init_store(temp/'db','microns',datastack='minnie65_phase3_v1',mat_version=1718)
        for name in ('label_hierarchies','experiments','runs','checkpoints'):
            e.st.append(store,name,e.st.scan(shared,name))
        context = dict(plan=e.read(e.RUN/'plan.json'), digest=original['plan_sha256'],
            timestamp=e.read(e.STATE/'run_identity.json')['timestamp'], bundle=temp/'bundle',
            hierarchy=e.LabelHierarchy.from_tree(original['hierarchy_id'],
                {'neuron':e.read(e.RUN/'manifests/fold0.json')['hierarchy_tree']['neuron']}),
            score_semantics=original['score_semantics'], models=deepcopy(original['models']))
        node_rows = {}
        for model in context['models']:
            fold = model['fold']
            model['experiment_id'] += '_parallel_test'
            model['classifier_run_id'] = model['experiment_id']+'__'+context['timestamp']
            model['prediction_run_id'] += '.parallel_test'
            model_dir = temp/'models'/f'fold{fold}'
            model_dir.mkdir(parents=True)
            (model_dir/'checkpoint_best.pt').symlink_to(model['shared_checkpoint'])
            model['shared_checkpoint'] = str(model_dir/'checkpoint_best.pt')
            table = next(pq.ParquetFile(Path(original['shared_bundle'])/f'all_lcpn_predictions/fold{fold}.parquet').iter_batches(batch_size=64))
            table = pa.Table.from_batches([table])
            mapping = pq.read_table(Path(original['shared_bundle'])/f'anchor_mappings/fold{fold}.parquet').slice(0,len(table))
            model['windows'] = len(table)
            for directory, value in (('all_lcpn_predictions',table),('anchor_mappings',mapping)):
                dest = context['bundle']/directory/f'fold{fold}.parquet'
                dest.parent.mkdir(parents=True,exist_ok=True)
                pq.write_table(value,dest)
            for row in mapping.to_pylist():
                rid, node = row['presynaptic_root_id'], row['canonical_anchor_node_id']
                with (ROOT/'data/skeleton_cache'/f'{rid}.pkl').open('rb') as f:
                    coords = pickle.load(f).coords[node]
                node_rows[rid,node] = dict(root_id=rid,node_id=node,x_nm=float(coords[0]),
                    y_nm=float(coords[1]),z_nm=float(coords[2]),node_set_id=e.NODE_SET)
        e.st.append(store,'nodes',pa.Table.from_pylist(list(node_rows.values())))
        writer = e.register(store,context,context['models'])
        # Simulate a crash after labels but before logits for two different folds.
        for model in context['models'][:2]:
            data = e.payload(e.load_fold(context,model),model)
            writer.add_predictions(model['prediction_run_id'],**dict(data,logits=None))
        results = list(e.publish_parallel(temp/'db',context['bundle'],context['models'],5))
        assert len(results) == 5 and len({r['worker_pid'] for r in results}) > 1
        before = {name:e.st.open_table(store,name).version for name in ('predictions','prediction_logits')}
        repeated = list(e.publish_parallel(temp/'db',context['bundle'],context['models'],5))
        assert len(repeated) == 5
        assert before == {name:e.st.open_table(store,name).version for name in before}, 'retry rewrote correct rows'
        for model in context['models']:
            assert e.verify(store,model['prediction_run_id'],e.payload(e.load_fold(context,model),model))[0]
        # An invalid fold must fail the parallel stage, not produce a completion.
        broken = deepcopy(context['models'][0]); broken['windows'] += 1
        try:
            list(e.publish_parallel(temp/'db',context['bundle'],[broken],1))
        except AssertionError:
            pass
        else:
            raise AssertionError('worker failure was hidden')
        report = dict(passed=True, workers=5, worker_pids=sorted({r['worker_pid'] for r in results}),
            folds=5, partial_write_recovery=True, exact_roundtrip=True,
            retry_without_new_table_versions=True, failure_propagated=True,
            shared_table_versions_before=original_versions,
            shared_table_versions_after={name:e.st.open_table(shared,name).version for name in original_versions})
    e.save(ROOT/'logs/parallel_radius_export_test.json',report)
    print('PARALLEL EXPORT TEST PASSED',report,flush=True)


if __name__ == '__main__':
    main()
