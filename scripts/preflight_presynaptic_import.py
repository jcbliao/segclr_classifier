"""Read-only inventory for importing the frozen presynaptic deployment."""
import importlib.util
import json
import os
from pathlib import Path
import sys

checkout = Path(os.environ['SEGCLR_IMPORT_CHECKOUT'])
spec = importlib.util.spec_from_file_location('segclr_db', checkout/'src/__init__.py', submodule_search_locations=[str(checkout/'src')])
module = importlib.util.module_from_spec(spec)
sys.modules['segclr_db'] = module
spec.loader.exec_module(module)
from segclr_db import store as st
from segclr_db.migrations import extend_v7_agg_specs
import pyarrow.parquet as pq

root = Path('/orcd/compute/sdorkenw/001/segclr-db')
source = Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/incoming_presynaptic_fragments_k10_collina_v2')
store = st.open_store(root, 'microns', check_schema=False)
report = {'schema_drift': st.schema_drift(store), 'aggregation_extension': extend_v7_agg_specs(root), 'registries': {}}
for name in ('prediction_runs', 'agg_specs', 'label_hierarchies', 'node_sets', 'experiments', 'runs', 'checkpoints'):
    report['registries'][name] = st.scan(store, name).to_pylist()
for name in ('nodes', 'predictions', 'prediction_logits', 'synapse_partners', 'synapses'):
    ds = st.open_table(store, name)
    report[name] = {'rows': ds.count_rows(), 'version': ds.version, 'schema': str(ds.schema)}
plan = json.loads((source/'cell_types/plan.json').read_text())
report['model_artifacts'] = [{'path': m['path'], 'exists': Path(m['path']).exists(), 'manifest_exists': Path(m['manifest']).exists()} for m in plan['models']]
part = next((source/'cell_types/parts').glob('*/*.parquet'))
report['sample_part'] = {'path': str(part), 'schema': str(pq.read_schema(part)), 'metadata': {k.decode(): v.decode() for k,v in (pq.read_schema(part).metadata or {}).items()}}
output = Path('logs/presynaptic_import/inventory.json')
output.write_text(json.dumps(report, indent=2, default=str))
print('inventory:', output, flush=True)
print('schema drift:', report['schema_drift'], flush=True)
print('aggregation extension:', report['aggregation_extension'], flush=True)
print('prediction runs:', [r['prediction_run_id'] for r in report['registries']['prediction_runs']], flush=True)
