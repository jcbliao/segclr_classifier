"""Audit cached CAVE partner skeletons for every neuronal incoming synapse.

Read-only against segclr_db. Pins table versions and writes the exact cohort,
including explicit exclusions; never treats an unreadable embedding as absent.
"""
import argparse
import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segclr_db import store as st
from scripts.subcompartment_predict_all import DB_ROOT, EMBED_RUN, EMBED_CKPT, NEURONS, atomic_json

DEFAULT_OUT = Path('/orcd/scratch/orcd/013/jcbliao/incoming_presynaptic_fragments_k10_v1')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect(str(a.output/'audit.duckdb'))
    c.execute('SET threads=4')
    c.execute("SET memory_limit='16GB'")
    c.execute('SET temp_directory=?', [str(a.output/'tmp')])
    store = st.open_store(DB_ROOT, 'microns')
    labels = st.scan(store, 'cell_labels', filter="label_set = 'cell_type'").to_pandas()
    neurons = labels.loc[labels.label.isin(NEURONS), ['root_id']].drop_duplicates()
    c.register('neurons', neurons)
    source = ROOT/'data/synapse_cache/postsynaptic_sites.parquet'
    c.read_parquet(str(source)).create_view('sites')
    c.execute('CREATE OR REPLACE TABLE selected AS SELECT s.* FROM sites s JOIN neurons n ON s.cell_root_id=n.root_id')
    assert c.sql('SELECT count(*)=count(DISTINCT synapse_id) FROM selected').fetchone()[0]
    c.execute('CREATE OR REPLACE TABLE partners AS SELECT partner_root_id AS root_id, count(*) AS synapses FROM selected GROUP BY partner_root_id')
    report = dict(source=str(source), source_size=source.stat().st_size, source_mtime_ns=source.stat().st_mtime_ns,
                  n_neurons=len(neurons), synapses=c.sql('SELECT count(*) FROM selected').fetchone()[0],
                  partner_roots=c.sql('SELECT count(*) FROM partners').fetchone()[0],
                  run_id=EMBED_RUN, checkpoint_id=EMBED_CKPT, k=10, tables={})
    print(json.dumps(report), flush=True)
    for name, columns in [('skeleton_manifest', ['root_id','n_nodes','n_edges'])]:
        ds = st.open_table(store, name)
        report['tables'][name] = dict(version=ds.version, uri=store.table_uri(name))
        c.register('source_table', ds.scanner(columns=columns, batch_size=65536, fragment_readahead=2).to_reader())
        c.execute(f'CREATE OR REPLACE TABLE {name} AS SELECT s.* FROM source_table s JOIN partners p USING(root_id)')
        c.unregister('source_table')
    report['missing_skeleton_roots'] = c.sql('SELECT count(*) FROM partners p LEFT JOIN skeleton_manifest s USING(root_id) WHERE s.root_id IS NULL').fetchone()[0]
    report['synapses_missing_skeleton'] = c.sql('SELECT coalesce(sum(p.synapses),0) FROM partners p LEFT JOIN skeleton_manifest s USING(root_id) WHERE s.root_id IS NULL').fetchone()[0]
    report['roots_with_fewer_than_10_nodes'] = c.sql('SELECT count(*) FROM skeleton_manifest WHERE n_nodes<10').fetchone()[0]
    for table in ['selected','partners','skeleton_manifest']:
        dest=a.output/f'{table}.parquet'
        c.execute(f'COPY (SELECT * FROM {table}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)', [str(dest)])
    c.execute('COPY (SELECT p.* FROM partners p LEFT JOIN skeleton_manifest s USING(root_id) WHERE s.root_id IS NULL) TO ? (FORMAT PARQUET)', [str(a.output/'missing_skeleton_roots.parquet')])
    atomic_json(a.output/'audit.json',report)
    print(json.dumps(report),flush=True)
    try:
        ds=st.open_table(store,'node_embeddings',64)
        report['tables']['node_embeddings']=dict(version=ds.version,uri=store.table_uri('node_embeddings',64))
        # Only identifiers are decoded; no 64D vectors are loaded for this audit.
        scanner=ds.scanner(columns=['root_id','node_id'],
            filter=f"run_id = '{EMBED_RUN}' AND checkpoint_id = '{EMBED_CKPT}'",
            batch_size=65536,fragment_readahead=2,batch_readahead=1,use_scalar_index=False)
        c.register('embedding_source',scanner.to_reader())
        c.execute('CREATE OR REPLACE TABLE embedding_counts AS SELECT e.root_id,count(*) AS embedding_rows,count(DISTINCT node_id) AS embedded_nodes FROM embedding_source e JOIN partners p USING(root_id) GROUP BY e.root_id')
        report['roots_without_embeddings']=c.sql('SELECT count(*) FROM partners p LEFT JOIN embedding_counts e USING(root_id) WHERE e.root_id IS NULL').fetchone()[0]
        report['roots_with_embeddings']=c.sql('SELECT count(*) FROM embedding_counts').fetchone()[0]
        c.execute('COPY embedding_counts TO ? (FORMAT PARQUET, COMPRESSION ZSTD)',[str(a.output/'embedding_counts.parquet')])
        report['embedding_audit']='root coverage complete; exact ten-node membership coverage still requires fragment construction'
    except Exception as exc:
        report['embedding_audit']='failed to read; coverage unknown'
        report['embedding_error']=repr(exc)
    atomic_json(a.output/'audit.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
