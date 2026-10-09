"""Extract incoming synapses for every neuronal root in the cell label table."""
import json
from pathlib import Path
import duckdb
import lance

OUT = Path(__file__).resolve().parent
REPO = OUT.parent.parent
LABELS = '/orcd/compute/sdorkenw/001/segclr-db/microns/dims/cell_labels.lance'
labels = lance.dataset(LABELS).to_table().to_pandas()
labels = labels[(labels.label_set == 'cell_type') & ~labels.label.isin(['astrocyte', 'oligo', 'microglia', 'OPC'])]
assert set(labels.mat_version) == {1718}
c = duckdb.connect()
c.execute('SET threads=4')
c.register('labels', labels)
c.execute("CREATE TABLE neurons AS SELECT root_id, list_sort(list(distinct label)) AS cell_types FROM labels GROUP BY root_id")
sources = [str(REPO / 'data/synapse_cache/postsynaptic_sites.parquet')]
extra = OUT / 'additional_incoming.parquet'
if extra.exists():
    sources.append(str(extra))
c.read_parquet(sources).create_view('sites')
assert c.sql("SELECT count(*) FROM sites WHERE mode <> 'incoming'").fetchone()[0] == 0
c.execute('CREATE VIEW points AS SELECT s.*, n.cell_types FROM sites s JOIN neurons n ON s.cell_root_id=n.root_id')
assert c.sql('SELECT count(*)=count(distinct synapse_id) FROM points').fetchone()[0]
c.execute('COPY points TO ? (FORMAT PARQUET, COMPRESSION ZSTD)', [str(OUT/'points.parquet')])
c.execute('COPY (SELECT n.root_id,n.cell_types,count(p.synapse_id) AS n_incoming_points FROM neurons n LEFT JOIN points p ON n.root_id=p.cell_root_id GROUP BY 1,2 ORDER BY 1) TO ? (HEADER)', [str(OUT/'per_cell.csv')])
c.execute('COPY (SELECT l.label AS cell_type,count(distinct l.root_id) AS n_cells,count(distinct p.cell_root_id) AS n_cells_with_points,count(p.synapse_id) AS n_incoming_points FROM (SELECT DISTINCT root_id,label FROM labels) l LEFT JOIN points p ON l.root_id=p.cell_root_id GROUP BY 1 ORDER BY 1) TO ? (HEADER)', [str(OUT/'per_type.csv')])
c.execute('COPY (SELECT * FROM neurons WHERE root_id NOT IN (SELECT DISTINCT cell_root_id FROM sites)) TO ? (HEADER)', [str(OUT/'cells_without_cached_points.csv')])
summary = dict(label_table=LABELS, sources=sources, datastack='minnie65_public', materialization_version=1718, synapse_table='synapses_pni_2', coordinate_units='nm', postsynaptic_coordinates=['cell_x_nm','cell_y_nm','cell_z_nm'], n_neurons=c.sql('select count(*) from neurons').fetchone()[0], n_points=c.sql('select count(*) from points').fetchone()[0], n_neurons_with_points=c.sql('select count(distinct cell_root_id) from points').fetchone()[0], multiple_labels='cell_types list preserves all labels without duplicating synapses', coverage_note='Cells without cached points are not confirmed to have zero synapses unless queried separately.')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
print((OUT/'per_type.csv').read_text())
