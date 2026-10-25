import json,sys
from pathlib import Path
import pyarrow.parquet as pq
import duckdb
root=Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/latest100k_morpho6_centers_v1')
p=json.loads((root/'plan.json').read_text());valid=[];rows=0;invalid=[]
for item in p['chunks']:
 path=root/'embeddings'/Path(item['path']).name
 if not path.exists():continue
 try:
  meta=pq.ParquetFile(path).metadata
  if meta.num_rows!=item['rows'] or (meta.metadata or {}).get(b'checkpoint_sha256')!=p['checkpoint_sha256'].encode():raise ValueError('provenance or row-count mismatch')
  valid.append(str(path));rows+=meta.num_rows
 except Exception as e:invalid.append([str(path),str(e)])
c=duckdb.connect();c.execute("SET memory_limit='8GB'");c.execute('SET threads=2')
if valid:
 c.read_parquet(valid).create_view('generated')
 count,unique,finite=c.execute('SELECT count(*),count(DISTINCT point_id),count(*) FILTER (WHERE embedding IS NULL OR len(embedding)!=64 OR NOT isfinite(list_sum(embedding))) FROM generated').fetchone()
 assert count==unique==rows and finite==0
report={'valid_saved_shards':len(valid),'valid_saved_embeddings':rows,'invalid_shards':invalid,'planned_missing_points':p['missing_points'],'not_yet_saved':p['missing_points']-rows}
out=Path('/orcd/scratch/orcd/013/jcbliao/latest_embedding_coverage_audit/local_shards.json');out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if '--require-complete' in sys.argv and (invalid or report['not_yet_saved']):
 raise ValueError('Embedding generation is incomplete or contains invalid shards')
