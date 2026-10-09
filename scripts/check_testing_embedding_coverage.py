"""Read-only coverage of the saved 466-cell cohort in CAVE-node embeddings."""
import json
from pathlib import Path
import pandas as pd
from segclr_db import store as st
from segclr_db.skeletons import SkeletonCache

REPO = Path('/home/jcbliao/rotation/segclr/gnn_classifier')
cohort = pd.read_csv('/orcd/data/sdorkenw/001/collina/segclr_downstream_analysis/round_1/cells.csv')
roots = cohort.pt_root_id.tolist()
store = st.open_store('/orcd/compute/sdorkenw/001/segclr-db', 'microns')
counts = SkeletonCache(store).node_counts(roots)
work = st.scan(store, 'work_units', root_ids=roots).to_pandas()
work.to_csv(REPO/'analysis/testing_embedding_work_units.csv',index=False)
print('WORK RECORDS',flush=True)
if len(work):
    print(work.groupby(['table','scope_id','status']).agg(cells=('root_id','nunique')).to_string(),flush=True)
frames = []
for dim in st.read_meta(store).embedding_dims:
    df = st.open_table(store, 'node_embeddings', dim=dim).to_table(
        columns=['root_id','node_id','run_id','checkpoint_id'],
        filter='root_id IN (' + ','.join(map(str,roots)) + ')',
        use_scalar_index=False).to_pandas()
    print(f'd{dim}: {len(df):,} embedding rows in cohort', flush=True)
    if df.empty:
        continue
    grouped = df.groupby(['run_id','checkpoint_id','root_id']).node_id.agg(
        embedding_rows='size', embedded_nodes='nunique', min_node='min', max_node='max').reset_index()
    grouped['dim'] = dim
    grouped['cave_nodes'] = grouped.root_id.map(counts)
    grouped['valid_node_range'] = (grouped.min_node >= 0) & (grouped.max_node < grouped.cave_nodes)
    grouped['complete'] = grouped.valid_node_range & (grouped.embedded_nodes == grouped.cave_nodes)
    frames.append(grouped)
all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
present = set(all_rows.root_id) if len(all_rows) else set()
complete = set(all_rows.loc[all_rows.complete, 'root_id']) if len(all_rows) else set()
cohort['has_cave_skeleton'] = cohort.pt_root_id.isin(counts)
cohort['has_embeddings'] = cohort.pt_root_id.isin(present)
cohort['complete_any_checkpoint'] = cohort.pt_root_id.isin(complete)
archive = Path('/orcd/scratch/orcd/013/jcbliao/skeletons/segclr/skeletons')
cohort['has_existing_teasar'] = [(archive/f'{r}.npz').exists() for r in roots]
summary = dict(total=len(roots), cave_skeletons=len(counts), any_embeddings=len(present),
               complete_any_checkpoint=len(complete), no_embeddings=sorted(set(roots)-present),
               embedded_with_existing_teasar=int((cohort.has_embeddings & cohort.has_existing_teasar).sum()),
               embedded_without_existing_teasar=int((cohort.has_embeddings & ~cohort.has_existing_teasar).sum()))
out = REPO/'analysis/testing_embedding_coverage'
out.mkdir(exist_ok=True)
all_rows.to_csv(out/'per_checkpoint_cell.csv',index=False)
cohort.to_csv(out/'per_cell.csv',index=False)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
if len(all_rows):
    print(all_rows.groupby(['run_id','checkpoint_id','dim']).agg(
        cells=('root_id','nunique'), complete_cells=('complete','sum'),
        invalid_node_ranges=('valid_node_range',lambda x: (~x).sum())).to_string(),flush=True)
print(cohort.groupby('cell_type')[['has_cave_skeleton','has_embeddings','complete_any_checkpoint','has_existing_teasar']].sum().to_string(),flush=True)
