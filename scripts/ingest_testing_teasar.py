"""Serial, resumable named-skeleton ingestion after parallel generation."""
from pathlib import Path
import json
import sys
import os
import fcntl
import numpy as np
from segclr_db import store as st
from segclr_db.skeletons import SkeletonCache
from segclr_db.results import Skeleton

OUT = Path(os.environ.get('OUT', '/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_testing_teasar_111nm_20260910'))
# Partial and final jobs may overlap; each named identity has only one writer.
lock = (OUT / '.ingestion.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX)
allow_partial = '--allow-partial' in sys.argv
store = st.open_store('/orcd/compute/sdorkenw/001/segclr-db', 'microns')
cache = SkeletonCache(store)
roots = [int(x) for x in Path(os.environ.get('ROOTS_FILE', str(OUT / 'roots.txt'))).read_text().split()]
before = cache.node_counts(roots)
if '--preflight' in sys.argv:
    st.ensure_table(store, 'named_skeletons')
    print(f'v5 store opened; named table ready; {len(before)} default skeletons found', flush=True)
    sys.exit(0)
names = {'skeletons': os.environ.get('RAW_NAME', 'teasar_testing_20260910'),
         'resampled_111nm': os.environ.get('DENSE_NAME', 'teasar_testing_111nm_20260910')}
missing = []
written = 0
per_name = {name: 0 for name in names.values()}
for rid in roots:
    for directory, name in names.items():
        source_dir = Path(os.environ['SOURCE_DIR']) if directory == 'skeletons' and 'SOURCE_DIR' in os.environ else OUT / directory
        source = source_dir / f'{rid}.npz'
        if not source.exists():
            missing.append([rid, name])
            continue
        with np.load(source) as raw:
            value = Skeleton(root_id=rid, coords=raw['vertices'], edges=raw['edges'],
                             radii=raw['radius'], skeleton_version=0)
        cache.store_named_skeleton(name, value)
        got = cache.get_named_skeleton(rid, name)
        for field in ('coords','edges','radii','compartments'):
            np.testing.assert_array_equal(getattr(value,field),getattr(got,field))
        written += 1
        per_name[name] += 1
        print(f'Verified {rid}/{name}: {len(value)} nodes',flush=True)
assert cache.node_counts(roots) == before, 'Default skeleton counts changed'
summary = dict(expected_cells=len(roots),verified_named_skeletons=written,
               verified_per_name=per_name,partial=allow_partial,
               names=names,missing=missing,default_node_counts_unchanged=True,
               database=str(store.path))
(OUT/('partial_ingestion_report.json' if allow_partial else 'ingestion_report.json')).write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
if missing and not allow_partial:
    raise RuntimeError(f'{len(missing)} named skeletons still missing; rerun failed tasks then ingestion')
