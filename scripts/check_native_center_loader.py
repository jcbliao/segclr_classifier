"""Build one real centered cell and verify feature IDs and graph membership in the loader."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.build_registered_presynaptic import NATIVE, ROUTES, build, SCALE_WINDOWS
from data.native_centered_windows import CENTER_POLICY
from data.dataset_presynaptic import PresynapticWindowDataset

out=NATIVE.parent/'training_local_center_pilot'
routes={r['root_id']:r for r in json.loads(ROUTES.read_text()) if r['priority']}
rid=min(routes)
build(rid,routes[rid],out,[32])
database=out/'scale32'/f'k{SCALE_WINDOWS[32]}'
manifest=json.loads((NATIVE/'manifest.json').read_text())
manifest['cells']={str(rid):manifest['cells'][str(rid)]}
ds=PresynapticWindowDataset(manifest,manifest['cells'][str(rid)]['split'],'new',database=database)
assert len(ds)>0
with np.load(database/'cells'/f'{rid}.npz') as z:
    assert str(z['center_policy'])==CENTER_POLICY
    with np.load(str(z['embedding_path'])) as emb:
        ids=emb['node_ids']; x=emb['embeddings']
        for i in range(min(len(ds),100)):
            row=ds.index_window_rows[i]
            original=z['new_center_original_node_ids'][row]
            data=ds[i]
            assert data.num_nodes==SCALE_WINDOWS[32]
            assert data.has_segclr.all()
            np.testing.assert_array_equal(data.rel_pos[0].numpy(),np.zeros(3))
            location=np.searchsorted(ids,original)
            assert ids[location]==original
            np.testing.assert_array_equal(data.x[0].numpy(),x[location])
            # Connected tree: each undirected edge is represented twice.
            assert data.edge_index.shape[1]==2*(data.num_nodes-1)
print(f'PASS: real cell {rid}, {len(ds)} unique centered windows; first 100 have exact center embeddings, K nodes, connected graphs',flush=True)
