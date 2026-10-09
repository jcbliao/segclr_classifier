"""Check the real native pilot through the training loader and all three models."""
import json
from pathlib import Path
import sys
import torch
from torch_geometric.data import Batch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset_presynaptic import PresynapticWindowDataset
from gnn.model import ModelConfig, WindowClassifier
from data.build_registered_presynaptic import NATIVE
from data.skeleton_subsampling import NATIVE_SCALE_WINDOWS

p = Path(sys.argv[1])
m = json.loads((NATIVE/'manifest.json').read_text())
ids = {f.stem for f in (p/'cells').glob('*.npz')}
m['cells'] = {r:v for r,v in m['cells'].items() if r in ids}
assert len(m['cells']) == 1
split = next(iter(m['cells'].values()))['split']
ds = PresynapticWindowDataset(m, split, 'new', database=p, cell_cache_size=1)
assert len(ds) > 0
batch = Batch.from_data_list([ds[i] for i in range(min(4,len(ds)))])
assert batch.x.shape[1] == 64 and torch.isfinite(batch.x).all()
assert all(ds[i].num_nodes == NATIVE_SCALE_WINDOWS[32] for i in range(min(4,len(ds))))
for architecture in ['mean','graph_transformer','pointwise_mlp']:
    model = WindowClassifier(ModelConfig(architecture=architecture), hierarchy=ds.hierarchy)
    output = model(batch.x, batch.edge_index, batch.batch,
                   pos_enc=batch.pos_enc, rel_pos=batch.rel_pos, has_segclr=batch.has_segclr)
    def check(value):
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()
        elif isinstance(value, dict):
            for v in value.values(): check(v)
        elif isinstance(value, (tuple,list)):
            for v in value: check(v)
    check(output)
    print(architecture, 'forward passed', flush=True)
print('Pilot windows:',len(ds),flush=True)
