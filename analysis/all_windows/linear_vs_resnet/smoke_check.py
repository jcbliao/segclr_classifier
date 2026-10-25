"""One actual cached window: verify every new encoder and head receives gradients."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from data.dataset_lcpn import load_hierarchy
from data.geodesic_window import extract_window_subgraph
from gnn.model import ModelConfig,WindowClassifier
from torch_geometric.data import Batch
base=Path(__file__).resolve().parent;plan=json.loads((base/'plan.json').read_text());manifest=json.loads((base/'manifest.json').read_text());h=load_hierarchy(manifest)
r=next(r for r in json.loads((base/'preflight.json').read_text())['roots'] if r['split']=='train' and not r['missing'])
rid=r['root_id'];data=torch.load(ROOT/'data/graph_cache'/f'{rid}.pt',weights_only=False)
path=h.label_paths[manifest['cells'][str(rid)]['cell_type']];data.y_levels=torch.tensor([[h.level_maps[i][v] for i,v in enumerate(path)]])
a=json.loads((ROOT/plan['jobs'][0]['config']).read_text())
with np.load(Path(a['neighborhood_root'])/'neighborhoods/n10'/f'{rid}.npz') as z:
    ids=z['cache_index'];offsets=z['offsets'];members=ids[z['members']]
    # Center 0 uses the mapped member set; the center relative coordinates are attached below.
    graph=extract_window_subgraph(data,0,offsets,members)
batch=Batch.from_data_list([graph])
rows=[]
for job in plan['jobs']:
    a=json.loads((ROOT/job['config']).read_text())
    c=ModelConfig(architecture=a['architecture'],cls_head_resnet=False,use_position=a['position'],use_lpe=a['lpe'],mpnn_layers=a['mpnn_layers'],gt_depth=a['gt_depth'])
    model=WindowClassifier(c,h)
    assert all(p.requires_grad for p in model.parameters())
    hidden=model(batch.x,batch.edge_index,batch.batch,pos_enc=batch.pos_enc,rel_pos=batch.rel_pos)
    loss=model.cls_head.compute_loss(hidden,batch.y_levels);loss.backward()
    encoder=next((v for v in [model.encoder,model.graph_transformer,model.pointwise_mlp] if v is not None),None)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.cls_head.parameters())
    if encoder is not None:assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in encoder.parameters())
    rows.append(dict(architecture=job['architecture'],loss=float(loss),encoder_and_head_trainable=True))
(base/'smoke_check.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
