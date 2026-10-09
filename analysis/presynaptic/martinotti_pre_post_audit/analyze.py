"""Recover per-cell Martinotti/pyramidal window votes from completed best checkpoints."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch_geometric.loader import DataLoader
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from data.dataset_presynaptic import PresynapticWindowDataset, AttentionBudgetBatchSampler
from gnn.model import WindowClassifier
from scripts.train_gnn import evaluate
FAMILIES = ('native_skeletons_pre_post', 'native_pre_post_axon_filtered', 'native_pre_post_axon_filtered_equal_sampling')
OUT = Path(__file__).resolve().parent

def result_paths():
    return sorted(p for family in FAMILIES for p in (ROOT / 'results/presynaptic' / family).glob('scale*/k*/conf*/fold*/*/gnn_*.json'))

def run(path):
    result = json.loads(path.read_text())
    args = result['args']; classes = result['classes']
    family = next(f for f in FAMILIES if f in path.parts)
    run_id = family + '__' + path.stem
    dest = OUT / 'runs' / f'{run_id}.csv'
    if dest.exists():
        return
    manifest = json.loads(Path(args['presynaptic_database'], 'manifest.json').read_text())
    manifest['cells'] = {r:v for r,v in manifest['cells'].items() if v['split'] == 'test' and v['cell_type'] == 'MartFam'}
    dataset = PresynapticWindowDataset(manifest, 'test', 'new', database=args['presynaptic_database'],
        postsynaptic_cache=args.get('postsynaptic_cache'), use_postsynaptic=args.get('use_postsynaptic',False),
        compartment_filter=args.get('presynaptic_compartment_filter'), memmap_root=args.get('presynaptic_memmap_root'), cell_cache_size=32)
    assert dataset.classes == classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path.parent / path.stem / 'checkpoint_best.pt', map_location='cpu', weights_only=False)
    model = WindowClassifier(checkpoint['config'], dataset.hierarchy).to(device)
    model.load_state_dict(checkpoint['model_state'])
    sampler = AttentionBudgetBatchSampler(dataset, attention_budget=args['attention_budget'], max_windows=512, shuffle=False)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, persistent_workers=True)
    print('INFER',family,path.parent.name,args['architecture'],len(dataset),'windows',device,flush=True)
    with torch.inference_mode():
        labels,predictions,roots = evaluate(model,loader,device,amp=args.get('amp',False))
    j=classes.index('MartFam'); k=classes.index('pyramidal')
    cm=np.asarray(result['window_test_metrics']['confusion_matrix'])
    recovered=np.bincount(predictions,minlength=len(classes))
    # Same AMP and checkpoint as final evaluation; verify all Martinotti window counts.
    assert recovered.sum() == cm[j].sum()
    max_fraction_difference = float(np.max(np.abs(recovered - cm[j])) / recovered.sum())
    assert max_fraction_difference < 0.002, (recovered, cm[j])
    print("Population fraction difference from saved AMP evaluation:", max_fraction_difference, flush=True)
    rows=[]
    for rid in np.unique(roots):
        counts=np.bincount(predictions[roots==rid],minlength=len(classes)); n=int(counts.sum())
        row=dict(experiment=family,condition=path.parent.name,architecture=args['architecture'],run=path.stem,
            root_id=str(rid),n_windows=n,cell_prediction=classes[int(counts.argmax())],
            martinotti_windows=int(counts[j]),pyramidal_windows=int(counts[k]),
            martinotti_fraction=float(counts[j]/n),pyramidal_fraction=float(counts[k]/n),best_epoch=checkpoint['epoch'],inference_device=str(device),max_population_fraction_difference=max_fraction_difference)
        row.update({f'votes_{c}':int(counts[i]) for i,c in enumerate(classes)})
        rows.append(row)
    cell_counts=np.bincount([classes.index(r['cell_prediction']) for r in rows],minlength=len(classes))
    np.testing.assert_array_equal(cell_counts,np.asarray(result['test_metrics']['confusion_matrix'])[j])
    dest.parent.mkdir(exist_ok=True)
    with dest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print('DONE',[(r['root_id'],round(r['pyramidal_fraction'],3),round(r['martinotti_fraction'],3)) for r in rows if r['cell_prediction']=='pyramidal'],flush=True)

def summarize():
    rows=[]
    for p in sorted((OUT/'runs').glob('*.csv')):
        with p.open() as f: rows.extend(csv.DictReader(f))
    for name, selected in [('all_martinotti_cells.csv',rows),('martinotti_called_pyramidal.csv',[r for r in rows if r['cell_prediction']=='pyramidal'])]:
        if not rows: continue
        with (OUT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(selected)
    population=[]
    for p in result_paths():
        z=json.loads(p.read_text()); family=next(f for f in FAMILIES if f in p.parts)
        cm=np.asarray(z['window_test_metrics']['confusion_matrix']); j=z['classes'].index('MartFam'); k=z['classes'].index('pyramidal')
        matched=[r for r in rows if r['experiment']==family and r['run']==p.stem]
        record=dict(experiment=family,condition=p.parent.name,architecture=z['args']['architecture'],
            n_martinotti_cells=int(np.asarray(z['test_metrics']['confusion_matrix'])[j].sum()),
            n_martinotti_windows=int(cm[j].sum()),martinotti_fraction=float(cm[j,j]/cm[j].sum()),pyramidal_fraction=float(cm[j,k]/cm[j].sum()),
            all_test_windows=int(cm.sum()),all_test_martinotti_fraction=float(cm[:,j].sum()/cm.sum()),all_test_pyramidal_fraction=float(cm[:,k].sum()/cm.sum()))
        if matched:
            record.update(cell_average_martinotti_fraction=float(np.mean([float(r['martinotti_fraction']) for r in matched])),
                cell_average_pyramidal_fraction=float(np.mean([float(r['pyramidal_fraction']) for r in matched])))
        population.append(record)
    fields=list(dict.fromkeys(k for r in population for k in r))
    with (OUT/'population.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(population)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--index',type=int); parser.add_argument('--summarize',action='store_true'); args=parser.parse_args()
    torch.set_num_threads(2)
    if not args.summarize:
        paths=result_paths()
        for p in ([paths[args.index]] if args.index is not None else paths): run(p)
    summarize()
