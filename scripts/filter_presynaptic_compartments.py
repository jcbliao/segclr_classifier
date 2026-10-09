"""GPU compartment votes for every native presynaptic window; export raw means."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'subcompartment_classification'))
from train_compartment_gpu import make_model

BASE = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons')
DATABASE = BASE / 'casey_coarse_confidence_with_tc_folds/scale16/k17/conf0.7/fold0'
MEMMAP = BASE / 'casey_k17_memmap/scale16/k17'
MODEL = Path('/orcd/scratch/orcd/013/jcbliao/segclr/subcompartment_geodesic_exclusion20um_v1/models_resnet_inverse_sqrt_100epochs/geodesic/model.pt')
OUTPUT = BASE / 'compartment_filtered/scale16/k17/conf0.7/fold0/geodesic_exclusion20um'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database', type=Path, default=DATABASE)
    p.add_argument('--memmap', type=Path, default=MEMMAP)
    p.add_argument('--model', type=Path, default=MODEL)
    p.add_argument('--output', type=Path, default=OUTPUT)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Compartment inference requires CUDA')
    device = torch.device('cuda')
    checkpoint = torch.load(a.model, map_location='cpu', weights_only=True)
    classes = checkpoint['classes']
    dendrite = classes.index('dendrite')
    state = checkpoint['state_dict']
    model = make_model(checkpoint['head'], state['mean'], state['scale'])
    model.load_state_dict(state)
    model.to(device).eval()
    provenance = dict(database=str(a.database.resolve()), variant='new', model=str(a.model.resolve()),
        model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(), classes=classes,
        rule='drop iff dendrite votes > half of observed window embeddings; retain ties',
        averaging='raw SegCLR mean per unique retained window', deduplication='complete sorted node set within root')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'cells').mkdir(exist_ok=True)
    marker = a.output / 'provenance.json'
    if marker.exists() and json.loads(marker.read_text()) != provenance:
        raise ValueError('Output contains different provenance')
    marker.write_text(json.dumps(provenance, indent=2) + '\n')
    manifest = json.loads((a.database / 'manifest.json').read_text())
    audit = []
    for rid, info in tqdm(sorted(manifest['cells'].items()), desc='Compartment inference', file=sys.stderr):
        dest = a.output / 'cells' / f'{rid}.npz'
        if dest.exists():
            with np.load(dest) as z:
                audit.append(json.loads(str(z['audit'])))
            continue
        cache = a.memmap / rid
        x = np.load(cache / 'observed_x.npy', mmap_mode='r')
        ids = np.load(cache / 'observed_ids.npy', mmap_mode='r')
        members = np.load(cache / 'members.npy', mmap_mode='r')
        offsets = np.load(cache / 'offsets.npy', mmap_mode='r')
        with np.load(a.database / 'cells' / f'{rid}.npz') as z:
            valid = z['new_valid_k_window'].copy()
            synapse_ids = z['new_synapse_id'].copy()
        if not np.isfinite(x).all():
            raise ValueError(f'Nonfinite embeddings in {rid}')
        pred = np.empty(len(x), np.int8)
        with torch.inference_mode():
            for start in range(0, len(x), 65536):
                features = torch.from_numpy(np.array(x[start:start+65536], dtype=np.float32)).to(device)
                pred[start:start+len(features)] = model(features).argmax(1).cpu().numpy()
        lookup = np.full(int(max(ids.max(initial=0), members.max(initial=0)))+1, -1, np.int32)
        lookup[ids] = np.arange(len(ids))
        votes = np.zeros((len(valid), len(classes)), np.int16)
        keep = valid.copy()
        rows = []; means = []; seen = set(); before = set(); dropped_unique = set()
        for row in np.flatnonzero(valid):
            nodes = members[offsets[row]:offsets[row+1]]
            observed = lookup[nodes]; observed = observed[observed >= 0]
            if not len(observed):
                raise ValueError(f'No embeddings in valid window {rid}/{row}')
            votes[row] = np.bincount(pred[observed], minlength=len(classes))
            key = np.sort(nodes).tobytes()
            before.add(key)
            keep[row] = 2 * votes[row, dendrite] <= len(observed)
            if not keep[row]:
                dropped_unique.add(key)
            elif key not in seen:
                seen.add(key); rows.append(row)
                means.append(np.asarray(x[observed], np.float32).mean(axis=0))
        means = np.asarray(means, np.float32).reshape(-1, 64)
        record = dict(root_id=rid, cell_type=info['cell_type'], split=info['split'],
            valid_sites=int(valid.sum()), removed_sites=int((valid & ~keep).sum()),
            unique_before=len(before), unique_removed=len(dropped_unique), unique_retained=len(rows))
        assert record['unique_before'] == record['unique_removed'] + record['unique_retained']
        tmp = dest.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, synapse_ids=synapse_ids, keep=keep, votes=votes,
            node_predictions=pred, observed_node_ids=ids, window_rows=np.asarray(rows, np.int32),
            mean_embeddings=means, cell_mean=means.mean(0) if len(means) else np.full(64, np.nan),
            audit=json.dumps(record))
        tmp.replace(dest); audit.append(record)
    frame = pd.DataFrame(audit)
    frame.to_csv(a.output / 'cell_audit.csv', index=False)
    totals = frame.groupby('cell_type')[['valid_sites','removed_sites','unique_before','unique_removed','unique_retained']].sum()
    totals['removed_fraction'] = totals.unique_removed / totals.unique_before
    totals.to_csv(a.output / 'cell_type_audit.csv')
    summary = dict(**provenance, n_cells=len(frame), **{col:int(frame[col].sum()) for col in
        ['valid_sites','removed_sites','unique_before','unique_removed','unique_retained']})
    tmp = a.output / 'summary.tmp.json'
    tmp.write_text(json.dumps(summary, indent=2)+'\n'); tmp.replace(a.output / 'summary.json')
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
