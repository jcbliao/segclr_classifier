"""Align verified post embeddings to CAVE k10 sites and keep the five folds."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.dataset_lcpn import load_hierarchy
from data.dataset_presynaptic import PresynapticWindowDataset
from gnn.model import ModelConfig, WindowClassifier
from prepare_native_three_class import GROUPS

BASE = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons')
DATABASE = BASE / 'casey_confidence_cave/k10/conf0.7'
POST = Path('/orcd/scratch/orcd/013/jcbliao/segclr/postsynaptic_site_embeddings_v1718')
OUTPUT = BASE / 'cave_skeletons_pre_post/k10/conf0.7/postsynaptic'
ANALYSIS = ROOT / 'analysis/presynaptic/cave_skeletons_pre_post/three_class'


def prepare():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / 'manifests').mkdir(exist_ok=True)
    manifests = []
    for fold in range(5):
        source = DATABASE / f'fold{fold}/manifest.json'
        original = json.loads(source.read_text())
        reference = json.loads((ROOT / f'analysis/presynaptic/native_single_pre_post/three_class/manifests/fold{fold}.json').read_text())
        assert original['cells'] == reference['cells'], f'CAVE fold {fold} differs from original cohort'
        hierarchy = load_hierarchy(original)
        buckets = {name: [] for name in set(GROUPS.values())}
        for label, path in hierarchy.label_paths.items():
            buckets[GROUPS[path[-1]]].append(label)
        manifest = deepcopy(original)
        manifest['hierarchy_tree'] = {name: {'_labels_': sorted(labels)} for name, labels in buckets.items()}
        manifest['hierarchy_levels_dropped'] = 0
        manifest['three_class_provenance'] = dict(source_manifest=str(source), source_class_mapping=GROUPS)
        manifest['postsynaptic_cache'] = str(OUTPUT)
        assert load_hierarchy(manifest).depth == 1
        (ANALYSIS / f'manifests/fold{fold}.json').write_text(json.dumps(manifest, indent=2) + '\n')
        manifests.append(manifest)

    embedding_manifest = json.loads((POST / 'manifest.json').read_text())
    verification = json.loads((POST / 'summary.json').read_text())
    assert verification['n_sites'] == embedding_manifest['n_sites']
    assert embedding_manifest['materialization_version'] == 1718
    roots = pd.DataFrame({'presynaptic_root_id': [int(r) for r in manifests[0]['cells']]})
    assert all(set(m['cells']) == set(manifests[0]['cells']) for m in manifests)
    con = duckdb.connect()
    con.execute('SET threads=4')
    frame = con.execute('''select e.synapse_id, e.presynaptic_root_id,
        e.postsynaptic_root_id, e.status, e.embedding
        from read_parquet(?) e join roots using(presynaptic_root_id)''',
        [str(POST / 'embeddings/*.parquet')]).df()
    assert not frame.duplicated(['presynaptic_root_id', 'synapse_id']).any()
    assert not ((frame.postsynaptic_root_id != 0) & ~frame.status.isin(['ok', 'empty_mask'])).any()
    grouped = frame.groupby('presynaptic_root_id').indices
    (OUTPUT / 'cells').mkdir(parents=True, exist_ok=True)
    marker = OUTPUT / 'complete.json'
    counts = dict(n_cells=0, n_site_rows=0, n_eligible_valid_sites=0)
    for rid in sorted(int(r) for r in manifests[0]['cells']):
        with np.load(DATABASE / f'fold0/cells/{rid}.npz') as z:
            ids, valid = z['cave_synapse_id'].copy(), z['cave_valid_k_window'].copy()
        rows = frame.iloc[grouped.get(rid, [])].set_index('synapse_id')
        positions = rows.index.get_indexer(ids)
        if (valid & (positions < 0)).any():
            raise ValueError(f'{rid}: valid CAVE windows lack post inference; cannot train')
        vectors = np.zeros((len(ids), 64), np.float32)
        eligible = np.zeros(len(ids), bool)
        present = np.flatnonzero(positions >= 0)
        aligned = rows.iloc[positions[present]]
        good = aligned.postsynaptic_root_id.ne(0).to_numpy() & aligned.status.eq('ok').to_numpy()
        eligible[present[good]] = True
        if good.any():
            vectors[present[good]] = np.stack(aligned.embedding.to_numpy()[good]).astype(np.float32)
        assert np.isfinite(vectors).all()
        destination = OUTPUT / f'cells/{rid}.npz'
        temporary = destination.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, synapse_ids=ids, eligible=eligible, post_x=vectors)
        temporary.replace(destination)
        counts['n_cells'] += 1
        counts['n_site_rows'] += len(ids)
        counts['n_eligible_valid_sites'] += int((valid & eligible).sum())
        if counts['n_cells'] % 100 == 0:
            print(counts, flush=True)
    provenance = dict(format='postsynaptic-native-site-cache-v1', variant='cave',
        database=str(DATABASE), embeddings=str(POST), embedding_dim=64,
        checkpoint_sha256=embedding_manifest['checkpoint_sha256'], **counts)
    # The loader requires a marker. Remove it if any real-data checks fail.
    marker.write_text(json.dumps(provenance, indent=2) + '\n')
    try:
        for fold, manifest in enumerate(manifests):
            for split in ('train', 'test'):
                rid = next(r for r, info in manifest['cells'].items() if info['split'] == split)
                probe = dict(manifest, cells={rid: manifest['cells'][rid]})
                ds = PresynapticWindowDataset(probe, split, 'cave', database=DATABASE / f'fold{fold}',
                    postsynaptic_cache=OUTPUT, use_postsynaptic=True)
                batch = Batch.from_data_list([ds[0], ds[0]])
                assert batch.x.shape == (20, 64)
                assert batch.postsynaptic_embedding.shape == (2, 64)
                for architecture in ('pointwise_mlp', 'graph_transformer'):
                    model = WindowClassifier(ModelConfig(architecture=architecture, postsynaptic_dim=64), ds.hierarchy)
                    features = model(batch.x, batch.edge_index, batch.batch,
                        pos_enc=batch.pos_enc, rel_pos=batch.rel_pos,
                        postsynaptic_embedding=batch.postsynaptic_embedding)
                    loss = model.cls_head.compute_loss(features, batch.y_levels)
                    assert torch.isfinite(loss)
                    loss.backward()
                print(f'Passed fold {fold}, {split}, k10 MLP and GT forward/backward', flush=True)
    except BaseException:
        marker.unlink(missing_ok=True)
        raise
    (ANALYSIS / 'validation.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('CAVE k10 pre + post preparation and validation passed', flush=True)


if __name__ == '__main__':
    prepare()
