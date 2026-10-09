"""Check paired-site filtering, presynaptic deduplication, and post-pooling fusion."""
from pathlib import Path
import json
import sys
import tempfile
import numpy as np
import torch
from torch_geometric.loader import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset_presynaptic import PresynapticWindowDataset
from gnn.model import ModelConfig, WindowClassifier
from scripts.prepare_native_pre_post import prepare
from scripts.train_gnn import evaluate


def main():
    torch.set_num_threads(1)
    source = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_coarse_confidence_with_tc_folds/scale16/k17/conf0.7/fold0')
    manifest = json.loads((source / 'manifest.json').read_text())
    rid, info = next((r, v) for r, v in manifest['cells'].items() if v['split'] == 'train')
    manifest['cells'] = {rid: info}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        db, cache = root / 'database', root / 'post'
        (db / 'cells').mkdir(parents=True)
        (cache / 'cells').mkdir(parents=True)
        (db / 'manifest.json').write_text(json.dumps(manifest))
        (db / 'metadata.json').write_text(json.dumps({'factor': 16, 'k_observed': 17}))
        rng = np.random.default_rng(0)
        # Rows 0–2 have the same presynaptic node set; row 0 is unresolved,
        # rows 1 and 2 have distinct post points/vectors but remain duplicates.
        np.savez(db / 'cells' / f'{rid}.npz', new_synapse_id=np.array([10, 11, 12, 13]),
            new_valid_k_window=np.ones(4, bool), new_window_offsets=np.array([0, 3, 6, 9, 12]),
            new_window_members=np.array([0, 1, 2, 0, 1, 2, 2, 0, 1, 1, 2, 3]),
            new_window_lpe=rng.normal(size=(12, 8)).astype('float32'),
            new_pos_nm=rng.normal(size=(4, 3)).astype('float32'),
            new_edges=np.array([[0, 1], [1, 2], [2, 3]]),
            new_observed_node_ids=np.arange(4), new_observed_x=rng.normal(size=(4, 64)).astype('float32'),
            new_nearest_observed_node=np.array([0, 0, 0, 1]))
        post_x = rng.normal(size=(4, 64)).astype('float32')
        import pandas as pd
        embeddings = root / 'inference'
        (embeddings / 'embeddings').mkdir(parents=True)
        (embeddings / 'summary.json').write_text(json.dumps({'n_sites': 4}))
        (embeddings / 'manifest.json').write_text(json.dumps({'n_sites': 4, 'materialization_version': 1718,
            'checkpoint_sha256': 'fixture'}))
        pd.DataFrame(dict(synapse_id=[13, 10, 12, 11], presynaptic_root_id=[int(rid)]*4,
            postsynaptic_root_id=[103, 0, 102, 101], status=['ok', 'unresolved_root', 'ok', 'ok'],
            embedding=[post_x[3], None, post_x[2], post_x[1]])).to_parquet(embeddings / 'embeddings/part-00000.parquet', index=False)
        prepare(db, embeddings, cache)
        pre = PresynapticWindowDataset(manifest, 'train', 'new', db, postsynaptic_cache=cache)
        post = PresynapticWindowDataset(manifest, 'train', 'new', db, postsynaptic_cache=cache, use_postsynaptic=True)
        assert len(pre) == len(post) == 2
        np.testing.assert_array_equal(pre.index_window_rows, [1, 3])
        np.testing.assert_array_equal(pre.index_window_rows, post.index_window_rows)
        assert int(post[0].synapse_id) == 11
        np.testing.assert_array_equal(post[0].postsynaptic_embedding.numpy()[0], post_x[1])
        assert not hasattr(pre[0], 'postsynaptic_embedding')
        batch = next(iter(DataLoader(post, batch_size=2)))
        assert batch.postsynaptic_embedding.shape == (2, 64)
        for arch in ('mean', 'pointwise_mlp', 'graph_transformer'):
            for with_post in (False, True):
                config = ModelConfig(architecture=arch, postsynaptic_dim=64 if with_post else 0,
                    gt_depth=4, gt_heads=4)
                model = WindowClassifier(config, post.hierarchy)
                kwargs = dict(pos_enc=batch.pos_enc, rel_pos=batch.rel_pos, has_segclr=batch.has_segclr)
                features = batch.postsynaptic_embedding.clone().requires_grad_(True)
                pooled = model._encode(batch.x, batch.edge_index, batch.batch, **kwargs)
                fused = model(batch.x, batch.edge_index, batch.batch,
                    postsynaptic_embedding=features if with_post else None, **kwargs)
                assert fused.shape[1] == pooled.shape[1] + (64 if with_post else 0)
                if with_post:
                    torch.testing.assert_close(fused[:, -64:], features)
                model.cls_head.compute_loss(fused, batch.y_levels).backward()
                if with_post:
                    assert features.grad is not None and features.grad.abs().sum() > 0
                predictions = model.cls_head.predict_top_down(fused.detach())
                assert predictions.shape[0] == 2
                dataset = post if with_post else pre
                labels, predictions, roots = evaluate(model, DataLoader(dataset, batch_size=2), torch.device('cpu'))
                assert len(labels) == len(predictions) == len(roots) == 2
                print('PASS', arch, 'pre+post' if with_post else 'pre-only', flush=True)
        for arch in ('mean', 'pointwise_mlp', 'graph_transformer'):
            control = WindowClassifier(ModelConfig(architecture=arch, append_presynaptic_mean=True), post.hierarchy)
            paired = WindowClassifier(ModelConfig(architecture=arch, postsynaptic_dim=64), post.hierarchy)
            kwargs = dict(pos_enc=batch.pos_enc, rel_pos=batch.rel_pos, has_segclr=batch.has_segclr)
            fused = control(batch.x, batch.edge_index, batch.batch, **kwargs)
            expected = torch.stack([batch.x[batch.batch == i].mean(0) for i in range(batch.num_graphs)])
            torch.testing.assert_close(fused[:, -64:], expected)
            assert fused.shape[1] == (128 if arch == 'mean' else 192)
            assert sum(p.numel() for p in control.parameters()) == sum(p.numel() for p in paired.parameters())
            control.cls_head.compute_loss(fused, batch.y_levels).backward()
            evaluate(control, DataLoader(pre, batch_size=2), torch.device('cpu'))
            print('PASS', arch, 'raw pre mean control: exact means, matched parameter count, loss and evaluation')
        # Exercise the newly authorized empty-mask exclusion independently.
        input_path = embeddings / 'embeddings/part-00000.parquet'
        frame = pd.read_parquet(input_path)
        frame.loc[frame.synapse_id == 12, 'status'] = 'empty_mask'
        frame.loc[frame.synapse_id == 12, 'embedding'] = None
        frame.to_parquet(input_path, index=False)
        empty_cache = root / 'post_with_empty_mask'
        prepare(db, embeddings, empty_cache)
        with np.load(empty_cache / 'cells' / f'{rid}.npz') as z:
            np.testing.assert_array_equal(z['eligible'], [False, True, False, True])
        marker = json.loads((empty_cache / 'complete.json').read_text())
        assert marker['n_unresolved_excluded'] == marker['n_empty_mask_excluded'] == 1
        print('PASS empty-mask exclusion')
    print('PASS unresolved exclusion, unchanged presynaptic uniqueness, matched datasets and paired vector alignment')

if __name__ == '__main__':
    main()
