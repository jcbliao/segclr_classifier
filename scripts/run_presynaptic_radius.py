"""GPU preflight and training for frozen presynaptic neighborhood experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_presynaptic_radius import DEFAULT_STATE, DEFAULT_SOURCE, save, sha


def main():
    import numpy as np
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('smoke', 'train', 'classify'))
    parser.add_argument('--fold', type=int, default=0, choices=range(5))
    parser.add_argument('--state', type=Path, default=DEFAULT_STATE)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Radius model execution requires the allocated GPU')
    torch.set_num_threads(1)
    plan = json.loads((args.state / 'plan.json').read_text())
    model_record = plan['models'][args.fold]
    config = Path(model_record['config'])
    values = json.loads(config.read_text())
    if (sha(config) != model_record['config_sha256'] or
            sha(values['manifest']) != model_record['manifest_sha256']):
        raise ValueError('Frozen training inputs changed')
    if args.stage == 'train':
        smoke = json.loads((args.state / 'gpu_preflight.json').read_text())
        if not smoke['passed'] or smoke['plan_sha256'] != sha(args.state / 'plan.json'):
            raise ValueError('GPU preflight does not cover this plan')
        os.execv(sys.executable, [sys.executable, '-u', str(ROOT / 'scripts/train_gnn.py'), '--config', str(config)])
    from scripts.train_gnn import build_training_parser, build_training_datasets
    from scripts.training_config import parse_training_args
    from scripts.predict_incoming_fragment_cell_types import load_checkpoint
    parsed = parse_training_args(build_training_parser(), ['--config', str(config)])
    train, test, _ = build_training_datasets(parsed)
    if args.stage == 'classify':
        from scripts.export_presynaptic_lcpn import export_fold
        return export_fold(args.state, model_record, test)
    old_plan = json.loads((DEFAULT_SOURCE / 'plan.json').read_text())
    weights = Path(old_plan['models'][0]['checkpoint'])
    reference, _ = load_checkpoint(weights)
    from gnn.model import WindowClassifier
    model = WindowClassifier(reference.config, train.hierarchy).cuda().eval()
    # Check the existing aggregator/trunk with new independently initialized
    # local heads for the restored full hierarchy; this is not fine-tuning.
    model.pointwise_mlp.load_state_dict(reference.pointwise_mlp.state_dict())
    model.cls_head.trunk.load_state_dict(reference.cls_head.trunk.state_dict())
    expected_heads = plan['hierarchy']['local_heads']
    assert model.cls_head.nodes == expected_heads
    checked = []
    for split, dataset in [('train', train), ('test', test)]:
        indices = np.unique(np.r_[0, len(dataset) - 1, int(dataset.window_sizes.argmin()),
                                 int(dataset.window_sizes.argmax()),
                                 np.unique(dataset.index_labels, return_index=True)[1]])
        batch = dataset.__getitems__(indices).to('cuda')
        assert batch.x.shape[0] == int(dataset.window_sizes[indices].sum())
        assert batch.has_segclr.all() and torch.isfinite(batch.x).all()
        with torch.no_grad():
            hidden = model(batch.x, batch.edge_index, batch.batch, postsynaptic_embedding=batch.postsynaptic_embedding)
            means = torch.stack([model.pointwise_mlp(batch.x[batch.ptr[i]:batch.ptr[i+1]]).mean(0)
                                 for i in range(len(indices))])
            expected = torch.cat((means, batch.postsynaptic_embedding), dim=-1)
            torch.testing.assert_close(hidden, expected, rtol=2e-5, atol=2e-5)
            distribution = model.cls_head.predict_distribution(hidden)
            assert len(distribution['head_logits']) == len(expected_heads)
            trunk = model.cls_head.trunk(hidden)
            for logits, head, spec in zip(distribution['head_logits'], model.cls_head.heads, expected_heads, strict=True):
                assert logits.shape == (len(indices), spec['n_children'])
                torch.testing.assert_close(logits, head(trunk))
            from scripts.export_presynaptic_lcpn import score_columns
            import pyarrow as pa
            import pyarrow.parquet as pq
            import tempfile
            table = pa.table(score_columns(distribution, expected_heads))
            assert table.num_rows == len(indices)
            assert all(f'head_{i}_logits' in table.column_names for i in range(len(expected_heads)))
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'heads.parquet'
                pq.write_table(table, output)
                assert pq.read_table(output).equals(table)
            assert torch.isfinite(distribution['level_probabilities'][-1]).all()
            torch.testing.assert_close(distribution['level_probabilities'][-1].sum(1),
                                       torch.ones(len(indices), device='cuda'))
        model.zero_grad(set_to_none=True)
        hidden = model(batch.x, batch.edge_index, batch.batch, postsynaptic_embedding=batch.postsynaptic_embedding)
        loss = model.cls_head.compute_loss(hidden, batch.y_levels)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert all(p.grad is not None for head in model.cls_head.heads for p in head.parameters())
        checked.append(dict(split=split, counts=dataset.window_sizes[indices].tolist(),
                            roots=dataset.index_root_ids[indices].tolist(), loss=float(loss.detach())))
    save(args.state / 'gpu_preflight.json', dict(passed=True, device=torch.cuda.get_device_name(),
         plan_sha256=sha(args.state / 'plan.json'), reference_checkpoint=str(weights),
         reference_checkpoint_sha256=sha(weights), checks=checked, local_heads=expected_heads,
         scope='forward and backward on presynaptic bags; explicit phi-then-mean parity; not a performance evaluation'))
    print('GPU RADIUS PREFLIGHT PASSED', checked, flush=True)


if __name__ == '__main__':
    main()
