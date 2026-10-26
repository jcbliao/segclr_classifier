"""Certify all five completed folds, checkpoints, and all-head score exports."""
import csv
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import pyarrow.parquet as pq
from scripts.prepare_presynaptic_radius import DEFAULT_STATE, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=DEFAULT_STATE)
    state = parser.parse_args().state
    plan = json.loads((state / 'plan.json').read_text())
    folds = []
    for record in plan['models']:
        paths = list(Path(record['results_dir']).glob('*/checkpoint_last.pt'))
        if len(paths) != 1:
            raise ValueError(f"Fold {record['fold']} lacks a unique checkpoint")
        last_path = paths[0]
        last = torch.load(last_path, map_location='cpu', weights_only=False)
        assert last['epoch'] == 29
        assert all(torch.isfinite(value).all() for value in last['model_state'].values())
        best_path = last_path.with_name('checkpoint_best.pt')
        best = torch.load(best_path, map_location='cpu', weights_only=False)
        assert all(torch.isfinite(value).all() for value in best['model_state'].values())
        report = json.loads(last_path.parent.with_suffix('.json').read_text())
        config = json.loads(Path(record['config']).read_text())
        assert sha(record['config']) == record['config_sha256']
        assert sha(config['manifest']) == record['manifest_sha256']
        assert all(report['args'].get(k) == v for k, v in config.items())
        with (last_path.parent / 'epoch_metrics.csv').open() as stream:
            epochs = list(csv.DictReader(stream))
        assert sorted({int(row['epoch']) for row in epochs}) == list(range(30))
        scores = state / 'all_lcpn_predictions' / f"fold{record['fold']}.parquet"
        meta = json.loads(scores.with_suffix('.json').read_text())
        assert meta['checkpoint_sha256'] == sha(best_path)
        assert meta['parquet_sha256'] == sha(scores)
        assert meta['plan_sha256'] == sha(state / 'plan.json')
        assert meta['local_heads'] == plan['hierarchy']['local_heads']
        assert pq.ParquetFile(scores).metadata.num_rows == record['test_windows']
        folds.append(dict(fold=record['fold'], completed_epochs=30, best_epoch=best['epoch'],
                          checkpoint=str(best_path), checkpoint_sha256=meta['checkpoint_sha256'],
                          predictions=str(scores), classified_windows=meta['windows'],
                          local_heads=len(meta['local_heads']), window_metrics=report['window_test_metrics'],
                          cell_metrics=report['test_metrics'],
                          local_head_metrics=meta['local_head_metrics_conditioned_on_true_parent']))
    assert len(folds) == 5 and {item['fold'] for item in folds} == set(range(5))
    save(state / 'completion.json', dict(complete=True, window_nm=plan['window_nm'],
         hierarchy=plan['hierarchy'], plan_sha256=sha(state / 'plan.json'), folds=folds,
         evaluation='held-out folds also used for checkpoint selection',
         classified_windows=sum(item['classified_windows'] for item in folds)))
    print('ALL FIVE FOLDS COMPLETE; ALL LOCAL HEADS EXPORTED', flush=True)


if __name__ == '__main__':
    main()
