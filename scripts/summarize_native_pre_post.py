"""Write paired pre-only/pre+post classification metrics and differences."""
import csv
import argparse
import json
from pathlib import Path

ROOT = Path('results/presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold0')
parser = argparse.ArgumentParser()
parser.add_argument('--include-controls', action='store_true')
parser.add_argument('--root', type=Path, default=ROOT)
args = parser.parse_args()
ROOT = args.root
conditions = ('pre_only', 'pre_mean_control', 'pre_post') if args.include_controls else ('pre_only', 'pre_post')
references = ('pre_only', 'pre_mean_control') if args.include_controls else ('pre_only',)
rows = []
dataset_counts = None
for condition in conditions:
    found = {}
    for path in (ROOT / condition).glob('gnn_*.json'):
        result = json.loads(path.read_text())
        if dataset_counts is None:
            dataset_counts = result['dataset_counts']
        assert result['dataset_counts'] == dataset_counts, 'Comparison datasets differ'
        assert result['use_postsynaptic'] == (condition == 'pre_post')
        architecture = result['args']['architecture']
        if architecture in found:
            raise ValueError(f'Duplicate result for {condition}/{architecture}')
        row = dict(condition=condition, architecture=architecture, best_epoch=result['best_epoch'])
        for granularity, key in (('cell', 'test_metrics'), ('window', 'window_test_metrics')):
            for metric in ('accuracy', 'balanced_accuracy', 'macro_precision', 'macro_f1'):
                row[f'{granularity}_{metric}'] = result[key][metric]
        found[architecture] = row
    assert set(found) == {'mean', 'pointwise_mlp', 'graph_transformer'}, f'Missing {condition} results'
    rows.extend(found.values())
with (ROOT / 'comparison.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
differences = []
for architecture in ('mean', 'pointwise_mlp', 'graph_transformer'):
    post = next(r for r in rows if r['architecture'] == architecture and r['condition'] == 'pre_post')
    for reference in references:
        pre = next(r for r in rows if r['architecture'] == architecture and r['condition'] == reference)
        differences.append(dict(architecture=architecture, reference=reference,
            **{k: post[k] - pre[k] for k in pre if k.startswith(('cell_', 'window_'))}))
with (ROOT / 'post_minus_pre.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(differences[0]))
    writer.writeheader()
    writer.writerows(differences)
print(json.dumps(differences, indent=2))
