"""Prepare three-class manifests without changing the original five splits."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.dataset_lcpn import load_hierarchy, split_cells

SOURCE = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_coarse_confidence_with_tc_folds/scale16/k17/conf0.7')
GROUPS = {
    'BasketFam': 'BasketFam',
    'BipFam': 'nonBasketInhibitory',
    'MartFam': 'nonBasketInhibitory',
    'NglFam': 'nonBasketInhibitory',
    'pyramidal': 'Excitatory',
    'thalamocortical': 'Excitatory',
}
EXPERIMENTS = ('native_single_pre_post', 'native_skeletons_pre_post')


def prepare():
    report = []
    for fold in range(5):
        source = SOURCE / f'fold{fold}' / 'manifest.json'
        original = json.loads(source.read_text())
        hierarchy = load_hierarchy(original)
        buckets = {name: [] for name in sorted(set(GROUPS.values()))}
        for label, path in hierarchy.label_paths.items():
            if path[-1] not in GROUPS:
                raise ValueError(f'Unexpected source class: {path[-1]}')
            buckets[GROUPS[path[-1]]].append(label)
        manifest = deepcopy(original)
        manifest['hierarchy_tree'] = {
            name: {'_labels_': sorted(labels)} for name, labels in buckets.items()
        }
        manifest['hierarchy_levels_dropped'] = 0
        manifest['three_class_provenance'] = {
            'source_manifest': str(source), 'source_class_mapping': GROUPS,
            'split_policy': 'Unchanged source folds and cell assignments',
            'classifier_policy': 'One LCPN root head over the three merged classes',
        }
        merged = load_hierarchy(manifest)
        assert merged.depth == 1
        assert set(merged.level_classes[-1]) == set(GROUPS.values())
        assert manifest['cells'] == original['cells']
        # Exercise the actual target encoding and classifier, including a
        # backward pass, before any dependent GPU jobs are allowed to start.
        import torch
        from gnn.lcpn import LCPNHead
        head = LCPNHead(merged, in_dim=8)
        assert len(head.nodes) == 1 and head.nodes[0]['n_children'] == 3
        targets = torch.tensor([
            [merged.level_maps[0][path[0]]] for path in merged.label_paths.values()
        ])
        features = torch.randn(len(targets), 8, requires_grad=True)
        loss = head.compute_loss(features, targets)
        assert torch.isfinite(loss)
        loss.backward()
        assert features.grad is not None and torch.isfinite(features.grad).all()
        predictions = head.predict_top_down(features.detach())
        assert predictions.shape == targets.shape
        assert ((predictions >= 0) & (predictions < 3)).all()
        for split in ('train', 'test'):
            before = split_cells(original, split, hierarchy)
            after = split_cells(manifest, split, merged)
            assert before == after, f'Cohort changed in fold {fold}, {split}'
            counts = Counter(merged.label_paths[info['cell_type']][-1] for _, info in after)
            assert set(counts) == set(GROUPS.values()), (fold, split, counts)
            report.append(dict(fold=fold, split=split, cell_counts=dict(counts)))
        for experiment in EXPERIMENTS:
            destination = ROOT / 'analysis/presynaptic' / experiment / 'three_class/manifests' / f'fold{fold}.json'
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(manifest, indent=2) + '\n')
    for experiment in EXPERIMENTS:
        destination = ROOT / 'analysis/presynaptic' / experiment / 'three_class/validation.json'
        destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    prepare()
