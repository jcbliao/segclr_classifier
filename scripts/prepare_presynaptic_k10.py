"""Reuse the latest fixed-ten bags with the full morphology hierarchy."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_presynaptic_radius import DEFAULT_SOURCE, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    args.state = args.state.resolve()
    from scripts.train_gnn import build_training_parser, build_training_datasets
    from scripts.training_config import parse_training_args
    from gnn.hierarchy import get_local_classifier_nodes
    models = []
    reference = None
    for fold in range(5):
        config = json.loads((DEFAULT_SOURCE/f'configs/{fold:02d}.json').read_text())
        manifest = json.loads(Path(config['manifest']).read_text())
        assert config['num_embeddings'] == 10 and config['use_postsynaptic'] and config['epochs'] == 30
        assert manifest['fold_index'] == fold and manifest['casey_coarse_confidence_min'] == 0
        if reference is None:
            reference = deepcopy(manifest)
        assert set(manifest['cells']) == set(reference['cells'])
        assert manifest['feature_run_id'] == reference['feature_run_id']
        manifest['hierarchy_levels_dropped'] = 0
        path = args.state/f'manifests/fold{fold}.json'
        save(path, manifest)
        config.update(manifest=str(path), resume=True,
            results_dir=str(ROOT/f'results/presynaptic_latest100k/k10/full_hierarchy/conf0/fold{fold}'))
        config.pop('presynaptic_memmap_root', None)
        target = args.state/f'configs/{fold:02d}.json'
        save(target, config)
        parsed = parse_training_args(build_training_parser(), ['--config', str(target)])
        train, test, _ = build_training_datasets(parsed)
        assert (train.window_sizes == 10).all() and (test.window_sizes == 10).all()
        hierarchy = dict(level_classes=train.hierarchy.level_classes,
            local_heads=get_local_classifier_nodes(train.hierarchy), hierarchy_levels_dropped=0)
        models.append(dict(fold=fold, config=str(target), config_sha256=sha(target),
            manifest_sha256=sha(path), train_windows=len(train), test_windows=len(test),
            train_cells=len(train.cell_types), test_cells=len(test.cell_types), results_dir=config['results_dir']))
        del train, test
    save(args.state/'plan.json', dict(complete=True, format='presynaptic-fixed-k-source-v1',
        cells=len(reference['cells']), k_observed=10, window_nm=config['window_nm'],
        window_nm_semantics='legacy config value; fixed K=10 determines membership, not a radius cutoff',
        selection='source ten nearest observed embeddings by skeleton geodesic distance, presynaptic anchors only',
        anchor_match_policy='preserve source per-site eligibility, including historical 2um and 5um cutoffs',
        soma_exclusion='preserve source geometry, including explicitly soma-free records',
        feature_run_id=reference['feature_run_id'], feature_checkpoint_sha256=reference['feature_checkpoint_sha256'],
        hierarchy=hierarchy, models=models, minimum_embeddings=10, maximum_embeddings=10,
        deduplication='existing per-cell membership-set deduplication',
        source_manifests_sha256=[sha(json.loads((DEFAULT_SOURCE/f'configs/{f:02d}.json').read_text())['manifest']) for f in range(5)]))
    print('FULL HIERARCHY K10 READY', sum(m['test_windows'] for m in models), flush=True)


if __name__ == '__main__':
    main()
