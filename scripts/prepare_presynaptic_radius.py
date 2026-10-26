"""Rebuild latest CAVE presynaptic bags at a geodesic radius; preserve five folds."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.presynaptic_radius import FORMAT, radius_windows

DEFAULT_SOURCE = ROOT / 'analysis/db_deployment/presynaptic/latest_segclr/morpho6'
DEFAULT_STATE = ROOT / 'analysis/presynaptic/latest100k_geodesic20um'
DEFAULT_DATABASE = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons_latest100k/geodesic20um_sourceanchors/conf0')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def build_cell(task):
    import numpy as np
    rid, source, output, radius, cutoff, feature_run, feature_hash = task
    digest = sha(source)
    target = output / 'cells' / f'{rid}.npz'
    record = output / 'records' / f'{rid}.json'
    identity = dict(source_sha256=digest, window_nm=radius, anchor_match_cutoff_nm=cutoff, format=FORMAT,
                    selection_code_sha256=sha(ROOT / 'data/presynaptic_radius.py'),
                    feature_run_id=feature_run, feature_checkpoint_sha256=feature_hash)
    if target.exists() and record.exists():
        old = json.loads(record.read_text())
        if any(old.get(k) != v for k, v in identity.items()):
            raise ValueError(f'Stale radius inputs for {rid}; choose a new output directory')
        if old['output_sha256'] != sha(target):
            raise ValueError(f'Radius output changed for {rid}')
        return old
    with np.load(source) as src:
        if str(src['feature_run_id']) != feature_run or str(src['feature_checkpoint_sha256']) != feature_hash:
            raise ValueError(f'Embedding checkpoint mismatch for {rid}')
        names = ('pos_nm', 'edges', 'observed_node_ids', 'observed_x', 'synapse_id',
                 'synapse_xyz_nm', 'nearest_observed_node', 'nearest_observed_distance_nm')
        arrays = {'cave_' + name: src['cave_' + name].copy() for name in names}
        cutoff_key = 'cave_within_match_cutoff' if 'cave_within_match_cutoff' in src else 'cave_within_2um'
        arrays['cave_within_match_cutoff'] = src[cutoff_key].copy()
        previous_valid = src['cave_valid_k_window'].copy()
    vectors = arrays['cave_observed_x']
    if vectors.shape != (len(arrays['cave_observed_node_ids']), 64) or not np.isfinite(vectors).all():
        raise ValueError(f'Invalid raw features for {rid}')
    offsets, members, valid = radius_windows(
        arrays['cave_pos_nm'], arrays['cave_edges'], arrays['cave_observed_node_ids'],
        arrays['cave_nearest_observed_node'], arrays['cave_nearest_observed_distance_nm'], radius,
        anchor_eligible=arrays['cave_within_match_cutoff'])
    if np.any(previous_valid & ~valid):
        raise ValueError(f'Previously eligible anchors were lost for {rid}')
    arrays.update(cave_window_offsets=offsets, cave_window_members=members,
                  cave_valid_k_window=valid,  # compatibility field, means valid radius bag here
                  cave_window_lpe=np.zeros((len(members), 8), np.float32),
                  window_nm=np.asarray(radius), window_selection=np.asarray(FORMAT),
                  feature_run_id=np.asarray(feature_run), feature_checkpoint_sha256=np.asarray(feature_hash))
    temporary = target.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(target)
    sizes = np.diff(offsets)[valid]
    unique = len({np.sort(members[offsets[i]:offsets[i+1]]).tobytes() for i in np.flatnonzero(valid)})
    result = dict(identity, root_id=rid, output_sha256=sha(target), sites=len(valid),
                  old_k10_valid=int(previous_valid.sum()), valid=int(valid.sum()),
                  newly_valid=int((valid & ~previous_valid).sum()), unique_windows=unique,
                  membership_count=int(sizes.sum()), maximum_embeddings=int(sizes.max(initial=0)),
                  below_ten=int((sizes < 10).sum()), above_ten=int((sizes > 10).sum()))
    save(record, result)
    return result


def prepare(args):
    args.state = args.state.resolve()
    args.database = args.database.resolve()
    from tqdm import tqdm
    configs = [json.loads((args.source / 'configs' / f'{fold:02d}.json').read_text()) for fold in range(5)]
    manifests = [json.loads(Path(c['manifest']).read_text()) for c in configs]
    first = manifests[0]
    roots = sorted(first['cells'])
    source_metadata = json.loads((Path(configs[0]['presynaptic_database']) / 'metadata.json').read_text())
    cutoff = float(source_metadata['match_cutoff_nm'])
    for fold, (config, manifest) in enumerate(zip(configs, manifests, strict=True)):
        if (config['architecture'] != 'pointwise_mlp' or config['num_embeddings'] != 10 or
                not config['use_postsynaptic'] or config['epochs'] != 30 or
                manifest['fold_index'] != fold or manifest['casey_coarse_confidence_min'] != 0 or
                set(manifest['cells']) != set(roots) or
                manifest['feature_run_id'] != first['feature_run_id']):
            raise ValueError('Source is not the latest matched five-fold CAVE setup')
    output = args.database
    for name in ('cells', 'records'):
        (output / name).mkdir(parents=True, exist_ok=True)
    identity = dict(format=FORMAT, window_nm=args.window_nm,
                    source_declared_match_cutoff_nm=cutoff,
                    anchor_match_policy='preserve per-site cave_within_match_cutoff; source records include 2um and 5um cutoffs',
                    feature_run_id=first['feature_run_id'],
                    feature_checkpoint_sha256=first['feature_checkpoint_sha256'],
                    source_manifests_sha256=[sha(c['manifest']) for c in configs])
    marker = output / 'build.json'
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError('Output directory belongs to a different radius or source')
    save(marker, identity)
    source_cells = Path(configs[0]['presynaptic_database']) / 'cells'
    tasks = [(rid, source_cells / f'{rid}.npz', output, args.window_nm, cutoff,
              first['feature_run_id'], first['feature_checkpoint_sha256']) for rid in roots]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(tqdm(pool.map(build_cell, tasks), total=len(tasks), desc='Radius cells', mininterval=10))
    metadata = dict(identity, complete=True, missing_root_ids=[], cells=len(roots),
                    selection='all observed embeddings at skeleton distance <= window_nm from existing nearest observed anchor',
                    soma_exclusion='source CAVE skeleton cut, unchanged',
                    minimum_embeddings=1, k_observed=None, geometry_features_prepared=False,
                    eligibility='valid anchor and existing postsynaptic eligibility; no K10 minimum',
                    deduplication='existing per-cell membership-set deduplication')
    save(output / 'metadata.json', metadata)
    models = []
    for fold, (config, manifest) in enumerate(zip(configs, manifests, strict=True)):
        manifest = deepcopy(manifest)
        manifest.update(k_observed=None, window_nm=args.window_nm, hierarchy_levels_dropped=0,
                        radius_provenance=metadata,
                        source_database=str(output))
        manifest_path = args.state / 'manifests' / f'fold{fold}.json'
        save(manifest_path, manifest)
        config.update(num_embeddings=None, window_nm=args.window_nm, manifest=str(manifest_path),
                      presynaptic_database=str(output),
                      results_dir=str(ROOT / f'results/presynaptic_latest100k/geodesic{args.window_nm:g}nm/full_hierarchy/conf0/fold{fold}'),
                      resume=True)
        config.pop('presynaptic_memmap_root', None)
        target = args.state / 'configs' / f'{fold:02d}.json'
        save(target, config)
        from scripts.train_gnn import build_training_parser, build_training_datasets
        from scripts.training_config import parse_training_args
        parsed = parse_training_args(build_training_parser(), ['--config', str(target)])
        train, test, _ = build_training_datasets(parsed)
        from gnn.hierarchy import get_local_classifier_nodes
        hierarchy = dict(level_classes=train.hierarchy.level_classes,
                         local_heads=get_local_classifier_nodes(train.hierarchy),
                         hierarchy_levels_dropped=0)
        models.append(dict(fold=fold, config=str(target), config_sha256=sha(target),
                           manifest_sha256=sha(manifest_path), train_windows=len(train), test_windows=len(test),
                           train_cells=len(train.cell_types), test_cells=len(test.cell_types),
                           results_dir=config['results_dir']))
        del train, test
    totals = {key: sum(record[key] for record in records) for key in
              ('sites', 'valid', 'old_k10_valid', 'newly_valid', 'unique_windows', 'membership_count', 'below_ten', 'above_ten')}
    save(args.state / 'plan.json', dict(metadata, totals=totals, models=models, hierarchy=hierarchy,
                                      maximum_embeddings=max(r['maximum_embeddings'] for r in records)))
    print(json.dumps(dict(totals=totals, models=models), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--state', type=Path, default=DEFAULT_STATE)
    parser.add_argument('--database', type=Path, default=DEFAULT_DATABASE)
    parser.add_argument('--window-nm', type=float, default=20000.0)
    parser.add_argument('--workers', type=int, default=8)
    prepare(parser.parse_args())
