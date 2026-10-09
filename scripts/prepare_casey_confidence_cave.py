"""Pair the native Casey confidence cohorts/folds with the existing CAVE K=10 database."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

BASE = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons')
CONFIDENCES = (0, 0.3, 0.5, 0.7, 0.9)


def prepare(source: Path, database: Path, output: Path) -> dict:
    metadata = json.loads((database / 'metadata.json').read_text())
    if metadata['format'] != 'presynaptic-axon-npz-v1' or metadata['k_observed'] != 10:
        raise ValueError('Expected CAVE K=10 presynaptic database')
    # Validate every input before writing any cohort.
    inputs = []
    for confidence in CONFIDENCES:
        for fold in range(5):
            directory = source / f'conf{confidence:g}' / f'fold{fold}'
            manifest = json.loads((directory / 'manifest.json').read_text())
            rows = list(csv.DictReader((directory / 'cohort.csv').open()))
            if {row['root_id'] for row in rows} != set(manifest['cells']):
                raise ValueError(f'Cohort/manifest mismatch: {directory}')
            for rid, info in manifest['cells'].items():
                if not (database / 'cells' / f'{rid}.npz').is_file():
                    raise FileNotFoundError(database / 'cells' / f'{rid}.npz')
                row = next((r for r in rows if r['root_id'] == rid), None)
                if info['split'] != row['split'] or (info['split'] == 'test') != (int(row['held_out_fold']) == fold):
                    raise ValueError(f'Fold mismatch: {directory}, {rid}')
            inputs.append((directory, manifest))
    for directory, original in inputs:
        destination = output / directory.relative_to(source)
        destination.mkdir(parents=True, exist_ok=True)
        link = destination / 'cells'
        target = (database / 'cells').resolve()
        if link.is_symlink() and link.resolve() != target:
            raise ValueError(f'Wrong cells symlink: {link}')
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
        elif link.resolve() != target:
            raise ValueError(f'Wrong cells directory: {link}')
        manifest = {**original, 'source_database': str(database.resolve()),
                    'native_cohort_manifest': str((directory / 'manifest.json').resolve()),
                    'presynaptic_variant': 'cave', 'k_observed': 10}
        (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        for name in ('cohort.csv',):
            (destination / name).write_text((directory / name).read_text())
    (output / 'fold_assignments.csv').write_text((source / 'fold_assignments.csv').read_text())
    report = {'n_cohorts': len(inputs), 'k_observed': 10,
              'cell_counts': {f'conf{confidence:g}': len(inputs[i*5][1]['cells'])
                              for i, confidence in enumerate(CONFIDENCES)},
              'native_cohorts': str(source.resolve()), 'cave_database': str(database.resolve())}
    (output / 'preparation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=BASE/'casey_coarse_confidence_with_tc_folds/scale16/k17')
    parser.add_argument('--database', type=Path, default=BASE/'k10')
    parser.add_argument('--output', type=Path, default=BASE/'casey_confidence_cave/k10')
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.database, args.output), indent=2))
