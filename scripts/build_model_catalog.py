"""Build factual model/run documentation from saved final evaluation reports.

Run from any directory. No model imports, inference, checkpoint loading, or
metric recomputation. Existing analysis and result files are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['model_id', 'experiment', 'run', 'architecture', 'dataset', 'classifier',
          'num_embeddings', 'fold', 'seed', 'best_epoch', 'classes',
          'window_macro_f1', 'window_balanced_accuracy', 'cell_macro_f1',
          'cell_balanced_accuracy', 'report', 'report_sha256']


def build(root: Path = ROOT) -> int:
    models = root / 'models'
    models.mkdir(exist_ok=True)
    rows = []
    entries = {}
    # Only sibling <run>.json reports count as final evaluations. In-progress
    # best_metrics.json and auxiliary summaries do not enter this table.
    for path in sorted((root / 'results').rglob('*.json')):
        if path.name == 'best_metrics.json':
            continue
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or not all(
            key in payload for key in ('args', 'classes', 'window_test_metrics', 'test_metrics')
        ):
            continue
        args = payload['args']
        if not isinstance(args, dict) or not args.get('architecture'):
            continue
        rel = path.relative_to(root)
        experiment = path.parent.relative_to(root / 'results').as_posix()
        # A readable namespace plus run name avoids conflating identical names
        # in different experiment directories.
        model_id = f'{experiment}/{path.stem}'
        card_dir = models / experiment / path.stem
        card_dir.mkdir(parents=True, exist_ok=True)
        report_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        row = dict(model_id=model_id, experiment=experiment, run=path.stem,
                   architecture=args['architecture'], dataset=args.get('dataset', ''),
                   classifier=args.get('classifier', ''), num_embeddings=args.get('num_embeddings', ''),
                   fold=payload.get('fold', args.get('fold_index', '')),
                   seed=args.get('seed', ''), best_epoch=payload.get('best_epoch', ''),
                   classes=json.dumps(payload['classes'], separators=(',', ':')),
                   report=rel.as_posix(), report_sha256=report_hash)
        for scope, key in [('window', 'window_test_metrics'), ('cell', 'test_metrics')]:
            for metric in ('macro_f1', 'balanced_accuracy'):
                row[f'{scope}_{metric}'] = payload[key].get(metric, '')
        rows.append(row)
        (card_dir / 'config.json').write_text(json.dumps(args, indent=2) + '\n')
        metrics = {k: v for k, v in payload.items() if k != 'args'}
        (card_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        # Keep publication metadata separate so regeneration cannot erase a
        # manually supplied checkpoint URL or the original training commit.
        artifact = card_dir / 'artifact.json'
        publication = (json.loads(artifact.read_text()) if artifact.exists() else
                       dict(checkpoint_url=None, checkpoint_sha256=None, training_commit=None))
        publication.update(source_report=rel.as_posix(), source_report_sha256=report_hash)
        artifact.write_text(json.dumps(publication, indent=2) + '\n')
        report_link = Path(os.path.relpath(path, card_dir)).as_posix()
        lines = [f'# {path.stem}', '', f'Experiment: `{experiment}`. Architecture: `{args["architecture"]}`.', '',
                 '## Saved evaluation', '', '| Scope | Macro F1 | Balanced accuracy |', '|---|---:|---:|']
        for scope in ('window', 'cell'):
            scores = [row[f'{scope}_{metric}'] for metric in ('macro_f1', 'balanced_accuracy')]
            formatted = [f'{value:.4f}' if isinstance(value, (float, int)) else 'not recorded' for value in scores]
            lines.append(f'| {scope.title()} | {formatted[0]} | {formatted[1]} |')
        lines += ['', f'Source: [evaluation report]({report_link}). Values are copied without recomputation.', '',
                  f'Classes: {", ".join(f"`{c}`" for c in payload["classes"])}.', '',
                  f'Fold as recorded: `{row["fold"] if row["fold"] not in ("", None) else "not recorded"}`. '
                  f'Best epoch as recorded: `{row["best_epoch"] if row["best_epoch"] not in ("", None) else "not recorded"}`.', '',
                  '## Configuration and artifacts', '',
                  '- [Recorded training arguments](config.json)', '- [Full evaluation metadata and per-class metrics](metrics.json)',
                  '- [Checkpoint publication metadata](artifact.json)', '',
                  'The artifact metadata records verified checkpoint publication details; null fields are unverified.', '',
                  'Training arguments are historical records. Dataset paths must be supplied for your environment; '
                  'older reports may omit options introduced later. See [reproduction guidance](' +
                  Path(os.path.relpath(root / 'docs/reproducibility.md', card_dir)).as_posix() + ').', '']
        (card_dir / 'README.md').write_text('\n'.join(lines))
        entries.setdefault(experiment, []).append(
            f'| [{path.stem}]({experiment}/{path.stem}/README.md) | `{args["architecture"]}` |')
    summary = root / 'results/summary.csv'
    with summary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    catalog_lines = [
        '# Model catalog', '',
        'Architectures are implemented in [`gnn/`](../gnn/). Each entry below is a trained run '
        'with a saved final evaluation report; an entry does not imply that its weights have been published.', '',
        'Runs retain their experiment namespace. The table is sorted by path and makes no cross-experiment ranking.', '',
        '[Machine-readable results](../results/summary.csv) · [Architecture reference](../docs/architectures.md) · '
        '[Reproduction](../docs/reproducibility.md)', '',
        'Regenerate with `python scripts/build_model_catalog.py` (through Slurm on the lab cluster). '
        'Generated cards, configs, and metrics are overwritten; publication fields in `artifact.json` are preserved. '
        'Entries for removed reports are not automatically deleted.', '',
        f'{len(rows)} final evaluation reports across {len(entries)} experiment directories.', '']
    for experiment, group in entries.items():
        catalog_lines += [f'## {experiment}', '', '| Run | Architecture |', '|---|---|', *group, '']
    (models / 'README.md').write_text('\n'.join(catalog_lines))
    notebook_lines = ['# Analysis notebooks', '',
        'Notebooks remain grouped with their experiment helpers and caches. '
        'This index describes locations only; the notebook prose and scientific interpretation are maintained by the author.', '',
        '[Execution and environment instructions](../docs/reproducibility.md#notebooks)', '',
        '| Experiment directory | Notebook |', '|---|---|']
    for base in ('analysis', 'archive', '20260811', '20260812_level4'):
        for notebook in sorted((root / base).rglob('*.ipynb')):
            if '.ipynb_checkpoints' in notebook.parts:
                continue
            target = quote(Path(os.path.relpath(notebook, root / 'analysis')).as_posix())
            notebook_lines.append(f'| `{notebook.parent.relative_to(root)}` | [{notebook.stem}]({target}) |')
    (root / 'analysis/README.md').write_text('\n'.join([*notebook_lines, '']))
    return len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    print(f'Cataloged {build(parser.parse_args().root.resolve())} final reports')
