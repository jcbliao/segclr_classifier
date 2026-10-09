"""Check catalog integrity, Python/notebook syntax and local documentation links.

This check does not import project modules, run notebook cells, or access data
services. Full notebook execution is a separate Slurm job.
"""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    failures = []
    notebooks = sorted(p for base in ('analysis', 'archive', '20260811', '20260812_level4')
                       for p in (root / base).rglob('*.ipynb') if '.ipynb_checkpoints' not in p.parts)
    from IPython.core.inputtransformer2 import TransformerManager
    transform = TransformerManager()
    for path in notebooks:
        nb = json.loads(path.read_text())
        if nb.get('nbformat') != 4:
            failures.append(f'{path.relative_to(root)}: expected notebook format 4')
        for index, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                try:
                    ast.parse(transform.transform_cell(source))
                except SyntaxError as exc:
                    failures.append(f'{path.relative_to(root)} cell {index}: {exc}')
    for base in ('gnn', 'data', 'scripts', 'analysis'):
        for path in (root / base).rglob('*.py'):
            if '__pycache__' not in path.parts:
                try:
                    ast.parse(path.read_text())
                except SyntaxError as exc:
                    failures.append(f'{path.relative_to(root)}: {exc}')
    for path in [root / 'README.md', root / 'analysis/README.md', root / 'results/README.md',
                 root / 'configs/README.md', *sorted((root / 'models').rglob('README.md')),
                 root / 'docs/architectures.md', root / 'docs/reproducibility.md',
                 root / 'docs/repository_verification.md']:
        if not path.exists():
            failures.append(f'missing document {path.relative_to(root)}')
            continue
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            destination = (path.parent / unquote(target.split('#')[0])).resolve()
            if not destination.exists():
                failures.append(f'{path.relative_to(root)}: broken link {target}')
    summary = root / 'results/summary.csv'
    with summary.open() as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        mid = row['model_id']
        if mid in seen:
            failures.append(f'duplicate model ID: {mid}')
        seen.add(mid)
        report = root / row['report']
        if hashlib.sha256(report.read_bytes()).hexdigest() != row['report_sha256']:
            failures.append(f'stale catalog: {mid}')
        config = json.loads((root / 'models' / mid / 'config.json').read_text())
        if config != json.loads(report.read_text())['args']:
            failures.append(f'config differs from source: {mid}')
    print(f'Checked {len(notebooks)} notebooks, {len(rows)} catalog entries, Python syntax and documentation links')
    for failure in failures:
        print(f'FAIL {failure}')
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(check())
