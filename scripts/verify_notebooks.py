"""Execute analysis notebooks into a separate output tree, recording every result."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import nbformat
    from nbclient import NotebookClient
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks', nargs='*', type=Path)
    parser.add_argument('--timeout', type=int, default=2700, help='Per-cell timeout in seconds (default: 2700 for full-data analyses)')
    parser.add_argument('--include-archive', action='store_true')
    args = parser.parse_args()
    paths = args.notebooks or sorted((ROOT / 'analysis').rglob('*.ipynb'))
    if args.include_archive:
        paths += sorted((ROOT / 'archive').rglob('*.ipynb'))
        paths += sorted((ROOT / '20260811').rglob('*.ipynb'))
        paths += sorted((ROOT / '20260812_level4').rglob('*.ipynb'))
    output = ROOT / 'logs/notebook_checks' / os.environ.get('SLURM_JOB_ID', str(int(time.time())))
    output.mkdir(parents=True, exist_ok=True)
    # Use the exact interpreter/environment running this script rather than a
    # possibly unrelated user kernelspec named python3.
    import sys
    kernel_dir = output / 'kernels/repository-python'
    kernel_dir.mkdir(parents=True)
    (kernel_dir / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Repository Python', 'language': 'python'}))
    os.environ['JUPYTER_PATH'] = str(output) + os.pathsep + os.environ.get('JUPYTER_PATH', '')
    os.environ['MPLBACKEND'] = 'Agg'
    # Existing native-skeleton notebooks use the same workaround for slow
    # transparent-huge-page allocation on these cluster nodes. This affects
    # allocation behavior, not arrays, model inputs, or analysis parameters.
    os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
    results = []
    for path in paths:
        path = path.resolve()
        if '.ipynb_checkpoints' in path.parts:
            continue
        rel = path.relative_to(ROOT)
        print(f'RUN {rel}', flush=True)
        start = time.monotonic()
        nb = nbformat.read(path, as_version=4)
        entry = {'notebook': str(rel), 'status': 'running',
                 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        results.append(entry)
        (output / 'report.json').write_text(json.dumps(results, indent=2) + '\n')

        def cell_started(cell, cell_index):
            if cell.cell_type == 'code':
                print(f'  cell {cell_index + 1}/{len(nb.cells)}', flush=True)

        try:
            NotebookClient(nb, timeout=args.timeout, kernel_name='repository-python',
                           resources={'metadata': {'path': str(path.parent)}},
                           on_cell_start=cell_started).execute()
            entry['status'] = 'passed'
        except Exception as exc:
            entry.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        entry['seconds'] = round(time.monotonic() - start, 2)
        destination = output / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, destination)
        (output / 'report.json').write_text(json.dumps(results, indent=2) + '\n')
        print(f'{entry["status"].upper()} {rel} ({entry["seconds"]}s)', flush=True)
    failures = sum(r['status'] != 'passed' for r in results)
    print(f'{len(results) - failures}/{len(results)} passed; report: {output / "report.json"}', flush=True)
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
