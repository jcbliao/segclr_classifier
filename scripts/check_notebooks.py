"""Execute notebooks from their own folders with the project's dependency environment.

Reports and execution copies go under logs/notebook_checks; sources keep their outputs.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks', nargs='*', type=Path)
    parser.add_argument('--setup-only', action='store_true', help='Execute the first setup code cell only')
    parser.add_argument('--timeout', type=int, default=1800, help='Seconds per cell')
    parser.add_argument('--output', type=Path, default=ROOT/'logs/notebook_checks')
    args = parser.parse_args()
    paths = args.notebooks or sorted((ROOT/'analysis').rglob('*.ipynb'))
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMPY_MADVISE_HUGEPAGE='0')
    reports = []
    for path in paths:
        path = path.resolve()
        label = str(path.relative_to(ROOT))
        print(f'START {label}', flush=True)
        nb = nbformat.read(path, as_version=4)
        if args.setup_only:
            first = next(i for i,cell in enumerate(nb.cells) if cell.cell_type == 'code')
            nb.cells = nb.cells[:first+1]
        active = {'index': None}
        def record_cell(cell, cell_index, **kwargs):
            active['index'] = cell_index
            print(f'CELL {label} {cell_index}', flush=True)
        start = perf_counter()
        result = {'notebook': label}
        client = NotebookClient(nb, timeout=args.timeout, kernel_name='segclr_db', force_raise_errors=True,
                                resources={'metadata':{'path':str(path.parent)}},
                                on_cell_start=record_cell)
        try:
            client.execute()
            # Widget callbacks can capture exceptions without failing the cell.
            def errors(value):
                if isinstance(value, dict):
                    if value.get('output_type') == 'error':
                        yield f"{value.get('ename')}: {value.get('evalue')}"
                    for child in value.values():
                        yield from errors(child)
                elif isinstance(value, list):
                    for child in value:
                        yield from errors(child)
            captured = list(errors(nb))
            if captured:
                raise RuntimeError('Captured notebook/widget errors: '+ '; '.join(captured))
            result['status'] = 'passed' 
        except Exception as exc:
            result.update(status='failed',cell=active['index'],error=str(exc),exception=type(exc).__name__)
        result['seconds'] = round(perf_counter()-start,2)
        artifact = args.output / label
        artifact.parent.mkdir(parents=True,exist_ok=True)
        nbformat.write(nb,artifact)
        reports.append(result)
        (args.output/'report.json').write_text(json.dumps(reports,indent=2)+'\n')
        print(f"{result['status'].upper()} {label} ({result['seconds']}s)" ,flush=True)
        if result['status']=='failed': print(result['error'][-1200:],flush=True)
    return int(any(r['status']=='failed' for r in reports))


if __name__ == '__main__':
    raise SystemExit(main())
