"""Show in-progress folds and save executed outputs in active comparison notebooks."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
LOADER = '''def read_metrics(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None  # A training job may currently be writing this file.

status, rows, payloads, curves = [], [], {}, {}
metric_columns = [f'{scope}_{metric}' for scope in ('window', 'cell')
                  for metric in ('accuracy', 'balanced_accuracy', 'macro_precision', 'macro_f1')]
for fold in FOLDS:
    directory = RESULT_ROOT / f'fold{fold}' / 'pre_post'
    path = directory / f'{RUN}_fold{fold}.json'
    run_directory = directory / f'{RUN}_fold{fold}'
    curve_path = run_directory / 'epoch_metrics.csv'
    if curve_path.exists():
        try:
            frame = pd.read_csv(curve_path).dropna(subset=['epoch']).sort_values('epoch')
            if not frame.empty:
                curves[fold] = frame
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            pass
    payload = read_metrics(path)
    complete = payload is not None
    if payload is None:
        payload = read_metrics(run_directory / 'best_metrics.json')
    state = 'complete' if complete else ('in progress (best so far)' if payload else
            'in progress (epoch metrics)' if fold in curves else 'awaiting metrics')
    status.append(dict(fold=fold, status=state,
                       latest_epoch=int(curves[fold].epoch.max()) if fold in curves else None,
                       epochs_recorded=len(curves[fold]) if fold in curves else 0,
                       result=str((path if complete else run_directory).relative_to(ROOT))))
    if payload is None:
        continue
    if complete:
        assert payload['fold_index'] == fold, f'Fold mismatch: {path}'
    payload['best_epoch'] = payload.get('best_epoch', payload.get('epoch'))
    payloads[fold] = payload
    row = dict(fold=fold, complete=complete, best_epoch=payload['best_epoch'])
    for scope, key in [('window', 'window_test_metrics'), ('cell', 'test_metrics')]:
        for metric in ('accuracy', 'balanced_accuracy', 'macro_precision', 'macro_f1'):
            row[f'{scope}_{metric}'] = payload[key][metric]
        row[f'{scope}_support'] = np.asarray(payload[key]['confusion_matrix']).sum()
    rows.append(row)
print('Snapshot refreshed:', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
display(pd.DataFrame(status))
summary = pd.DataFrame(rows, columns=['fold', 'complete', 'best_epoch', *metric_columns,
                                     'window_support', 'cell_support']).set_index('fold').sort_index()
if not summary.empty:
    print('Saved best-checkpoint metrics; complete=False means provisional.')
    display(summary)
completed = summary[summary.complete.eq(True)]
print(f'Completed folds: {len(completed)}/5. Rerun all cells to refresh this snapshot.')
if not completed.empty:
    aggregate = completed[metric_columns].agg(['count', 'mean', 'std', 'min', 'max']).T
    aggregate.index.name = 'metric'
    print('Aggregate uses completed folds only (sample standard deviation, ddof=1).')
    display(aggregate)
else:
    print('Awaiting completed folds; training curves and available best-so-far metrics are shown below.')
'''


def update(path):
    nb = nbformat.read(path, as_version=4)
    nb.cells[0].source += '\n\nRunning folds appear with best-so-far metrics. Completed-fold aggregates exclude unfinished runs. Outputs are a saved snapshot; use Run All to refresh.' if 'Running folds appear' not in nb.cells[0].source else ''
    nb.cells[2].source = '## Fold status and best-checkpoint performance\n\nEach fold shows its latest recorded epoch and whether final results are available. Running folds use best_metrics.json; completed folds use the final result JSON. Aggregates include completed folds only.'
    nb.cells[3].source = LOADER
    nb.cells[4].source = nb.cells[4].source.replace("    ax.axhline(values.mean(), color='black', linestyle='--', label=f'Mean {values.mean():.3f}')", "    if not completed.empty:\n        mean = completed[metric].mean()\n        ax.axhline(mean, color='black', linestyle='--', label=f'Completed mean {mean:.3f}')")
    nb.cells[4].source = nb.cells[4].source.replace('saved held-out performance', 'best checkpoint performance (unfinished folds provisional)')
    nb.cells[7].source = '## Per-class recall and support\n\nRunning folds show provisional best-checkpoint results. Per-class mean/std below includes all displayed folds; support counts explain rare-class variability.'
    nb.cells[9].source = '## Confusion matrices across folds\n\nRows are true classes; columns are predicted classes. Each entry shows the row-normalized proportion and the raw count in parentheses. Running folds use provisional best-checkpoint results.'
    nb.cells[10].source = nb.cells[10].source.replace(
        "f'{value:.2f}'", "f'{value:.2f}\n({counts[i,j]:,})'")
    nb.cells[10].source = nb.cells[10].source.replace(
        r"f'{value:.2f}\\n({counts[i,j]:,})'",
        r"f'{value:.2f}\n({counts[i,j]:,})'")
    for index in (4, 8, 10):
        source = nb.cells[index].source
        if not source.startswith('if payloads:'):
            nb.cells[index].source = 'if payloads:\n' + '\n'.join('    ' + line for line in source.splitlines()) + "\nelse:\n    print('Awaiting saved best-checkpoint metrics.')"
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    # Save the corrected loader even if notebook execution later fails.
    nbformat.write(nb, path)
    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv[0] = str(ROOT / 'segclr_db/.venv/bin/python')
    NotebookClient(nb, timeout=120, km=km,
                   resources={'metadata': {'path': str(path.parent)}}).execute()
    nbformat.validate(nb)
    nbformat.write(nb, path)
    print('Refreshed:', path.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    for folder in ('cave_skeletons_pre_post', 'native_single_pre_post', 'native_skeletons_pre_post'):
        for path in sorted((ROOT / 'analysis/presynaptic' / folder).glob('*_5folds.ipynb')):
            update(path)
