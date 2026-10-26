"""Build a self-contained notebook for the completed five-fold radius run."""
from pathlib import Path
import argparse
import json
from textwrap import dedent
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/presynaptic/latest100k_geodesic20um'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=OUT)
    output = parser.parse_args().state.resolve()
    experiment = json.loads((output/'plan.json').read_text())
    cells = []

    def md(text):
        cells.append(nb.v4.new_markdown_cell(dedent(text).strip()))

    def code(text):
        cells.append(nb.v4.new_code_cell(dedent(text).strip()))

    md('''
    # Presynaptic 20 µm neighborhoods: five-fold morphology classification

    All five folds completed **30 epochs**. Each input is seeded at an existing
    **presynaptic site** and contains every available embedding within **20 µm
    geodesic distance of its skeleton anchor**. The model shares a pointwise MLP
    aggregation and ResNet trunk, appends the postsynaptic embedding, and trains
    **five local heads across the full morphology hierarchy (13 leaf classes)**.

    This notebook reads the completed reports, epoch curves, and saved all-head
    predictions. It does not train models or change the database.

    **Reading the results:** the held-out fold was also used to select the best
    epoch, so it is not an untouched final test set. Fold means give each fold
    equal weight; error bars show sample standard deviation across folds, not
    confidence intervals for independent replicates. Local-head metrics are
    conditioned on the **true parent**; end-to-end metrics include routing errors.
    Windows use the training pipeline's per-cell membership-set deduplication.
    A saved synapse ID identifies the representative of a unique window.

    **Metric convention:** ordinary accuracy is omitted. Balanced accuracy
    (mean recall over supported classes) is paired with **per-class precision**;
    macro precision and macro F1 summarize comparisons. Precision is set to zero
    when a class has no predictions, matching the trainer. Tables show predicted
    counts and the number of folds with no predictions to make these cases visible.
    ''')
    code('''
    %matplotlib inline
    import json
    import sys
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pyarrow.parquet as pq
    from IPython.display import display

    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                if (p / 'scripts/train_gnn.py').exists())
    RUN = ROOT / 'analysis/presynaptic/latest100k_geodesic20um'
    sys.path.insert(0, str(ROOT))
    from data.dataset_lcpn import load_hierarchy
    from gnn.metrics import balanced_accuracy, macro_precision, macro_f1, per_class_precision, per_class_recall
    completion = json.loads((RUN / 'completion.json').read_text())
    plan = json.loads((RUN / 'plan.json').read_text())
    folds = sorted(completion['folds'], key=lambda r: r['fold'])
    assert completion['complete'] and [r['fold'] for r in folds] == list(range(5))
    assert all(r['completed_epochs'] == 30 for r in folds)
    manifests = {i: json.loads((RUN / 'manifests' / f'fold{i}.json').read_text()) for i in range(5)}
    hierarchy = load_hierarchy(manifests[0])
    levels = completion['hierarchy']['level_classes']
    heads = completion['hierarchy']['local_heads']
    assert hierarchy.level_classes == levels and len(heads) == 5
    leaf_names = levels[-1]
    COLORS = plt.get_cmap('tab10').colors
    plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titlesize': 12})
    pd.set_option('display.max_colwidth', 80)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 20)
    def class_precision_table(matrices, names):
        rows = []
        for fold, matrix in enumerate(matrices):
            matrix = np.asarray(matrix)
            precision = per_class_precision(matrix)
            recall = per_class_recall(matrix)
            for i, name in enumerate(names):
                rows.append({'Fold': fold, 'Class': name, 'Precision': precision[i],
                             'Recall': recall[i], 'True count': int(matrix[i].sum()),
                             'Predicted count': int(matrix[:, i].sum()),
                             'No predictions': int(matrix[:, i].sum() == 0)})
        return pd.DataFrame(rows).groupby('Class', sort=False).agg(
            precision_mean=('Precision', 'mean'), precision_fold_SD=('Precision', 'std'),
            recall_mean=('Recall', 'mean'), recall_fold_SD=('Recall', 'std'),
            true_count=('True count', 'sum'), predicted_count=('Predicted count', 'sum'),
            folds_without_predictions=('No predictions', 'sum'))
    def predictions_path(fold):
        return RUN / 'all_lcpn_predictions' / f'fold{fold}.parquet'
    def checkpoint_path(record):
        # Prefer the current checkout when moved between lab mount aliases.
        suffix = str(record['checkpoint']).split('/results/', 1)[1]
        return ROOT / 'results' / suffix
    for record in folds:
        assert pq.ParquetFile(predictions_path(record['fold'])).metadata.num_rows == record['classified_windows']
    neighborhood = (f"K={plan['k_observed']} observed embeddings" if plan.get('k_observed') is not None
                    else f"{completion['window_nm'] / 1000:g} µm geodesic radius")
    display(pd.Series({'Neighborhood': neighborhood,
                       'Cells': plan['cells'], 'Folds': len(folds),
                       'Epochs per fold': 30, 'Learned local heads': len(heads),
                       'Leaf classes': len(leaf_names),
                       'Classified unique windows': completion['classified_windows']}).to_frame('Value'))
    ''')
    md('''
    ## Hierarchy and local classifiers

    Each branching parent has a separate learned output head. Unary branches
    (including the neuron-only root) pass through deterministically. Every head's
    logits are saved for every window, including heads outside its predicted path.
    ''')
    code('''
    head_table = pd.DataFrame([{'Head': i, 'Parent': h['parent_name'],
                                'Decision level': h['child_level'],
                                'Children': ', '.join(h['children'])} for i, h in enumerate(heads)])
    display(head_table.set_index('Head'))
    display(pd.DataFrame({'Level': range(len(levels)), 'Classes': [', '.join(x) for x in levels]}).set_index('Level'))
    ''')
    md('''
    ## Best-checkpoint performance by fold

    Window metrics evaluate individual unique neighborhoods. Cell metrics are
    the trainer's aggregation of predictions per cell. These tables use the
    **13 leaf classes**. Best-epoch indices are zero-based.
    Per-class precision for both evaluation units accompanies the aggregate scores.
    ''')
    code('''
    rows = []
    for record in folds:
        rows.append({'Fold': record['fold'], 'Best epoch (0-based)': record['best_epoch'],
                     'Held-out windows': record['classified_windows'],
                     **{f'{unit} {metric}': record[key][metric]
                        for unit, key in [('Window', 'window_metrics'), ('Cell', 'cell_metrics')]
                        for metric in ['balanced_accuracy', 'macro_precision', 'macro_f1']}})
    scores = pd.DataFrame(rows).set_index('Fold')
    display(scores.round(4))
    metric_columns = [c for c in scores if c.startswith(('Window ', 'Cell '))]
    display(scores[metric_columns].agg(['mean', 'std']).T.rename(columns={'std': 'fold SD'}).round(4))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, unit in zip(axes, ['Window', 'Cell']):
        for i, metric in enumerate(['balanced_accuracy', 'macro_precision', 'macro_f1']):
            values = scores[f'{unit} {metric}'].to_numpy()
            ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), fmt='o', color='black', capsize=5)
            ax.scatter(i + np.linspace(-.12, .12, 5), values, c=COLORS[:5], s=28)
        ax.set_xticks(range(3), ['Balanced accuracy', 'Macro precision', 'Macro F1'])
        ax.set_title(f'{unit}: mean ± fold SD; dots = folds')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    plt.show()
    leaf_precision = pd.concat({unit: class_precision_table(
        [record[key]['confusion_matrix'] for record in folds], leaf_names)
        for unit, key in [('Window', 'window_metrics'), ('Cell', 'cell_metrics')]},
        names=['Evaluation', 'Class'])
    display(leaf_precision.round(4))
    for record in folds:
        for key in ['window_metrics', 'cell_metrics']:
            actual = per_class_precision(np.asarray(record[key]['confusion_matrix']))
            np.testing.assert_allclose(actual, [record[key]['per_class_precision'][name] for name in leaf_names])
    ''')
    md('''
    ## Training curves

    Curves include all 30 epochs; markers locate the selected checkpoint in
    each fold. Validation here is the same cell-held-out fold reported above.
    ''')
    code('''
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for record in folds:
        frame = pd.read_csv(checkpoint_path(record).parent / 'epoch_metrics.csv')
        assert set(frame.epoch) == set(range(30))
        for ax, metric, title in zip(axes, ['train_loss', 'window_macro_f1', 'cell_macro_f1'],
                                     ['Training loss', 'Held-out window macro F1', 'Held-out cell macro F1']):
            ax.plot(frame.epoch + 1, frame[metric], color=COLORS[record['fold']], label=f"Fold {record['fold']}")
            best = frame.loc[frame.epoch == record['best_epoch']].iloc[-1]
            ax.scatter(best.epoch + 1, best[metric], color=COLORS[record['fold']], edgecolor='black', s=40, zorder=3)
            ax.set(title=title, xlabel='Epoch (1–30)')
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    ''')
    md('''
    ## End-to-end predictions at each hierarchy level

    We collapse the saved **window-level leaf confusion matrices** to each
    ancestor level. This measures the full predicted route, including upstream
    errors. It is not conditional local-head performance. The constant neuron-only
    root is omitted from the plot.
    Per-class precision is reported separately for each level's class vocabulary.
    ''')
    code('''
    def matrix_metrics(matrix):
        matrix = np.asarray(matrix)
        return {'balanced_accuracy': balanced_accuracy(matrix),
                'macro_precision': macro_precision(matrix), 'macro_f1': macro_f1(matrix)}
    level_matrices = {}
    level_rows = []
    for record in folds:
        leaf = np.asarray(record['window_metrics']['confusion_matrix'])
        assert leaf.shape == (len(leaf_names), len(leaf_names))
        for level, names in enumerate(levels):
            ancestors = np.array([hierarchy.level_maps[level][hierarchy.label_paths[name][level]] for name in leaf_names])
            collapsed = np.zeros((len(names), len(names)), dtype=np.int64)
            np.add.at(collapsed, (ancestors[:, None], ancestors[None, :]), leaf)
            assert collapsed.sum() == record['classified_windows']
            level_matrices[(record['fold'], level)] = collapsed
            level_rows.append({'Fold': record['fold'], 'Level': level, 'Classes': len(names), **matrix_metrics(collapsed)})
        assert np.isclose(matrix_metrics(leaf)['macro_f1'], record['window_metrics']['macro_f1'])
    level_scores = pd.DataFrame(level_rows)
    display(level_scores.groupby(['Level', 'Classes'])[['balanced_accuracy', 'macro_precision', 'macro_f1']].agg(['mean', 'std']).round(4))
    fig, ax = plt.subplots(figsize=(8, 4))
    for metric, label in [('balanced_accuracy', 'Balanced accuracy'), ('macro_precision', 'Macro precision'), ('macro_f1', 'Macro F1')]:
        grouped = level_scores.query('Level > 0').groupby('Level')[metric].agg(['mean', 'std'])
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], marker='o', capsize=4, label=label)
    ax.set(xticks=range(1, len(levels)), xlabel='Hierarchy level', ylabel='Window score', ylim=(0, 1),
           title='End-to-end hierarchy predictions: mean ± fold SD')
    ax.legend()
    ax.grid(alpha=.2)
    plt.show()
    level_precision = pd.concat({level: class_precision_table(
        [level_matrices[(fold, level)] for fold in range(5)], names)
        for level, names in enumerate(levels) if level > 0}, names=['Level', 'Class'])
    display(level_precision.round(4))
    ''')
    md('''
    ## Every local head, conditioned on its true parent

    A head is evaluated on windows whose true label lies under that parent,
    regardless of upstream predictions. This separates local discrimination
    from routing errors. Support counts sum across the disjoint held-out folds.
    Precision here is conditional on the true parent, just like the head's
    balanced accuracy; it is not end-to-end precision for that class.
    ''')
    code('''
    head_rows = []
    head_matrices = {}
    for record in folds:
        for i, metrics in enumerate(record['local_head_metrics']):
            assert metrics['parent'] == heads[i]['parent_name'] and metrics['children'] == heads[i]['children']
            matrix = np.asarray(metrics['confusion_matrix'])
            assert matrix.sum() == metrics['support']
            head_matrices[(record['fold'], i)] = matrix
            head_rows.append({'Fold': record['fold'], 'Head': i, 'Parent': metrics['parent'],
                              'Support': metrics['support'], **matrix_metrics(matrix)})
    head_scores = pd.DataFrame(head_rows)
    display(head_scores.groupby('Parent', sort=False).agg(
        held_out_support=('Support', 'sum'), balanced_accuracy=('balanced_accuracy', 'mean'),
        balanced_accuracy_fold_SD=('balanced_accuracy', 'std'),
        macro_precision=('macro_precision', 'mean'), macro_precision_fold_SD=('macro_precision', 'std'),
        macro_F1=('macro_f1', 'mean'), macro_F1_fold_SD=('macro_f1', 'std')).round(4))
    head_precision = pd.concat({head['parent_name']: class_precision_table(
        [head_matrices[(fold, i)] for fold in range(5)], head['children'])
        for i, head in enumerate(heads)}, names=['True parent', 'Child class'])
    display(head_precision.round(4))
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, head in enumerate(heads):
        group = head_scores[head_scores.Head == i]
        values = group.macro_f1.to_numpy()
        ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), fmt='o', color='black', capsize=5)
        ax.scatter(i + np.linspace(-.12, .12, 5), values, c=COLORS[:5])
    ax.set(xticks=range(len(heads)), xticklabels=[h['parent_name'] for h in heads],
           ylabel='Conditional macro F1', ylim=(0, 1), title='All local heads: mean ± fold SD')
    ax.grid(axis='y', alpha=.2)
    plt.show()
    ''')
    md('''
    ## Confusion matrices and per-class precision

    Rows are true classes and columns are predictions. Each fold is normalized
    by its own row support **before** taking the equal-weight mean. Unsupported
    rows are omitted from that row's mean rather than counted as zero recall.
    These row-normalized matrices show recall; the precision panels below use
    predicted-class counts as their denominator.
    ''')
    code('''
    def mean_normalized(matrices):
        normalized = []
        for matrix in matrices:
            matrix = np.asarray(matrix, dtype=float)
            support = matrix.sum(1, keepdims=True)
            normalized.append(np.divide(matrix, support, out=np.full_like(matrix, np.nan), where=support > 0))
        return np.nanmean(normalized, axis=0)
    def draw_matrix(ax, matrix, names, title, annotate=False):
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
        ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names,
               xlabel='Predicted', ylabel='True', title=title)
        plt.setp(ax.get_xticklabels(), rotation=65, ha='right', fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)
        if annotate:
            for (i, j), value in np.ndenumerate(matrix):
                if np.isfinite(value):
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=8,
                            color='white' if value > .55 else 'black')
        return image
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    for i, head in enumerate(heads):
        matrix = mean_normalized([head_matrices[(f, i)] for f in range(5)])
        image = draw_matrix(axes.flat[i], matrix, head['children'], f"Local head: {head['parent_name']}", True)
    axes.flat[-1].axis('off')
    fig.colorbar(image, ax=list(axes.flat[:-1]), shrink=.65, label='Mean row fraction')
    plt.show()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
    for ax, key, title in zip(axes, ['window_metrics', 'cell_metrics'], ['Leaf windows', 'Leaf cells']):
        matrix = mean_normalized([record[key]['confusion_matrix'] for record in folds])
        image = draw_matrix(ax, matrix, leaf_names, title)
    fig.colorbar(image, ax=axes.tolist(), shrink=.7, label='Mean row fraction')
    plt.show()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')
    for ax, key, title in zip(axes, ['window_metrics', 'cell_metrics'], ['Window precision', 'Cell precision']):
        precision = pd.DataFrame({f"Fold {r['fold']}": r[key]['per_class_precision'] for r in folds}).reindex(leaf_names)
        image = ax.imshow(precision.to_numpy(), vmin=0, vmax=1, cmap='Blues', aspect='auto')
        ax.set(xticks=range(5), xticklabels=precision.columns, yticks=range(len(leaf_names)),
               yticklabels=leaf_names, title=title)
        for (i, j), value in np.ndenumerate(precision.to_numpy()):
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if value > .55 else 'black')
    fig.colorbar(image, ax=axes.tolist(), shrink=.8, label='Per-class precision')
    plt.show()
    ''')
    md('''
    ## Number of embeddings per window

    This reads only the `n_embeddings` column, in batches, from the saved
    predictions. Counts describe eligible, deduplicated held-out windows rather
    than every raw synapse. The ten-embedding line is a reference to the previous
    selection rule, not a matched model-performance comparison.
    ''')
    code('''
    size_counts = {}
    for record in folds:
        histogram = np.zeros(plan['maximum_embeddings'] + 1, np.int64)
        for batch in pq.ParquetFile(predictions_path(record['fold'])).iter_batches(columns=['n_embeddings'], batch_size=65536):
            sizes = batch.column(0).to_numpy()
            assert (sizes > 0).all() and sizes.max() <= plan['maximum_embeddings']
            histogram += np.bincount(sizes, minlength=len(histogram))
        assert histogram.sum() == record['classified_windows']
        size_counts[record['fold']] = histogram
    counts = sum(size_counts.values())
    x = np.arange(len(counts))
    quantiles = x[np.searchsorted(np.cumsum(counts), np.array([.1, .5, .9]) * counts.sum())]
    display(pd.Series({'Mean embeddings': np.dot(x, counts) / counts.sum(),
                       '10th percentile': quantiles[0], 'Median': quantiles[1],
                       '90th percentile': quantiles[2], 'Maximum': x[counts > 0].max(),
                       'Fraction below 10': counts[:10].sum() / counts.sum()}).to_frame('Value'))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(x, counts / counts.sum(), width=1, color=COLORS[0])
    ax.axvline(10, color='black', linestyle='--', label='Previous fixed count: 10')
    ax.set(xlabel='Embeddings in the 20 µm geodesic window', ylabel='Fraction of held-out windows')
    ax.legend()
    plt.show()
    ''')
    md('''
    ## Inspect saved logits from every head

    Edit `FOLD`, `N_ROWS` and `ROW` below. The preview takes the first rows for
    inspection, **not a representative random sample**. Logits are local raw
    scores; the associated softmax probabilities are conditional on the head's
    parent. Do not softmax logits from different heads together. Joint
    probabilities multiply the relevant probabilities along the hierarchy.
    An off-path head still has logits, but its conditional decision is not a
    claim that the window belongs to that parent.
    ''')
    code('''
    FOLD = 0
    N_ROWS = 8
    ROW = 0
    assert 0 <= FOLD < 5 and 0 <= ROW < N_ROWS
    sample = next(pq.ParquetFile(predictions_path(FOLD)).iter_batches(batch_size=N_ROWS)).to_pandas()
    preview = sample[['root_id', 'synapse_id', 'window_row', 'n_embeddings']].copy()
    preview['true_leaf'] = [manifests[FOLD]['cells'][str(int(rid))]['cell_type'] for rid in sample.root_id]
    preview['predicted_leaf'] = [leaf_names[int(path[-1])] for path in sample.top_down_predictions]
    display(preview)
    row = sample.iloc[ROW]
    truth = hierarchy.label_paths[preview.iloc[ROW].true_leaf]
    local_rows = []
    for i, head in enumerate(heads):
        logits = np.asarray(row[f'head_{i}_logits'])
        probabilities = np.asarray(row[f'head_{i}_probabilities'])
        assert len(logits) == head['n_children'] and np.isclose(probabilities.sum(), 1, atol=1e-6)
        on_path = head['parent_level'] < 0 or truth[head['parent_level']] == head['parent_name']
        for child, logit, probability in zip(head['children'], logits, probabilities):
            local_rows.append({'Head parent': head['parent_name'], 'On true parent path': on_path,
                               'Child': child, 'Logit': float(logit), 'Conditional probability': float(probability)})
    display(pd.DataFrame(local_rows).set_index(['Head parent', 'Child']).round(5))
    joint_rows = []
    for level, names in enumerate(levels):
        probabilities = np.asarray(row[f'level_{level}_joint_probabilities'])
        assert np.isclose(probabilities.sum(), 1, atol=1e-5)
        selected = int(row.top_down_predictions[level])
        for i, (name, probability) in enumerate(zip(names, probabilities)):
            joint_rows.append({'Level': level, 'Class': name, 'Joint probability': float(probability),
                               'Top-down selection': i == selected})
    display(pd.DataFrame(joint_rows).set_index(['Level', 'Class']).round(5))
    ''')
    md('''
    ## Files and provenance

    `completion.json` records verified checkpoint hashes and coverage.
    `all_lcpn_predictions/fold*.json` records head ordering, score semantics and
    Parquet hashes. Source anchor eligibility was preserved per site, including
    historical 2 µm/5 µm differences documented in the original 20 µm run's
    `source_anchor_audit.json` and the experiment notes.
    Source soma geometry, folds and newest 100k SegCLR features were retained:
    the 5 µm cut applies where present; explicitly soma-free records retain their
    full skeleton. The finer hierarchy changes the label task from the earlier
    six-class run, so its leaf scores should not be compared directly to that run.

    See [the experiment and upstream pipeline notes](../../../docs/presynaptic_geodesic20um.md).
    ''')
    code('''
    display(pd.DataFrame([{'Fold': r['fold'], 'Checkpoint': str(checkpoint_path(r)),
                           'Predictions': str(predictions_path(r['fold'])),
                           'Rows': r['classified_windows']} for r in folds]).set_index('Fold'))
    display(pd.Series({'Embedding run': plan['feature_run_id'],
                       'SegCLR checkpoint SHA256': plan['feature_checkpoint_sha256'],
                       'Anchor policy': plan['anchor_match_policy'],
                       'Completion plan SHA256': completion['plan_sha256']}).to_frame('Provenance'))
    ''')
    md('''
    ## segclr-db export

    The receipt below records the verified export of these held-out windows,
    with one prediction run per fold and raw logits from all five local heads.
    Database `root_id` is the postsynaptic target; `subject_root_id` is the
    classified presynaptic cell. Node IDs refer to its canonical CAVE skeleton.

    Database logits are stored in hierarchy-level class order. Apply softmax
    separately over each head's `child_global_indices`, never over a whole level.
    Full conditional and joint probabilities remain in the shared artifact
    bundle's `all_lcpn_predictions/fold*.parquet` files. The export does not run
    inference on additional synapses or integrate the production predictor.
    ''')
    code('''
    receipt_path = RUN / 'segclr_db_export/receipt.json'
    if receipt_path.exists():
        export = json.loads(receipt_path.read_text())
        assert export['complete'] and len(export['folds']) == 5
        display(pd.Series({key: export[key] for key in
                           ('database_root', 'dataset', 'shared_bundle', 'predictions',
                            'level_logit_rows', 'completed_utc')}).to_frame('Verified export'))
        display(pd.DataFrame(export['folds']).set_index('fold'))
    else:
        print('No completed segclr-db export receipt is present.')
    ''')
    for cell in cells:
        cell.source = cell.source.replace('analysis/presynaptic/latest100k_geodesic20um',
                                          str(output.relative_to(ROOT)))
        if experiment.get('k_observed') is None:
            cell.source = cell.source.replace('20 µm', f'{experiment["window_nm"]/1000:g} µm')
        else:
            cell.source = cell.source.replace('contains every available embedding within **20 µm\ngeodesic distance of its skeleton anchor**',
                'contains **10 observed embeddings selected by geodesic distance from its skeleton anchor**')
            cell.source = cell.source.replace('20 µm geodesic window', 'fixed-10 window').replace('20 µm neighborhoods', 'fixed-10 neighborhoods')
    notebook = nb.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'segclr_db (.venv)', 'language': 'python', 'name': 'segclr_db'},
        'language_info': {'name': 'python'}, 'experiment': output.name,
    })
    nb.validate(notebook)
    destination = output / 'five_fold_results.ipynb'
    nb.write(notebook, destination)
    print(destination, flush=True)


if __name__ == '__main__':
    main()
