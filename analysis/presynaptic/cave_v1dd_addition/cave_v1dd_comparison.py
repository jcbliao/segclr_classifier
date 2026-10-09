"""Notebook helpers for paired MICrONS/V1DD presynaptic comparisons."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = ROOT / 'results/presynaptic/cave_v1dd_addition'
CONDITIONS = ('no_v1dd', 'with_v1dd')
ARCHITECTURES = ('mean', 'pointwise_mlp', 'graph_transformer')
DATASETS = ('combined', 'microns', 'v1dd')
LABELS = {'no_v1dd': 'Without V1DD', 'with_v1dd': 'With V1DD',
          'mean': 'Mean', 'pointwise_mlp': 'Pointwise MLP', 'graph_transformer': 'GT',
          'combined': 'Combined', 'microns': 'MICrONS', 'v1dd': 'V1DD'}
COLORS = {'no_v1dd': '#4C78A8', 'with_v1dd': '#F58518'}
TARGET_CLASSES = ('BipFam', 'MartFam', 'NglFam')
TAGS = {'mean': 'mean', 'pointwise_mlp_L2': 'pointwise_mlp', 'gt_L4_H4': 'graph_transformer'}
METRICS = ('accuracy', 'balanced_accuracy', 'macro_precision', 'macro_f1')


def run_identity(run):
    match = re.fullmatch(r'gnn_lcpn_scratch_(mean|pointwise_mlp_L2|gt_L4_H4)_resnet4x128_n10_mixed16_sampled_fold(\d+)', run)
    if match is None:
        raise ValueError(f'Unexpected V1DD run: {run}')
    return TAGS[match[1]], int(match[2])


def load_training_curves(root=RESULT_ROOT):
    frames = []
    for condition in CONDITIONS:
        for path in sorted((root/condition).glob('*/epoch_metrics.csv')):
            architecture, fold = run_identity(path.parent.name)
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if not frame.empty:
                frame = frame.assign(condition=condition, architecture=architecture,
                                     fold=fold, run=path.parent.name)
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def domains(payload):
    return {'combined': {'cell': payload['test_metrics'], 'window': payload['window_test_metrics']},
            **payload.get('dataset_test_metrics', {})}


def load_summaries(root=RESULT_ROOT):
    rows, payloads = [], {}
    for condition in CONDITIONS:
        for path in sorted((root/condition).glob('gnn_*.json')):
            architecture, fold = run_identity(path.stem)
            payload = json.loads(path.read_text())
            if payload['args']['architecture'] != architecture:
                raise ValueError(f'Architecture mismatch: {path}')
            payloads[fold, architecture, condition] = payload
            for dataset, metrics in domains(payload).items():
                row = dict(condition=condition, architecture=architecture, fold=fold,
                           dataset=dataset, best_epoch=payload.get('best_epoch'), run=path.stem)
                for scope in ('cell', 'window'):
                    row[f'n_{scope}s'] = int(np.asarray(metrics[scope]['confusion_matrix']).sum())
                    for metric in METRICS:
                        row[f'{scope}_{metric}'] = metrics[scope][metric]
                rows.append(row)
    return pd.DataFrame(rows), payloads


def metric_deltas(summary):
    if summary.empty:
        return pd.DataFrame()
    keys = ['fold', 'architecture', 'dataset']
    cols = [f'{scope}_{metric}' for scope in ('cell', 'window') for metric in METRICS]
    pairs = summary[summary.condition.eq('with_v1dd')].merge(
        summary[summary.condition.eq('no_v1dd')], on=keys, suffixes=('_with', '_without'), validate='one_to_one')
    result = pairs[keys].copy()
    for scope in ('cell', 'window'):
        if not pairs[f'n_{scope}s_with'].equals(pairs[f'n_{scope}s_without']):
            raise ValueError(f'Paired {scope} test counts differ')
    for col in cols:
        result[col] = pairs[col+'_with'] - pairs[col+'_without']
    return result


def per_class_metrics(payloads):
    rows = []
    for (fold, architecture, condition), payload in payloads.items():
        for dataset, metrics in domains(payload).items():
            for scope in ('cell', 'window'):
                m = metrics[scope]
                support = np.asarray(m['confusion_matrix']).sum(axis=1)
                for index, label in enumerate(payload['classes']):
                    rows.append(dict(fold=fold, architecture=architecture, condition=condition,
                                     dataset=dataset, granularity=scope, cell_type=label,
                                     support=int(support[index]),
                                     recall=m['per_class_recall'][label] if support[index] else np.nan,
                                     precision=m['per_class_precision'][label]))
    return pd.DataFrame(rows)


def plot_training_curves(curves):
    if curves.empty:
        return None
    metrics = ('train_loss', 'window_macro_f1', 'cell_macro_f1')
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), squeeze=False, layout='constrained')
    for row, architecture in enumerate(ARCHITECTURES):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            for condition in CONDITIONS:
                group = curves[curves.architecture.eq(architecture) & curves.condition.eq(condition)].sort_values('epoch')
                if not group.empty:
                    ax.plot(group.epoch, group[metric], marker='o', markersize=3,
                            color=COLORS[condition], label=LABELS[condition])
            ax.set(title=f'{LABELS[architecture]}: {metric.replace("_", " ")}', xlabel='Epoch')
            ax.grid(alpha=.25)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=False, fontsize=8)
    return fig


def plot_metric_comparison(summary):
    if summary.empty:
        return None
    metrics = ('cell_macro_f1', 'cell_balanced_accuracy', 'window_macro_f1')
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), layout='constrained')
    x = np.arange(3)
    for row, dataset in enumerate(DATASETS):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            for index, condition in enumerate(CONDITIONS):
                group = summary[summary.dataset.eq(dataset) & summary.condition.eq(condition)]
                values = group.set_index('architecture')[metric].reindex(ARCHITECTURES)
                bars = ax.bar(x + (index-.5)*.36, values, .36, color=COLORS[condition], label=LABELS[condition])
                ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=8)
            ax.set(title=f'{LABELS[dataset]}: {metric.replace("_", " ")}', ylim=(0, 1),
                   xticks=x, xticklabels=[LABELS[a] for a in ARCHITECTURES])
            ax.grid(axis='y', alpha=.25)
            if row == col == 0:
                ax.legend(frameon=False)
    return fig


def plot_per_class_recall(frame, dataset, targets=TARGET_CLASSES):
    group = frame[frame.dataset.eq(dataset) & frame.granularity.eq('cell')]
    if group.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    x = np.arange(len(targets))
    for ax, architecture in zip(axes, ARCHITECTURES):
        for index, condition in enumerate(CONDITIONS):
            values = group[group.architecture.eq(architecture) & group.condition.eq(condition)].set_index('cell_type').recall.reindex(targets)
            ax.bar(x+(index-.5)*.36, values, .36, color=COLORS[condition], label=LABELS[condition])
        ax.set(title=LABELS[architecture], xticks=x, xticklabels=targets, ylim=(0,1), ylabel='Cell recall')
        ax.grid(axis='y', alpha=.25)
    axes[0].legend(frameon=False)
    fig.suptitle(f'{LABELS[dataset]} target-family recall')
    return fig


def plot_confusions(payloads, fold, architecture, dataset):
    items = [(condition, payloads[fold, architecture, condition]) for condition in CONDITIONS
             if (fold, architecture, condition) in payloads and dataset in domains(payloads[fold, architecture, condition])]
    if not items:
        return None
    fig, axes = plt.subplots(2, len(items), figsize=(10*len(items), 13), squeeze=False, layout='constrained')
    image = None
    for col, (condition, payload) in enumerate(items):
        classes = payload['classes']
        for row, scope in enumerate(('window', 'cell')):
            ax = axes[row, col]
            counts = np.asarray(domains(payload)[dataset][scope]['confusion_matrix'], dtype=int)
            supported = counts.sum(axis=1)>0
            counts = counts[supported]
            if not len(counts):
                ax.set_axis_off()
                continue
            fractions = counts / counts.sum(axis=1, keepdims=True)
            image = ax.imshow(fractions, vmin=0, vmax=1, cmap='Blues', aspect='auto')
            for (i,j), value in np.ndenumerate(fractions):
                ax.text(j,i,f'{value:.2f}\n({counts[i,j]:,})',ha='center',va='center',fontsize=6,
                        color='white' if value>.5 else 'black')
            ax.set(xticks=range(len(classes)), xticklabels=classes,
                   yticks=range(len(counts)), yticklabels=np.asarray(classes)[supported],
                   xlabel='Predicted', ylabel=f'{scope.title()}: true class', title=LABELS[condition])
            ax.tick_params(axis='x', rotation=60, labelsize=8)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=.7, label='Fraction within true class')
    fig.suptitle(f'Fold {fold}: {LABELS[architecture]} — {LABELS[dataset]}\nRow fraction (raw count)')
    return fig
