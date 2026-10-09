"""Read current native pre/post training metrics without loading checkpoints."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = ROOT / 'results/presynaptic/native_pre_post_axon_filtered_equal_sampling/scale16/k17/conf0.7'
ARCHITECTURES = ('mean', 'pointwise_mlp', 'graph_transformer')
ARCH_LABELS = {'mean': 'Mean', 'pointwise_mlp': 'Pointwise MLP', 'graph_transformer': 'GT'}
CONDITIONS = ('pre_only', 'pre_mean_control', 'pre_post')
LABELS = {'pre_only': 'Pre only', 'pre_mean_control': 'Pre + raw pre mean', 'pre_post': 'Pre + post'}
COLORS = {'pre_only': '#4C78A8', 'pre_mean_control': '#54A24B', 'pre_post': '#F58518'}
METRICS = ('window_macro_f1', 'window_balanced_accuracy', 'cell_macro_f1')


def identity(path):
    fold = int(path.parent.parent.name.removeprefix('fold'))
    condition = path.parent.name
    run = path.name
    architecture = next((arch for arch, prefix in (
        ('mean', 'gnn_lcpn_scratch_mean_'), ('pointwise_mlp', 'gnn_lcpn_scratch_pointwise_mlp_'),
        ('graph_transformer', 'gnn_lcpn_scratch_gt_')) if run.startswith(prefix)), None)
    if architecture is None or condition not in CONDITIONS:
        raise ValueError(f'Unrecognized run: {path}')
    return fold, architecture, condition


def load_training_curves(root=RESULT_ROOT):
    frames = []
    for path in sorted(root.glob('fold*/*/*/epoch_metrics.csv')):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        frame['fold'], frame['architecture'], frame['condition'] = identity(path.parent)
        frame['run'] = path.parent.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_summaries(root=RESULT_ROOT):
    rows, payloads = [], {}
    directories = sorted({p.parent for p in root.glob('fold*/*/*/best_metrics.json')} |
                         {p.parent / p.stem for p in root.glob('fold*/*/gnn_*.json')})
    for directory in directories:
        final = directory.parent / f'{directory.name}.json'
        best = directory / 'best_metrics.json'
        if not final.exists() and not best.exists():
            continue
        payload = json.loads((final if final.exists() else best).read_text())
        fold, architecture, condition = identity(directory)
        key = fold, architecture, condition
        if key in payloads:
            raise ValueError(f'Duplicate experiment: {key}')
        payloads[key] = payload
        row = dict(fold=fold, architecture=architecture, condition=condition,
                   complete=final.exists(), best_epoch=payload.get('best_epoch', payload.get('epoch')),
                   run=directory.name)
        for scope, name in (('window', 'window_test_metrics'), ('cell', 'test_metrics')):
            for metric in ('accuracy', 'balanced_accuracy', 'macro_precision', 'macro_f1'):
                row[f'{scope}_{metric}'] = payload[name][metric]
        rows.append(row)
    return pd.DataFrame(rows), payloads


def metric_deltas(summary):
    rows = []
    if summary.empty:
        return pd.DataFrame()
    metrics = [key for key in summary if key.startswith(('window_', 'cell_'))]
    for (fold, architecture), group in summary.groupby(['fold', 'architecture']):
        by_condition = group.set_index('condition')
        if 'pre_post' not in by_condition.index:
            continue
        post = by_condition.loc['pre_post']
        for reference in ('pre_only', 'pre_mean_control'):
            if reference not in by_condition.index:
                continue
            pre = by_condition.loc[reference]
            rows.append(dict(fold=fold, architecture=architecture, reference=reference,
                             both_complete=bool(pre.complete and post.complete),
                             **{metric: post[metric] - pre[metric] for metric in metrics}))
    return pd.DataFrame(rows)


def plot_training_curves(curves):
    if curves.empty:
        return None
    metrics = ('train_loss', 'window_macro_f1', 'cell_macro_f1')
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), squeeze=False)
    for row, arch in enumerate(ARCHITECTURES):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            for condition in CONDITIONS:
                group = curves[(curves.architecture == arch) & (curves.condition == condition)].sort_values('epoch')
                if not group.empty and metric in group:
                    ax.plot(group.epoch, group[metric], color=COLORS[condition], label=LABELS[condition], marker='o', markersize=2)
            ax.set(title=f'{ARCH_LABELS[arch]} · {metric.replace("_", " ")}', xlabel='Epoch')
            ax.grid(alpha=.25)
            if ax.lines:
                ax.legend(frameon=False)
    fig.suptitle('Filtered native pre/post training progress')
    fig.tight_layout()
    return fig


def plot_metric_comparison(summary):
    if summary.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(3)
    for ax, metric in zip(axes, METRICS):
        for i, condition in enumerate(CONDITIONS):
            selected = summary[summary.condition == condition].set_index('architecture')
            values = [selected.loc[a, metric] if a in selected.index else np.nan for a in ARCHITECTURES]
            bars = ax.bar(x + (i - 1) * .25, values, .25, color=COLORS[condition], label=LABELS[condition])
            ax.bar_label(bars, labels=[f'{v:.3f}' if np.isfinite(v) else '' for v in values], padding=2, fontsize=8)
        ax.set(title=metric.replace('_', ' ').title(), xticks=x,
               xticklabels=[ARCH_LABELS[a] for a in ARCHITECTURES], ylim=(0, 1))
        ax.grid(axis='y', alpha=.25)
    axes[0].legend(frameon=False)
    fig.suptitle('Held-out performance (best-so-far for unfinished models)')
    fig.tight_layout()
    return fig


def plot_per_class_recall(payloads, fold, scope='cell'):
    key = 'test_metrics' if scope == 'cell' else 'window_test_metrics'
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, arch in zip(axes, ARCHITECTURES):
        available = [(c, payloads[(fold, arch, c)]) for c in CONDITIONS if (fold, arch, c) in payloads]
        if not available:
            ax.set_title(f'{ARCH_LABELS[arch]}: awaiting metrics')
            continue
        classes = available[0][1]['classes']
        x = np.arange(len(classes))
        for condition, payload in available:
            assert payload['classes'] == classes
            i = CONDITIONS.index(condition)
            values = [payload[key]['per_class_recall'][name] for name in classes]
            ax.bar(x + (i - 1) * .25, values, .25, color=COLORS[condition], label=LABELS[condition])
        ax.set(title=ARCH_LABELS[arch], xticks=x, xticklabels=classes, ylim=(0, 1), ylabel='Recall')
        ax.tick_params(axis='x', rotation=35)
        ax.legend(frameon=False)
        ax.grid(axis='y', alpha=.25)
    fig.suptitle(f'{scope.title()} recall by class')
    fig.tight_layout()
    return fig


def plot_confusions(payloads, fold, architecture):
    available = [(c, payloads[(fold, architecture, c)]) for c in CONDITIONS if (fold, architecture, c) in payloads]
    if not available:
        return None
    fig, axes = plt.subplots(2, len(available), figsize=(6 * len(available), 11), squeeze=False, layout='constrained')
    for r, (scope, key) in enumerate((('Window', 'window_test_metrics'), ('Cell', 'test_metrics'))):
        for col, (condition, payload) in enumerate(available):
            ax = axes[r, col]
            counts = np.asarray(payload[key]['confusion_matrix'], np.int64)
            totals = counts.sum(axis=1, keepdims=True)
            values = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0)
            image = ax.imshow(values, vmin=0, vmax=1, cmap='Blues')
            for (i, j), value in np.ndenumerate(values):
                ax.text(j, i, f'{value:.2f}\n({counts[i,j]:,})', ha='center', va='center', fontsize=7,
                        color='white' if value > .5 else 'black')
            classes = payload['classes']
            ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes,
                   xlabel='Predicted', ylabel=f'{scope}\nTrue', title=LABELS[condition])
            ax.tick_params(axis='x', rotation=45)
    fig.colorbar(image, ax=axes, shrink=.8, label='Recall within true class')
    fig.suptitle(f'{ARCH_LABELS[architecture]} confusion matrices · fraction (raw count)')
    return fig
