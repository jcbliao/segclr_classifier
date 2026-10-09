"""Plots of per-cell CAVE/TEASAR agreement versus Casey label confidence."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DEFAULT_METRICS = {
    'symmetric_mean_um': 'Symmetric mean centerline distance (µm)',
    'teasar_cave_cable_ratio': 'TEASAR / CAVE cable length',
    'f1_0.25um': 'Cable overlap F1 at 0.25 µm',
    'f1_0.5um': 'Cable overlap F1 at 0.5 µm',
    'f1_1um': 'Cable overlap F1 at 1 µm',
    'f1_2um': 'Cable overlap F1 at 2 µm',
}
DEFAULT_CSV = Path('/orcd/scratch/orcd/013/jcbliao/skeleton_geometry_comparison/metrics.csv')


def load_metrics(path=DEFAULT_CSV, confidence='casey_coarse_confidence', cell_type='casey_coarse'):
    frame = pd.read_csv(path, dtype={'root_id': 'string', 'casey_root_id': 'string', 'casey_cell_id': 'string'})
    required = {'root_id', 'status', confidence, cell_type, *DEFAULT_METRICS}
    missing = required - set(frame)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if frame.root_id.duplicated().any():
        raise ValueError('Expected one row per cell')
    scores = pd.to_numeric(frame[confidence], errors='coerce')
    valid = np.isfinite(scores) & scores.between(0, 1)
    ok = frame.status.eq('ok')
    audit = pd.Series({'CSV rows': len(frame), 'successful cells': int(ok.sum()),
                       'successful cells missing/invalid confidence': int((ok & ~valid).sum()),
                       'plotted cells': int((ok & valid).sum())}, name='count')
    frame = frame.loc[ok & valid].copy()
    frame['confidence'] = scores.loc[frame.index]
    frame['cell_type'] = frame[cell_type].fillna('Unlabeled').replace('', 'Unlabeled')
    if frame.empty:
        raise ValueError('No successful cells with valid confidence')
    for metric in DEFAULT_METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors='raise')
        if not np.isfinite(frame[metric]).all():
            raise ValueError(f'Nonfinite metric: {metric}')
    return frame, audit


def correlation(frame, metric):
    if len(frame) < 3 or frame.confidence.nunique() < 2 or frame[metric].nunique() < 2:
        return np.nan
    return float(spearmanr(frame.confidence, frame[metric]).statistic)


def binned_medians(frame, metric, bins=10):
    index = np.minimum((frame.confidence.to_numpy() * bins).astype(int), bins - 1)
    data = frame.assign(_bin=index)
    return data.groupby('_bin', observed=True).agg(
        confidence=('confidence', 'median'), median=(metric, 'median'), n=(metric, 'size'))


def format_axis(ax, frame, metric, label, confidence_label, bins):
    trend = binned_medians(frame, metric, bins)
    ax.plot(trend.confidence, trend['median'], color='black', marker='o', markersize=4,
            linewidth=1.5, label='Confidence-bin median', zorder=5)
    rho = correlation(frame, metric)
    statistic = f'ρ = {rho:.2f}' if np.isfinite(rho) else 'ρ undefined (constant or insufficient data)'
    overlap = metric.startswith(('f1_', 'precision_', 'recall_'))
    ax.text(.03, .03 if overlap else .97, f'n = {len(frame):,}\n{statistic}', transform=ax.transAxes,
            va='bottom' if overlap else 'top', fontsize=8,
            bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
    ax.set(xlim=(-.02, 1.02), xlabel=confidence_label, ylabel=label)
    if metric == 'teasar_cave_cable_ratio':
        ax.axhline(1, color='gray', linestyle='--', linewidth=1)
    set_metric_limits(ax, frame, metric)
    ax.grid(alpha=.2)


def set_metric_limits(ax, frame, metric):
    low, high = float(frame[metric].min()), float(frame[metric].max())
    if metric.startswith(('f1_', 'precision_', 'recall_')):
        pad = max((high - low) * .08, .01)
        ax.set_ylim(max(-.02, low - pad), min(1.02, high + pad))
    elif metric == 'teasar_cave_cable_ratio':
        low, high = min(low, 1), max(high, 1)
        pad = max((high - low) * .08, .01)
        ax.set_ylim(max(0, low - pad), high + pad)
    else:
        ax.set_ylim(0, max(high * 1.08, 1e-8))


def palette(frame):
    names = sorted(frame.cell_type.unique())
    colors = plt.get_cmap('tab20', max(20, len(names)))
    return {name: colors(i) for i, name in enumerate(names)}


def plot_global(frame, metrics=DEFAULT_METRICS, confidence_label='Casey coarse confidence', bins=10):
    columns = 3
    fig, axes = plt.subplots(int(np.ceil(len(metrics) / columns)), columns,
                             figsize=(16, 4.7 * np.ceil(len(metrics) / columns)), squeeze=False)
    colors = palette(frame)
    for ax, (metric, label) in zip(axes.flat, metrics.items()):
        for name, group in frame.groupby('cell_type', sort=True):
            ax.scatter(group.confidence, group[metric], s=10, alpha=.35,
                       color=colors[name], edgecolors='none', label=name, rasterized=True)
        format_axis(ax, frame, metric, label, confidence_label, bins)
        ax.set_title(label, fontsize=11)
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(6, len(labels)),
               fontsize=8, frameon=False)
    fig.suptitle(f'CAVE vs TEASAR geometry · all {len(frame):,} cells', fontsize=14)
    fig.tight_layout(rect=(0, .10, 1, .96))
    return fig


def plot_by_cell_type(frame, metric, label, confidence_label='Casey coarse confidence', bins=10, columns=4):
    names = sorted(frame.cell_type.unique())
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 3.7 * rows),
                             squeeze=False, sharex=True, sharey=True)
    colors = palette(frame)
    for ax, name in zip(axes.flat, names):
        group = frame.loc[frame.cell_type.eq(name)]
        ax.scatter(group.confidence, group[metric], s=13, alpha=.4,
                   color=colors[name], edgecolors='none', rasterized=True)
        format_axis(ax, group, metric, label, confidence_label, bins)
        ax.set_title(name)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    # Setting a lower limit disables autoscaling on shared axes. Set the final
    # range from all cells so later facets cannot be clipped by the first type.
    set_metric_limits(axes.flat[0], frame, metric)
    fig.suptitle(f'{label} vs confidence · by cell type', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    return fig


def correlation_table(frame, metrics=DEFAULT_METRICS):
    rows = []
    for name, group in [('All cells', frame), *list(frame.groupby('cell_type', sort=True))]:
        for metric in metrics:
            rows.append(dict(cell_type=name, metric=metric, n=len(group),
                             confidence_values=group.confidence.nunique(),
                             spearman_rho=correlation(group, metric),
                             median_metric=group[metric].median()))
    return pd.DataFrame(rows)


def save_figure(fig, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f'{name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(output / f'{name}.pdf', bbox_inches='tight')
