"""Compare GraphTransformer training across embedding augmentation policies."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = ROOT / "results" / "presynaptic" / "cave_embedding_augmentation_conf0.7"
CONDITIONS = ("clean", "gray", "flip", "structured_low")
LABELS = {
    "clean": "Clean",
    "gray": "Gray",
    "flip": "Flip",
    "structured_low": "Structured low",
}
COLORS = dict(zip(CONDITIONS, ("#4C78A8", "#F58518", "#54A24B", "#E45756"), strict=True))
METRICS = ("window_macro_f1", "window_balanced_accuracy", "cell_macro_f1")


def condition_of(run: str) -> str:
    match = re.search(r"_embaug_(clean|gray|flip|structured_low)_sampled_fold\d+$", run)
    if match is None:
        raise ValueError(f"Unrecognized augmentation condition in run name: {run}")
    return match.group(1)


def load_training_curves(root: Path = RESULT_ROOT) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*/epoch_metrics.csv")):
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, OSError):
            continue
        if frame.empty:
            continue
        frame["run"] = path.parent.name
        frame["condition"] = condition_of(path.parent.name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    curves = pd.concat(frames, ignore_index=True)
    curves["condition"] = pd.Categorical(curves["condition"], CONDITIONS, ordered=True)
    return curves.sort_values(["condition", "epoch"]).reset_index(drop=True)


def load_summaries(root: Path = RESULT_ROOT) -> tuple[pd.DataFrame, dict[str, dict]]:
    rows, payloads = [], {}
    for path in sorted(root.glob("*/best_metrics.json")):
        payload = json.loads(path.read_text())
        if not all(key in payload for key in ("window_test_metrics", "test_metrics", "classes")):
            continue
        run = path.parent.name
        condition = condition_of(run)
        payloads[condition] = payload
        row = {"condition": condition, "label": LABELS[condition],
               "run": run, "best_epoch": payload.get("epoch")}
        for scope, key in (("window", "window_test_metrics"), ("cell", "test_metrics")):
            for metric in ("accuracy", "balanced_accuracy", "macro_precision", "macro_f1"):
                row[f"{scope}_{metric}"] = payload[key][metric]
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["condition"] = pd.Categorical(summary["condition"], CONDITIONS, ordered=True)
        summary = summary.sort_values("condition").reset_index(drop=True)
    return summary, payloads


def metric_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "clean" not in summary.condition.astype(str).values:
        return pd.DataFrame()
    metrics = [column for column in summary if column.startswith(("window_", "cell_"))]
    clean = summary.loc[summary.condition.astype(str) == "clean", metrics].iloc[0]
    delta = summary[["condition", "label", *metrics]].copy()
    delta[metrics] = delta[metrics] - clean
    return delta


def plot_training_curves(curves: pd.DataFrame):
    if curves.empty:
        return None
    metrics = ("train_loss", "window_macro_f1", "cell_macro_f1")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric in zip(axes, metrics, strict=True):
        for condition in CONDITIONS:
            group = curves[curves.condition.astype(str) == condition]
            if not group.empty and metric in group:
                ax.plot(group.epoch, group[metric], marker="o", markersize=3,
                        color=COLORS[condition], label=LABELS[condition])
        ax.set(title=metric.replace("_", " ").title(), xlabel="Epoch")
        ax.grid(alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .98),
               ncol=4, frameon=False)
    fig.suptitle("Embedding augmentation training progress", y=1.08)
    fig.tight_layout(rect=(0, 0, 1, .88))
    return fig


def plot_metric_comparison(summary: pd.DataFrame):
    if summary.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    labels = summary["label"].tolist()
    colors = [COLORS[str(condition)] for condition in summary.condition]
    for ax, metric in zip(axes, METRICS, strict=True):
        bars = ax.bar(labels, summary[metric], color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Clean held-out performance by training augmentation", y=1.04)
    fig.tight_layout()
    return fig


def plot_per_class_recall(payloads: dict[str, dict], scope: str = "cell"):
    if not payloads:
        return None
    key = "test_metrics" if scope == "cell" else "window_test_metrics"
    classes = next(iter(payloads.values()))["classes"]
    x = np.arange(len(classes)); width = .8 / len(CONDITIONS)
    fig, ax = plt.subplots(figsize=(max(10, 1.35 * len(classes)), 4.8))
    for index, condition in enumerate(CONDITIONS):
        if condition not in payloads:
            continue
        recall = payloads[condition][key]["per_class_recall"]
        values = [recall[name] for name in classes]
        ax.bar(x - .4 + width / 2 + index * width, values, width,
               color=COLORS[condition], label=LABELS[condition])
    ax.set(xticks=x, xticklabels=classes, ylim=(0, 1), ylabel="Recall",
           title=f"{scope.title()} recall by class")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    return fig


def plot_confusions(payloads: dict[str, dict], normalize: bool = True):
    """Plot cell and window confusion matrices as a single 2 x 4 panel.

    Each entry includes its raw number of cells/windows.  When ``normalize``
    is true, the first line is the within-true-class fraction and the second
    line is the count used to compute it.
    """
    if not payloads:
        return None
    items = [(condition, payloads[condition]) for condition in CONDITIONS if condition in payloads]
    scopes = (("Window", "window_test_metrics"), ("Cell", "test_metrics"))
    fig, axes = plt.subplots(
        2, len(items), figsize=(5.2 * len(items) + 1, 10.2), squeeze=False,
        layout="constrained",
    )
    shared_image = None
    for scope_row, (scope, key) in enumerate(scopes):
        for column, (condition, payload) in enumerate(items):
            ax = axes[scope_row, column]
            counts = np.asarray(payload[key]["confusion_matrix"], dtype=np.int64)
            denom = counts.sum(axis=1, keepdims=True)
            values = (np.divide(counts, denom, out=np.zeros_like(counts, dtype=float),
                                where=denom != 0) if normalize else counts.astype(float))
            image = ax.imshow(values, vmin=0, vmax=1 if normalize else None, cmap="Blues")
            shared_image = image
            for (row, col), value in np.ndenumerate(values):
                label = f"{value:.2f}\n({counts[row, col]:,})" if normalize else f"{counts[row, col]:,}"
                ax.text(col, row, label, ha="center", va="center", fontsize=7,
                        color="white" if image.norm(value) > .5 else "black")
            classes = payload["classes"]
            ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
                   xticklabels=classes, yticklabels=classes,
                   xlabel="Predicted" if scope_row == 1 else "",
                   ylabel=f"{scope}\nTrue class" if column == 0 else "",
                   title=LABELS[condition] if scope_row == 0 else "")
            ax.tick_params(axis="x", rotation=45)
    fig.colorbar(shared_image, ax=axes, shrink=.82, pad=.015,
                 label="Recall within true class" if normalize else "Count")
    fig.suptitle("Confusion matrices on clean held-out data\n"
                 "row-normalized fraction (raw count)")
    return fig
