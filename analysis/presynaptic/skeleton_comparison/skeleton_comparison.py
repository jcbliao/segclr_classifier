"""Readers and plots comparing presynaptic Mean, Graph Transformer, and TEASAR GT runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOTS = {
    "CAVE skeleton": ROOT / "results" / "presynaptic" / "cave_skeletons",
    "New skeleton (TEASAR)": ROOT / "results" / "presynaptic" / "new_skeletons",
}


def model_label(run: str) -> str:
    if "_mean_" in run:
        return "Mean"
    if "_gt_teasar_" in run:
        return "TEASAR GT"
    if "_gt_" in run:
        return "Graph Transformer"
    return run


def load_training_curves() -> pd.DataFrame:
    frames = []
    for skeleton, root in RESULT_ROOTS.items():
        for path in sorted(root.glob("*/epoch_metrics.csv")):
            try:
                frame = pd.read_csv(path)
            except (pd.errors.EmptyDataError, OSError):
                continue
            if frame.empty:
                continue
            frame["skeleton"] = skeleton
            frame["run"] = path.parent.name
            frame["model"] = model_label(path.parent.name)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_summaries() -> tuple[pd.DataFrame, dict]:
    rows, payloads = [], {}
    for skeleton, root in RESULT_ROOTS.items():
        for path in sorted(root.glob("*.json")):
            payload = json.loads(path.read_text())
            run = path.stem
            payloads[(skeleton, run)] = payload
            row = {"model": model_label(run), "skeleton": skeleton, "run": run}
            for scope, key in (("window", "window_test_metrics"), ("cell", "test_metrics")):
                metrics = payload[key]
                for metric in ("accuracy", "balanced_accuracy", "macro_precision", "macro_f1"):
                    row[f"{scope}_{metric}"] = metrics[metric]
            rows.append(row)
    return pd.DataFrame(rows), payloads


def plot_training_curves(curves: pd.DataFrame):
    if curves.empty:
        return None
    metrics = ["train_loss", "window_macro_f1", "cell_macro_f1"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric in zip(axes, metrics, strict=True):
        for (model, run), group in curves.groupby(["model", "run"]):
            ax.plot(group.epoch, group[metric], marker="o", label=model)
        ax.set(title=metric.replace("_", " ").title(), xlabel="Epoch")
        ax.grid(alpha=.25)
        ax.set_xlim(0, 30)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .9))
    return fig


def plot_metric_comparison(summary: pd.DataFrame):
    if summary.empty:
        return None
    metrics = ["window_macro_f1", "window_balanced_accuracy", "cell_macro_f1"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, metrics, strict=True):
        values = summary.set_index("model")[metric]
        ax.bar(values.index, values.values, color=["#4C78A8", "#F58518", "#54A24B"])
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    return fig


def plot_confusions(payloads: dict, scope: str = "window", normalize: bool = True):
    if not payloads:
        return None
    key = "window_test_metrics" if scope == "window" else "test_metrics"
    items = list(payloads.items())
    fig, axes = plt.subplots(1, len(items), figsize=(6 * len(items), 5), squeeze=False)
    for ax, ((skeleton, run), payload) in zip(axes[0], items, strict=True):
        matrix = np.asarray(payload[key]["confusion_matrix"], dtype=float)
        if normalize:
            denom = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom != 0)
        image = ax.imshow(matrix, vmin=0, vmax=1 if normalize else None, cmap="Blues")
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, f"{value:.2f}" if normalize else f"{value:g}",
                    ha="center", va="center",
                    color="white" if image.norm(value) > 0.5 else "black")
        classes = payload["classes"]
        ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
               xticklabels=classes, yticklabels=classes,
               xlabel="Predicted", ylabel="True", title=model_label(run))
        ax.tick_params(axis="x", rotation=45)
        fig.colorbar(image, ax=ax, fraction=.046)
    fig.suptitle(f"{scope.title()} confusion matrices")
    fig.tight_layout()
    return fig
