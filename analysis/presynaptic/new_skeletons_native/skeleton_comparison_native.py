"""Compare Mean, Pointwise-MLP, and GT on native presynaptic skeletons."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = ROOT / "results" / "presynaptic" / "new_skeletons_native" / "local_center"
MIXED_RESULT_ROOT = ROOT / "results" / "presynaptic" / "new_skeletons_native" / "mixed_cell_b32"
MODELS = ("Mean", "Pointwise-MLP", "GT")
COLORS = dict(zip(MODELS, ("#4C78A8", "#F58518", "#54A24B"), strict=True))
METRICS = ("window_macro_f1", "window_balanced_accuracy", "cell_macro_f1")


def model_label(run: str) -> str:
    if "_pointwise_mlp_" in run:
        return "Pointwise-MLP"
    if "_mean_" in run:
        return "Mean"
    if "_gt_" in run:
        return "GT"
    raise ValueError(f"Unrecognized model in run name: {run}")


def scale_of(path: Path) -> int:
    return int(next(part.removeprefix("scale") for part in path.parts
                    if part.startswith("scale") and part[5:].isdigit()))


def load_training_curves(root: Path = RESULT_ROOT) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("scale*/k*/*/epoch_metrics.csv")):
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, OSError):
            continue
        if frame.empty:
            continue
        frame["scale"] = scale_of(path)
        frame["run"] = path.parent.name
        frame["model"] = model_label(path.parent.name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    curves = pd.concat(frames, ignore_index=True)
    curves["model"] = pd.Categorical(curves["model"], MODELS, ordered=True)
    return curves


def load_summaries() -> tuple[pd.DataFrame, dict]:
    rows, payloads = [], {}
    for path in sorted(RESULT_ROOT.glob("scale*/k*/*.json")):
        payload = json.loads(path.read_text())
        if not all(key in payload for key in ("window_test_metrics", "test_metrics", "classes")):
            continue
        scale, run = scale_of(path), path.stem
        payloads[(scale, run)] = payload
        row = {"scale": scale, "model": model_label(run), "run": run}
        for scope, key in (("window", "window_test_metrics"), ("cell", "test_metrics")):
            for metric in ("accuracy", "balanced_accuracy", "macro_precision", "macro_f1"):
                row[f"{scope}_{metric}"] = payload[key][metric]
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["model"] = pd.Categorical(summary["model"], MODELS, ordered=True)
        summary = summary.sort_values(["scale", "model"], ascending=[False, True])
        summary = summary.reset_index(drop=True)
    return summary, payloads


def plot_training_curves(curves: pd.DataFrame, experiment: str = ""):
    """Yield one three-panel figure per scale, from coarsest to densest."""
    if curves.empty:
        return
    metrics = ("train_loss", "window_macro_f1", "cell_macro_f1")
    for scale in sorted(curves.scale.unique(), reverse=True):
        subset = curves[curves.scale == scale]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, metric in zip(axes, metrics, strict=True):
            for model in MODELS:
                for _, group in subset[subset.model == model].groupby("run", sort=True):
                    if metric in group:
                        ax.plot(group.epoch, group[metric], marker="o", color=COLORS[model], label=model)
            ax.set(title=metric.replace("_", " ").title(), xlabel="Epoch")
            ax.grid(alpha=.25)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94),
                   ncol=3, frameon=False)
        label = f"{experiment} " if experiment else ""
        fig.suptitle(f"b{scale} {label}training progress", y=1.05)
        fig.tight_layout(rect=(0, 0, 1, .82))
        yield fig


def plot_metric_comparison(summary: pd.DataFrame):
    """Yield one three-panel metrics figure per scale, from coarsest to densest."""
    if summary.empty:
        return
    for scale in sorted(summary.scale.unique(), reverse=True):
        subset = summary[summary.scale == scale]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, metric in zip(axes, METRICS, strict=True):
            ax.bar(subset.model.astype(str), subset[metric],
                   color=[COLORS[model] for model in subset.model.astype(str)])
            ax.set_title(metric.replace("_", " ").title())
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=18)
            ax.grid(axis="y", alpha=.25)
        fig.suptitle(f"b{scale} final metrics", y=1.04)
        fig.tight_layout()
        yield fig


def plot_confusions(payloads: dict, scope: str = "window", normalize: bool = True):
    """Yield one confusion-matrix figure per scale, from coarsest to densest."""
    if scope not in ("window", "cell"):
        raise ValueError("scope must be 'window' or 'cell'")
    key = "window_test_metrics" if scope == "window" else "test_metrics"
    for scale in sorted({scale for scale, _ in payloads}, reverse=True):
        items = [(run, payload) for (run_scale, run), payload in payloads.items()
                 if run_scale == scale]
        items.sort(key=lambda item: MODELS.index(model_label(item[0])))
        fig, axes = plt.subplots(1, len(items), figsize=(6 * len(items), 5), squeeze=False)
        for ax, (run, payload) in zip(axes[0], items, strict=True):
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
        fig.suptitle(f"b{scale} {scope} confusion matrices")
        fig.tight_layout()
        yield fig
