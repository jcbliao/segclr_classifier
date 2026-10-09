"""Compare Mean, Pointwise-MLP, and GT across Casey confidence thresholds at CAVE K=10."""

from __future__ import annotations

import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = ROOT / "results/presynaptic/casey_confidence_cave/k10"
COHORT_ROOT = Path(
    "/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/"
    "casey_confidence_cave/k10"
)
CASEY_COMPARISON = ROOT / "data/v1718_extended_axon_neurons_casey_comparison.csv"
CONFIDENCES = (0, 0.3, 0.5, 0.7, 0.9)
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


def confidence_of(path: Path) -> float:
    for part in path.parts:
        if part.startswith("conf"):
            try:
                confidence = float(part.removeprefix("conf"))
            except ValueError:
                continue
            if confidence in CONFIDENCES:
                return confidence
    raise ValueError(f"No sweep confidence in {path}")


def load_cohort_counts(root: Path = COHORT_ROOT, fold: int = 0) -> pd.DataFrame:
    rows = []
    for confidence in CONFIDENCES:
        path = root / f"conf{confidence:g}" / f"fold{fold}" / "cohort.csv"
        if not path.exists():
            continue
        cohort = pd.read_csv(path)
        cohort["class"] = np.where(
            cohort["original_cell_type"] == "thalamocortical", "thalamocortical",
            np.where(cohort["casey_class"] == "excitatory", "pyramidal", cohort["casey_coarse"]),
        )
        counts = cohort.groupby(["class", "split"]).size().unstack(fill_value=0)
        for class_name, count in counts.iterrows():
            rows.append({
                "confidence": confidence, "class": class_name,
                "train_cells": int(count.get("train", 0)),
                "test_cells": int(count.get("test", 0)),
            })
    return pd.DataFrame(rows)


def load_fine_cohort_counts(
    root: Path = COHORT_ROOT, comparison: Path = CASEY_COMPARISON, fold: int = 0
) -> pd.DataFrame:
    """Train/test cell counts by Casey fine label for each coarse confidence cut."""
    fine = pd.read_csv(comparison, usecols=["root_id", "casey_fine"], dtype={"root_id": str})
    if fine.root_id.duplicated().any():
        raise ValueError("Duplicate roots in Casey comparison")
    rows = []
    for confidence in CONFIDENCES:
        path = root / f"conf{confidence:g}" / f"fold{fold}" / "cohort.csv"
        if not path.exists():
            continue
        cohort = pd.read_csv(path, dtype={"root_id": str})
        cohort = cohort.merge(fine, on="root_id", how="left", validate="one_to_one")
        is_tc = cohort.original_cell_type == "thalamocortical"
        if cohort.loc[~is_tc, "casey_fine"].isna().any():
            raise ValueError(f"Missing Casey fine labels in {path}")
        cohort["fine_class"] = cohort.casey_fine.where(~is_tc, "thalamocortical")
        cohort["coarse_class"] = cohort.casey_coarse.where(~is_tc, "thalamocortical")
        counts = cohort.groupby(["coarse_class", "fine_class", "split"]).size().unstack(fill_value=0)
        for (coarse_class, fine_class), count in counts.iterrows():
            rows.append({
                "confidence": confidence, "coarse_class": coarse_class,
                "fine_class": fine_class,
                "train_cells": int(count.get("train", 0)),
                "test_cells": int(count.get("test", 0)),
            })
    return pd.DataFrame(rows)


def fine_split_matrix(fine_counts: pd.DataFrame) -> pd.DataFrame:
    """Fine types by confidence, with train/test cell counts in each column pair."""
    if fine_counts.empty:
        return pd.DataFrame()
    matrix = fine_counts.pivot_table(
        index="fine_class", columns="confidence",
        values=["train_cells", "test_cells"], aggfunc="sum", fill_value=0,
    )
    matrix = matrix.swaplevel(0, 1, axis=1)
    matrix = matrix.rename(columns={"train_cells": "Train", "test_cells": "Test"}, level=1)
    columns = pd.MultiIndex.from_product(
        [CONFIDENCES, ("Train", "Test")], names=["Minimum confidence", "Split"]
    )
    matrix = matrix.reindex(columns=columns, fill_value=0)
    matrix = matrix.astype(int)
    matrix.index.name = "Casey fine type"
    return matrix


def load_coarse_cohort_counts(root: Path = COHORT_ROOT, fold: int = 0) -> pd.DataFrame:
    """Train/test counts by Casey coarse type, planned or finalized."""
    if fold not in range(5):
        raise ValueError("fold must be 0 through 4")
    rows = []
    for confidence in CONFIDENCES:
        path = root / f"conf{confidence:g}" / f"fold{fold}" / "cohort.csv"
        cohort = pd.read_csv(path)
        cohort["coarse_type"] = cohort["casey_coarse"].where(
            cohort["original_cell_type"] != "thalamocortical", "thalamocortical"
        )
        counts = cohort.groupby(["coarse_type", "split"]).size().unstack(fill_value=0)
        for coarse_type, count in counts.iterrows():
            rows.append({"confidence": confidence, "coarse_type": coarse_type,
                         "train_cells": int(count.get("train", 0)),
                         "test_cells": int(count.get("test", 0))})
    result = pd.DataFrame(rows)
    result.attrs["planned"] = False
    return result


def coarse_split_matrix(coarse_counts: pd.DataFrame) -> pd.DataFrame:
    """Casey coarse types by confidence, with train/test cell counts."""
    if coarse_counts.empty:
        return pd.DataFrame()
    matrix = coarse_counts.pivot_table(
        index="coarse_type", columns="confidence",
        values=["train_cells", "test_cells"], aggfunc="sum", fill_value=0,
    ).swaplevel(0, 1, axis=1)
    matrix = matrix.rename(columns={"train_cells": "Train", "test_cells": "Test"}, level=1)
    matrix = matrix.reindex(columns=pd.MultiIndex.from_product(
        [CONFIDENCES, ("Train", "Test")], names=["Minimum confidence", "Split"]
    ), fill_value=0).astype(int)
    matrix.index.name = "Casey coarse type"
    return matrix


def fold_of(path: Path) -> int:
    return int(next(part[4:] for part in path.parts if part.startswith("fold") and part[4:].isdigit()))


def load_training_curves(root: Path = RESULT_ROOT) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("conf*/fold*/*/epoch_metrics.csv")):
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, OSError):
            continue
        if frame.empty:
            continue
        frame["confidence"] = confidence_of(path)
        frame["fold"] = fold_of(path)
        frame["run"] = path.parent.name
        frame["model"] = model_label(path.parent.name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    curves = pd.concat(frames, ignore_index=True)
    curves["model"] = pd.Categorical(curves["model"], MODELS, ordered=True)
    return curves


def load_summaries(root: Path = RESULT_ROOT) -> tuple[pd.DataFrame, dict]:
    rows, payloads = [], {}
    # Read the checkpoint-selection record directly.  The top-level result
    # JSON is also evaluated with the reloaded best checkpoint, but older
    # files omit its epoch and therefore make the plot look like a final-epoch
    # summary.
    for path in sorted(root.glob("conf*/fold*/*/best_metrics.json")):
        payload = json.loads(path.read_text())
        if not all(key in payload for key in ("window_test_metrics", "test_metrics", "classes")):
            continue
        confidence, fold, run = confidence_of(path), fold_of(path), path.parent.name
        model = model_label(run)
        payloads[(confidence, fold, model)] = payload
        row = {"confidence": confidence, "fold": fold, "model": model, "run": run,
               "best_epoch": payload.get("epoch")}
        for scope, key in (("window", "window_test_metrics"), ("cell", "test_metrics")):
            for metric in ("accuracy", "balanced_accuracy", "macro_precision", "macro_f1"):
                row[f"{scope}_{metric}"] = payload[key][metric]
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["model"] = pd.Categorical(summary["model"], MODELS, ordered=True)
        summary = summary.sort_values(["confidence", "fold", "model"]).reset_index(drop=True)
    return summary, payloads


def plot_training_curves(curves: pd.DataFrame):
    """Yield one three-panel progress figure per confidence threshold."""
    for confidence in CONFIDENCES:
        subset = curves[curves.confidence == confidence]
        if subset.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, metric in zip(axes, ("train_loss", "window_macro_f1", "cell_macro_f1"), strict=True):
            for model in MODELS:
                for _, group in subset[subset.model == model].groupby("run", sort=True):
                    ax.plot(group.epoch, group[metric], marker="o", color=COLORS[model], label=model)
            ax.set(title=metric.replace("_", " ").title(), xlabel="Epoch")
            ax.grid(alpha=.25)
        handles, labels = axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        fig.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(.5, .97),
                   ncol=3, frameon=False)
        fig.suptitle(f"Casey coarse confidence ≥ {confidence:g}: K=10 CAVE training progress", y=1.08)
        fig.tight_layout(rect=(0, 0, 1, .84))
        yield fig


def plot_metric_comparison(summary: pd.DataFrame):
    """Plot best-window-F1-checkpoint metrics against the confidence cutoff."""
    if summary.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    for ax, metric in zip(axes, METRICS, strict=True):
        for model in MODELS:
            subset = summary[summary.model == model]
            if not subset.empty:
                grouped = subset.groupby("confidence", observed=True)[metric].agg(["mean", "std"])
                ax.errorbar(grouped.index.to_numpy(), grouped["mean"],
                            yerr=grouped["std"].fillna(0), fmt="o-", capsize=3,
                            color=COLORS[model], label=model)
                # Epoch is common to all three metric panels because one
                # checkpoint supplies all three values. Label it once to keep
                # the other panels readable. With multiple folds, show the
                # mean selected epoch.
                if ax is axes[0] and "best_epoch" in subset:
                    epochs = subset.groupby("confidence", observed=True)["best_epoch"].mean()
                    for confidence, value in grouped["mean"].items():
                        if confidence in epochs and pd.notna(epochs[confidence]):
                            label = (f"e{int(epochs[confidence])}" if float(epochs[confidence]).is_integer()
                                     else f"ē{epochs[confidence]:.1f}")
                            ax.annotate(label, (confidence, value), xytext=(3, 5),
                                        textcoords="offset points", fontsize=7,
                                        color=COLORS[model])
        ax.set(title=metric.replace("_", " ").title(), xlabel="Minimum Casey coarse confidence",
               xticks=CONFIDENCES, ylim=(0, 1))
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Test score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .97), ncol=3, frameon=False)
    fig.suptitle("K=10 CAVE: best-checkpoint performance across Casey confidence thresholds", y=1.08)
    fig.tight_layout(rect=(0, 0, 1, .84))
    return fig


def plot_confusions(payloads: dict, scope: str = "window", normalize: bool = True):
    """Yield one confusion-matrix figure per confidence threshold."""
    if scope not in ("window", "cell"):
        raise ValueError("scope must be 'window' or 'cell'")
    key = "window_test_metrics" if scope == "window" else "test_metrics"
    for confidence in CONFIDENCES:
        items = [(model, [payload for (conf, _, name), payload in payloads.items()
                          if conf == confidence and name == model]) for model in MODELS]
        items = [(model, values) for model, values in items if values]
        if not items:
            continue
        fig, axes = plt.subplots(1, len(items), figsize=(6 * len(items), 5), squeeze=False)
        for ax, (model, values) in zip(axes[0], items, strict=True):
            classes = values[0]["classes"]
            if any(payload["classes"] != classes for payload in values):
                raise ValueError(f"Class order differs between {model} folds")
            matrix = sum((np.asarray(payload[key]["confusion_matrix"], dtype=float)
                          for payload in values), np.zeros((len(classes), len(classes))))
            if normalize:
                denom = matrix.sum(axis=1, keepdims=True)
                matrix = np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom != 0)
            image = ax.imshow(matrix, vmin=0, vmax=1 if normalize else None, cmap="Blues")
            for (row, col), value in np.ndenumerate(matrix):
                ax.text(col, row, f"{value:.2f}" if normalize else f"{value:g}",
                        ha="center", va="center",
                        color="white" if image.norm(value) > 0.5 else "black")
            ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
                   xticklabels=classes, yticklabels=classes,
                   xlabel="Predicted", ylabel="True", title=model)
            ax.tick_params(axis="x", rotation=45)
            fig.colorbar(image, ax=ax, fraction=.046)
        fig.suptitle(f"Casey coarse confidence ≥ {confidence:g}: {scope} confusion matrices")
        fig.tight_layout()
        yield fig
