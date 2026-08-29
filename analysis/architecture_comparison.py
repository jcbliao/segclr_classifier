"""Lightweight readers for architecture_comparison.ipynb (no inference)."""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

COUNTS = (10, 20, 40)
ARCHITECTURES = ("Mean", "Fully connected", "MPNN", "GT")

def describe_run(run: str):
    match = re.search(r"_n(10|20|40)$", run)
    if not match:
        return None
    n = int(match.group(1))
    if "_mean_" in run:
        return "Mean", "Mean", n
    if "_fc_L" in run:
        architecture, base = "Fully connected", "Fully connected"
        position, lpe = "_position" in run, "_lpe" in run
    elif "_mpnn_L" in run:
        architecture, base = "MPNN", "MPNN"
        position, lpe = "_position" in run, "_lpe" in run
    elif "_gt_L" in run:
        architecture, base = "GT", "GT"
        position, lpe = "_norelpos" not in run, "_nolpe" not in run
    else:
        return None
    features = [name for name, enabled in (("position", position), ("LPE", lpe)) if enabled]
    return architecture, base + ((" + " + " + ".join(features)) if features else ""), n

def load_best_epochs(results: Path) -> pd.DataFrame:
    """One row per started run, selected by best window macro-F1 so far."""
    rows = []
    for csv_path in sorted(results.glob("*/epoch_metrics.csv")):
        run = csv_path.parent.name
        description = describe_run(run)
        if description is None:
            continue
        try:
            epochs = pd.read_csv(csv_path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if epochs.empty or "window_macro_f1" not in epochs:
            continue
        valid = epochs.dropna(subset=["window_macro_f1"])
        if valid.empty:
            continue
        best = valid.loc[valid["window_macro_f1"].idxmax()]
        architecture, model, n = description
        row = {"run": run, "architecture": architecture, "model": model,
               "n_embeddings": n, "best_epoch": int(best["epoch"]),
               "epochs_available": int(epochs["epoch"].max()) + 1,
               "complete": (results / f"{run}.json").is_file()}
        for scope in ("window", "cell"):
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                row[f"{scope}_{metric}"] = float(best.get(f"{scope}_{metric}", np.nan))
        rows.append(row)
    columns = ["run", "architecture", "model", "n_embeddings", "best_epoch",
               "epochs_available", "complete", "window_accuracy", "window_balanced_accuracy",
               "window_macro_f1", "cell_accuracy", "cell_balanced_accuracy", "cell_macro_f1"]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["n_embeddings", "architecture", "model"]).reset_index(drop=True)

def load_confusions(results: Path) -> dict[str, dict]:
    payloads = {}
    # In-progress sweep runs publish this whenever their best epoch improves.
    for path in sorted(results.glob("*/best_metrics.json")):
        if describe_run(path.parent.name) is None:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "window_test_metrics" in data:
            payloads[path.parent.name] = data
    # A completed run's final evaluation supersedes its in-progress payload.
    for path in sorted(results.glob("*_n*.json")):
        if describe_run(path.stem) is None:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "window_test_metrics" in data:
            payloads[path.stem] = data
    return payloads

def plot_within_count(frame, metric="window_macro_f1"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharey=True)
    colors = dict(zip(ARCHITECTURES, ("#555555", "#F58518", "#4C78A8", "#54A24B")))
    for ax, n in zip(axes, COUNTS):
        part = frame[frame.n_embeddings == n].sort_values(metric, ascending=False)
        if part.empty:
            ax.text(.5, .5, "No epochs available", ha="center", va="center")
            ax.set_axis_off()
            continue
        ax.barh(part.model, part[metric], color=[colors[a] for a in part.architecture])
        ax.invert_yaxis(); ax.set_title(f"{n} embeddings")
        ax.set_xlabel(metric.replace("_", " ")); ax.grid(axis="x", alpha=.25)
        for y, value in enumerate(part[metric]):
            ax.text(value + .003, y, f"{value:.3f}", va="center", fontsize=8)
    fig.suptitle("Best available epoch for each run"); fig.tight_layout()
    return fig

def plot_across_counts(frame, metric="window_macro_f1"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=True)
    for ax, architecture in zip(axes.flat, ARCHITECTURES):
        part = frame[frame.architecture == architecture]
        for model, series in part.groupby("model"):
            series = series.sort_values("n_embeddings")
            ax.plot(series.n_embeddings, series[metric], marker="o", label=model)
        ax.set_title(architecture); ax.set_xticks(COUNTS); ax.grid(alpha=.25)
        ax.set_xlabel("Number of embeddings"); ax.set_ylabel(metric.replace("_", " "))
        if not part.empty:
            ax.legend(fontsize=8)
    fig.suptitle("Matched architecture/configuration across embedding counts")
    fig.tight_layout()
    return fig

def show_confusion(payloads, run, scope="window", normalize=True):
    import matplotlib.pyplot as plt
    data = payloads[run]
    field = "window_test_metrics" if scope == "window" else "test_metrics"
    cm = np.asarray(data[field]["confusion_matrix"], dtype=float)
    classes = data["classes"]
    if normalize:
        denominator = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, denominator, out=np.zeros_like(cm), where=denominator != 0)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="Predicted class", ylabel="True class", title=f"{run} - {scope} level")
    fig.colorbar(image, ax=ax); fig.tight_layout(); plt.show()

def confusion_selector(payloads):
    if not payloads:
        print("No completed result JSONs with confusion matrices yet.")
        return
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets unavailable; use show_confusion(payloads, sorted(payloads)[0])")
        return
    run = widgets.Dropdown(options=sorted(payloads), description="Run:",
                           layout=widgets.Layout(width="750px"))
    scope = widgets.ToggleButtons(options=["window", "cell"], description="Level:")
    normalize = widgets.Checkbox(value=True, description="Row normalize")
    output = widgets.interactive_output(
        lambda run, scope, normalize: show_confusion(payloads, run, scope, normalize),
        {"run": run, "scope": scope, "normalize": normalize})
    display(widgets.VBox([run, scope, normalize]), output)
