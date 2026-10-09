"""Lightweight readers for architecture_comparison.ipynb (no inference).

Everything these read is `fold_0`: one 80/20 split of whole cells at split_seed 0.
There is no cross-validation in this pipeline, so a score here is a single-fold
estimate and a margin between two runs is a margin on that one split -- label any
figure or table taken from this module accordingly.

Runs differ along three axes, and the vocabulary is used consistently below:
the **architecture** is the aggregation method (`describe_run`'s first field),
the **model** is one architecture with a particular set of input ablations (its
second field), and the **embedding count** is how many nodes the neighborhood
holds. A figure fixes two of the three and draws the last.

`hierarchy_class_groups` is the one place this module reaches outside itself:
it imports the trained-on tree from `data.dataset_lcpn`, which pulls in torch,
and falls back to a flat class list when that import fails. Nothing else here
needs more than json, re, numpy and pandas.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

COUNTS = (10, 20, 40)
# A run name ends in the fold and carries the embedding count somewhere before
# it (other tags -- _noemb, _frozenagg -- may sit between the two). Both are
# required: a name without a fold predates the fold tag and is not a run this
# module can place on a split.
RUN_RE = re.compile(r"_n(10|20|40)(?=_|$).*_fold(\d+)$")
#: The aggregation families, in ladder order. A concrete **architecture** is a
#: family plus its depth (`MPNN L2`, `GT L4`) -- two depths of one family are
#: two architectures, not two ablations of one, because depth changes the model
#: rather than what it is fed. `ARCHITECTURE_RE` is what splits the two apart.
ARCHITECTURE_FAMILIES = ("Mean", "Pointwise MLP", "MPNN complete", "MPNN", "GT")
# One colour per family, in ARCHITECTURE_FAMILIES order -- zipped, so the two
# sequences have to stay aligned. Depths of one family share its colour; they
# never share a panel, and where they share a chart the labels tell them apart.
COLORS = ("#555555", "#B279A2", "#F58518", "#4C78A8", "#54A24B")
#: An architecture label is a family and an optional depth.
ARCHITECTURE_RE = re.compile(r"^(?P<family>.+?)(?: L(?P<depth>\d+))?$")


def architecture_family(architecture: str) -> str:
    """The aggregation family behind an architecture label, depth stripped."""
    return ARCHITECTURE_RE.match(architecture)["family"]


def architecture_rank(architecture: str) -> tuple[int, int, str]:
    """Sort key putting families in ladder order and depths in numeric order.

    Numeric, not lexical: `L10` sorts after `L2` here, where sorting the label
    would put it before.
    """
    match = ARCHITECTURE_RE.match(architecture)
    family, depth = match["family"], match["depth"]
    order = {name: index for index, name in enumerate(ARCHITECTURE_FAMILIES)}
    return (order.get(family, len(order)), int(depth) if depth else 0, family)


def architecture_color(architecture: str) -> str:
    """The family's colour, shared by all of its depths."""
    colors = dict(zip(ARCHITECTURE_FAMILIES, COLORS, strict=True))
    return colors.get(architecture_family(architecture), "#888888")


def architectures_in(frame) -> list[str]:
    """The architectures present in a frame, in ladder-then-depth order.

    Derived from the data rather than from a fixed tuple: a new depth is a new
    architecture, so the panel grid has to grow when one appears instead of
    dropping it off the end of a hardcoded list.
    """
    return sorted({str(a) for a in frame.architecture.dropna().unique()},
                  key=architecture_rank)

def is_flat(run: str) -> bool:
    """True for a run trained with the flat-softmax head instead of the LCPN.

    `gnn_flat_scratch_*` differs from the rest of the sweep by its
    classification objective, not by its aggregation, so putting it on the
    architecture comparison would answer a different question than the panel
    claims to. It is kept out of every table and plot here and left only in the
    confusion matrices, where the head is the thing being looked at.
    """
    return "_flat_scratch_" in run


def fold_of(run: str) -> int | None:
    """The split seed this run trained on, or None if the name carries no fold."""
    match = RUN_RE.search(run)
    return None if match is None else int(match.group(2))


def describe_run(run: str):
    """(architecture, model, embedding count) for a run name, or None to skip it.

    `_linear_` runs return None. A node-wise Linear commutes with the mean, so
    those runs are algebraically mean pooling followed by a linear projection
    and carry no aggregation the rest of the ladder does not already show; they
    are kept out of every table and plot here.
    """
    match = RUN_RE.search(run)
    if not match:
        return None
    n = int(match.group(1))

    def depth_of(pattern: str) -> str:
        """The layer count in the run name, as a ` L<d>` suffix for the label."""
        found = re.search(pattern, run)
        return f" L{found.group(1)}" if found else ""

    # The depth belongs to the ARCHITECTURE, not to the ablation: it changes the
    # model rather than what the model is fed, and two depths of one family are
    # two architectures to compare. It is also what keeps `mean_over_folds` from
    # averaging a depth-2 run and a depth-4 run at the same embedding count as
    # if they were two folds of one model -- MODEL_KEYS groups on these labels,
    # and the `folds` column would still have said 1.
    if "_mean_" in run:
        return "Mean", "Mean", n
    if "_pointwise_mlp_L" in run:
        # No position/LPE variants: the model refuses those switches for this
        # architecture, so depth is the only thing that can vary.
        architecture = "Pointwise MLP" + depth_of(r"_pointwise_mlp_L(\d+)")
        return architecture, architecture, n
    if "_mpnn_complete_L" in run:
        architecture = "MPNN complete" + depth_of(r"_mpnn_complete_L(\d+)")
        position, lpe = "_position" in run, "_lpe" in run
    elif "_mpnn_L" in run:
        architecture = "MPNN" + depth_of(r"_mpnn_L(\d+)")
        position, lpe = "_position" in run, "_lpe" in run
    elif "_gt_L" in run:
        architecture = "GT" + depth_of(r"_gt_L(\d+)")
        position, lpe = "_norelpos" not in run, "_nolpe" not in run
    else:
        return None
    features = [name for name, enabled in (("position", position), ("LPE", lpe)) if enabled]
    return architecture, architecture + ((" + " + " + ".join(features)) if features else ""), n

def load_best_epochs(results: Path, payloads: dict[str, dict] | None = None) -> pd.DataFrame:
    """One row per started LCPN run, selected by best window macro-F1 so far.

    Flat-softmax runs are excluded -- see `is_flat`.

    Pooled metrics come from `epoch_metrics.csv`, which every started run has.
    Per-class metrics come from the run's confusion matrix (`payloads`, loaded
    here if not passed), because the CSV holds recall and precision only at the
    finest level and neither combines upward into a coarser hierarchy class
    without the counts behind it. The payload is written at the same best epoch
    the CSV selects, and `payload_epoch` records which epoch it actually was so
    a stale one is visible rather than silently averaged in.
    """
    payloads = load_confusions(results) if payloads is None else payloads
    rows = []
    for csv_path in sorted(results.glob("*/epoch_metrics.csv")):
        run = csv_path.parent.name
        description = describe_run(run)
        if description is None or is_flat(run):
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
               "n_embeddings": n, "fold": fold_of(run), "best_epoch": int(best["epoch"]),
               "epochs_available": int(epochs["epoch"].max()) + 1,
               "complete": (results / f"{run}.json").is_file()}
        for scope in ("window", "cell"):
            for metric in ("accuracy", "balanced_accuracy", "macro_precision", "macro_f1"):
                row[f"{scope}_{metric}"] = float(best.get(f"{scope}_{metric}", np.nan))
        payload = payloads.get(run)
        if payload is not None:
            epoch = payload.get("epoch")
            row["payload_epoch"] = np.nan if epoch is None else int(epoch)
            row.update(per_class_columns(payload))
        rows.append(row)
    identity = ["run", "architecture", "model", "n_embeddings", "fold", "best_epoch",
                "payload_epoch", "epochs_available", "complete"]
    pooled = [f"{scope}_{metric}" for scope in ("window", "cell")
              for metric in ("accuracy", "balanced_accuracy", "macro_precision", "macro_f1")]
    # Union rather than the first row's keys: a run whose payload predates a
    # class would otherwise decide the column set for every run beside it, and
    # a run with no payload at all has no per-class keys to contribute.
    per_class = sorted({key for row in rows for key in row} - set(identity) - set(pooled))
    return pd.DataFrame(rows, columns=identity + pooled + per_class).sort_values(
        ["n_embeddings", "architecture", "model", "fold"]).reset_index(drop=True)


#: The pooled metrics `load_best_epochs` always writes. The per-class columns
#: beside them are discovered from the frame instead (`metric_names`), since
#: which classes exist is a property of the hierarchy the runs were trained
#: against, not of this module.
METRICS = ["window_accuracy", "window_balanced_accuracy", "window_macro_precision",
           "window_macro_f1", "cell_accuracy", "cell_balanced_accuracy",
           "cell_macro_precision", "cell_macro_f1"]
#: What identifies a model across folds: the same aggregation, feature set and
#: embedding count, trained on a different split.
MODEL_KEYS = ["architecture", "model", "n_embeddings"]
#: Columns that identify a run rather than score it; everything else in a frame
#: from `load_best_epochs` is a metric and is averaged over folds.
IDENTITY = ["run", "architecture", "model", "n_embeddings", "fold", "best_epoch",
            "payload_epoch", "epochs_available", "complete"]


def metric_names(frame: pd.DataFrame) -> list[str]:
    """Every scored column in `frame`, pooled and per class alike."""
    return [c for c in frame.columns if c not in IDENTITY and not c.endswith("_std")]


def mean_over_folds(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per model, averaged over whichever folds are present.

    Every plot goes through this, so a second fold changes the figures by
    appearing on disk and nothing else. With one fold the mean is that fold's
    number and every `_std` column is NaN, which is the honest reading: a
    single-fold estimate has no spread to report. With several, `folds` says
    how many went into the mean and `_std` is the sample standard deviation
    across them -- read a margin between two models against that, since this
    pipeline has no other estimate of its own noise.

    `folds` is counted per model, not per page: a model trained on two folds at
    n=20 and one at n=40 averages over what it has at each count rather than
    being dropped or silently mixed.
    """
    if frame.empty:
        return frame.assign(folds=pd.Series(dtype=int),
                            **{f"{metric}_std": pd.Series(dtype=float) for metric in METRICS})
    metrics = metric_names(frame)
    grouped = frame.groupby(MODEL_KEYS, sort=False)
    table = grouped[metrics].mean()
    for metric in metrics:
        table[f"{metric}_std"] = grouped[metric].std(ddof=1)
    table["folds"] = grouped["fold"].nunique()
    # The weakest fold's progress and completion, not the best: a mean over
    # folds is only as finished as its least finished member.
    table["epochs_available"] = grouped["epochs_available"].min()
    table["complete"] = grouped["complete"].all()
    return table.reset_index().sort_values(MODEL_KEYS).reset_index(drop=True)


def fold_label(frame: pd.DataFrame) -> str:
    """How to describe the split(s) behind a figure, in its own title."""
    folds = sorted({int(f) for f in frame["fold"].dropna().unique()})
    if not folds:
        return "no fold recorded"
    if len(folds) == 1:
        return f"fold_{folds[0]}"
    return "mean of " + ", ".join(f"fold_{fold}" for fold in folds)

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
            # The final summary records the split and the args but not which
            # epoch the reloaded `checkpoint_best.pt` came from; the in-progress
            # payload it supersedes does. Carrying that across is what lets
            # `payload_epoch` be checked against the CSV's best epoch for a
            # finished run instead of going missing exactly where the run is
            # most complete.
            if "epoch" not in data and "epoch" in payloads.get(path.stem, {}):
                data["epoch"] = payloads[path.stem]["epoch"]
            payloads[path.stem] = data
    return payloads

def plot_within_count(frame, metric="window_macro_f1"):
    """One panel per embedding count, models sorted by `metric` within each.

    Takes the per-run frame and averages folds itself (`mean_over_folds`), so a
    model with several folds is one bar carrying its across-fold error bar
    rather than one bar per split.

    Each panel carries its own y axis. Sharing it would put every panel on one
    categorical registry -- matplotlib propagates the same `UnitData` to shared
    axes -- so panel 2 and 3 would draw their bars at panel 1's ordering while
    sorting their own rows independently, and the value labels would land beside
    the wrong bars. Bars and labels are placed on explicit numeric positions for
    the same reason.
    """
    import matplotlib.pyplot as plt
    averaged = mean_over_folds(frame)
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    for ax, n in zip(axes, COUNTS):
        part = averaged[averaged.n_embeddings == n].sort_values(metric, ascending=False)
        if part.empty:
            ax.text(.5, .5, "No epochs available", ha="center", va="center")
            ax.set_axis_off()
            continue
        y = np.arange(len(part))
        # NaN is what a single-fold model's spread is; matplotlib would drop the
        # whole errorbar collection over one, so it becomes a zero-length bar.
        spread = part[f"{metric}_std"].fillna(0.).to_numpy()
        ax.barh(y, part[metric], color=[architecture_color(a) for a in part.architecture],
                xerr=spread if spread.any() else None,
                error_kw={"ecolor": "#333333", "elinewidth": 1, "capsize": 3})
        ax.set_yticks(y, part.model)
        ax.invert_yaxis(); ax.set_title(f"{n} embeddings")
        ax.set_xlabel(metric_label(metric)); ax.grid(axis="x", alpha=.25)
        for position, value, error, folds in zip(y, part[metric], spread, part.folds):
            label = f"{value:.3f}" if folds < 2 else f"{value:.3f} \u00b1{error:.3f}"
            ax.text(value + error + .003, position, label, va="center", fontsize=8)
    fig.suptitle(f"Best available epoch for each run ({fold_label(frame)})")
    fig.tight_layout()
    return fig


def plot_across_counts(frame, metric="window_macro_f1"):
    """One panel per architecture, `metric` against embedding count.

    Folds are averaged first, so a point is a model's mean over its folds and
    its error bar is the spread across them (nothing drawn where there is one
    fold, which is the current state of every run).
    """
    import matplotlib.pyplot as plt
    averaged = mean_over_folds(frame)
    # One panel per architecture PRESENT, not per entry of a fixed tuple: a
    # depth is an architecture, so a new one has to grow the grid rather than
    # fall off the end of a hardcoded list. (A fixed 2x2 once silently dropped
    # whichever architecture zip() ran past.)
    architectures = architectures_in(averaged)
    ncols = 3
    nrows = max(1, -(-len(architectures) // ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 5 * nrows), sharex=True, sharey=True,
        squeeze=False
    )
    for ax in axes.flat[len(architectures):]:
        ax.set_axis_off()
    for ax, architecture in zip(axes.flat, architectures, strict=False):
        part = averaged[averaged.architecture == architecture]
        for model, series in part.groupby("model"):
            series = series.sort_values("n_embeddings")
            ax.errorbar(series.n_embeddings, series[metric],
                        yerr=series[f"{metric}_std"].fillna(0.).to_numpy(),
                        marker="o", capsize=3, label=model)
        ax.set_title(architecture); ax.set_xticks(COUNTS); ax.grid(alpha=.25)
        ax.set_xlabel("Number of embeddings"); ax.set_ylabel(metric_label(metric))
        if not part.empty:
            ax.legend(fontsize=8)
    fig.suptitle(f"{metric_label(metric)} across embedding counts "
                 f"({fold_label(frame)})")
    fig.tight_layout()
    return fig


def run_title(run: str) -> str:
    """A readable label for a run name, for plot titles and selector entries.

    `gnn_lcpn_scratch_`, the `_resnet4x128` trunk tag and the fold carry nothing
    a title needs -- every run compared here shares the trunk, the scratch/LCPN
    setup and the split, so all the name distinguishes is the aggregation, which
    is what `describe_run` already parses. The flat-softmax runs are marked
    explicitly because `describe_run` reads only their aggregation and would
    otherwise label one identically to its LCPN counterpart.
    """
    described = describe_run(run)
    if described is None:
        return run
    _, model, n = described
    parts = [model, f"n={n}"]
    if is_flat(run):
        parts.append("flat softmax")
    return " \u00b7 ".join(parts)


def show_confusion(payloads, run, scope="window", normalize=True):
    """One run's confusion matrix, with its headline metrics above it.

    Macro F1 leads the metric line because it is what checkpoints are selected
    on and what the class imbalance makes raw accuracy misleading about; row
    normalization puts each class's recall on the diagonal.
    """
    import matplotlib.pyplot as plt
    data = payloads[run]
    field = "window_test_metrics" if scope == "window" else "test_metrics"
    metrics = data[field]
    cm = np.asarray(metrics["confusion_matrix"], dtype=float)
    classes = data["classes"]
    if normalize:
        denominator = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, denominator, out=np.zeros_like(cm), where=denominator != 0)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="Predicted class", ylabel="True class")
    ax.set_title(
        f"macro F1 {metrics['macro_f1']:.3f}   balanced accuracy "
        f"{metrics['balanced_accuracy']:.3f}   accuracy {metrics['accuracy']:.3f}\n"
        f"{run_title(run)} \u00b7 {scope} level", fontsize=11)
    threshold = 0.5 if normalize else (cm.max() / 2 if cm.size else 0)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            label = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j]):,}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > threshold else "black")
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
    # (label, value) pairs: the dropdown reads as the plot titles do, while the
    # value stays the run name every payload is keyed by.
    run = widgets.Dropdown(options=[(run_title(r), r) for r in sorted(payloads)],
                           description="Run:", layout=widgets.Layout(width="750px"))
    scope = widgets.ToggleButtons(options=["window", "cell"], description="Level:")
    normalize = widgets.Checkbox(value=True, description="Row normalize")
    output = widgets.interactive_output(
        lambda run, scope, normalize: show_confusion(payloads, run, scope, normalize),
        {"run": run, "scope": scope, "normalize": normalize})
    display(widgets.VBox([run, scope, normalize]), output)


#: Cache for `hierarchy_class_groups`, keyed by the finest-level class tuple a
#: run reports. Parsing the tree is cheap, but it is asked for once per figure
#: redraw and the answer cannot change while the kernel is up.
_GROUP_CACHE: dict[tuple[str, ...], list[dict]] = {}


def hierarchy_class_groups(classes) -> list[dict]:
    """Every class of the trained-on hierarchy, breadth-first, over `classes`.

    `classes` is the finest level a run actually reports -- the 8-class level
    the LCPN's last head emits. This walks the tree back up from there, so the
    selector offers `neuron` / `non_neuron` first, then `excitatory` /
    `inhibitory`, then the eight themselves, and a coarse selection is scored by
    summing the confusion matrix's block rather than by re-running anything.

    Falls back to the flat list -- one group per reported class, no tree -- when
    the hierarchy cannot be imported (`data.dataset_lcpn` pulls in torch) or
    when its finest level does not match what the run reports. A mismatch means
    the runs on disk were trained against a different taxonomy than this
    checkout defines, and grouping their classes by this tree would silently
    score the wrong blocks.
    """
    key = tuple(classes)
    if key in _GROUP_CACHE:
        return _GROUP_CACHE[key]
    flat = [{"name": name, "level": 0, "leaves": (name,)} for name in key]
    try:
        from data.dataset_lcpn import (
            ACTIVE_HIERARCHY_TREE, DROP_LABELS, HIERARCHY_LEVELS_DROPPED)
        from gnn.hierarchy import (
            hierarchy_groups, parse_hierarchy, truncate_hierarchy, with_dropped_labels)
    except ImportError:
        return _GROUP_CACHE.setdefault(key, flat)
    hierarchy = with_dropped_labels(
        truncate_hierarchy(parse_hierarchy(ACTIVE_HIERARCHY_TREE), HIERARCHY_LEVELS_DROPPED),
        DROP_LABELS)
    if tuple(hierarchy.level_classes[-1]) != key:
        return _GROUP_CACHE.setdefault(key, flat)
    return _GROUP_CACHE.setdefault(key, hierarchy_groups(hierarchy))


def group_scores(confusion, classes, groups) -> dict[str, dict[str, float]]:
    """Recall, precision and F1 for every hierarchy class, from one matrix.

    A group's true positives are the whole `leaves` x `leaves` block, not just
    the diagonal: at the `excitatory` node, a thalamocortical window predicted
    pyramidal is a correct excitatory call, and counting only the diagonal would
    score the coarse question with the fine question's errors still in it.

    Everything is derived from the confusion matrix rather than read from the
    trainer's `per_class_recall` / `per_class_precision`, because those are
    defined only at the finest level and neither recall nor precision can be
    combined across classes without the counts behind them.
    """
    index = {name: i for i, name in enumerate(classes)}
    confusion = np.asarray(confusion, dtype=float)
    scores = {}
    for group in groups:
        rows = [index[leaf] for leaf in group["leaves"] if leaf in index]
        if not rows:
            continue
        block = confusion[np.ix_(rows, rows)].sum()
        actual = confusion[rows, :].sum()
        predicted = confusion[:, rows].sum()
        recall = block / actual if actual else np.nan
        precision = block / predicted if predicted else 0.0
        total = recall + precision
        scores[group["name"]] = {
            "recall": float(recall), "precision": float(precision),
            "f1": 0.0 if not total or np.isnan(total) else float(2 * recall * precision / total)}
    return scores


def per_class_columns(payload) -> dict[str, float]:
    """`{scope}_{stat}_{class}` for every hierarchy class, at both levels.

    Both scopes come off the same payload, so the window score and the cell
    score in the selector are always the same epoch's -- the epoch the payload
    was written at, which `load_best_epochs` reports as `payload_epoch`.
    """
    classes = payload["classes"]
    groups = hierarchy_class_groups(classes)
    columns = {}
    for scope, field in (("window", "window_test_metrics"), ("cell", "test_metrics")):
        metrics = payload.get(field)
        if not metrics or "confusion_matrix" not in metrics:
            continue
        for name, scores in group_scores(metrics["confusion_matrix"], classes, groups).items():
            for stat, value in scores.items():
                columns[f"{scope}_{stat}_{name}"] = value
    return columns


def class_names(payloads) -> list[str]:
    """The classes a selector offers, breadth-first, or [] with no payloads.

    Takes the payload dict rather than a metrics frame: the class list is a
    property of the hierarchy the runs were trained against, which the payloads
    carry and a frame of derived columns only implies.
    """
    for payload in payloads.values():
        return [group["name"] for group in hierarchy_class_groups(payload["classes"])]
    return []


#: Every metric a per-class selection can ask for, and what it resolves to when
#: no class is selected. Per-class accuracy is recall by definition -- the
#: windows of one class are exactly the windows whose target is that class -- so
#: it maps there rather than being refused or left off the menu.
STAT_ALIASES = {"f1": "macro_f1", "recall": "balanced_accuracy",
                "precision": "macro_precision", "accuracy": "accuracy"}


def metric_column(stat: str = "f1", scope: str = "window",
                  class_name: str | None = None) -> str:
    """The column holding `stat` at `scope`, for one class or pooled."""
    if class_name is None:
        return f"{scope}_{STAT_ALIASES[stat]}"
    return f"{scope}_{'recall' if stat == 'accuracy' else stat}_{class_name}"


def metric_label(column: str) -> str:
    """An axis label for a metric column, with the class named where there is one."""
    match = re.match(r"^(window|cell)_(f1|recall|precision)_(.+)$", column)
    if match:
        scope, stat, name = match.groups()
        return f"{scope} {stat} \u00b7 {name}"
    return column.replace("_", " ")


def model_label(run: str) -> str:
    """The aggregation-plus-feature-variant label for a run, flat runs marked.

    `describe_run` reads only the aggregation, so a flat-softmax run and its
    LCPN twin would otherwise carry the same label and collide wherever runs
    are grouped by it. This is the single parser for that label --
    `feature_prediction_correlation.model_of` delegates here rather than
    matching the name a second time.
    """
    described = describe_run(run)
    if described is None:
        return run
    return described[1] + (" (flat softmax)" if is_flat(run) else "")


def architecture_grid(frame, stat="f1", scope="window", class_name=None):
    """`plot_across_counts` for one hierarchy class, or pooled when it is None.

    One panel per architecture, one line per ablation, embedding count on x --
    so a panel answers "does this ablation keep paying as the neighborhood
    grows", and comparing panels answers "does that depend on the aggregation".

    The class selection asks the same question of one class at a time, at any
    level of the tree. The macro average pools eight classes with wildly
    different window counts, so an ablation that helps only a rare class, or
    only helps `pyramidal`, is invisible in it -- and a coarse selection like
    `neuron` asks whether the ablation helps at all before the fine distinctions
    are attempted.
    """
    import matplotlib.pyplot as plt
    column = metric_column(stat, scope, class_name)
    if column not in frame.columns:
        print(f"{column} is not available -- no run in this frame has a confusion matrix "
              "carrying that class")
        return
    plot_across_counts(frame, column)
    plt.show()


def ablation_label(architecture: str, model: str) -> str:
    """What a model carries beyond its architecture -- its input ablations.

    The two input switches this sweep varies are the **position** features and
    the **Laplacian PE**, and `describe_run` already writes them into the model
    label; this strips the architecture off the front so the ablation dropdown
    says "position + LPE" rather than repeating "MPNN" in every entry.

    Anything else the label carries stays: a GT run's depth is not an input
    ablation, but two depths are two different models, so `GT L2 + LPE` becomes
    `L2 + LPE` and never collides with `L4 + LPE` in the same dropdown.
    """
    rest = model[len(architecture):].strip() if model.startswith(architecture) else model
    rest = rest.removeprefix("+").strip()
    return rest or "no position, no LPE"


def linked_model_pickers(pairs, architecture_label="Architecture:",
                         model_label_text="Ablation:"):
    """Two dropdowns: architecture, then its ablations -- the second follows the first.

    Similar models group themselves this way rather than being flattened into
    one long list: the architectures are the model *types* and the second
    dropdown holds that type's input ablations -- position, LPE, both or
    neither -- which is the distinction every "for each model" panel selects
    over. A single dropdown would put `GT + LPE` next to `MPNN` purely because
    of how the labels sort.

    The second dropdown's *labels* are the ablations alone (`ablation_label`)
    while its *values* stay the full model label every table is keyed by, so
    nothing downstream has to reassemble the two halves.

    Returns `(architecture_widget, model_widget)`; the caller wires them into
    whatever it renders, and the model widget's `.value` is always a model of
    the currently selected architecture.
    """
    import ipywidgets as widgets

    grouped: dict[str, list[str]] = {}
    for architecture, model in pairs:
        grouped.setdefault(architecture, [])
        if model not in grouped[architecture]:
            grouped[architecture].append(model)
    names = sorted(grouped, key=architecture_rank)

    def options_for(name):
        return [(ablation_label(name, model), model) for model in sorted(grouped[name])]

    architecture = widgets.Dropdown(options=names, description=architecture_label,
                                    layout=widgets.Layout(width="380px"))
    model = widgets.Dropdown(options=options_for(names[0]) if names else [],
                             description=model_label_text,
                             layout=widgets.Layout(width="420px"))

    def follow(change):
        # Options are replaced wholesale, which resets `.value` to the first
        # entry. That is what should happen: the previous ablation belongs to
        # the previous architecture and is not a valid selection here.
        model.options = options_for(change["new"])

    architecture.observe(follow, names="value")
    return architecture, model


def architecture_grid_selector(frame, payloads):
    """Class, level and metric dropdowns driving `architecture_grid`."""
    names = class_names(payloads)
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets unavailable; call architecture_grid(df, class_name=...) directly")
        return
    if not names:
        print("No confusion matrices on disk, so no per-class breakdown; "
              "showing the pooled metrics only")
    # (label, value): None is the pooled selection, and it has to be a real
    # option rather than an empty string, since `metric_column` switches on it.
    class_name = widgets.Dropdown(
        options=[("All classes (macro)", None)] + [(n, n) for n in names],
        description="Class:", layout=widgets.Layout(width="380px"))
    scope = widgets.ToggleButtons(options=["window", "cell"], description="Level:")
    stat = widgets.ToggleButtons(options=["f1", "recall", "precision", "accuracy"],
                                 description="Metric:")
    output = widgets.interactive_output(
        lambda stat, scope, class_name: architecture_grid(frame, stat, scope, class_name),
        {"stat": stat, "scope": scope, "class_name": class_name})
    display(widgets.VBox([class_name, scope, stat]), output)


def runs_by_count(payloads, model: str, counts=COUNTS) -> dict[int, str | None]:
    """The run of `model` at each embedding count, or None where there is none.

    Keyed by count rather than filtered down to what exists, so a confusion grid
    keeps an empty column where a run has not finished instead of silently
    shifting n=40's matrix into n=20's slot.
    """
    chosen = {n: None for n in counts}
    for run in sorted(payloads):
        described = describe_run(run)
        if described is None or model_label(run) != model:
            continue
        if described[2] in chosen and chosen[described[2]] is None:
            chosen[described[2]] = run
    return chosen


def plot_confusion_grid(payloads, model: str, scope="window", normalize=True,
                        counts=COUNTS):
    """One model's confusion matrices side by side, n increasing to the right.

    Row-normalized by default, which puts each class's recall on the diagonal
    and puts every panel on the same 0-1 scale -- without it the three panels
    carry raw window counts that differ by embedding count, and the colour of a
    cell would say more about how many windows the count produced than about
    the model.
    """
    import matplotlib.pyplot as plt
    chosen = runs_by_count(payloads, model, counts)
    field = "window_test_metrics" if scope == "window" else "test_metrics"
    fig, axes = plt.subplots(1, len(counts), figsize=(7.5 * len(counts), 7.5))
    axes = np.atleast_1d(axes)
    image = None
    for index, (ax, n) in enumerate(zip(axes, counts)):
        run = chosen[n]
        if run is None:
            ax.text(.5, .5, f"n={n}\nno result yet", ha="center", va="center")
            ax.set_axis_off()
            continue
        payload = payloads[run]
        metrics = payload[field]
        cm = np.asarray(metrics["confusion_matrix"], dtype=float)
        classes = payload["classes"]
        if normalize:
            denominator = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, denominator, out=np.zeros_like(cm), where=denominator != 0)
        image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
        # The panels share one class order and one row-normalized scale, so the
        # true-class names belong to the row of panels rather than to any one of
        # them. Only the leftmost carries them; the rest keep the tick marks so
        # the grid still lines up, and give the width back to the matrices.
        ax.set_yticks(range(len(classes)), classes if index == 0 else [], fontsize=8)
        ax.set(xlabel="Predicted class", ylabel="True class" if index == 0 else "")
        ax.set_title(f"n={n} · macro F1 {metrics['macro_f1']:.3f} · "
                     f"balanced acc {metrics['balanced_accuracy']:.3f}", fontsize=10)
        threshold = 0.5 if normalize else (cm.max() / 2 if cm.size else 0)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j]):,}"
                ax.text(j, i, text, ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > threshold else "black")
    if image is not None:
        fig.colorbar(image, ax=axes.tolist(), fraction=.02)
    fig.suptitle(f"{model} · {scope} level"
                 + ("" if normalize else " · raw counts"), fontsize=13)
    return fig


def confusion_grid_selector(payloads, counts=COUNTS):
    """Architecture and ablation dropdowns over `plot_confusion_grid`.

    Two dropdowns rather than one list of run names: the architectures are the
    model types, and picking one narrows the second to that type's ablations,
    which is the grouping `linked_model_pickers` exists for. The embedding
    counts are not a dropdown at all -- they are the grid, which is the whole
    point of the panel.
    """
    import matplotlib.pyplot as plt
    pairs = sorted({(describe_run(run)[0], model_label(run))
                    for run in payloads if describe_run(run) is not None})
    if not pairs:
        print("No completed result JSONs with confusion matrices yet.")
        return
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print(f"ipywidgets unavailable; use plot_confusion_grid(payloads, {pairs[0][1]!r})")
        return
    architecture, model = linked_model_pickers(pairs)
    scope = widgets.ToggleButtons(options=["window", "cell"], description="Level:")
    normalize = widgets.Checkbox(value=True, description="Row normalize")

    def render(model, scope, normalize, architecture):
        fig = plot_confusion_grid(payloads, model, scope, normalize, counts)
        plt.show()
        # Closed after display: the dropdowns redraw on every change and each
        # figure is three full matrices, which crosses pyplot's 20-open-figure
        # warning and keeps every one of them alive for the kernel's lifetime.
        plt.close(fig)

    # `architecture` is passed in unused so that changing it re-renders: it
    # rewrites the model options, and when the new first ablation happens to
    # have the same label as the old one the model widget alone fires nothing.
    output = widgets.interactive_output(
        render, {"model": model, "scope": scope, "normalize": normalize,
                 "architecture": architecture})
    display(widgets.VBox([architecture, model, scope, normalize]), output)
