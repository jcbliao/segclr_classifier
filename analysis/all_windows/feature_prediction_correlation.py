"""Build per-window prediction/geometry caches for feature_prediction_correlation.ipynb.

Every cache is built from a run's held-out `fold_0` windows -- one 80/20 split of whole
cells at split_seed 0. There is no cross-validation in this pipeline, so a correlation
computed here is a single-fold estimate; label it accordingly.

Performance is scored as **F1**, never as raw correctness. The window population is
dominated by a few populous classes, so the share of windows a bin gets right rises and
falls with that bin's class mix rather than with the feature being plotted.

Every score is at the level the model is actually trained on -- `ACTIVE_HIERARCHY_TREE`
truncated by `HIERARCHY_LEVELS_DROPPED` and then pruned by `DROP_LABELS`, currently the
8-class level (pyramidal, thalamocortical, the three interneuron families, astrocyte,
oligo, microglia; OPC is dropped), NOT the granular Allen-style label the manifest
carries. Groups are therefore keyed by classes of that tree: a per-granular-label
breakdown would split L2IT from L5IT in the plot while the model emits one `pyramidal`
for both, and the F1 shown under each would be the same class's, computed on a different
slice.

A group is any class of the hierarchy, not only a level-2 one -- `neuron`,
`non_neuron`, `excitatory` and `inhibitory` are scored by pooling their descendants'
codes. That is exact rather than an approximation: the model emits a level-2 code, so
"did the prediction land inside this node" is a question the cached predictions already
answer, and no coarser model has to be trained to ask it.

Groups also go the other way, down to the manifest's **granular** labels -- `L4IT`,
`AltBasket`, `NMC`. Those change which windows are being scored and nothing else: the
model can never emit `AltBasket`, so an AltBasket group's truth comes from the manifest
while its predicted side is still `putative_cge`, and the F1 stays a level-2 F1. Read
`accuracy` on such a group as "how often this subtype is called correctly" and the
precision term as how specific that call is -- its false positives include the other
subtypes under the same level-2 class, which the model has no way to tell apart.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "all_windows"
CACHE = ROOT / "analysis" / "all_windows" / "feature_prediction_cache"
NB_ROOT = Path("/orcd/scratch/orcd/013/jcbliao/embedding_paths/r5um")
NEW_SKEL = Path("/orcd/scratch/orcd/013/jcbliao/skeletons/segclr/skeletons")
VOLUME = ROOT / "data" / "mask_volume_cache"
NUCLEI = ROOT / "data" / "nucleus_positions.json"
NM3_PER_UM3 = 1e9


def usable_runs() -> list[str]:
    """Every run a cache can be built from: it has best weights and a parseable name.

    The gate is `checkpoint_best.pt`, not the `<run>.json` a run writes when it
    finishes. Everything `build_cache` needs -- the embedding count, the window
    `pos_dim`, whether thickness was on -- comes from the run name and the
    checkpoint's own `ModelConfig`, so requiring the final summary would lock
    out every architecture whose sweep has not finished (all of `mpnn_complete`,
    and GT at n=40) for no reason. A still-training run's best checkpoint moves,
    so a cache built from one is a snapshot of that epoch.

    Names `describe_run` refuses -- no fold tag, or an architecture kept out of
    the comparison -- are dropped here rather than at each call site, so the
    list the notebook prints is the list that can actually be built.
    """
    return sorted(path.parent.name for path in RESULTS.glob("*/checkpoint_best.pt")
                  if count_of(path.parent.name) is not None)


def count_of(run_name: str) -> int | None:
    """This run's fixed embedding count, from the same parser everything else uses."""
    from analysis.all_windows.architecture_comparison import describe_run

    described = describe_run(run_name)
    return None if described is None else described[2]


def architecture_of(run_name: str) -> str | None:
    """This run's architecture label, or None if the name isn't a fixed-node run.

    Delegates to architecture_comparison.describe_run rather than matching the
    name again here. Two independent parsers of the same run names is how one
    notebook silently stops seeing an architecture the other one shows.
    """
    from analysis.all_windows.architecture_comparison import describe_run

    described = describe_run(run_name)
    return None if described is None else described[0]


def model_of(run_name: str) -> str | None:
    """This run's model label -- the architecture plus its feature variant.

    `architecture_of` collapses "MPNN", "MPNN + position", "MPNN + LPE" and
    "MPNN + position + LPE" into one name; this is what tells them apart, and
    what `select_runs(models=...)` matches on.

    A flat-softmax run is marked, because `describe_run` reads only the
    aggregation: `gnn_flat_scratch_mpnn_L2_position_lpe_..._n40` and its LCPN
    twin would otherwise carry the same label, and the flat one sorts first, so
    a selection asking for the MPNN variant would silently get the run that
    differs by classification objective.
    """
    from analysis.all_windows.architecture_comparison import describe_run, model_label

    return None if describe_run(run_name) is None else model_label(run_name)


def has_summary(run_name: str) -> bool:
    """True when this run's compact summary is on disk and the notebook can plot it."""
    return (CACHE / f"{run_name}.summary.json").is_file()


def cached_runs(runs=None, counts=(10, 20, 40)) -> list[str]:
    """Every run the notebook can actually plot, in ladder order.

    This is what a "grid of all models" wants: one panel per model that has a
    summary on disk, rather than `select_runs`'s one-per-architecture, which
    collapses four GraphTransformer ablations at two depths into a single `GT`
    panel and picks whichever sorts first. With the ablations named in the
    labels, a grid of eight GT panels is the comparison; a grid of one is a
    coin toss over which ablation got shown.

    Ordered by the architecture ladder, then by model label, then by count, so
    the panels read mean-first and sibling ablations sit together.
    """
    from analysis.all_windows.architecture_comparison import architecture_rank, is_flat

    runs = usable_runs() if runs is None else runs
    keep = [run for run in runs if has_summary(run) and count_of(run) in counts]
    return sorted(keep, key=lambda run: (is_flat(run),
                                         architecture_rank(architecture_of(run)),
                                         model_of(run) or run,
                                         count_of(run)))


def select_runs(runs, architectures=("Mean", "Pointwise MLP"), counts=(10, 20, 40),
                models=None, prefer_cached=True) -> list[str]:
    """One run per (architecture, embedding count), in that nesting order.

    Defaults to the mean-pool baseline and Pointwise MLP: the pair that differs by
    a learned per-node transform and nothing else, so a feature whose
    correlation with F1 moves between them moved because of phi.
    Pass every architecture for the whole ladder -- but note each
    run costs one GPU inference pass over every held-out window unless its
    predictions are already in the shared cache, so widening this is not free.

    `models` selects on the finer label instead -- "MPNN + LPE",
    "GT + position + LPE" -- so one feature variant can be pinned without
    writing out a run string. It replaces `architectures` when given, and is
    the way to reach a variant that the architecture-level default would
    resolve to some other run of the same architecture.

    Where several runs share the selected label at one count, a run whose
    summary is already built wins, and the first by name breaks the remaining
    tie; name a run explicitly to pin a particular one. `prefer_cached=False`
    goes back to name order alone, which is what to pass when the question is
    "which run *would* be selected" rather than "which can be plotted".

    That preference is load-bearing once a second GraphTransformer depth exists:
    `gt_L2_...` sorts ahead of `gt_L4_...`, so the moment a depth-2 run writes
    its first checkpoint it takes the whole `GT` slot -- and since it has no
    feature cache yet, the notebook's missing-cache filter then drops GT from
    the grid entirely instead of falling back to the depth-4 run that can
    actually be drawn. Preferring a cached run keeps an architecture on the page
    whenever anything of that architecture can be shown at all.
    """
    from analysis.all_windows.architecture_comparison import is_flat

    label_of, wanted = ((model_of, models) if models is not None
                        else (architecture_of, architectures))
    # Flat-softmax runs sort last, not by name: `architecture_of` reads only the
    # aggregation, so at n=40 the flat MPNN run would otherwise win the "MPNN"
    # slot purely because `gnn_flat_` precedes `gnn_lcpn_`. It stays selectable
    # by its own `models=` label, which says "flat softmax" on it.
    candidates = sorted(runs, key=lambda run: (is_flat(run),
                                               prefer_cached and not has_summary(run),
                                               run))
    selected = []
    for label in wanted:
        for n in counts:
            match = next(
                (
                    run
                    for run in candidates
                    if label_of(run) == label and count_of(run) == n
                ),
                None,
            )
            if match is not None:
                selected.append(match)
    return selected


def _reduce(values, offsets, op):
    out = np.full(len(offsets) - 1, np.nan, np.float32)
    for i, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        x = values[lo:hi]
        x = x[np.isfinite(x)]
        if len(x):
            out[i] = op(x)
    return out


def _soma_distances(data, nucleus_xyz):
    pos = data.pos.numpy().astype(np.float64)
    spatial = np.linalg.norm(pos - nucleus_xyz, axis=1).astype(np.float32)
    anchor = int(np.argmin(spatial))
    ei = data.edge_index.numpy()
    ew = data.edge_attr.numpy().reshape(-1)
    graph = coo_matrix((ew, (ei[0], ei[1])), shape=(len(pos), len(pos))).tocsr()
    path = dijkstra(graph, directed=False, indices=anchor).astype(np.float32)
    return spatial, path


def _new_skeleton_radius(root_id, positions):
    path = NEW_SKEL / f"{root_id}.npz"
    if not path.exists():
        return np.full(len(positions), np.nan, np.float32)
    with np.load(path) as z:
        vertices = z["vertices"].astype(np.float64)
        radius = z["radius"].astype(np.float32)
    _, nearest = cKDTree(vertices).query(positions, k=1)
    return radius[nearest]


def _volume(root_id, data):
    path = VOLUME / f"{root_id}.npz"
    if not path.exists():
        return np.full(data.num_nodes, np.nan, np.float32)
    with np.load(path) as z:
        if not np.array_equal(z["orig_node_ids"], data.orig_node_ids.numpy()):
            raise RuntimeError(f"volume/node-id mismatch for {root_id}")
        scale = int(z["voxel_volume_nm3"]) / NM3_PER_UM3
        return (z["voxel_count"] * scale).astype(np.float32)


@torch.no_grad()
def build_cache(run_name: str, batch_size=1024, num_workers=15, force=False) -> Path:
    """Run held-out inference and join per-window physical features once.

    Everything the dataset has to match the model on comes from the
    checkpoint's own `ModelConfig` -- the window `pos_dim` and whether the
    thickness channels were on -- and the embedding count comes from the run
    name. Nothing here reads `<run>.json`, which a run writes only when it
    finishes: an unfinished run is a snapshot, not an error, and requiring that
    file would have made every `mpnn_complete` run and GT at n=40 unreachable.
    """
    # Keep the GPU/model stack out of module import.  Plotting and HTML export
    # only read the compact summaries and should work in a lightweight Python
    # environment (including the GitHub Pages publisher).
    from torch_geometric.loader import DataLoader
    from data.dataset_lcpn import load_hierarchy, load_manifest
    from data.dataset_windowed import WindowedGraphDatasetLCPN
    from data.window_prediction_cache import (
        DEFAULT_CACHE_DIR, load_prediction_cache, save_prediction_cache,
    )
    from gnn.model import WindowClassifier

    CACHE.mkdir(exist_ok=True)
    dest = CACHE / f"{run_name}.npz"
    summary = CACHE / f"{run_name}.summary.json"
    if dest.exists() and not force:
        if not summary.exists():
            summarize_cache(run_name)
        return dest
    n = count_of(run_name)
    if n is None:
        raise ValueError(
            f"{run_name} is not a fixed-node run name with a fold tag "
            "(expected ..._n{10,20,40}..._fold{seed})"
        )
    checkpoint = torch.load(RESULTS / run_name / "checkpoint_best.pt",
                            map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    manifest = load_manifest()
    hierarchy = load_hierarchy(manifest)
    # use_thickness is read off the checkpoint rather than left at its default:
    # a model trained with the thickness channels expects them at inference, and
    # a dataset built without them would hand it a batch with no `thickness`
    # attribute at all.
    ds = WindowedGraphDatasetLCPN(manifest, "test", pos_dim=config.gt_pos_dim,
                                  use_thickness=config.gt_use_thickness,
                                  num_embeddings=n, neighborhood_root=NB_ROOT)
    try:
        shared = load_prediction_cache(run_name, DEFAULT_CACHE_DIR, required_splits=("test",))
        keep = shared["split"].astype(str) == "test"
        if not np.array_equal(shared["root_id"][keep].astype(np.int64), ds.index_root_ids):
            raise ValueError("test root ordering differs from current dataset")
        if not np.array_equal(shared["center_index"][keep].astype(np.int64), ds.index_centers):
            raise ValueError("test center ordering differs from current dataset")
        predictions = shared["prediction"][keep].astype(np.int16)
        print(f"using shared predictions from {DEFAULT_CACHE_DIR}", flush=True)
    except (FileNotFoundError, ValueError):
        model = WindowClassifier(config, hierarchy)
        model.load_state_dict(checkpoint["model_state"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            persistent_workers=num_workers > 0)
        predictions = []
        for batch in tqdm(loader, desc=f"infer {run_name}"):
            batch = batch.to(device)
            g = model(batch.x, batch.edge_index, batch.batch, batch.pos_enc, batch.rel_pos,
                      getattr(batch, "thickness", None))
            pred = model.cls_head.predict_top_down(g)[:, -1]
            predictions.append(pred.cpu().numpy())
        predictions = np.concatenate(predictions).astype(np.int16)
        xyz = np.empty((len(ds), 3), np.float32)
        for rid in np.unique(ds.index_root_ids):
            rows = np.flatnonzero(ds.index_root_ids == rid)
            xyz[rows] = ds.cell_data[int(rid)].pos[ds.index_centers[rows]].numpy()
        save_prediction_cache(run_name, {
            "split": np.full(len(ds), "test", dtype="U5"),
            "root_id": ds.index_root_ids.astype(np.uint64),
            "center_index": ds.index_centers.astype(np.int32),
            "center_xyz": xyz, "prediction": predictions,
            "target": ds.index_labels.astype(np.int16),
            "num_embeddings": np.array([n], np.int16),
        }, DEFAULT_CACHE_DIR)

    # Both branches keep the class codes, not just the boolean: an F1 needs the
    # predicted class of a wrong window, which `correct` has already discarded.
    targets = ds.index_labels.astype(np.int16)
    correct = predictions == targets

    nuclei = {int(k): np.asarray(v, np.float64) for k, v in json.loads(NUCLEI.read_text())["positions"].items()}
    columns = {k: [] for k in (
        "root_id", "cell_type", "geodesic_radius_um", "path_distance_um",
        "node_density_per_um", "spatial_radius_um", "spatial_density_per_um3",
        "soma_path_um", "soma_spatial_um", "volume_sum_um3", "volume_mean_um3",
        "volume_median_um3", "radius_mean_nm", "radius_median_nm", "radius_min_nm",
        "radius_max_nm")}
    for root_id, data in tqdm(ds.cell_data.items(), desc=f"features n={n}", unit="cell"):
        with np.load(NB_ROOT / "neighborhoods" / f"n{n}" / f"{root_id}.npz") as z:
            offsets = z["offsets"].astype(np.int64)
            restricted_members = z["members"].astype(np.int64)
            cache_index = z["cache_index"].astype(np.int64)
            cable = z["cable_nm"].astype(np.float32)
            geodesic_radius = z["radius_nm"].astype(np.float32)
        members = cache_index[restricted_members]
        centers = cache_index
        pos = data.pos.numpy().astype(np.float64)
        repeated_centers = np.repeat(pos[centers], np.diff(offsets), axis=0)
        euclid = np.linalg.norm(pos[members] - repeated_centers, axis=1)
        spatial_radius = _reduce(euclid, offsets, np.max)
        volumes = _volume(root_id, data)[members]
        new_radius = _new_skeleton_radius(root_id, pos)[members]
        if root_id in nuclei:
            soma_spatial, soma_path = _soma_distances(data, nuclei[root_id])
            soma_spatial, soma_path = soma_spatial[centers], soma_path[centers]
        else:
            soma_spatial = soma_path = np.full(len(centers), np.nan, np.float32)
        cell_type = manifest["cells"][str(root_id)]["cell_type"]
        columns["root_id"].append(np.full(len(centers), root_id, np.int64))
        columns["cell_type"].append(np.full(len(centers), cell_type, dtype=f"U{max(16, len(cell_type))}"))
        columns["geodesic_radius_um"].append(geodesic_radius / 1000)
        columns["path_distance_um"].append(cable / 1000)
        columns["node_density_per_um"].append(n / np.maximum(cable / 1000, 1e-6))
        columns["spatial_radius_um"].append(spatial_radius / 1000)
        columns["spatial_density_per_um3"].append(n / np.maximum(4/3*np.pi*(spatial_radius/1000)**3, 1e-9))
        columns["soma_path_um"].append(soma_path / 1000)
        columns["soma_spatial_um"].append(soma_spatial / 1000)
        columns["volume_sum_um3"].append(_reduce(volumes, offsets, np.sum))
        columns["volume_mean_um3"].append(_reduce(volumes, offsets, np.mean))
        columns["volume_median_um3"].append(_reduce(volumes, offsets, np.median))
        columns["radius_mean_nm"].append(_reduce(new_radius, offsets, np.mean))
        columns["radius_median_nm"].append(_reduce(new_radius, offsets, np.median))
        columns["radius_min_nm"].append(_reduce(new_radius, offsets, np.min))
        columns["radius_max_nm"].append(_reduce(new_radius, offsets, np.max))
    arrays = {k: np.concatenate(v) for k, v in columns.items()}
    if len(correct) != len(arrays["root_id"]):
        raise RuntimeError(f"prediction/feature row mismatch: {len(correct)} != {len(arrays['root_id'])}")
    np.savez_compressed(dest, correct=correct, prediction=predictions, target=targets,
                        n_embeddings=np.array([n]), **arrays)
    summarize_cache(run_name)
    return dest


def _classes_from_shared_cache(run_name: str, root_id, correct) -> dict:
    """Per-window class codes for a cache built before they were stored.

    The npz predates the F1 rewrite and holds only the boolean. The shared
    prediction cache holds the codes for the same run and the same held-out
    rows in the same order, so the codes are recoverable without re-running
    inference -- but only after the alignment is checked, which is what the two
    comparisons below do.
    """
    from data.window_prediction_cache import DEFAULT_CACHE_DIR, load_prediction_cache

    shared = load_prediction_cache(run_name, DEFAULT_CACHE_DIR, required_splits=("test",))
    keep = shared["split"].astype(str) == "test"
    prediction = shared["prediction"][keep].astype(np.int16)
    target = shared["target"][keep].astype(np.int16)
    if not np.array_equal(shared["root_id"][keep].astype(np.int64), np.asarray(root_id)):
        raise ValueError(f"{run_name}: shared prediction cache is not row-aligned to the feature cache")
    if not np.array_equal(prediction == target, np.asarray(correct)):
        raise ValueError(f"{run_name}: shared predictions disagree with the cached correctness")
    return {"prediction": prediction, "target": target}


def load_cache(run_name: str):
    import pandas as pd
    with np.load(CACHE / f"{run_name}.npz") as z:
        columns = {k: z[k] for k in z.files if k != "n_embeddings"}
    if "prediction" not in columns:
        columns.update(_classes_from_shared_cache(run_name, columns["root_id"], columns["correct"]))
    return pd.DataFrame(columns)


FEATURES = [
    "geodesic_radius_um", "path_distance_um", "node_density_per_um",
    "spatial_radius_um", "spatial_density_per_um3", "soma_path_um",
    "soma_spatial_um", "volume_sum_um3", "volume_mean_um3",
    "volume_median_um3", "radius_mean_nm", "radius_median_nm",
    "radius_min_nm", "radius_max_nm",
]
#: Where the center node sits on the cell, as opposed to what its neighborhood
#: looks like. Kept named because the notebook plots the two together and the
#: text calls them out as the pair that measures the same thing two ways.
SOMA_FEATURES = ("soma_path_um", "soma_spatial_um")
#: Every feature a per-class breakdown is computed for. All of them: a class
#: dropdown that covers only the soma pair cannot answer whether a class's F1
#: rises with cable, occupancy or clearance, which is the question each of the
#: notebook's panels asks. The cost is bounded -- one `pd.cut` of the held-out
#: population per (class, feature), against a table already in memory.
CLASS_FEATURES = tuple(FEATURES)


def hierarchy_class_groups(classes) -> list[dict]:
    """The BFS class list every panel's selector offers, from one shared parser.

    Delegates to `architecture_comparison`, which both notebooks read: two
    independent walks of the same tree is how one page silently starts scoring
    a different set of classes than the other.
    """
    from analysis.all_windows.architecture_comparison import hierarchy_class_groups as groups

    return groups(classes)


def class_names() -> list[str]:
    """Class name per target code, at the level the model is trained on.

    `load_hierarchy` already applies `HIERARCHY_LEVELS_DROPPED`, so this is the
    truncated finest level -- the same list `train_gnn.py` reports against, and
    the same codes `y_levels[:, -1]` and `predict_top_down(...)[:, -1]` carry.
    The manifest's granular `cell_type` is a finer label than any of these and
    is never what a prediction is compared to.
    """
    from data.dataset_lcpn import load_hierarchy, load_manifest
    return list(load_hierarchy(load_manifest()).level_classes[-1])


def _macro_scores(target, prediction) -> dict:
    """Macro precision/recall/F1 over the classes present in `target`.

    Macro recall is balanced accuracy by definition, so no separate column for
    it. Raw accuracy rides along because it is what the class imbalance
    distorts -- the point of showing both is the gap between them.
    """
    labels = np.unique(target)
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, prediction, labels=labels, average="macro", zero_division=0)
    return {"f1": float(f1), "precision": float(precision), "recall": float(recall),
            "accuracy": float(np.mean(target == prediction)), "n_classes": int(len(labels))}


def _group_scores(target, prediction, codes, own=None) -> dict:
    """The level-2 cell-type prediction, scored on one group of windows.

    `codes` are the level-2 class codes a prediction has to land in to be right.
    There is only ever one thing being measured here -- the level-2 prediction
    -- and the group only decides which windows it is measured on.

    Two shapes, because the two kinds of group differ in whether a false
    positive can exist at all:

    **A hierarchy class** (`pyramidal`, `excitatory`, `neuron`) is something the
    model emits, so it is scored against the whole held-out population: truth is
    `target in codes`, and the false positives are windows genuinely outside the
    class that were called it. That is the ordinary per-class F1, and it is the
    same quantity `window_macro_f1` averages.

    **A granular label** (`L5ET`, `AltBasket`) is not something the model can
    emit, and `own` marks its windows. Scoring happens inside that set, where
    the correct answer for every window is the same level-2 class -- so no
    window there can be a false positive, precision is 1 wherever the class is
    predicted at all, and the F1 is the harmonic mean of that with the share
    called correctly. Nothing about a sibling subtype enters it.

    An earlier version scored a granular group against the whole population,
    which put every correctly-classified sibling into its false-positive count:
    L5ET came out at F1 0.067 while being called correctly 80.5% of the time,
    because the number had become L5ET's share of the pyramidal population
    rather than anything about the model. Do not reintroduce that.
    """
    codes = np.asarray(codes)
    if own is not None:
        own = np.asarray(own)
        target, prediction = target[own], prediction[own]
    truth = np.isin(target, codes)
    predicted = np.isin(prediction, codes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average="binary", zero_division=0)
    return {"f1": float(f1), "precision": float(precision), "recall": float(recall),
            "accuracy": float(np.mean(predicted[truth])) if truth.any() else float("nan"),
            "n_classes": int(len(codes))}


#: Cache for `scoring_groups`, keyed by the level-2 class tuple. It reads the
#: manifest to get the granular-to-level-2 map, and a selector asks for it once
#: per redraw.
_SCORING_CACHE: dict[tuple[str, ...], list[dict]] = {}


def scoring_groups(names) -> list[dict]:
    """Every group a per-class panel can be keyed by, coarse to fine.

    Two kinds, in one list so the summary builder and the notebook agree on the
    order a selector shows:

    - **hierarchy** groups, breadth-first from the root (`neuron`, `non_neuron`,
      `excitatory`, `inhibitory`, then the level-2 classes the head emits). Its
      `codes` are the level-2 codes descending from the node.
    - **granular** groups, one per manifest label (`L4IT`, `AltBasket`, ...),
      whose `codes` are the single level-2 code that label maps to. These are
      finer than anything the model predicts, which is the point: the scoring
      stays at level 2 and only the population being scored gets finer.

    `granular` marks which kind a row is, and `coarse_class` names the level-2
    class a granular group is scored through, so a bar chart can group its
    subtypes under the class the model actually emits.
    """
    from data.dataset_lcpn import load_hierarchy, load_manifest

    key = tuple(str(name) for name in names)
    if key in _SCORING_CACHE:
        return _SCORING_CACHE[key]
    hierarchy = load_hierarchy(load_manifest())
    finest = hierarchy.depth - 1
    code_of = {str(name): index for index, name in enumerate(names)}
    groups = []
    coarse_rank = {}
    for group in hierarchy_class_groups(names):
        codes = tuple(code_of[leaf] for leaf in group["leaves"] if leaf in code_of)
        if codes:
            coarse_rank[group["name"]] = len(coarse_rank)
            groups.append({"name": group["name"], "codes": codes, "granular": False,
                           "coarse_class": None, "level": group["level"]})
    # Granular labels follow their own level-2 parent's position, not the
    # alphabet: sorted by name alone, `AltBasket` (cge) would land between
    # `astrocyte` (glia) and `L2IT` (pyramidal), and a selector or a bar chart
    # built on that order puts unrelated subtypes side by side.
    granular = sorted(
        ((coarse_rank.get(path[finest], len(coarse_rank)), label, path[finest])
         for label, path in hierarchy.label_paths.items() if path[finest] in code_of))
    for _, label, coarse in granular:
        groups.append({"name": label, "codes": (code_of[coarse],), "granular": True,
                       "coarse_class": coarse, "level": finest + 1})
    return _SCORING_CACHE.setdefault(key, groups)


#: The quantiles marked with a drop-line under each binned curve. They are read
#: off the group's own feature distribution, so "the 75th percentile" means
#: "75% of this group's windows sit to the left", not a position on the axis.
MARKED_QUANTILES = (25, 50, 75)


def summarize_cache(run_name: str, bins=10) -> Path:
    """Reduce the multi-million-row cache to login-node-safe plot tables.

    Five tables, all scored by F1 at the trained-on level-2 hierarchy (see
    `class_names`), including the granular groups -- the model emits a level-2
    code and can never say `L4IT`, so a granular group changes which windows are
    being scored, never what the prediction is compared against:

    ``overall``  one row per group -- macro F1 over the classes present for the
                 pooled run, and that group's own F1 otherwise.
    ``bins``     the same scores inside each of `bins` quantile bins of a
                 feature, which is what the notebook plots.
    ``quantiles``  each group's own p25/p50/p75 of every feature, which the
                 notebook drops as dotted lines from the curve to the axis.
                 Read off the raw windows, not off the bin edges: ten quantile
                 bins put edges at the deciles, and p25 is not one of them.
    ``correlations``  Spearman rho between a bin's median feature value and its
                 F1, over those bins. An F1 is a property of a group of windows,
                 not of one window, so this is a rank correlation over ~10
                 points rather than over millions -- read the bin curve, and
                 read rho as a summary of its monotonicity, not as a test with
                 the old per-window sample size behind it.
    ``cells``    per held-out cell, its own class's F1 and the recall behind it.

    A group's false positives are counted against the whole held-out population
    in the same feature range, not against the group: every window in the group
    carries a label inside the group, so precision computed inside it would be 1
    by construction. See `_group_scores`.

    Every feature gets the per-class breakdown (`CLASS_FEATURES`), not just the
    two soma ones -- a class dropdown that reached only distance-from-soma could
    not answer whether a class's F1 rises with cable, occupancy or clearance,
    which is what each of the notebook's panels asks.
    """
    import pandas as pd
    frame = load_cache(run_name)
    n_embeddings = count_of(run_name)
    if n_embeddings is None:
        raise ValueError(
            f"{run_name} is not a fixed-node run name with a fold tag "
            "(expected ..._n{10,20,40}..._fold{seed})"
        )
    names = np.asarray(class_names())
    if frame.target.max() >= len(names):
        raise ValueError(f"{run_name}: target code {int(frame.target.max())} outside "
                         f"the {len(names)}-class active hierarchy")
    frame["class_name"] = names[frame.target.to_numpy()]
    # The granular label as an integer, so the per-feature population frames
    # below carry one int16 column instead of a few million fixed-width strings.
    cell_types, cell_code = np.unique(frame.cell_type.to_numpy(), return_inverse=True)
    frame["cell_code"] = cell_code.astype(np.int16)
    granular_code = {str(name): index for index, name in enumerate(cell_types)}
    overall, correlations, binned, quantiles = [], [], [], []
    stamp = {"run": run_name, "n_embeddings": n_embeddings}
    # The population a group's false positives are counted against, one dropna'd
    # view per feature. Built once and reused by every group: the frame does not
    # depend on the group, only the mask the scoring uses does, so building it
    # inside the group loop would repeat the same reduction of several million
    # rows once per group per feature.
    populations = {feature: frame[[feature, "target", "prediction", "cell_code"]]
                   .replace([np.inf, -np.inf], np.nan).dropna()
                   for feature in CLASS_FEATURES}

    def truth_in(scope, group):
        """Which of `scope`'s rows belong to `group`."""
        if group["granular"]:
            return scope.cell_code.to_numpy() == granular_code[group["name"]]
        return np.isin(scope.target.to_numpy(), group["codes"])

    def add_group(part, group=None):
        label = None if group is None else group["name"]
        coarse = None if group is None else group["coarse_class"]
        # `own` only for a granular group: it moves `accuracy` onto that
        # subtype's windows while F1/precision/recall stay the level-2 class's.
        own = (truth_in(frame, group)
               if group is not None and group["granular"] else None)
        scores = (_macro_scores(part.target.to_numpy(), part.prediction.to_numpy())
                  if group is None else
                  _group_scores(frame.target.to_numpy(), frame.prediction.to_numpy(),
                                group["codes"], own))
        overall.append({**stamp, "class_name": label, "coarse_class": coarse,
                        "granular": bool(group and group["granular"]),
                        "n": len(part), **scores})
        for feature in FEATURES:
            if group is not None and feature not in CLASS_FEATURES:
                continue
            x = part[[feature, "target", "prediction"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(x):
                marks = np.percentile(x[feature].to_numpy(), MARKED_QUANTILES)
                quantiles.append({**stamp, "class_name": label, "feature": feature,
                                  "n": len(x),
                                  **{f"p{q}": float(v)
                                     for q, v in zip(MARKED_QUANTILES, marks)}})
            rows = []
            if len(x) >= bins and x[feature].nunique() >= 2:
                # Bin edges always come from the group's own quantiles. For a
                # group the scoring then moves to the population under those
                # edges, so a false positive is counted in the same feature
                # range the group's own windows were binned by; population
                # windows outside that range fall in no bin.
                assigned, edges = pd.qcut(x[feature], bins, duplicates="drop",
                                          labels=False, retbins=True)
                scope = x if group is None else populations[feature]
                assigned = (assigned.to_numpy() if group is None else
                            pd.cut(scope[feature], edges, labels=False,
                                   include_lowest=True).to_numpy())
                values = scope[feature].to_numpy()
                target = scope.target.to_numpy()
                prediction = scope.prediction.to_numpy()
                # Which of the scope's rows are the group's own. For a
                # hierarchy class that is `target in codes`; for a granular one
                # it is the manifest label, and only `accuracy` uses it.
                belongs = None if group is None else truth_in(scope, group)
                for b in range(len(edges) - 1):
                    selected = assigned == b
                    own = selected if group is None else (selected & belongs)
                    if not own.any():
                        continue
                    scored = (_macro_scores(target[selected], prediction[selected])
                              if group is None else
                              _group_scores(target[selected], prediction[selected],
                                            group["codes"],
                                            own[selected] if group["granular"] else None))
                    rows.append({**stamp, "class_name": label, "coarse_class": coarse,
                                 "feature": feature, "bin": len(rows),
                                 "feature_median": float(np.median(values[own])),
                                 "n": int(own.sum()), **scored})
            binned.extend(rows)
            rho, p = spearmanr([r["feature_median"] for r in rows],
                               [r["f1"] for r in rows]) if len(rows) > 2 else (np.nan, np.nan)
            correlations.append({**stamp, "class_name": label, "coarse_class": coarse,
                                 "feature": feature, "spearman_rho": float(rho),
                                 "p": float(p), "n_bins": len(rows), "n_windows": len(x)})

    add_group(frame)
    # Every hierarchy class breadth-first, then every granular label. A coarse
    # node is scored by pooling its descendants' codes and a granular one by
    # narrowing which windows count as its own; both compare against the same
    # level-2 prediction, which is the only thing the model emits.
    for group in scoring_groups(names):
        part = (frame[frame.cell_type == group["name"]] if group["granular"]
                else frame[frame.target.isin(group["codes"])])
        if part.empty:
            continue
        add_group(part, group)
    # Per cell: F1 for "predicted this cell's own class", over this cell's own
    # windows. Every window of one cell carries that cell's single label, so
    # inside the cell there is no window that could be a false positive -- FP is
    # 0 by construction, precision is 1 whenever the class is predicted at all,
    # and F1 = 2PR/(P+R) collapses to 2r/(1+r), or 0 where the class is never
    # predicted. That is the closed form of the count-based F1, not a stand-in
    # for it, so it is written out rather than re-derived per cell. Recall is
    # kept beside it: F1 is monotone in recall here, so the two rank cells
    # identically and only the axis values differ.
    # The granular `cell_type` rides along here and only here -- it is finer
    # than anything the model predicts, so it groups cells rather than scoring.
    cells = (frame.assign(hit=frame.prediction == frame.target)
             .groupby(["root_id", "cell_type", "class_name"], as_index=False)
             .agg(recall=("hit", "mean"), n_windows=("hit", "size")))
    cells["f1"] = np.where(cells.recall > 0, 2 * cells.recall / (1 + cells.recall), 0.0)
    cell_rows = [{**stamp, **row} for row in cells.to_dict(orient="records")]
    payload = {"run": run_name, "n_embeddings": n_embeddings, "classes": names.tolist(),
               "overall": overall, "correlations": correlations, "bins": binned,
               "quantiles": quantiles, "cells": cell_rows}
    out = CACHE / f"{run_name}.summary.json"
    out.write_text(json.dumps(payload))
    return out


def load_summary(run_name: str) -> dict:
    return json.loads((CACHE / f"{run_name}.summary.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_names", nargs="*", help="completed result run names; default: all")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=15)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true",
                        help="rebuild the summary tables from an existing feature cache; "
                             "no inference and no feature join")
    args = parser.parse_args()
    runs = args.run_names
    if not runs:
        # `--summarize-only` reduces a feature cache that is already on disk, so
        # its default is the runs that have one -- not every run with a
        # checkpoint, which would fail on the first that has never been built.
        # Both lists go through `usable_runs`, so a cache left behind by a run
        # whose results directory is gone (an architecture since removed) is
        # skipped rather than raising on its unparseable name.
        runs = [run for run in usable_runs()
                if not args.summarize_only or (CACHE / f"{run}.npz").exists()]
    if not runs:
        raise SystemExit("no fixed-node runs with a best checkpoint found")
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run}", flush=True)
        if args.summarize_only:
            print(summarize_cache(run), flush=True)
            continue
        print(build_cache(run, batch_size=args.batch_size, num_workers=args.num_workers,
                          force=args.force), flush=True)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# Notebook figures. Everything below reads only the compact summary tables --
# no window ever reaches the notebook, which is what keeps these safe to draw
# on a login node.
# --------------------------------------------------------------------------

#: One colour per embedding count, shared by every panel that puts the three on
#: the same axes. Keyed by count rather than zipped against a list, so a panel
#: missing n=20 still draws n=40 in n=40's colour.
COUNT_COLORS = {10: "#4C78A8", 20: "#F58518", 40: "#54A24B"}
#: One colour per statistic within a feature family (clearance radius, volume,
#: distance-from-soma), which are the panels that put several variants of the
#: same measurement on one axis instead of several counts.
VARIANT_COLORS = ("#4C78A8", "#E45756", "#54A24B", "#B279A2", "#F58518")

#: The feature families the notebook plots, and the variants inside each. A
#: family is drawn as one panel per embedding count with its variants coloured
#: against each other; a bare feature is drawn as one panel per model with the
#: counts coloured against each other.
FEATURE_FAMILIES = {
    # Mean and median only. The minimum and the maximum over k members are
    # order statistics of a small sample, so they move with k itself rather than
    # with the neighborhood's clearance, and the summed volume grows with k by
    # construction -- all three are still cached, they are just not what these
    # panels ask about.
    "clearance radius (nm)": ["radius_mean_nm", "radius_median_nm"],
    "embedding-box volume (um^3)": ["volume_mean_um3", "volume_median_um3"],
    # The soma pair is not two reductions of one measurement but two ways of
    # measuring the same distance -- along the cable and through space. It
    # shares the family layout because the question is the same: do these agree.
    "distance from soma (um)": ["soma_path_um", "soma_spatial_um"],
}


def load_frames(run_names) -> dict:
    """The four summary tables for `run_names`, labelled by architecture and model.

    Every table below is keyed by run, and two runs of the same architecture at
    the same embedding count are otherwise told apart only by reading the run
    string -- so the finer `model` label is attached here, once, rather than at
    each plotting call site.
    """
    import pandas as pd

    summaries = [load_summary(run) for run in run_names]

    def framed(key):
        # `.get(key, [])`, so a summary written before a table existed loads as
        # an empty frame rather than raising and taking every other run with it.
        frame = pd.DataFrame([row for summary in summaries for row in summary.get(key, [])])
        if len(frame):
            frame.insert(1, "architecture", frame.run.map(architecture_of))
            frame.insert(2, "model", frame.run.map(model_of))
        return frame

    keys = ("overall", "correlations", "bins", "quantiles", "cells")
    frames = {key: framed(key) for key in keys}
    # The finest-level class list, for ordering a class selector by the tree.
    # Every selected run is trained against the same hierarchy, so the first
    # summary's list is every summary's list.
    frames["classes"] = summaries[0]["classes"] if summaries else []
    return frames


def model_order(frame) -> list[str]:
    """Model labels in ladder order -- Mean first, GT last, variants alphabetical.

    `sorted()` would put "GT" before "Mean" before "MPNN", which reads as an
    arbitrary list rather than as the ladder of how much learned mixing happens
    before the readout that the architectures actually form.
    """
    from analysis.all_windows.architecture_comparison import architecture_rank

    pairs = frame[["architecture", "model"]].drop_duplicates()
    return [model for _, model in sorted(
        (architecture_rank(architecture), model)
        for architecture, model in pairs.itertuples(index=False))]


def class_options(frame, classes=None) -> list[str]:
    """Every class a per-class breakdown exists for, coarse to fine.

    `classes` is a run's level-2 class list (`summary['classes']`), which is
    what puts the answer in tree order -- neuron / non_neuron, then excitatory /
    inhibitory, then the eight classes the head emits, then the manifest's
    granular labels under the level-2 class each maps to. Without it the names
    come back alphabetically, which reads as an arbitrary list rather than a
    tree.
    """
    present = set(frame.class_name.dropna().unique())
    if classes is None:
        return sorted(present)
    return [group["name"] for group in scoring_groups(classes)
            if group["name"] in present]


def _select(bins, class_name):
    """The pooled rows when `class_name` is None, that class's rows otherwise."""
    return (bins[bins.class_name.isna()] if class_name is None
            else bins[bins.class_name == class_name])


def _score_label(class_name) -> str:
    return "macro F1" if class_name is None else f"F1 · {class_name} vs. rest"


#: Quantiles marked under a binned curve, matching what `summarize_cache`
#: stores. Read off the group's own feature distribution, so p75 means "75% of
#: this group's windows lie to the left", not a position on the drawn axis.
MARKED_LABELS = ("p25", "p50", "p75")


def _drop_lines(ax, marks):
    """Dotted verticals from each marked quantile on a curve down to the axis.

    `marks` is a list of `(x, y, color)`. Drawn after every curve is on the axes
    so the bottom of each line is the panel's settled y limit -- doing it inside
    the plotting loop would anchor early lines to an axis that later curves then
    rescale, leaving them floating above the frame.

    The curve is interpolated at the quantile rather than snapped to the nearest
    bin: the bins are deciles, so p25 falls between two of them and the nearest
    one is up to half a bin away.
    """
    if not marks:
        return
    bottom = ax.get_ylim()[0]
    for x, y, color in marks:
        ax.vlines(x, bottom, y, colors=color, linestyles=":", linewidth=1, alpha=.8)
        ax.plot([x], [y], marker="|", ms=7, color=color)
    ax.set_ylim(bottom=bottom)


def _quantile_marks(quantiles, curve, color, keys=MARKED_LABELS):
    """`(x, y, color)` for each marked quantile that falls inside `curve`.

    `curve` is the sorted bin table for one series; `quantiles` is its matching
    one-row frame. A quantile outside the drawn x range is dropped rather than
    clamped to the end of the curve, which would put a line under a value the
    panel never plotted.
    """
    if quantiles is None or not len(quantiles) or len(curve) < 2:
        return []
    x = curve.feature_median.to_numpy(dtype=float)
    y = curve.f1.to_numpy(dtype=float)
    row = quantiles.iloc[0]
    marks = []
    for key in keys:
        value = float(row[key])
        if np.isfinite(value) and x[0] <= value <= x[-1]:
            marks.append((value, float(np.interp(value, x, y)), color))
    return marks


def _quantile_row(quantiles, **match):
    """The one quantile row matching every keyword, or None."""
    if quantiles is None or not len(quantiles):
        return None
    part = quantiles
    for column, value in match.items():
        part = part[part[column].isna() if value is None else part[column] == value]
    return part if len(part) else None


def plot_feature_by_model(bins, feature, class_name=None, counts=(10, 20, 40),
                          models=None, ncols=3, quantiles=None):
    """One panel per model, one line per embedding count, F1 against `feature`.

    The panel grid is the comparison across aggregation methods and the colours
    are the comparison across neighborhood size, so a feature that stops mattering
    once the neighborhood is large shows up as three curves converging inside one
    panel, and a feature only one architecture is sensitive to shows up as one
    panel disagreeing with its neighbours.

    Every panel shares one y axis: the point is whether a curve sits higher than
    its neighbour, which per-panel autoscaling would hide by stretching a flat
    curve to fill its own panel.
    """
    import matplotlib.pyplot as plt

    # The grid is laid out from every model in `bins`, not from the models that
    # survive this feature and class: a panel that comes out empty says the
    # model has no bins here, whereas dropping it reshuffles every other panel
    # and makes two selections impossible to compare side by side.
    models = models if models is not None else model_order(bins)
    part = _select(bins, class_name)
    part = part[part.feature == feature]
    if not models:
        print(f"no binned rows for {feature}"
              + ("" if class_name is None else f" / {class_name}"))
        return None
    nrows = -(-len(models) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                            sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[len(models):]:
        ax.set_axis_off()
    for ax, model in zip(axes.flat, models):
        marks = []
        for n in counts:
            curve = part[(part.model == model) & (part.n_embeddings == n)].sort_values("bin")
            if curve.empty:
                continue
            color = COUNT_COLORS.get(n)
            ax.plot(curve.feature_median, curve.f1, marker="o", ms=4,
                    color=color, label=f"n={n}")
            marks += _quantile_marks(
                _quantile_row(quantiles, model=model, n_embeddings=n,
                              feature=feature, class_name=class_name), curve, color)
        # After the loop, so every line is anchored to the settled y limit.
        _drop_lines(ax, marks)
        ax.set_title(model, fontsize=10)
        ax.set(xlabel=feature.replace("_", " "), ylabel=_score_label(class_name))
        ax.grid(alpha=.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    subtitle = ("" if quantiles is None else
                " · dotted verticals at p25/p50/p75 of the feature")
    fig.suptitle(f"{_score_label(class_name)} against {feature.replace('_', ' ')}"
                 f"{subtitle} · fold_0", fontsize=13)
    fig.tight_layout()
    return fig


def plot_family_by_count(bins, family, model, class_name=None, counts=(10, 20, 40),
                         quantiles=None):
    """One model's F1 against every variant of one measurement, n across the panels.

    The variants of a family (mean / median / min / max of a per-node quantity,
    or the two ways of measuring distance from the soma) share a panel because
    they are the same measurement reduced differently -- what is being read off
    is which reduction the model tracks, and whether that answer survives a
    larger neighborhood, which is what moving right across the panels shows.

    They share an x axis only within a panel, not across families: a radius in
    nm and a volume in um^3 are not on one scale.
    """
    import matplotlib.pyplot as plt

    features = FEATURE_FAMILIES[family]
    part = _select(bins, class_name)
    part = part[(part.model == model) & part.feature.isin(features)]
    if part.empty:
        print(f"no binned rows for {model} / {family}"
              + ("" if class_name is None else f" / {class_name}"))
        return None
    fig, axes = plt.subplots(1, len(counts), figsize=(5.5 * len(counts), 4.2),
                            sharey=True, squeeze=False)
    axes = axes[0]
    for ax, n in zip(axes, counts):
        panel = part[part.n_embeddings == n]
        if panel.empty:
            ax.text(.5, .5, f"n={n}\nno cache", ha="center", va="center")
            ax.set_axis_off()
            continue
        marks = []
        for color, feature in zip(VARIANT_COLORS, features):
            curve = panel[panel.feature == feature].sort_values("bin")
            if curve.empty:
                continue
            ax.plot(curve.feature_median, curve.f1, marker="o", ms=4,
                    color=color, label=feature.replace("_", " "))
            marks += _quantile_marks(
                _quantile_row(quantiles, model=model, n_embeddings=n,
                              feature=feature, class_name=class_name), curve, color)
        # After the loop, so every line is anchored to the settled y limit.
        _drop_lines(ax, marks)
        ax.set_title(f"n={n}", fontsize=10)
        ax.set(xlabel=family, ylabel=_score_label(class_name))
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    subtitle = ("" if quantiles is None else
                " · dotted verticals at p25/p50/p75")
    fig.suptitle(f"{model} · {_score_label(class_name)} against {family}"
                 f"{subtitle} · fold_0", fontsize=13)
    fig.tight_layout()
    return fig


def _class_dropdown(bins, classes=None, description="Class:"):
    import ipywidgets as widgets

    # (label, value): None is the pooled selection and has to be a real option,
    # since every plotting function switches on it rather than on a name. The
    # names are indented by their depth in the tree, so the flat list of options
    # still reads as neuron/non_neuron splitting into its branches.
    names = class_options(bins, classes)
    depth = {group["name"]: group["level"]
             for group in (scoring_groups(classes) if classes is not None else [])}
    options = [("All classes (macro)", None)]
    options += [("  " * depth.get(name, 0) + name, name) for name in names]
    return widgets.Dropdown(options=options, description=description,
                            layout=widgets.Layout(width="380px"))


def _model_dropdowns(frame):
    """Architecture and ablation dropdowns over whatever models `frame` holds."""
    from analysis.all_windows.architecture_comparison import linked_model_pickers

    pairs = frame[["architecture", "model"]].drop_duplicates().itertuples(index=False)
    return linked_model_pickers(list(pairs))


def feature_selector(bins, features, classes=None, counts=(10, 20, 40), quantiles=None):
    """A class dropdown over `plot_feature_by_model`, one grid per feature.

    No model dropdown here on purpose: the grid *is* the comparison across
    models, and the counts are its colours, so the only thing left to select is
    which class the F1 is measured on.
    """
    import matplotlib.pyplot as plt

    features = [features] if isinstance(features, str) else list(features)
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets unavailable; call plot_feature_by_model(bins, feature) directly")
        return

    def render(class_name):
        for feature in features:
            fig = plot_feature_by_model(bins, feature, class_name, counts,
                                        quantiles=quantiles)
            if fig is not None:
                plt.show()
                # Closed after display: the dropdown redraws every figure on
                # each change, which crosses pyplot's 20-open-figure warning and
                # keeps every one of them alive for the kernel's lifetime.
                plt.close(fig)

    class_name = _class_dropdown(bins, classes)
    output = widgets.interactive_output(render, {"class_name": class_name})
    display(class_name, output)


def family_selector(bins, family, classes=None, counts=(10, 20, 40), quantiles=None):
    """Architecture, ablation and class dropdowns over `plot_family_by_count`.

    Three of the four things that vary are selected and the fourth is drawn:
    the architecture and its ablation pin one model, the class pins what the F1
    measures, the family's variants are the coloured lines, and the embedding
    counts are the panels across the page.
    """
    import matplotlib.pyplot as plt

    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets unavailable; call plot_family_by_count(bins, family, model)")
        return

    architecture, model = _model_dropdowns(bins)
    class_name = _class_dropdown(bins, classes)

    def render(model, class_name, architecture):
        fig = plot_family_by_count(bins, family, model, class_name, counts,
                                   quantiles=quantiles)
        if fig is not None:
            plt.show()
            plt.close(fig)

    # `architecture` is passed in unused so that changing it re-renders: it
    # rewrites the model options, and when the new first ablation happens to
    # carry the same label as the old one the model widget alone fires nothing.
    output = widgets.interactive_output(
        render, {"model": model, "class_name": class_name, "architecture": architecture})
    display(widgets.VBox([architecture, model, class_name]), output)


def _cdf(values):
    """Points of the empirical CDF of `values`, as (score, fraction at or below).

    Drawn with `drawstyle="steps-post"`, which is what an empirical CDF is: the
    fraction is constant between observed values and jumps at each one. Joining
    the points with straight lines would interpolate cells that do not exist.
    """
    ordered = np.sort(np.asarray(values, dtype=float))
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


def plot_score_cdf(cells, model, overall=None, counts=(10, 20, 40), ncols=5,
                   threshold=0.5, score="f1"):
    """One model's per-cell F1 distribution: pooled, then split by cell type.

    The distribution is over **held-out cells**, one point each, and the score
    is that cell's F1 for its own class over its own windows. Inside a cell
    there is no window that could be a false positive -- every window carries
    that cell's single label -- so precision is 1 wherever the class is
    predicted at all and the F1 is `2r/(1+r)`. It is therefore monotone in
    recall: the curves rank cells exactly as a recall CDF would, and what
    changes is the axis the 0.5 line is read against. `score="recall"` plots
    that instead.

    The run's macro F1 over all held-out windows is annotated in the pooled
    panel's legend, which is where a single F1 for the whole model belongs --
    it is a different quantity from any point on the curve.

    The small panels are keyed by the manifest's **granular** `cell_type`, not by
    the 8-class level the model predicts -- so `pyramidal` fans out into L2IT,
    L5ET and the rest, and a class the model does well on overall can still show
    one constituent type sitting near zero. That split is only legible per cell,
    since it is finer than anything a prediction is compared against.

    The vertical line at `threshold` reads off directly as "this fraction of
    held-out cells score below 0.5", which is the number the curves exist to
    compare across neighborhood size.
    """
    import matplotlib.pyplot as plt

    part = cells[cells.model == model]
    if part.empty:
        print(f"no per-cell rows for {model}")
        return None
    if score not in part.columns:
        print(f"{score} is not in these summaries; rebuild them with --summarize-only")
        return None
    axis_label = f"per-cell {'F1' if score == 'f1' else score}"
    types = sorted(part.cell_type.unique())
    nrows = -(-len(types) // ncols)
    fig = plt.figure(figsize=(4 * ncols, 5.5 + 2.8 * nrows))
    # The pooled panel spans the full width and is given twice a small panel's
    # height: it is the headline, and the grid under it is the breakdown.
    grid = fig.add_gridspec(nrows + 2, ncols, height_ratios=[1] * 2 + [1] * nrows,
                            hspace=.55, wspace=.25)
    big = fig.add_subplot(grid[0:2, :])
    for n in counts:
        subset = part[part.n_embeddings == n]
        if subset.empty:
            continue
        x, y = _cdf(subset[score])
        label = f"n={n} \u00b7 {len(subset)} cells"
        if overall is not None:
            macro = overall[(overall.model == model) & (overall.n_embeddings == n)
                            & overall.class_name.isna()]
            if len(macro):
                label += f" \u00b7 window macro F1 {float(macro.f1.iloc[0]):.3f}"
        big.plot(x, y, drawstyle="steps-post", color=COUNT_COLORS.get(n), label=label)
    big.axvline(threshold, color="#333333", ls="--", lw=1)
    big.set(xlim=(0, 1), ylim=(0, 1), xlabel=axis_label,
            ylabel="fraction of held-out cells", title="All held-out cells")
    big.grid(alpha=.25)
    big.legend(fontsize=9)
    for index, cell_type in enumerate(types):
        ax = fig.add_subplot(grid[2 + index // ncols, index % ncols])
        typed = part[part.cell_type == cell_type]
        for n in counts:
            subset = typed[typed.n_embeddings == n]
            if subset.empty:
                continue
            x, y = _cdf(subset[score])
            ax.plot(x, y, drawstyle="steps-post", color=COUNT_COLORS.get(n))
        ax.axvline(threshold, color="#333333", ls="--", lw=.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=axis_label,
               title=f"{cell_type} \u00b7 {typed.root_id.nunique()} cells")
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.25)
    fig.suptitle(f"{model} \u00b7 per-cell {axis_label.split()[-1]} CDF by granular "
                 "cell type \u00b7 fold_0", fontsize=14)
    return fig


def score_cdf_selector(cells, overall=None, counts=(10, 20, 40), threshold=0.5,
                       score="f1"):
    """Architecture and ablation dropdowns over `plot_score_cdf`.

    No class dropdown: the small panels already *are* the class breakdown, and
    at a finer grain than a selector could offer, since they split by the
    manifest's granular `cell_type` rather than by the coarse class a
    prediction is compared against.
    """
    import matplotlib.pyplot as plt

    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print(f"ipywidgets unavailable; use plot_score_cdf(cells, {model_order(cells)[0]!r})")
        return

    architecture, model = _model_dropdowns(cells)

    def render(model, architecture):
        fig = plot_score_cdf(cells, model, overall, counts, threshold=threshold,
                             score=score)
        if fig is not None:
            plt.show()
            plt.close(fig)

    # `architecture` is passed in unused so that changing it re-renders; see
    # `family_selector`.
    output = widgets.interactive_output(
        render, {"model": model, "architecture": architecture})
    display(widgets.VBox([architecture, model]), output)


#: Tables and columns a summary only has once `summarize_cache` has been rerun
#: for the granular groups and the quantile marks. A cache written before that
#: is still readable -- every coarse panel works off it -- so the notebook says
#: what is missing rather than failing to open.
SUMMARY_REQUIREMENTS = {
    "overall": ("granular", "coarse_class"),
    "bins": ("coarse_class",),
    "quantiles": ("p25", "p50", "p75"),
}


def stale_summaries(frames) -> list[str]:
    """Which summary tables in `frames` predate the current schema."""
    missing = []
    for table, columns in SUMMARY_REQUIREMENTS.items():
        frame = frames.get(table)
        if frame is None or not len(frame):
            missing.append(f"{table} (absent)")
            continue
        absent = [column for column in columns if column not in frame.columns]
        if absent:
            missing.append(f"{table} (no {', '.join(absent)})")
    return missing


def _granular_rows(frame):
    """The granular rows of a summary table, or none at all on an old cache."""
    if "granular" not in getattr(frame, "columns", ()):
        return frame.iloc[0:0] if hasattr(frame, "iloc") else frame
    return frame[frame.granular.fillna(False).astype(bool)]


def granular_order(overall, classes=None) -> list[str]:
    """Granular labels grouped under their level-2 class, coarse classes in tree order.

    Sorting the granular labels alphabetically would interleave `AltBasket`
    (cge) with `astrocyte` (glia) and `L2IT` (pyramidal); grouping by the class
    the model actually emits puts each subtype beside the ones it is genuinely
    confusable with, which is what the bars are being compared for.
    """
    part = _granular_rows(overall)
    if part.empty:
        return []
    present = set(part.class_name.dropna().unique())
    if classes is None:
        pairs = part[["coarse_class", "class_name"]].drop_duplicates()
        return [name for _, name in sorted(
            (str(parent), str(name)) for parent, name in pairs.itertuples(index=False))]
    return [group["name"] for group in scoring_groups(classes)
            if group["granular"] and group["name"] in present]


def plot_class_bars(overall, model, classes=None, counts=(10, 20, 40),
                    metrics=(("f1", "window F1 of the level-2 prediction"),)):
    """One model's per-granular-class scores, three bars per class for the node counts.

    The x axis is the manifest's **finest** label -- `L4IT`, `AltBasket`, `NMC`
    -- grouped under the level-2 class the model emits for it, with a divider
    and a heading between groups. The scoring itself never leaves level 2: the
    model cannot say `AltBasket`, so a subtype's bar asks how the `putative_cge`
    call performs on exactly that subtype's windows.

    One panel, because there is one thing being measured: the level-2
    prediction. A bar is the F1 of "this window was given the right level-2
    class", over that subtype's windows alone -- inside the group every window
    has the same correct answer, so no window there can be a false positive,
    precision is 1 and the bar is sibling-independent. A second "correctness"
    panel would be a monotone restatement of this one, not a second question.
    """
    import matplotlib.pyplot as plt

    part = _granular_rows(overall)
    part = part[part.model == model] if len(part) else part
    if part.empty:
        print(f"no granular rows for {model} -- this summary predates them. Rebuild with\n"
              "  sbatch scripts/sbatch/summarize_feature_prediction_cache.sh")
        return None
    names = [name for name in granular_order(overall, classes)
             if name in set(part.class_name)]
    if not names:
        print(f"no granular classes present for {model}")
        return None
    parents = {str(row.class_name): str(row.coarse_class)
               for row in part.itertuples(index=False)}
    positions = np.arange(len(names), dtype=float)
    width = 0.8 / max(len(counts), 1)
    fig, axes = plt.subplots(len(metrics), 1,
                            figsize=(max(12, 0.55 * len(names)), 1 + 4.5 * len(metrics)),
                            sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (column, title) in zip(axes, metrics):
        for offset, n in enumerate(counts):
            scores = part[part.n_embeddings == n].set_index("class_name")[column]
            values = [float(scores.get(name, np.nan)) for name in names]
            ax.bar(positions + (offset - (len(counts) - 1) / 2) * width, values,
                   width=width, color=COUNT_COLORS.get(n), label=f"n={n}")
        ax.set(ylabel=title, ylim=(0, 1))
        ax.grid(axis="y", alpha=.25)
        ax.legend(fontsize=9, ncol=len(counts))
        # A divider wherever the level-2 class changes, so the three-bar groups
        # read as subtypes of one class rather than as one flat list.
        for index in range(1, len(names)):
            if parents[names[index]] != parents[names[index - 1]]:
                ax.axvline(index - .5, color="#999999", lw=.8, alpha=.6)
    start = 0
    for index in range(len(names) + 1):
        if index == len(names) or (index and parents[names[index]] != parents[names[start]]):
            axes[0].text((start + index - 1) / 2, 1.02, parents[names[start]],
                         ha="center", va="bottom", fontsize=9, color="#444444")
            start = index
    axes[-1].set_xticks(positions, names, rotation=60, ha="right", fontsize=8)
    fig.suptitle(f"{model} · per-granular-class window scores · fold_0", fontsize=13)
    fig.tight_layout()
    return fig


def class_bar_selector(overall, classes=None, counts=(10, 20, 40)):
    """Architecture and ablation dropdowns over `plot_class_bars`."""
    import matplotlib.pyplot as plt

    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets unavailable; call plot_class_bars(overall_df, model) directly")
        return

    architecture, model = _model_dropdowns(overall)

    def render(model, architecture):
        fig = plot_class_bars(overall, model, classes, counts)
        if fig is not None:
            plt.show()
            plt.close(fig)

    # `architecture` is passed in unused so that changing it re-renders; see
    # `family_selector`.
    output = widgets.interactive_output(
        render, {"model": model, "architecture": architecture})
    display(widgets.VBox([architecture, model]), output)
