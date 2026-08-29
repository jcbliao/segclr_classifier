"""Loads the segclr_db-store dataset (data/build_dataset_from_store.py) into
PyG-ready form: data/manifest.json (labels from segclr_db's own registered
cell_labels table, flat Allen-style strings) + data/graph_cache/*.pt (raw
64-dim resnet_860b_reshuffled node embeddings + skeleton edges, no
aggregation baked in).

The cached Data objects carry no y_levels; hierarchy labels are computed here
at load time from the manifest's flat `cell_type` string
(gnn/hierarchy.py::parse_hierarchy(ACTIVE_HIERARCHY_TREE).label_paths) and
attached onto the loaded Data object in memory, once per cell.

Coverage: 2335 cells, 23 granular labels, spanning both branches of the
hierarchy -- the four glia (astrocyte 77, microglia 30, oligo 16, OPC 3) and
thalamocortical (17) all carry cells, so every class at the 9-class level
this trains at receives gradient. Chandelier cells (ChC) are dropped outright
(n=1, too few to train or hold out on) -- see EXCLUDED_LABELS in
build_dataset_from_store.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset

from gnn.hierarchy import (
    HIERARCHY_V2_TREE,
    ParsedHierarchy,
    parse_hierarchy,
    truncate_hierarchy,
    with_dropped_labels,
)

#: The taxonomy every consumer of this module classifies against --
#: gnn/hierarchy.py::HIERARCHY_V2_TREE, the `hierarchy_v2` row of the shared
#: v3 store's label_hierarchies table. LAB_HIERARCHY_TREE (the store's
#: `original_hierarchy`) is the other registered option; swapping this
#: constant moves the LCPN heads, the class weights, the per-window targets
#: and the reported class list together, since all four derive from it.
ACTIVE_HIERARCHY_TREE = HIERARCHY_V2_TREE

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"
GRAPH_CACHE_DIR = Path(__file__).resolve().parent / "graph_cache"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


#: Finest levels of ACTIVE_HIERARCHY_TREE dropped before the hierarchy is
#: used. hierarchy_v2 is 5 levels deep (sizes [2, 3, 9, 12, 16], coarse to
#: fine); dropping 2 classifies at its 9-class level 2 -- pyramidal,
#: thalamocortical, the three interneuron families, and the four glia.
#: That level is where the classes that have data carry real train support:
#: the granular levels leave ChC with no training windows at all and OPC with
#: 1,724 from 2 cells, and they are more imbalanced because the L*IT merge
#: lands entirely on the head of the distribution.
#: See scripts/check_hierarchy_levels.py.
#:
#: All nine carry cells here, though support is steeply imbalanced: 1740
#: cells / 8.04M windows (pyramidal) down to 3 cells / 2562 windows (OPC),
#: a ~3100x window-count spread that the per-node class weights exist to
#: counteract. OPC lands 2 train / 1 test cell, so its CELL-level recall can
#: only ever read 0.0 or 1.0. Regenerate with
#: scripts/check_hierarchy_v2_parse.py.
HIERARCHY_LEVELS_DROPPED = 2


#: Granular labels excluded from training and evaluation, in the sense
#: segCLR_cell_classification's hierarchy `drop` key means: their cells are
#: filtered out of every split (see split_cells below), while the tree itself
#: is untouched. Applied from outside the tree rather than as a `drop` key
#: inside it, because ACTIVE_HIERARCHY_TREE is pinned verbatim to the store's
#: registered row and must keep comparing equal to it.
#:
#: ChC (n=1) is not listed here: it is already absent from manifest.json,
#: dropped at build time by build_dataset_from_store.py::EXCLUDED_LABELS.
#: This is the lever for excluding a label WITHOUT a dataset rebuild -- unlike
#: that one, it costs nothing to change and takes effect on the next run.
#:
#: OPC is dropped because 3 cells (2 train / 1 test) cannot support a class at
#: the level this trains at, and its noise was steering the whole run. It is
#: one ninth of a 9-class macro average estimated from a single held-out cell,
#: so its per-epoch swing of 0.53-0.77 window recall accounted for ~60% of the
#: epoch-to-epoch movement in the checkpoint-selection metric while every
#: other class stayed within +/-0.03. The class-balanced sampler made it worse
#: rather than better: OPC's 1,724 train windows are drawn ~36x per epoch, so
#: the model memorizes two specific cells and generalizes to the third worse
#: over time. Dropping it prunes the class outright (see
#: gnn/hierarchy.py::with_dropped_labels), leaving 8 classes.
DROP_LABELS: set[str] = {"OPC"}


def load_hierarchy(manifest: dict | None = None) -> ParsedHierarchy:
    """Resolves to ACTIVE_HIERARCHY_TREE unless the manifest carries its own
    `hierarchy_tree` key. Kept as a function (not a bare constant) so call
    sites don't need to change if a future manifest ever does carry one.

    The returned hierarchy is truncated by HIERARCHY_LEVELS_DROPPED
    (gnn/hierarchy.py::truncate_hierarchy), so classification, class
    weighting and reported metrics all operate at that level, and carries
    DROP_LABELS in its `drop_labels` set."""
    tree = (
        manifest["hierarchy_tree"]
        if manifest and "hierarchy_tree" in manifest
        else ACTIVE_HIERARCHY_TREE
    )
    hierarchy = truncate_hierarchy(parse_hierarchy(tree), HIERARCHY_LEVELS_DROPPED)
    return with_dropped_labels(hierarchy, DROP_LABELS)


def split_cells(manifest: dict, split: str, hierarchy: ParsedHierarchy) -> list[tuple[int, dict]]:
    """(root_id, info) for every cell of `split` the hierarchy admits.

    Two exclusions, both ported from segCLR_cell_classification's
    HierarchyCellTypingDataset: the hierarchy's own `drop_labels`, and any
    label the tree does not cover at all. The second is the load-bearing one
    -- an uncovered label has no entry in `label_paths`, so building its
    per-level targets raises a bare KeyError deep in dataset construction.
    Dropping it with a warning instead means a manifest that gains a new label
    (a relabeled cell, a new source table) degrades to "that cell sits out"
    rather than taking the run down.
    """
    cells = [
        (int(root_id), info)
        for root_id, info in manifest["cells"].items()
        if info["split"] == split
    ]
    # drop_labels is subtracted before computing `unknown`: a dropped label is
    # removed from label_paths (gnn/hierarchy.py::with_dropped_labels), so
    # without this it would be reported as a label the hierarchy has never
    # heard of rather than as one deliberately excluded.
    dropped = set(hierarchy.drop_labels)
    unknown = {info["cell_type"] for _, info in cells} - set(hierarchy.label_paths) - dropped
    dropped |= unknown
    if not dropped:
        return cells

    kept = [(rid, info) for rid, info in cells if info["cell_type"] not in dropped]
    import logging

    logger = logging.getLogger(__name__)
    if unknown:
        logger.warning(
            "split %r: %d label(s) absent from the hierarchy will be dropped: %s",
            split, len(unknown), sorted(unknown),
        )
    logger.info(
        "split %r: dropped %d/%d cells with labels %s",
        split, len(cells) - len(kept), len(cells), sorted(dropped),
    )
    return kept


def train_window_counts_by_label(
    manifest: dict, hierarchy: ParsedHierarchy | None = None
) -> dict[str, float]:
    """{granular_label: total WINDOW count in the train split}, from
    manifest.json's per-cell n_nodes_covered -- cheap (manifest-only, no
    .pt loading) since a window's label is its cell's label and its count
    is exactly that cell's node count. Feeds
    gnn/lcpn.py::compute_node_class_weights; see that function's docstring
    for why cell counts alone would be the wrong thing to weight from.

    Pass `hierarchy` to count only the cells the datasets actually keep --
    without it, a dropped label still contributes to the weights of a run
    that never sees it."""
    cells = (
        split_cells(manifest, "train", hierarchy)
        if hierarchy is not None
        else [(int(r), i) for r, i in manifest["cells"].items() if i["split"] == "train"]
    )
    counts: dict[str, float] = {}
    for _, info in cells:
        counts[info["cell_type"]] = counts.get(info["cell_type"], 0.0) + info["n_nodes_covered"]
    return counts


class SegCLRGraphDatasetLCPN(TorchDataset):
    """One split ("train"/"test") of the segclr_db-store dataset, at
    WHOLE-CELL granularity. Training and evaluation use the per-window
    dataset (data/dataset_windowed.py) instead; this one is for diagnostics
    and anything that needs a cell's full skeleton at once."""

    def __init__(self, manifest: dict, split: str):
        self.hierarchy = load_hierarchy(manifest)
        self.classes = self.hierarchy.level_classes[-1]  # finest-level names, for logging/metrics
        self.items = split_cells(manifest, split, self.hierarchy)

    def __len__(self) -> int:
        return len(self.items)

    def _y_levels(self, cell_type: str) -> torch.Tensor:
        path = self.hierarchy.label_paths[cell_type]
        levels = [self.hierarchy.level_maps[lvl][path[lvl]] for lvl in range(self.hierarchy.depth)]
        return torch.tensor(levels, dtype=torch.long).unsqueeze(0)  # (1, depth)

    def __getitem__(self, i: int):
        root_id, info = self.items[i]
        data = torch.load(GRAPH_CACHE_DIR / f"{root_id}.pt", weights_only=False)
        data.y_levels = self._y_levels(info["cell_type"])
        data.y = data.y_levels[:, -1]  # finest-level alias
        return data
