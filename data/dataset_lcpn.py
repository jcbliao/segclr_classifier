"""Loads the segclr_db-store dataset (data/build_dataset_from_store.py) into
PyG-ready form: data/manifest.json (labels from segclr_db's own registered
cell_labels table, flat Allen-style strings) + data/graph_cache/*.pt (raw
64-dim resnet_860b_reshuffled node embeddings + skeleton edges, no
aggregation baked in).

The cached Data objects carry no y_levels; hierarchy labels are computed here
at load time from the manifest's flat `cell_type` string
(gnn/hierarchy.py::parse_hierarchy(LAB_HIERARCHY_TREE).label_paths) and
attached onto the loaded Data object in memory, once per cell.

Coverage: 2192 cells, 18 granular classes, all under the `neuron` branch of
LAB_HIERARCHY_TREE. `thalamocortical` and the four glia classes
(astrocyte/oligo/microglia/OPC) have zero examples here -- not a labeling gap
(segclr_db's cell_labels table does have them), but an embedding one: the
resnet_860b_reshuffled experiment this pipeline reads has zero embedding rows
for any non-neuron cell (scripts/check_new_cells_embedding_coverage.py), so
those LCPN head branches never receive gradient. Chandelier cells (ChC) are
dropped outright (n=1, too few to train or hold out on) -- see
EXCLUDED_LABELS in build_dataset_from_store.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset

from gnn.hierarchy import LAB_HIERARCHY_TREE, ParsedHierarchy, parse_hierarchy

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"
GRAPH_CACHE_DIR = Path(__file__).resolve().parent / "graph_cache"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def load_hierarchy(manifest: dict | None = None) -> ParsedHierarchy:
    """Resolves to LAB_HIERARCHY_TREE unless the manifest carries its own
    `hierarchy_tree` key. Kept as a function (not a bare constant) so call
    sites don't need to change if a future manifest ever does carry one."""
    tree = manifest["hierarchy_tree"] if manifest and "hierarchy_tree" in manifest else LAB_HIERARCHY_TREE
    return parse_hierarchy(tree)


def train_window_counts_by_label(manifest: dict) -> dict[str, float]:
    """{granular_label: total WINDOW count in the train split}, from
    manifest.json's per-cell n_nodes_covered -- cheap (manifest-only, no
    .pt loading) since a window's label is its cell's label and its count
    is exactly that cell's node count. Feeds
    gnn/lcpn.py::compute_node_class_weights; see that function's docstring
    for why cell counts alone would be the wrong thing to weight from."""
    counts: dict[str, float] = {}
    for info in manifest["cells"].values():
        if info["split"] == "train":
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
        self.items = [
            (int(root_id), info)
            for root_id, info in manifest["cells"].items()
            if info["split"] == split
        ]

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
