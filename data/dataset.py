"""Loads the local dataset built by data/build_dataset.py into PyG-ready form.

Reads data/manifest.json (per-cell cell_type + split) and data/graph_cache/*.pt
(one torch_geometric.data.Data per cell). No segclr_db store involved -- this
is the whole data layer for now (see data/cave_skeletons.py for why).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"
GRAPH_CACHE_DIR = Path(__file__).resolve().parent / "graph_cache"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def label_vocab(manifest: dict, depth: int | None = None) -> tuple[list[str], dict[str, int]]:
    """Class vocabulary at a given hierarchy depth.

    cell_type strings are dash-separated, coarse-to-fine (e.g.
    "C-N-I-BC-Pvalb" = cell / neuron / inhibitory / basket cell / Pvalb+).
    depth=2 gives "C-N"/"C-G" (coarsest, neuron-vs-glia); depth=3 adds E/I or
    a glia subtype; depth=None uses the full string (finest, ~20 classes,
    several with under 10 examples in the labeled set -- see
    build_dataset.stratified_split's thin-class warning).

    Vocabulary is built from every cell in the manifest, not just train: it's
    a fixed label space, not a data-derived statistic.
    """
    labels = set()
    for info in manifest["cells"].values():
        ct = info["cell_type"]
        if depth is not None:
            ct = "-".join(ct.split("-")[:depth])
        labels.add(ct)
    classes = sorted(labels)
    return classes, {c: i for i, c in enumerate(classes)}


class SegCLRGraphDataset(TorchDataset):
    """One split ("train"/"val"/"test") of the local dataset."""

    def __init__(self, manifest: dict, split: str, depth: int | None = None):
        self.depth = depth
        self.classes, self.class_to_idx = label_vocab(manifest, depth)
        self.items = [
            (int(root_id), info)
            for root_id, info in manifest["cells"].items()
            if info["split"] == split
        ]

    def __len__(self) -> int:
        return len(self.items)

    def _label_id(self, cell_type: str) -> int:
        ct = cell_type if self.depth is None else "-".join(cell_type.split("-")[: self.depth])
        return self.class_to_idx[ct]

    def __getitem__(self, i: int):
        root_id, info = self.items[i]
        data = torch.load(GRAPH_CACHE_DIR / f"{root_id}.pt", weights_only=False)
        data.y = torch.tensor([self._label_id(info["cell_type"])], dtype=torch.long)
        return data
