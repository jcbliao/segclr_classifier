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
    a fixed label space, not a data-derived statistic. Class weights, which
    ARE data-derived, are computed from train alone in gnn/losses.py.
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


class ReplacementPool:
    """Samples x_r replacement embeddings for gnn/masking.py's NodeMasker, per
    the spec's "drawn from D_train" requirement -- built from a train-split
    SegCLRGraphDataset only, never val/test.

    Supports the spec's three variants:
      "random"     -- any other neuron in train, regardless of class
      "diff_class" -- a neuron whose cell_type differs from the node's own cell
      "same_class" -- a neuron of the same cell_type

    "diff_class"/"same_class" need the class of the graph being masked, which
    the generic per-node NodeMasker.forward doesn't track. So replacement here
    is bound per-graph via sampler_for(label) -- call it once per graph in the
    training loop and pass the result as NodeMasker's replacement_source. All
    replacement draws within one masked graph then respect that graph's own
    same/different-class relationship, which is what "from another skeleton"
    naturally means when the pool is organized by class.
    """

    def __init__(
        self,
        train_dataset: SegCLRGraphDataset,
        strategy: str = "random",
        seed: int = 0,
        device: torch.device | str | None = None,
    ):
        if strategy not in ("random", "diff_class", "same_class"):
            raise ValueError(f"unknown strategy {strategy!r}; expected random/diff_class/same_class")
        self.strategy = strategy
        # A plain (CPU) generator on purpose -- torch.randint with a CUDA
        # generator requires the *output* tensor to be on that same device,
        # which would force self.all_x/by_class onto GPU memory just to
        # sample indices. Indices are sampled on CPU either way (they're just
        # ints), so the generator device is unrelated to where the pool's
        # embedding tensors themselves live.
        self.rng = torch.Generator().manual_seed(seed)

        by_class: dict[int, list[torch.Tensor]] = {}
        all_x: list[torch.Tensor] = []
        for i in range(len(train_dataset)):
            data = train_dataset[i]
            label = int(data.y.item())
            by_class.setdefault(label, []).append(data.x)
            all_x.append(data.x)
        self.all_x = torch.cat(all_x, dim=0)
        self.by_class = {c: torch.cat(xs, dim=0) for c, xs in by_class.items()}
        if device is not None:
            self.all_x = self.all_x.to(device)
            self.by_class = {c: x.to(device) for c, x in self.by_class.items()}

    def sampler_for(self, label: int):
        """Returns callable(n) -> (n, D) tensor, bound to one graph's class --
        pass directly as gnn.masking.NodeMasker.forward's replacement_source."""
        if self.strategy == "random":
            pool = self.all_x
        elif self.strategy == "same_class":
            pool = self.by_class.get(label, self.all_x)
        else:  # diff_class
            others = [x for c, x in self.by_class.items() if c != label]
            pool = torch.cat(others, dim=0) if others else self.all_x

        def _sample(n: int) -> torch.Tensor:
            idx = torch.randint(0, len(pool), (n,), generator=self.rng)
            return pool[idx]

        return _sample
