"""Runs the lab's actual train.py (segCLR_cell_classification/train.py)
unchanged, with two behavioral patches applied via monkeypatching -- their
vendored source is never edited directly (matches the segclr_db convention:
prefer patching from a sibling location over editing someone else's repo):

1. CellTypingDataset._split_by_cells uses OUR exact train/val/test partition
   (the "splits" field written by scripts/generate_lab_baseline_hdf5.py, a
   copy of data/manifest.json's split) instead of their internal random
   2-way split. Per explicit user direction: keep our own split so the
   baseline (this) and the GNN are evaluated on identical cells.

   Test rows (splits == "test") are excluded from both train_indices and
   val_indices here -- held out entirely, evaluated separately by
   scripts/evaluate_lab_baseline_on_test.py against the trained checkpoint.

2. CellTypingDataset.__init__'s class-weight computation is vectorized. Their
   original does `[self.train_dataset[i][1] for i in range(len(...))]` --
   8.2M individual Python-level __getitem__ calls (each slicing a (N, 64)
   tensor just to read one label) to build a list Python then converts with
   torch.tensor(). Replaced with one vectorized np.searchsorted call against
   self.class_labels_in_order (sorted, so searchsorted reproduces
   celltype_map's mapping exactly) -- same result, no measurable per-example
   Python overhead. This block also has a latent bug in their own reference
   config (configs/resnet.yaml sets weight_imbalanced_classes: true, a YAML
   bool that matches neither `== "sample"` nor `== "loss"`, leaving `sampler`
   unassigned -> UnboundLocalError; our config uses the string "loss"
   instead, confirmed by actually hitting that crash first).

Everything else -- model (DeepResNet), trainer, loss, optimizer, their
cell_level_accuracy majority-vote metric -- stays exactly their code, called
through their unmodified train.py.

Usage (same CLI as their train.py):
    python scripts/lab_baseline_train.py configs/our_resnet.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

LAB_REPO = Path(__file__).resolve().parent.parent / "segCLR_cell_classification"
sys.path.insert(0, str(LAB_REPO))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402

from cell_classification.data.datasets import (  # noqa: E402
    CellTypingDataset,
    HDF5CellDatasetEager,
    HDF5CellDatasetLazy,
)


def _split_by_our_cells(self, seg_ids, all_labels):
    with h5py.File(self.config.path, "r") as f:
        splits = np.array([s.decode() if isinstance(s, bytes) else s for s in f["splits"][:]])
    train_indices = np.nonzero(splits == "train")[0].tolist()
    val_indices = np.nonzero(splits == "val")[0].tolist()
    n_test = int((splits == "test").sum())
    print(
        f"[our split, not the lab's internal random split] "
        f"train: {len(train_indices)} rows, val: {len(val_indices)} rows, "
        f"test: {n_test} rows (excluded here, evaluated separately)"
    )
    return train_indices, val_indices


def _init_vectorized(self, config):
    """Copy of CellTypingDataset.__init__ with the class-weight computation
    vectorized (see module docstring, point 2) -- everything else identical
    to their source (src/data/datasets.py) as of the commit this was cloned
    from."""
    self.config = config
    self.train_split = self.config.train_split
    self.batch_size = self.config.batch_size
    self.normalize = self.config.normalize
    self.use_coarse = self.config.use_coarse
    self.weight_imbalanced_classes = self.config.weight_imbalanced_classes
    self.lazy_loading = self.config.lazy_loading
    self.is_ood_dataset = False

    print(f"Loading from {config.path} (lazy={self.lazy_loading})...", flush=True)

    with h5py.File(config.path, "r") as f:
        seg_ids = torch.from_numpy(f["seg_ids"][:])
        label_key = "coarse_cell_types" if self.use_coarse else "cell_types"
        all_labels = np.array([s.decode("utf-8") if isinstance(s, bytes) else s for s in f[label_key][:]])
        self._embedding_dim = f["embeddings"].shape[1]
        all_nodes = torch.from_numpy(f["nodes"][:]) if "nodes" in f else None
        if not self.lazy_loading:
            print("Loading all embeddings into memory...")
            embeddings = torch.from_numpy(f["embeddings"][:]).to(torch.float32)

    print(f"Loaded metadata for {len(seg_ids)} embeddings", flush=True)

    valid_indices_mask = None
    from cell_classification.data.datasets import _apply_class_transforms

    if not self.use_coarse and (config.merge_classes or config.drop_classes):
        all_labels, valid_mask = _apply_class_transforms(
            all_labels, config.merge_classes, config.drop_classes
        )
        valid_indices_mask = np.where(valid_mask)[0]
        seg_ids = seg_ids[valid_indices_mask]
        all_labels = all_labels[valid_indices_mask]
        if all_nodes is not None:
            all_nodes = all_nodes[valid_indices_mask]
        if not self.lazy_loading:
            embeddings = embeddings[valid_indices_mask]
        print(
            f"After class transforms: {len(seg_ids)} embeddings remain "
            f"({valid_mask.sum()} / {len(valid_mask)})",
            flush=True,
        )

    self.class_labels_in_order = sorted(set(all_labels))
    self.celltype_map = {ct: idx for idx, ct in enumerate(self.class_labels_in_order)}
    self.reverse_celltype_map = {idx: ct for ct, idx in self.celltype_map.items()}

    train_indices, val_indices = self._split_by_cells(seg_ids, all_labels)

    self.val_node_ids = all_nodes[np.array(val_indices)] if all_nodes is not None else None

    if valid_indices_mask is not None and self.lazy_loading:
        train_indices = valid_indices_mask[train_indices].tolist()
        val_indices = valid_indices_mask[val_indices].tolist()

    val_ids = seg_ids[val_indices].unique()
    with open("val_ids.json", "w") as f:
        json.dump(val_ids.tolist(), f)

    if self.lazy_loading:
        self.train_dataset = HDF5CellDatasetLazy(
            h5_path=config.path, indices=train_indices, seg_ids=seg_ids[train_indices],
            labels=all_labels[train_indices], celltype_map=self.celltype_map, normalize=self.normalize,
        )
        self.val_dataset = HDF5CellDatasetLazy(
            h5_path=config.path, indices=val_indices, seg_ids=seg_ids[val_indices],
            labels=all_labels[val_indices], celltype_map=self.celltype_map, normalize=self.normalize,
        )
        num_workers, persistent_workers = 4, True
    else:
        self.train_dataset = HDF5CellDatasetEager(
            embeddings=embeddings[train_indices], seg_ids=seg_ids[train_indices],
            labels=all_labels[train_indices], celltype_map=self.celltype_map, normalize=self.normalize,
        )
        self.val_dataset = HDF5CellDatasetEager(
            embeddings=embeddings[val_indices], seg_ids=seg_ids[val_indices],
            labels=all_labels[val_indices], celltype_map=self.celltype_map, normalize=self.normalize,
        )
        num_workers, persistent_workers = 0, False

    if not self.weight_imbalanced_classes:
        sampler, shuffle = None, True
    else:
        # Vectorized: class_labels_in_order is sorted, so searchsorted finds
        # each label's celltype_map index directly -- same result as
        # `[celltype_map[lbl] for lbl in all_labels[train_indices]]`, no
        # per-example Python-level __getitem__ calls.
        sorted_labels = np.array(self.class_labels_in_order)
        train_labels_np = np.searchsorted(sorted_labels, all_labels[train_indices])
        train_labels = train_labels_np.tolist()
        class_counts = torch.bincount(
            torch.from_numpy(train_labels_np), minlength=len(self.celltype_map)
        )
        self.class_weights = torch.nan_to_num(1.0 / class_counts, nan=1.0, posinf=1.0)
        if self.weight_imbalanced_classes == "sample":
            sample_weights = self.class_weights[train_labels_np]
            sampler = WeightedRandomSampler(
                weights=sample_weights, num_samples=len(sample_weights), replacement=True
            )
            shuffle = False
        elif self.weight_imbalanced_classes == "loss":
            sampler, shuffle = None, True
        else:
            raise ValueError(
                f"weight_imbalanced_classes must be a JSON/YAML string \"sample\" or \"loss\" "
                f"(got {self.weight_imbalanced_classes!r} of type {type(self.weight_imbalanced_classes).__name__}) "
                f"-- a bare `true` matches neither branch and used to silently crash later "
                f"with UnboundLocalError on `sampler`."
            )

    self._train_loader = DataLoader(
        self.train_dataset, sampler=sampler, shuffle=shuffle, batch_size=self.batch_size,
        num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=True,
    )
    self._val_loader = DataLoader(
        self.val_dataset, batch_size=self.batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=True,
    )


CellTypingDataset._split_by_cells = _split_by_our_cells
CellTypingDataset.__init__ = _init_vectorized

# Import their actual train.py as a module and run its main() unchanged --
# this happens AFTER the monkeypatches above so setup_data() picks them up.
_spec = importlib.util.spec_from_file_location("lab_train", LAB_REPO / "train.py")
lab_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config YAML")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    lab_train.main(args.config, args.debug)
