"""One shared node/view bank; sample fresh views in packed CPU/GPU gathers."""

import copy
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from data.batched_presynaptic import ragged_indices
from data.segmentation_augmentation import TRAINING_POLICIES, training_choice_indices


def enabled(dataset):
    return getattr(dataset, "embedding_augmentation", None) in TRAINING_POLICIES


def attach(dataset, workers, max_bytes):
    """Store each distinct node's seven views once, plus window occurrence IDs."""
    packed = dataset._packed
    roots, starts, counts = np.unique(
        dataset.index_root_ids, return_index=True, return_counts=True
    )
    tasks = []
    labels = None
    total = 1  # zero row for nodes lacking an observed embedding
    for root, start, count in zip(roots, starts, counts):
        path = Path(dataset.augmentation_database) / "segmask" / "cells" / f"{root}.npz"
        with np.load(path, allow_pickle=False) as z:
            current = z["choice_labels"].copy()
            size = len(z["node_ids"])
        if labels is not None and not np.array_equal(labels, current):
            raise ValueError("augmentation view order differs between cells")
        labels = current
        tasks.append((int(root), int(start), int(count), total, size, path))
        total += size
    if labels is None:
        raise ValueError("cannot pack an empty augmented dataset")
    required = total * len(labels) * 64 * 4 + int(packed["ptr"][-1]) * 8
    if dataset.packing_bytes + required > max_bytes:
        raise MemoryError(
            f"Packed windows and augmentation bank need {(dataset.packing_bytes + required) / 2**30:.1f} GiB"
        )
    workdir = getattr(dataset, "_packing_workdir", None)
    shape = (total, len(labels), 64)
    bank = (
        np.lib.format.open_memmap(
            Path(workdir) / "augmentation_bank.npy",
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
        if workdir
        else np.empty(shape, np.float32)
    )
    bank[0] = 0
    rows = np.zeros(int(packed["ptr"][-1]), np.int64)
    clean_column = list(labels).index("clean")

    def pack(task):
        root, start, count, offset, size, path = task
        local = copy.copy(dataset)
        local._cache = OrderedDict()
        local.cell_cache_size = 1
        local.embedding_augmentation = None
        cell = local._load_cell(root)
        with np.load(path, allow_pickle=False) as z:
            ids = z["node_ids"]
            values = z["embedding_choices"]
            if values.shape != (size, len(labels), 64) or len(np.unique(ids)) != size:
                raise ValueError(f"invalid augmentation bank: {path}")
            bank[offset : offset + size] = values
        observed = np.full(len(cell["pos"]), -1, np.int64)
        observed[cell["observed_ids"]] = np.arange(len(cell["observed_ids"]))
        if size and (
            ids.min() < 0 or ids.max() >= len(observed) or np.any(observed[ids] < 0)
        ):
            raise ValueError(f"augmented node absent from clean inputs: {root}")
        bank[offset : offset + size, clean_column] = cell["observed_x"][observed[ids]]
        lookup = np.zeros(len(cell["pos"]), np.int64)
        lookup[ids] = np.arange(offset, offset + size)
        selected = dataset.index_window_rows[start : start + count]
        lengths = dataset.window_sizes[start : start + count].astype(np.int64)
        nodes = (
            np.asarray(cell["centers"][selected], np.int64)
            if dataset.single_presynaptic_embedding
            else cell["members"][ragged_indices(cell["offsets"][selected], lengths)]
        )
        if np.any((observed[nodes] >= 0) & (lookup[nodes] == 0)):
            raise ValueError(
                f"augmentation coverage missing for observed window nodes: {root}"
            )
        lo, hi = map(int, packed["ptr"][[start, start + count]])
        rows[lo:hi] = lookup[nodes]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(pack, tasks))
    packed.update(
        augmentation_bank=bank,
        augmentation_rows=rows,
        augmentation_labels=np.asarray(labels),
    )
    dataset.packing_bytes += (
        bank.nbytes + rows.nbytes + packed["augmentation_labels"].nbytes
    )
    print(
        f"Packed stochastic augmentation bank: {total - 1:,} distinct nodes, {len(labels)} views ({required / 2**30:.2f} GiB)",
        flush=True,
    )


def sample(dataset, indices, device=False, choices=None):
    p = dataset._packed
    if device:
        bank, rows = p["device_augmentation_bank"], p["device_augmentation_rows"]
        columns = p["device_augmentation_columns"]
    else:
        bank, rows = (
            torch.from_numpy(p["augmentation_bank"]),
            torch.from_numpy(p["augmentation_rows"]),
        )
        columns = torch.tensor(
            training_choice_indices(
                dataset.embedding_augmentation, p["augmentation_labels"].tolist()
            ),
            dtype=torch.long,
        )
    indices = torch.as_tensor(indices, dtype=torch.long, device=bank.device)
    chosen = (
        torch.randint(len(columns), (indices.numel(),), device=bank.device)
        if choices is None
        else torch.as_tensor(choices, dtype=torch.long, device=bank.device)
    )
    return bank[rows.index_select(0, indices), columns.index_select(0, chosen)]
