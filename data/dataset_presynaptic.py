"""Lazy presynaptic-window datasets with exact membership deduplication.

The source database stores ragged windows per cell.  This loader keeps only a
compact global index in memory and caches a few decompressed cells per worker.
Windows are unique within their root ID by their complete sorted node set.  In
particular, ``variant='new'`` includes non-SegCLR TEASAR nodes in the key.
"""

from __future__ import annotations

import random
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data

from data.dataset_lcpn import load_hierarchy, split_cells
from data.geodesic_window import REL_POS_SCALE_NM
from gnn.hierarchy import with_dropped_labels

DEFAULT_DATABASE = Path("/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/k10")
DEFAULT_AUGMENTATION_DATABASE = Path(
    "/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/"
    "cave_embedding_training_choices/conf0.7/fold0/cutoff5000_fp16"
)
NON_NEURON_LABELS = {"astrocyte", "microglia", "oligo", "OPC"}


def load_cell_arrays(path: Path, variant: str) -> dict[str, np.ndarray]:
    """Decode one cell exactly as the training loader does before caching it."""
    with np.load(path) as z:
        if "geometry_path" in z:
            with np.load(str(z['geometry_path'])) as geo, np.load(str(z['embedding_path'])) as emb:
                original = geo['original_node_ids']
                ids = emb['node_ids']
                locations = np.searchsorted(ids, original)
                if len(locations) and (locations.max() >= len(ids) or not np.array_equal(ids[locations], original)):
                    raise ValueError(f'Missing named embeddings for cell {path.stem}')
                cell = dict(pos=geo['pos_nm'].copy(), edges=geo['edges'].copy(),
                            observed_ids=np.arange(len(original), dtype=np.int32),
                            observed_x=emb['embeddings'][locations].copy(),
                            original_node_ids=original.copy())
        else:
            cell = {
                "pos": z[f"{variant}_pos_nm"].copy(),
                "edges": z[f"{variant}_edges"].copy(),
                "observed_ids": z[f"{variant}_observed_node_ids"].copy(),
                "observed_x": z[f"{variant}_observed_x"].copy(),
            }
        cell.update({
            "centers": z[f"{variant}_nearest_observed_node"].copy(),
            "offsets": z[f"{variant}_window_offsets"].copy(),
            "members": z[f"{variant}_window_members"].copy(),
            "lpe": z[f"{variant}_window_lpe"].copy(),
        })
        if f'{variant}_window_edge_offsets' in z:
            cell['window_edge_offsets'] = z[f'{variant}_window_edge_offsets'].copy()
            cell['window_edges'] = z[f'{variant}_window_edges'].copy()
    return cell


class PresynapticWindowDataset(Dataset):
    """Unique CAVE or new-skeleton presynaptic windows for one cell split."""

    def __init__(
        self,
        manifest: dict,
        split: str,
        variant: str,
        database: str | Path = DEFAULT_DATABASE,
        cell_cache_size: int = 2,
        memmap_root: str | Path | None = None,
        embedding_augmentation: str | None = None,
        augmentation_database: str | Path = DEFAULT_AUGMENTATION_DATABASE,
        postsynaptic_cache: str | Path | None = None,
        use_postsynaptic: bool = False,
        compartment_filter: str | Path | None = None,
        single_presynaptic_embedding: bool = False,
    ):
        if variant not in ("cave", "new"):
            raise ValueError("variant must be 'cave' or 'new'")
        if memmap_root is not None and variant != "new":
            raise ValueError("The memory-mapped cache currently supports only variant='new'")
        if embedding_augmentation is not None and variant != "cave":
            raise ValueError("embedding augmentation is supported only for variant='cave'")
        self.postsynaptic_cache = Path(postsynaptic_cache) if postsynaptic_cache else None
        self.use_postsynaptic = use_postsynaptic
        if use_postsynaptic and self.postsynaptic_cache is None:
            raise ValueError('Postsynaptic features require a prepared cache')
        if self.postsynaptic_cache is not None:
            marker = json.loads((self.postsynaptic_cache / 'complete.json').read_text())
            if marker['format'] != 'postsynaptic-native-site-cache-v1':
                raise ValueError('Invalid postsynaptic cache')
        self.variant = variant
        self.single_presynaptic_embedding = single_presynaptic_embedding
        self.compartment_filter = Path(compartment_filter) if compartment_filter else None
        if self.compartment_filter:
            provenance = json.loads((self.compartment_filter / 'summary.json').read_text())
            if Path(provenance['database']).resolve() != Path(database).resolve() or provenance['variant'] != variant:
                raise ValueError('Compartment filter belongs to a different database or variant')
        self.database = Path(database)
        self.cell_dir = self.database / "cells"
        self.cell_cache_size = cell_cache_size
        self.memmap_root = Path(memmap_root) if memmap_root is not None else None
        self.embedding_augmentation = embedding_augmentation
        self.augmentation_database = Path(augmentation_database)
        # The proofread-axon database is neuron-only. Prune unsupported glial
        # outputs rather than leaving never-positive logits in the LCPN head.
        self.hierarchy = with_dropped_labels(load_hierarchy(manifest), NON_NEURON_LABELS)
        self.classes = self.hierarchy.level_classes[-1]
        allowed = split_cells(manifest, split, self.hierarchy)

        roots, rows, sizes, labels = [], [], [], []
        cell_types: dict[int, str] = {}
        for root_id, info in allowed:
            path = self.cell_dir / f"{root_id}.npz"
            if not path.is_file():
                continue
            with np.load(path) as z:
                valid = np.flatnonzero(z[f"{variant}_valid_k_window"])
                if self.compartment_filter:
                    with np.load(self.compartment_filter / 'cells' / f'{root_id}.npz') as votes:
                        if not np.array_equal(votes['synapse_ids'], z[f'{variant}_synapse_id']):
                            raise ValueError(f'Compartment filter site order mismatch for {root_id}')
                        keep = votes['keep']
                    valid = valid[keep[valid]]
                if self.postsynaptic_cache is not None:
                    with np.load(self.postsynaptic_cache / 'cells' / f'{root_id}.npz') as post:
                        if not np.array_equal(post['synapse_ids'], z[f'{variant}_synapse_id']):
                            raise ValueError(f'Postsynaptic cache site order mismatch for {root_id}')
                        eligible = post['eligible']
                    valid = valid[eligible[valid]]
                offsets = z[f"{variant}_window_offsets"]
                members = z[f"{variant}_window_members"]
                seen: set[bytes] = set()
                keep_rows, keep_sizes = [], []
                for row in valid:
                    a, b = int(offsets[row]), int(offsets[row + 1])
                    nodes = members[a:b]
                    # Node IDs are cell-local, so uniqueness is intentionally
                    # scoped to root_id. Sorting makes the key set-based while
                    # the stored order remains available for LPE alignment.
                    key = np.sort(nodes).tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    keep_rows.append(int(row))
                    keep_sizes.append(b - a)
            if not keep_rows:
                continue
            path_labels = self.hierarchy.label_paths[info["cell_type"]]
            finest = self.hierarchy.level_maps[-1][path_labels[-1]]
            roots.extend([root_id] * len(keep_rows))
            rows.extend(keep_rows)
            sizes.extend(keep_sizes)
            labels.extend([finest] * len(keep_rows))
            cell_types[root_id] = info["cell_type"]

        self.index_root_ids = np.asarray(roots, dtype=np.int64)
        self.index_window_rows = np.asarray(rows, dtype=np.int32)
        self.window_sizes = np.asarray(sizes, dtype=np.int32)
        if self.single_presynaptic_embedding:
            self.window_sizes[:] = 1
        self.index_labels = np.asarray(labels, dtype=np.int16)
        self.cell_types = cell_types
        self._memmap_keys: dict[int, list[str]] = {}
        if self.memmap_root is not None:
            missing = [root_id for root_id in cell_types
                       if not (self.memmap_root / str(root_id) / 'complete.json').is_file()]
            if missing:
                raise FileNotFoundError(f'Missing memory-mapped cache for {len(missing)} cells; first {missing[0]}')
            for root_id in cell_types:
                marker = json.loads((self.memmap_root / str(root_id) / 'complete.json').read_text())
                if marker['root_id'] != root_id or marker['variant'] != variant:
                    raise ValueError(f'Wrong memory-mapped cache for cell {root_id}')
                self._memmap_keys[root_id] = marker['arrays']
        self._cache: OrderedDict[int, dict[str, np.ndarray | torch.Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index_root_ids)

    def _targets(self, root_id: int) -> torch.Tensor:
        path = self.hierarchy.label_paths[self.cell_types[root_id]]
        levels = [self.hierarchy.level_maps[level][path[level]]
                  for level in range(self.hierarchy.depth)]
        return torch.tensor(levels, dtype=torch.long).unsqueeze(0)

    def _load_cell(self, root_id: int) -> dict[str, np.ndarray | torch.Tensor]:
        if root_id in self._cache:
            self._cache.move_to_end(root_id)
            return self._cache[root_id]
        if self.memmap_root is None:
            cell = load_cell_arrays(self.cell_dir / f"{root_id}.npz", self.variant)
        else:
            directory = self.memmap_root / str(root_id)
            cell = {name: np.load(directory / f'{name}.npy', mmap_mode='c')
                    for name in self._memmap_keys[root_id]}
        if self.embedding_augmentation is not None:
            path = (self.augmentation_database / self.embedding_augmentation /
                    "cells" / f"{root_id}.npz")
            if not path.is_file():
                raise FileNotFoundError(f"Missing packed augmentation choices: {path}")
            with np.load(path, allow_pickle=False) as z:
                node_ids = z["node_ids"].copy()
                choices = z["embedding_choices"].copy()
                policy = str(z["augmentation_id"])
            if (policy != self.embedding_augmentation or choices.ndim != 3
                    or choices.shape[0] != len(node_ids) or choices.shape[1] < 2
                    or choices.shape[2] != 64):
                raise ValueError(f"Invalid packed augmentation choices: {path}")
            cell["augmentation_choices"] = choices
            cell["augmentation_lookup"] = {
                int(node): row for row, node in enumerate(node_ids)
            }
        if self.postsynaptic_cache is not None:
            with np.load(self.postsynaptic_cache / 'cells' / f'{root_id}.npz') as post:
                cell['synapse_ids'] = post['synapse_ids'].copy()
                if self.use_postsynaptic:
                    cell['post_x'] = post['post_x'].copy()
        cell['targets'] = self._targets(root_id)
        # Full-cell edges are scanned only once: adjacency makes induced-edge
        # extraction O(sum degree in the window), not O(all cell edges).
        if 'window_edges' not in cell:
            adjacency: list[list[int]] = [[] for _ in range(len(cell["pos"]))]
            for u, w in cell["edges"]:
                adjacency[int(u)].append(int(w)); adjacency[int(w)].append(int(u))
            cell["adjacency"] = adjacency
        cell["observed_lookup"] = {
            int(node): row for row, node in enumerate(cell["observed_ids"])
        }
        self._cache[root_id] = cell
        self._cache.move_to_end(root_id)
        while len(self._cache) > self.cell_cache_size:
            self._cache.popitem(last=False)
        return cell

    def __getitem__(self, index: int) -> Data:
        root_id = int(self.index_root_ids[index])
        row = int(self.index_window_rows[index])
        cell = self._load_cell(root_id)
        a, b = int(cell["offsets"][row]), int(cell["offsets"][row + 1])
        nodes = cell["members"][a:b]
        if self.single_presynaptic_embedding:
            nodes = np.asarray([cell['centers'][row]])
        n = len(nodes)
        local = {int(node): i for i, node in enumerate(nodes)}
        edge_pairs = []
        if self.single_presynaptic_embedding:
            pass
        elif 'window_edges' in cell:
            ea,eb=cell['window_edge_offsets'][row:row+2]
            edge_pairs=cell['window_edges'][ea:eb].tolist()
        else:
            for node in nodes:
                u = local[int(node)]
                for neighbor in cell["adjacency"][int(node)]:
                    w = local.get(neighbor)
                    if w is not None:
                        edge_pairs.append((u, w))
        edge_index = (
            torch.tensor(edge_pairs, dtype=torch.long).T.contiguous()
            if edge_pairs else torch.empty((2, 0), dtype=torch.long)
        )

        x = torch.zeros((n, cell["observed_x"].shape[1]), dtype=torch.float32)
        has_segclr = torch.zeros(n, dtype=torch.bool)
        lookup = cell["observed_lookup"]
        for i, node in enumerate(nodes):
            embedding_row = lookup.get(int(node))
            if embedding_row is not None:
                if self.embedding_augmentation is None:
                    x[i] = torch.from_numpy(cell["observed_x"][embedding_row])
                else:
                    augmented_row = cell["augmentation_lookup"].get(int(node))
                    if augmented_row is None:
                        raise KeyError(
                            f"node {int(node)} in root {root_id} has no "
                            f"{self.embedding_augmentation} augmentation choices"
                        )
                    choice = int(torch.randint(cell["augmentation_choices"].shape[1], ()).item())
                    x[i] = torch.from_numpy(cell["augmentation_choices"][augmented_row, choice])
                has_segclr[i] = True

        if self.single_presynaptic_embedding and not bool(has_segclr.all()):
            raise ValueError(f'Nearest observed node lacks an embedding: {root_id}/{row}')

        pos = cell["pos"][nodes]
        center = int(cell["centers"][row])
        data = Data(
            x=x,
            edge_index=edge_index,
            pos=torch.from_numpy(pos).float(),
            rel_pos=torch.from_numpy((pos - cell["pos"][center]) / REL_POS_SCALE_NM).float(),
            pos_enc=(torch.zeros((1, cell['lpe'].shape[1]), dtype=torch.float32)
                     if self.single_presynaptic_embedding else torch.from_numpy(cell["lpe"][a:b]).float()),
            has_segclr=has_segclr,
            y_levels=cell["targets"],
            root_id=torch.tensor([root_id], dtype=torch.long),
            center_index=torch.tensor([center], dtype=torch.long),
        )
        if self.postsynaptic_cache is not None:
            data.synapse_id = torch.tensor([int(cell['synapse_ids'][row])], dtype=torch.long)
        if self.use_postsynaptic:
            data.postsynaptic_embedding = torch.from_numpy(cell['post_x'][row]).float().unsqueeze(0)
        data.y = data.y_levels[:, -1]
        return data


class AttentionBudgetBatchSampler(Sampler[list[int]]):
    """Size-bucketed batches bounded by dense attention cost.

    Windows are assembled into batches within a cell so a worker can reuse its
    decompressed cell record.  Short groups of those batches are then shuffled
    globally: this prevents a large cell (and therefore one class) from
    occupying thousands of consecutive optimizer steps while retaining most
    of the cache locality.
    """

    def __init__(
        self,
        dataset: PresynapticWindowDataset,
        attention_budget: int = 1_806_336,
        max_windows: int = 4096,
        shuffle: bool = True,
        seed: int = 0,
        cell_batch_group: int = 8,
        balance_classes: bool = False,
        class_balance_power: float = 0.5,
    ):
        self.dataset = dataset
        self.attention_budget = attention_budget
        self.max_windows = max_windows
        self.shuffle = shuffle
        self.seed = seed
        self.cell_batch_group = cell_batch_group
        self.balance_classes = balance_classes
        self.class_balance_power = class_balance_power
        self.epoch = 0
        self._plan: list[list[int]] | None = None
        by_cell: dict[int, list[int]] = defaultdict(list)
        for index, root_id in enumerate(dataset.index_root_ids):
            by_cell[int(root_id)].append(index)
        self.by_cell = by_cell

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._plan = None

    def _build_plan(self) -> list[list[int]]:
        if self._plan is not None:
            return self._plan
        rng = random.Random(self.seed + self.epoch)
        by_cell = self.by_cell
        if self.balance_classes:
            # Match segCLR_cell_classification's hierarchical sampler: draw
            # len(dataset) windows with replacement using a 1/sqrt(class
            # count) weight by default, or 1/count for equal expected class shares.
            # Sampling happens before the
            # attention-aware packing, so balancing does not sacrifice the
            # variable-size batch budget or per-cell cache locality.
            labels = self.dataset.index_labels.astype(np.int64, copy=False)
            counts = np.bincount(labels)
            class_weights = np.zeros_like(counts, dtype=np.float64)
            present = counts > 0
            class_weights[present] = counts[present].astype(np.float64) ** (-self.class_balance_power)
            probabilities = class_weights[labels]
            probabilities /= probabilities.sum()
            sampled = np.random.default_rng(self.seed + self.epoch).choice(
                len(labels), size=len(labels), replace=True, p=probabilities
            )
            sampled_by_cell: dict[int, list[int]] = defaultdict(list)
            for index in sampled:
                sampled_by_cell[int(self.dataset.index_root_ids[index])].append(int(index))
            by_cell = sampled_by_cell

        cells = list(by_cell)
        if self.shuffle:
            rng.shuffle(cells)
        groups: list[list[list[int]]] = []
        for root_id in cells:
            cell_batches: list[list[int]] = []
            indices = list(by_cell[root_id])
            if self.shuffle:
                rng.shuffle(indices)  # randomizes ties before the stable sort
            # Exact sorting avoids padding every 129-node graph as though it
            # had 256 nodes, which the old power-of-two bands did.  Greedy
            # batches therefore use the requested attention budget closely.
            indices.sort(key=lambda index: int(self.dataset.window_sizes[index]))
            batch, largest = [], 0
            for index in indices:
                size = int(self.dataset.window_sizes[index]) + 1  # CLS
                proposed_largest = max(largest, size)
                if batch and (len(batch) >= self.max_windows or
                              (len(batch) + 1) * proposed_largest**2 > self.attention_budget):
                    cell_batches.append(batch)
                    batch, largest = [], 0
                batch.append(index); largest = max(largest, size)
            if batch:
                cell_batches.append(batch)
            # Preserve a few adjacent batches per cell to amortize loading its
            # compressed record, but never let a giant cell monopolize a long
            # run of single-class optimizer steps.
            groups.extend(cell_batches[start : start + self.cell_batch_group]
                          for start in range(0, len(cell_batches), self.cell_batch_group))
        if self.shuffle:
            rng.shuffle(groups)
        self._plan = [batch for group in groups for batch in group]
        return self._plan

    def __iter__(self):
        yield from self._build_plan()

    def __len__(self) -> int:
        # Exact: tqdm must not claim an epoch ended thousands of batches ago.
        return len(self._build_plan())
